"""
Pseudo Token + ST Gumbel-Softmax Training for Sa2VA

Training loop (3 model calls):
1. Step 1: pseudo_toks = EMA_Sa2VA(image1, mask1) - teacher, stop-grad
2. Step 2: pseudo_toks_masked = random_mask(pseudo_toks, ratio=0.25)
3. Step 3: logits = Trainable_Sa2VA(image1, mask1, pseudo_toks_masked) - output text logits
4. Step 4: text_embeds = STGumbel(logits, tau=0.7) @ E - get [B,64,D] differentiable text embedding
5. Step 5: mask2' = Trainable_Sa2VA(image2, inputs_embeds=text_embeds) - predict mask
6. Step 6: L = L_mask(mask2', mask2), backprop only to trainable model

Trainable modules: proj layer, LLM LoRA, SAM2 decoder
Frozen modules: vision encoder, LLM base weights

Datasets:
- SAV: image1+mask1 -> text_embeds -> image2+mask2
- SA1B/OpenImage: image1+mask1 -> text_embeds -> image1+mask1
- RefCOCO: Only referring segmentation (direct tokenization, no gumbel)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import random

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from projects.llava_sam2.models.llava_sam2 import VideoLLaVASAMModel
from projects.llava_sam2.models import SAM2TrainRunner
from projects.llava_sam2.models.internvl import InternVL_Slowfast
from projects.llava_sam2.rl_train.ema_model import EMAModel

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from transformers import AutoTokenizer, GenerationConfig
from peft import LoraConfig
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Pseudo Token + ST Gumbel-Softmax Training')
    parser.add_argument('--pretrained_pth', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='./pretrained/InternVL2_5-4B')
    parser.add_argument('--output_dir', type=str, default='./work_dirs/pseudo_gumbel')

    # Dataset paths
    parser.add_argument('--sav_dir', type=str, default=None)
    parser.add_argument('--sa1b_dir', type=str, default=None)
    parser.add_argument('--openimage_dir', type=str, default=None)
    parser.add_argument('--refcoco_dir', type=str, default=None)
    parser.add_argument('--sa1b_max_samples', type=int, default=None)
    parser.add_argument('--sav_max_samples', type=int, default=None)
    parser.add_argument('--sav_repeats', type=int, default=1)
    parser.add_argument('--refcoco_repeats', type=int, default=1)

    # Training params
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--ema_decay', type=float, default=0.999)

    # Gumbel-Softmax params
    parser.add_argument('--gumbel_tau', type=float, default=0.7)
    parser.add_argument('--topk', type=int, default=128, help='Top-k for sparse Gumbel-Softmax')
    parser.add_argument('--mask_ratio', type=float, default=0.25, help='Ratio of tokens to mask')
    parser.add_argument('--max_caption_len', type=int, default=64)

    # LoRA params (same as sa2va_4b.py)
    parser.add_argument('--lora_r', type=int, default=128)
    parser.add_argument('--lora_alpha', type=int, default=256)

    # Other params
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()
    return args


def build_model(args):
    """Build VideoLLaVASAMModel with correct trainable parameters."""
    print(f"Building VideoLLaVASAMModel from {args.model_path}...")

    special_tokens = [
        '[SEG]', '<p>', '</p>',
        '<vp>', '</vp>',
        '<IMG_CONTEXT>',
        '<img>', '</img>'
    ]

    tokenizer_cfg = dict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=args.model_path,
        trust_remote_code=True,
        padding_side='right'
    )
    tokenizer = BUILDER.build(tokenizer_cfg)

    model_cfg = dict(
        type=VideoLLaVASAMModel,
        special_tokens=special_tokens,
        frozen_sam2_decoder=False,  # SAM2 decoder trainable
        mllm=dict(
            type=InternVL_Slowfast,
            model_path=args.model_path,
            freeze_llm=True,  # LLM base frozen, LoRA trainable
            freeze_visual_encoder=True,  # Vision encoder frozen
            llm_lora=dict(
                type=LoraConfig,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.05,
                bias='none',
                task_type='CAUSAL_LM'
            ),
            special_tokens=special_tokens,
        ),
        tokenizer=tokenizer_cfg,
        grounding_encoder=dict(type=SAM2TrainRunner),
        loss_mask=dict(type=CrossEntropyLoss, use_sigmoid=True, reduction='mean', loss_weight=2.0),
        loss_dice=dict(type=DiceLoss, use_sigmoid=True, activate=True, reduction='mean',
                       naive_dice=True, eps=1.0, loss_weight=0.5),
        pretrained_pth=None,
        loss_sample_points=True,
        bs=args.batch_size,
    )

    model = BUILDER.build(model_cfg)
    print("✓ Model built")

    # Load pretrained weights
    if args.pretrained_pth and os.path.exists(args.pretrained_pth):
        if os.path.exists(os.path.join(args.pretrained_pth, 'model.safetensors.index.json')):
            print(f"Loading from HuggingFace format: {args.pretrained_pth}")
            try:
                from projects.llava_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel
                hf_model = Sa2VAChatModel.from_pretrained(
                    args.pretrained_pth, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
                model.load_state_dict(hf_model.state_dict(), strict=False)
                print("✓ Pretrained weights loaded from HF")
                del hf_model
            except Exception as e:
                print(f"Warning: Failed to load HF model: {e}")
        else:
            try:
                print(f"Loading from checkpoint: {args.pretrained_pth}")
                checkpoint = guess_load_checkpoint(args.pretrained_pth)
                model.load_state_dict(checkpoint, strict=False)
                print("✓ Pretrained weights loaded from checkpoint")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint: {e}")

    return model, tokenizer


def print_trainable_parameters(model, name="Model"):
    """Print trainable parameters to confirm correct freezing."""
    trainable_params = 0
    all_params = 0
    trainable_names = []

    for n, p in model.named_parameters():
        all_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
            trainable_names.append(n)

    print(f"\n{'='*60}")
    print(f"{name} Trainable Parameters:")
    print(f"{'='*60}")
    print(f"Total params: {all_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / all_params:.2f}%")
    print(f"\nTrainable parameter groups:")

    # Group by module
    groups = {}
    for n in trainable_names:
        parts = n.split('.')
        group = parts[0] if len(parts) > 0 else 'root'
        if 'lora' in n.lower():
            group = 'LoRA'
        elif 'text_hidden_fcs' in n:
            group = 'text_hidden_fcs (proj)'
        elif 'sam_mask_decoder' in n:
            group = 'SAM2 decoder'
        elif 'mlp1' in n:
            group = 'mlp1 (proj)'
        if group not in groups:
            groups[group] = 0
        groups[group] += 1

    for g, count in groups.items():
        print(f"  {g}: {count} parameters")
    print(f"{'='*60}\n")


def build_datasets(args):
    """Build datasets with correct OpenImage path."""
    from projects.llava_sam2.mask_caption_sft.dataset_builder import build_mask_caption_dataset

    # OpenImage: CORRECT path is /data/xyc/openv7/data
    openimage_config = None
    if args.openimage_dir and os.path.exists(args.openimage_dir):
        openimage_config = {
            'annotation_csv': os.path.join(args.openimage_dir, 'train-annotations-object-segmentation.csv'),
            'label_csv': os.path.join(args.openimage_dir, 'oidv7-class-descriptions.csv'),
            'image_dir': os.path.join(args.openimage_dir, 'images', 'train'),
            'mask_dir': os.path.join(args.openimage_dir, 'masks', 'train'),
        }
        print(f"✓ OpenImage: {args.openimage_dir}")
    elif args.openimage_dir:
        print(f"⚠ OpenImage not found: {args.openimage_dir}")

    refcoco_config = None
    if args.refcoco_dir:
        refcoco_config = {
            'data_root': args.refcoco_dir,
            'split': 'train',
            'dataset_name': 'refcoco',
        }

    dataset = build_mask_caption_dataset(
        sav_dir=args.sav_dir,
        sa1b_dir=args.sa1b_dir,
        openimage_config=openimage_config,
        refcoco_config=refcoco_config,
        target_size=(1024, 1024),
        sa1b_max_samples=args.sa1b_max_samples,
        sav_max_samples=args.sav_max_samples,
        sav_repeats=args.sav_repeats,
        refcoco_repeats=args.refcoco_repeats,
    )

    print(f"✓ Dataset: {len(dataset)} samples")
    return dataset


class PseudoGumbelTrainer:
    """
    Trainer implementing the pseudo token + Gumbel-Softmax training loop.

    Three model calls per step:
    1. EMA generates pseudo tokens (stop-grad)
    2. Trainable produces text logits given masked pseudo tokens
    3. Trainable predicts mask from Gumbel-softmax text embeddings
    """

    def __init__(self, model, ema_model, tokenizer, train_dataloader,
                 optimizer, device, output_dir, args):
        self.model = model
        self.ema_model = ema_model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.args = args

        self.global_step = 0
        self.gumbel_tau = args.gumbel_tau
        self.topk = args.topk
        self.mask_ratio = args.mask_ratio
        self.max_caption_len = args.max_caption_len

        # Get actual model (handle DDP)
        self.actual_model = model.module if hasattr(model, 'module') else model
        self.ema_actual = ema_model.model

        # Special tokens
        self.seg_token_id = self.actual_model.seg_token_idx
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.img_context_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')

        # Embedding layer for Gumbel-Softmax
        self.embedding_layer = self.actual_model.mllm.model.language_model.get_input_embeddings()
        self.vocab_size = self.actual_model.mllm.model.language_model.config.vocab_size
        self.hidden_size = self.embedding_layer.embedding_dim

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))

        # For gradient checking
        self.text_embeds_grad = None

        print(f"✓ Trainer initialized")
        print(f"  Gumbel tau: {self.gumbel_tau}, Top-k: {self.topk}")
        print(f"  Mask ratio: {self.mask_ratio}, Max caption len: {self.max_caption_len}")
        print(f"  Vocab: {self.vocab_size}, Hidden: {self.hidden_size}")

    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        self.ema_model.eval()

        epoch_losses = []
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(pbar):
            if batch is None:
                continue

            # Move to device
            images1 = batch['image1'].to(self.device, dtype=torch.bfloat16)
            masks1 = batch['mask1'].to(self.device, dtype=torch.bfloat16)

            if batch['image2'] is not None:
                images2 = batch['image2'].to(self.device, dtype=torch.bfloat16)
                masks2 = batch['mask2'].to(self.device, dtype=torch.bfloat16)
            else:
                images2 = images1
                masks2 = masks1

            dataset_types = batch.get('dataset_types', ['unknown'] * images1.shape[0])
            captions_gt = batch.get('captions', None)

            # Check if RefCOCO (has GT caption)
            is_refcoco = captions_gt is not None and len(captions_gt) > 0 and captions_gt[0] is not None

            if is_refcoco:
                # RefCOCO: Direct referring segmentation, no pseudo-token loop
                loss_dict = self.refcoco_step(images2, captions_gt, masks2)
            else:
                # Full pseudo-token + Gumbel loop
                loss_dict = self.pseudo_gumbel_step(images1, masks1, images2, masks2)

            if loss_dict is None:
                continue

            loss = loss_dict['loss']

            # Backward
            loss_scaled = loss / self.args.gradient_accumulation_steps
            loss_scaled.backward()

            # Gradient sanity checks
            if self.global_step % self.args.log_interval == 0:
                self.check_gradients()

            # Update weights
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update EMA
                student = self.model.module if hasattr(self.model, 'module') else self.model
                self.ema_model.update(student)

                self.global_step += 1

            # Logging
            epoch_losses.append(loss.item())
            if self.global_step % self.args.log_interval == 0 and self.global_step > 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/mask_loss', loss_dict.get('mask_loss', 0), self.global_step)
                self.writer.add_scalar('train/dice_loss', loss_dict.get('dice_loss', 0), self.global_step)

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'mask': f"{loss_dict.get('mask_loss', 0):.4f}",
                    'dice': f"{loss_dict.get('dice_loss', 0):.4f}",
                })

            # Save
            if self.global_step > 0 and self.global_step % self.args.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pth")

        return {'loss': sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0}

    def pseudo_gumbel_step(self, images1, masks1, images2, masks2):
        """
        Full pseudo-token + Gumbel-Softmax training step.

        Step 1: EMA generates pseudo_toks (stop-grad)
        Step 2: Random mask pseudo_toks (25%)
        Step 3: Trainable produces logits from masked pseudo_toks
        Step 4: Top-k Gumbel-Softmax → text_embeds
        Step 5: Trainable predicts mask from text_embeds
        Step 6: Mask loss, backprop
        """
        batch_size = images1.shape[0]

        # ============ Step 1: EMA generates pseudo tokens ============
        with torch.no_grad():
            pseudo_toks = self.generate_pseudo_tokens(images1, masks1)
            # pseudo_toks: [B, max_caption_len]

        # ============ Step 2: Random mask 25% of pseudo tokens ============
        pseudo_toks_masked = self.random_mask_tokens(pseudo_toks)

        # ============ Step 3: Trainable produces logits ============
        # Input: image1 + mask1 + pseudo_toks_masked
        # Output: logits [B, max_caption_len, V]
        logits = self.trainable_forward_logits(images1, masks1, pseudo_toks_masked)

        # ============ Step 4: Top-k Gumbel-Softmax ============
        # Get text_embeds [B, max_caption_len, D]
        text_embeds = self.topk_gumbel_softmax(logits)

        # Register hook for gradient checking
        text_embeds.retain_grad()
        self.text_embeds_ref = text_embeds

        # ============ Step 5: Trainable predicts mask from text_embeds ============
        loss_dict = self.trainable_forward_mask(images2, text_embeds, masks2)

        return loss_dict

    @torch.no_grad()
    def generate_pseudo_tokens(self, images, masks):
        """
        Step 1: EMA model generates pseudo tokens from image + mask.

        Uses visual prompting: image + <vp>mask</vp> -> caption tokens
        """
        batch_size = images.shape[0]

        # Prepare images for vision encoder (448x448)
        images_448 = F.interpolate(images, size=(448, 448), mode='bilinear', align_corners=False)

        # Prepare visual prompt masks
        GRID_SIZE = 16
        masks_unsqueezed = masks.unsqueeze(1)
        masks_448 = F.interpolate(masks_unsqueezed, size=(448, 448), mode='nearest')
        pooled = F.adaptive_avg_pool2d(masks_448, (GRID_SIZE, GRID_SIZE))
        prompt_masks = (pooled > 0.5).to(torch.uint8).squeeze(1)  # [B, 16, 16]

        # Build input prompt
        IMG_CONTEXT = '<IMG_CONTEXT>'
        NUM_IMG_TOKENS = 256

        input_ids_list = []
        for i in range(batch_size):
            K = max(int(prompt_masks[i].sum().item()), 1)
            img_str = f'<img>{IMG_CONTEXT * NUM_IMG_TOKENS}</img>'
            vp_str = f'<vp>{IMG_CONTEXT * K}</vp>'
            prompt = f"<|user|>\n{img_str} Region: {vp_str}. Describe this region briefly.<|end|>\n<|assistant|>\n"
            ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            input_ids_list.append(ids)

        # Pad
        max_len = max(len(ids) for ids in input_ids_list)
        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)

        for i, ids in enumerate(input_ids_list):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            attention_mask[i, :len(ids)] = True

        # Prepare pixel values
        pixel_values = [images_448[i] for i in range(batch_size)]
        prompt_masks_list = [prompt_masks[i:i+1] for i in range(batch_size)]
        vp_overall_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        # Generate with EMA
        gen_config = GenerationConfig(
            max_new_tokens=self.max_caption_len,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

        outputs = self.ema_actual.mllm.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_masks=prompt_masks_list,
            vp_overall_mask=vp_overall_mask,
            generation_config=gen_config,
        )

        # Extract generated tokens (remove prompt)
        prompt_len = input_ids.shape[1]
        generated = outputs[:, prompt_len:]

        # Pad/truncate to max_caption_len
        if generated.shape[1] < self.max_caption_len:
            pad_len = self.max_caption_len - generated.shape[1]
            generated = F.pad(generated, (0, pad_len), value=self.pad_token_id)
        else:
            generated = generated[:, :self.max_caption_len]

        return generated  # [B, max_caption_len]

    def random_mask_tokens(self, tokens):
        """
        Step 2: Randomly mask tokens with mask_ratio.
        Replace masked tokens with [MASK] token or random token.
        """
        masked = tokens.clone()
        batch_size, seq_len = tokens.shape

        # Create mask
        mask = torch.rand(batch_size, seq_len, device=self.device) < self.mask_ratio

        # Don't mask padding
        mask = mask & (tokens != self.pad_token_id)

        # Replace with random tokens
        random_tokens = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        masked[mask] = random_tokens[mask]

        return masked

    def trainable_forward_logits(self, images, masks, pseudo_toks_masked):
        """
        Step 3: Trainable model produces logits given image + mask + masked pseudo tokens.

        This is a conditional generation: given the masked tokens, predict the original tokens.
        The output logits will be used for Gumbel-Softmax.
        """
        batch_size = images.shape[0]

        # Prepare images
        images_448 = F.interpolate(images, size=(448, 448), mode='bilinear', align_corners=False)

        # Prepare visual prompt masks
        GRID_SIZE = 16
        masks_unsqueezed = masks.unsqueeze(1)
        masks_448 = F.interpolate(masks_unsqueezed, size=(448, 448), mode='nearest')
        pooled = F.adaptive_avg_pool2d(masks_448, (GRID_SIZE, GRID_SIZE))
        prompt_masks = (pooled > 0.5).to(torch.uint8).squeeze(1)

        # Build input: image + mask + masked_pseudo_tokens
        IMG_CONTEXT = '<IMG_CONTEXT>'
        NUM_IMG_TOKENS = 256

        input_ids_list = []
        for i in range(batch_size):
            K = max(int(prompt_masks[i].sum().item()), 1)
            img_str = f'<img>{IMG_CONTEXT * NUM_IMG_TOKENS}</img>'
            vp_str = f'<vp>{IMG_CONTEXT * K}</vp>'

            # Decode masked tokens to string (for context)
            masked_text = self.tokenizer.decode(pseudo_toks_masked[i], skip_special_tokens=True)

            prompt = f"<|user|>\n{img_str} Region: {vp_str}. The description is: {masked_text}. Complete:<|end|>\n<|assistant|>\n"
            ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            input_ids_list.append(ids)

        # Pad
        max_len = max(len(ids) for ids in input_ids_list)
        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)

        for i, ids in enumerate(input_ids_list):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            attention_mask[i, :len(ids)] = True

        position_ids = torch.arange(max_len, device=self.device).unsqueeze(0).expand(batch_size, -1)

        # Prepare data dict
        pixel_values = [images_448[i] for i in range(batch_size)]
        prompt_masks_list = [prompt_masks[i:i+1] for i in range(batch_size)]
        vp_overall_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        data = {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'labels': None,  # No LM loss
            'vp_overall_mask': vp_overall_mask,
            'prompt_masks': prompt_masks_list,
        }

        # Forward through trainable model's LLM
        output = self.actual_model.mllm(data, data_samples=None, mode='loss')
        logits = output.logits  # [B, seq_len, V]

        # Get logits for caption positions (last max_caption_len positions)
        caption_logits = logits[:, -self.max_caption_len:, :]  # [B, max_caption_len, V]

        return caption_logits

    def topk_gumbel_softmax(self, logits):
        """
        Step 4: Top-k sparse Gumbel-Softmax.

        Args:
            logits: [B, T, V] - full vocabulary logits

        Returns:
            text_embeds: [B, T, D] - differentiable text embeddings
        """
        B, T, V = logits.shape
        k = self.topk

        # Get top-k logits and indices
        topk_vals, topk_idx = logits.topk(k, dim=-1)  # [B, T, k]

        # Gumbel-Softmax on top-k
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(topk_vals) + 1e-10) + 1e-10)
        y_soft = F.softmax((topk_vals + gumbel_noise) / self.gumbel_tau, dim=-1)  # [B, T, k]

        # Straight-through: hard forward, soft backward
        index = y_soft.argmax(dim=-1, keepdim=True)  # [B, T, 1]
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        y = y_hard - y_soft.detach() + y_soft  # [B, T, k]

        # Get embeddings for top-k tokens
        E = self.embedding_layer.weight  # [V, D]
        E_topk = E[topk_idx]  # [B, T, k, D]

        # Weighted sum
        text_embeds = (y.unsqueeze(-1) * E_topk).sum(dim=-2)  # [B, T, D]

        return text_embeds

    def trainable_forward_mask(self, images, text_embeds, gt_masks):
        """
        Step 5: Trainable model predicts mask from image + text_embeds.

        CRITICAL: Must use inputs_embeds, not input_ids!
        """
        batch_size = images.shape[0]

        # Prepare images
        images_448 = F.interpolate(images, size=(448, 448), mode='bilinear', align_corners=False)

        # Build prompt template for referring segmentation
        IMG_CONTEXT = '<IMG_CONTEXT>'
        NUM_IMG_TOKENS = 256

        # Template: "<img>...</img> Please segment: [text_embeds] [SEG]"
        prefix = f"<|user|>\n<img>{IMG_CONTEXT * NUM_IMG_TOKENS}</img>Please segment: "
        suffix = " [SEG]<|end|>\n<|assistant|>\nIt is [SEG]."

        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=True)
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)

        caption_len = text_embeds.shape[1]  # T
        total_len = len(prefix_ids) + caption_len + len(suffix_ids)

        # Build input_ids (placeholders for text_embeds positions)
        input_ids = torch.zeros(batch_size, total_len, dtype=torch.long, device=self.device)
        labels = torch.full((batch_size, total_len), -100, dtype=torch.long, device=self.device)

        for i in range(batch_size):
            input_ids[i, :len(prefix_ids)] = torch.tensor(prefix_ids, device=self.device)
            # Placeholder for text_embeds (will be replaced in embeddings)
            input_ids[i, len(prefix_ids):len(prefix_ids)+caption_len] = self.pad_token_id
            input_ids[i, len(prefix_ids)+caption_len:] = torch.tensor(suffix_ids, device=self.device)

            # Labels: only supervise [SEG] output
            labels[i, len(prefix_ids)+caption_len:] = torch.tensor(suffix_ids, device=self.device)

        attention_mask = torch.ones(batch_size, total_len, dtype=torch.bool, device=self.device)
        position_ids = torch.arange(total_len, device=self.device).unsqueeze(0).expand(batch_size, -1)

        # Get base embeddings
        base_embeds = self.embedding_layer(input_ids)  # [B, total_len, D]

        # Replace text_embeds positions
        soft_start = len(prefix_ids)
        soft_end = soft_start + caption_len
        input_embeds = base_embeds.clone()
        input_embeds[:, soft_start:soft_end, :] = text_embeds

        # Replace image token positions with vision embeddings
        concat_images = torch.stack([images_448[i] for i in range(batch_size)], dim=0)
        vit_embeds = self.actual_model.mllm.model.extract_feature(concat_images)
        vit_embeds = vit_embeds.to(input_embeds.dtype)

        # Find IMG_CONTEXT positions
        B, N, C = input_embeds.shape
        input_embeds_flat = input_embeds.reshape(B * N, C)
        input_ids_flat = input_ids.reshape(B * N)
        selected = (input_ids_flat == self.img_context_id)

        n_selected = selected.sum()
        vit_flat = vit_embeds.reshape(-1, C)
        if n_selected > len(vit_flat):
            expand = n_selected // len(vit_flat) + 1
            vit_flat = torch.cat([vit_flat] * expand, dim=0)
        input_embeds_flat[selected] = vit_flat[:n_selected]
        input_embeds = input_embeds_flat.reshape(B, N, C)

        # Forward through LLM with inputs_embeds
        outputs = self.actual_model.mllm.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get [SEG] token hidden states
        hidden_states = outputs.hidden_states[-1]
        hidden_states_transformed = self.actual_model.text_hidden_fcs(hidden_states)

        # Find [SEG] positions
        seg_mask = input_ids == self.seg_token_id
        seg_counts = seg_mask.int().sum(-1)

        if seg_counts.sum() == 0:
            return {'loss': torch.tensor(0.0, device=self.device, requires_grad=True),
                    'mask_loss': 0.0, 'dice_loss': 0.0}

        pred_embeddings = hidden_states_transformed[seg_mask]

        # SAM2 forward
        g_pixel_values = torch.stack([
            self.actual_model.grounding_encoder.preprocess_image(images[i])
            for i in range(batch_size)
        ])

        num_objs = 1
        sam_states = self.actual_model.grounding_encoder.get_sam2_embeddings(
            g_pixel_values, expand_size=num_objs)

        # Prepare language embeddings
        pred_list = torch.split(pred_embeddings, seg_counts.tolist(), dim=0)
        pred_list = [item for item in pred_list if len(item) > 0]

        if len(pred_list) == 0:
            return {'loss': torch.tensor(0.0, device=self.device, requires_grad=True),
                    'mask_loss': 0.0, 'dice_loss': 0.0}

        lang_embeds = torch.stack([emb[0] for emb in pred_list], dim=0)[:, None]

        # Predict masks
        pred_masks = self.actual_model.grounding_encoder.inject_language_embd(
            sam_states, lang_embeds, nf_nobj=(batch_size, num_objs))

        # Compute loss
        pred_size = pred_masks.shape[-2:]
        gt_resized = F.interpolate(gt_masks.unsqueeze(1).float(), size=pred_size, mode='nearest').squeeze(1)

        pred_flat = pred_masks.flatten(0, 1).squeeze(1)
        loss_mask = F.binary_cross_entropy_with_logits(pred_flat, gt_resized, reduction='mean') * 2.0

        pred_prob = torch.sigmoid(pred_flat)
        inter = (pred_prob * gt_resized).sum(dim=(1, 2))
        union = pred_prob.sum(dim=(1, 2)) + gt_resized.sum(dim=(1, 2))
        dice = (2 * inter + 1.0) / (union + 1.0)
        loss_dice = (1 - dice).mean() * 0.5

        total_loss = loss_mask + loss_dice

        return {
            'loss': total_loss,
            'mask_loss': loss_mask.item(),
            'dice_loss': loss_dice.item(),
        }

    def refcoco_step(self, images, captions, gt_masks):
        """RefCOCO: Direct referring segmentation with GT caption.

        Process one sample at a time to avoid batch issues with frames_per_batch.
        """
        batch_size = images.shape[0]
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_mask_loss = 0.0
        total_dice_loss = 0.0
        count = 0

        for i in range(batch_size):
            try:
                image = images[i:i+1]  # [1, 3, H, W]
                caption = captions[i]
                gt_mask = gt_masks[i:i+1]  # [1, H, W]

                image_448 = F.interpolate(image, size=(448, 448), mode='bilinear', align_corners=False)

                IMG_CONTEXT = '<IMG_CONTEXT>'
                NUM_IMG_TOKENS = 256

                img_str = f'<img>{IMG_CONTEXT * NUM_IMG_TOKENS}</img>'
                user_input = f"{img_str}Please segment: {caption}"
                user_text = f"<|user|>\n{user_input}<|end|>\n<|assistant|>\n"
                output_text = "It is [SEG].<|end|>"

                user_ids = self.tokenizer.encode(user_text, add_special_tokens=True)
                output_ids = self.tokenizer.encode(output_text, add_special_tokens=False)

                full_ids = user_ids + output_ids
                full_labels = [-100] * len(user_ids) + output_ids

                seq_len = len(full_ids)
                input_ids = torch.zeros(1, seq_len, dtype=torch.long, device=self.device)
                labels = torch.full((1, seq_len), -100, dtype=torch.long, device=self.device)
                attention_mask = torch.ones(1, seq_len, dtype=torch.bool, device=self.device)

                input_ids[0] = torch.tensor(full_ids)
                labels[0] = torch.tensor(full_labels)

                position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

                pixel_values = [image_448[0]]
                g_pixel_values = [image[0]]
                masks_list = [gt_mask]  # List of [1, H, W]

                data = {
                    'pixel_values': pixel_values,
                    'g_pixel_values': g_pixel_values,
                    'input_ids': input_ids,
                    'position_ids': position_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                    'masks': masks_list,
                    'frames_per_batch': [1],  # Single image
                }

                loss_dict = self.actual_model(data, data_samples=None, mode='loss')

                sample_loss = (
                    loss_dict.get('loss_mask', 0) +
                    loss_dict.get('loss_dice', 0) +
                    loss_dict.get('llm_loss', 0)
                )

                total_loss = total_loss + sample_loss
                total_mask_loss += loss_dict.get('loss_mask', torch.tensor(0.0)).item() if torch.is_tensor(loss_dict.get('loss_mask', 0)) else 0
                total_dice_loss += loss_dict.get('loss_dice', torch.tensor(0.0)).item() if torch.is_tensor(loss_dict.get('loss_dice', 0)) else 0
                count += 1

            except (NotImplementedError, RuntimeError) as e:
                # Skip failed samples (e.g., when genetate_video_pred_embeddings fails)
                print(f"Warning: RefCOCO sample {i} failed with {type(e).__name__}: {e}, skipping...")
                continue

        if count == 0:
            # All samples failed, return zero loss
            return {
                'loss': torch.tensor(0.0, device=self.device, requires_grad=True),
                'mask_loss': 0.0,
                'dice_loss': 0.0,
            }

        # Average over successful samples
        avg_loss = total_loss / count
        avg_mask_loss = total_mask_loss / count
        avg_dice_loss = total_dice_loss / count

        return {
            'loss': avg_loss,
            'mask_loss': avg_mask_loss,
            'dice_loss': avg_dice_loss,
        }

    def check_gradients(self):
        """Sanity check: verify gradients flow through Gumbel-Softmax."""
        # Check 1: text_embeds gradient
        if hasattr(self, 'text_embeds_ref') and self.text_embeds_ref is not None:
            if self.text_embeds_ref.grad is not None:
                grad_norm = self.text_embeds_ref.grad.norm().item()
                self.writer.add_scalar('grad/text_embeds_norm', grad_norm, self.global_step)
                if grad_norm == 0:
                    print(f"⚠ Warning step {self.global_step}: text_embeds grad is ZERO!")
            else:
                print(f"⚠ Warning step {self.global_step}: text_embeds.grad is None!")

        # Check 2: LoRA gradients
        lora_grad_norm = 0
        lora_count = 0
        for n, p in self.actual_model.named_parameters():
            if 'lora' in n.lower() and p.grad is not None:
                lora_grad_norm += p.grad.norm().item()
                lora_count += 1

        if lora_count > 0:
            avg_lora_grad = lora_grad_norm / lora_count
            self.writer.add_scalar('grad/lora_avg_norm', avg_lora_grad, self.global_step)
            if avg_lora_grad == 0:
                print(f"⚠ Warning step {self.global_step}: LoRA grad is ZERO!")

        # Check 3: SAM2 decoder gradients
        sam_grad_norm = 0
        sam_count = 0
        for n, p in self.actual_model.grounding_encoder.named_parameters():
            if 'sam_mask_decoder' in n and p.grad is not None:
                sam_grad_norm += p.grad.norm().item()
                sam_count += 1

        if sam_count > 0:
            avg_sam_grad = sam_grad_norm / sam_count
            self.writer.add_scalar('grad/sam2_decoder_avg_norm', avg_sam_grad, self.global_step)

        # Check 4: Projector (mlp1, text_hidden_fcs) gradients
        proj_grad_norm = 0
        proj_count = 0
        for n, p in self.actual_model.named_parameters():
            if ('mlp1' in n or 'text_hidden_fcs' in n) and p.grad is not None:
                proj_grad_norm += p.grad.norm().item()
                proj_count += 1

        if proj_count > 0:
            avg_proj_grad = proj_grad_norm / proj_count
            self.writer.add_scalar('grad/proj_avg_norm', avg_proj_grad, self.global_step)

    def save_checkpoint(self, filename):
        """Save checkpoint."""
        import torch.distributed as dist

        if hasattr(self.model, 'module'):
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank != 0:
                return
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        path = self.output_dir / filename
        torch.save({
            'model': model_state,
            'ema_model': self.ema_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
        }, path)
        print(f"✓ Saved: {path}")


def main():
    args = parse_args()

    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    args.local_rank = local_rank

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{args.local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")

    # Build model
    model, tokenizer = build_model(args)
    model = model.to(device, dtype=torch.bfloat16)

    # Print trainable parameters
    print_trainable_parameters(model, "VideoLLaVASAMModel")

    # DDP
    if args.local_rank != -1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,
                    broadcast_buffers=False, static_graph=True)
        print(f"✓ DDP on rank {args.local_rank}")

    # EMA
    print("Creating EMA model...")
    ema_model = EMAModel(model.module if hasattr(model, 'module') else model, decay=args.ema_decay)
    ema_model = ema_model.to(device)
    print(f"✓ EMA model created")

    # Dataset
    train_dataset = build_datasets(args)

    from projects.llava_sam2.mask_caption_sft.dataset_builder import collate_fn_mask_caption
    from torch.utils.data.distributed import DistributedSampler

    if args.local_rank != -1:
        sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        shuffle=shuffle, num_workers=args.num_workers,
        collate_fn=collate_fn_mask_caption, pin_memory=True)
    print(f"✓ Dataloader: {len(train_loader)} batches")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.05)
    print("✓ Optimizer")

    # Trainer
    trainer = PseudoGumbelTrainer(
        model=model, ema_model=ema_model, tokenizer=tokenizer,
        train_dataloader=train_loader, optimizer=optimizer,
        device=device, output_dir=args.output_dir, args=args)

    # Train
    print("=" * 80)
    print("Starting Pseudo Token + ST Gumbel-Softmax Training")
    print("=" * 80)

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        if args.local_rank != -1:
            train_loader.sampler.set_epoch(epoch)

        results = trainer.train_epoch(epoch)
        print(f"Epoch {epoch + 1} done. Loss: {results['loss']:.4f}")

    print("=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
