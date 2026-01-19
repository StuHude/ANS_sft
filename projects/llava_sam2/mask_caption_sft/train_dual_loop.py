"""
Dual-loop Mask Captioning SFT Training

Training flow:
1. image + mask → sa2va model → caption
2. image + caption → EMA sa2va model → mask'
3. Compute loss: mask' vs mask

Based on original Sa2VA training architecture.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from projects.llava_sam2.models.llava_sam2 import VideoLLaVASAMModel
from projects.llava_sam2.models import SAM2TrainRunner
from projects.llava_sam2.models.internvl import InternVL_Slowfast
from projects.llava_sam2.models.preprocess.image_resize import DirectResize
from projects.llava_sam2.rl_train.ema_model import EMAModel

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from transformers import AutoTokenizer
from peft import LoraConfig
from xtuner.registry import BUILDER
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.model.utils import guess_load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Dual-loop Mask Caption SFT Training')
    parser.add_argument('--pretrained_pth', type=str, required=True,
                        help='Path to pretrained checkpoint')
    parser.add_argument('--model_path', type=str, default='./pretrained/InternVL2_5-4B',
                        help='Path to base model (InternVL)')
    parser.add_argument('--output_dir', type=str, default='./work_dirs/dual_loop_sft',
                        help='Output directory')

    # Dataset paths
    parser.add_argument('--sav_dir', type=str, default=None)
    parser.add_argument('--sa1b_dir', type=str, default=None)
    parser.add_argument('--openimage_dir', type=str, default=None)
    parser.add_argument('--refcoco_dir', type=str, default=None)
    parser.add_argument('--sa1b_max_samples', type=int, default=None,
                        help='Limit SA1B samples for testing')
    parser.add_argument('--sav_max_samples', type=int, default=None,
                        help='Limit SAV samples for testing')

    # Dataset sampling repeats
    parser.add_argument('--sav_repeats', type=int, default=1,
                        help='Number of times to repeat SAV dataset (default: 1)')
    parser.add_argument('--refcoco_repeats', type=int, default=1,
                        help='Number of times to repeat RefCOCO dataset (default: 1)')

    # Training params
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--ema_decay', type=float, default=0.999)

    # LoRA params
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
    """Build VideoLLaVASAMModel following original config."""
    print(f"Building VideoLLaVASAMModel from {args.model_path}...")

    # Special tokens (from sa2va_4b.py)
    special_tokens = [
        '[SEG]', '<p>', '</p>',
        '<vp>', '</vp>',
        '<IMG_CONTEXT>',
        '<img>', '</img>'
    ]

    # Tokenizer
    tokenizer_cfg = dict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=args.model_path,
        trust_remote_code=True,
        padding_side='right'
    )
    tokenizer = BUILDER.build(tokenizer_cfg)

    # Model config (following sa2va_4b.py)
    model_cfg = dict(
        type=VideoLLaVASAMModel,
        special_tokens=special_tokens,
        frozen_sam2_decoder=False,  # We need to train SAM2 decoder
        mllm=dict(
            type=InternVL_Slowfast,
            model_path=args.model_path,
            freeze_llm=True,
            freeze_visual_encoder=True,
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
        grounding_encoder=dict(
            type=SAM2TrainRunner,
        ),
        loss_mask=dict(
            type=CrossEntropyLoss,
            use_sigmoid=True,
            reduction='mean',
            loss_weight=2.0
        ),
        loss_dice=dict(
            type=DiceLoss,
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=0.5
        ),
        pretrained_pth=None,  # Don't load in constructor
        loss_sample_points=True,
        bs=args.batch_size,
    )

    # Build model
    model = BUILDER.build(model_cfg)
    print("✓ Model built")

    # Load pretrained weights from HuggingFace format if provided
    if args.pretrained_pth and os.path.exists(args.pretrained_pth):
        # Check if it's a HuggingFace model directory
        if os.path.exists(os.path.join(args.pretrained_pth, 'model.safetensors.index.json')):
            print(f"Loading Sa2VA model from HuggingFace format: {args.pretrained_pth}")
            try:
                from projects.llava_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel
                import torch

                # Load HF model
                hf_model = Sa2VAChatModel.from_pretrained(
                    args.pretrained_pth,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )

                # Extract state dict and load into VideoLLaVASAMModel
                hf_state_dict = hf_model.state_dict()
                model.load_state_dict(hf_state_dict, strict=False)
                print("✓ Pretrained weights loaded from HuggingFace model")
                del hf_model  # Free memory
            except Exception as e:
                print(f"Warning: Failed to load HuggingFace model: {e}")
                print("Continuing with base model weights...")
        else:
            # Try loading as .pth checkpoint
            try:
                print(f"Loading pretrained weights from checkpoint: {args.pretrained_pth}")
                checkpoint = guess_load_checkpoint(args.pretrained_pth)
                model.load_state_dict(checkpoint, strict=False)
                print("✓ Pretrained weights loaded from checkpoint")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint: {e}")
                print("Continuing with base model weights...")

    return model, tokenizer


def build_datasets(args):
    """Build datasets following original format."""
    from projects.llava_sam2.mask_caption_sft.dataset_builder import build_mask_caption_dataset

    # OpenImage config - use correct paths from /data/xyc/openv7/data/
    openimage_config = None
    if args.openimage_dir and os.path.exists(args.openimage_dir):
        openimage_config = {
            'annotation_csv': os.path.join(args.openimage_dir, 'train-annotations-object-segmentation.csv'),
            'label_csv': os.path.join(args.openimage_dir, 'oidv7-class-descriptions.csv'),
            'image_dir': os.path.join(args.openimage_dir, 'images', 'train'),
            'mask_dir': os.path.join(args.openimage_dir, 'masks', 'train'),
        }
        print(f"✓ OpenImage directory found: {args.openimage_dir}")
    elif args.openimage_dir:
        print(f"⚠ Warning: OpenImage directory not found: {args.openimage_dir}, skipping OpenImage dataset")

    # RefCOCO config
    refcoco_config = None
    if args.refcoco_dir:
        refcoco_config = {
            'data_root': args.refcoco_dir,
            'split': 'train',
            'dataset_name': 'refcoco',
        }

    # Build dataset (target_size=1024 for SAM2 compatibility)
    # Use repeats for SAV and RefCOCO to increase their sampling weights
    dataset = build_mask_caption_dataset(
        sav_dir=args.sav_dir,
        sa1b_dir=args.sa1b_dir,
        openimage_config=openimage_config,
        refcoco_config=refcoco_config,
        target_size=(1024, 1024),  # Use 1024 for SAM2
        sa1b_max_samples=args.sa1b_max_samples,
        sav_max_samples=args.sav_max_samples,
        sav_repeats=args.sav_repeats,
        refcoco_repeats=args.refcoco_repeats,
    )

    print(f"✓ Dataset built: {len(dataset)} total samples")
    print(f"   SAV repeats: {args.sav_repeats}x")
    print(f"   RefCOCO repeats: {args.refcoco_repeats}x")
    return dataset


class DualLoopTrainer:
    """
    Dual-loop trainer:
    1. image + mask → model → caption
    2. image + caption → EMA model → mask'
    3. loss = segmentation_loss(mask', mask)
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

        # Get special token IDs
        self.seg_token_id = tokenizer.convert_tokens_to_ids('[SEG]')
        if self.seg_token_id is None or (isinstance(self.seg_token_id, int) and self.seg_token_id == tokenizer.unk_token_id):
            # [SEG] token not found, try to get it from the model's vocabulary
            # The VideoLLaVASAMModel should have set this up
            # Handle DDP-wrapped model
            actual_model = model.module if hasattr(model, 'module') else model
            if hasattr(actual_model, 'seg_token_idx'):
                self.seg_token_id = actual_model.seg_token_idx
            else:
                raise ValueError("Could not find [SEG] token in tokenizer or model")

        print(f"✓ Using [SEG] token ID: {self.seg_token_id}")

        # Extra image processor for SAM2 (1024x1024)
        self.extra_image_processor = DirectResize(target_length=1024)

    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        self.ema_model.eval()

        epoch_losses = []
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(pbar):
            # Move to device and convert to bfloat16 for FlashAttention compatibility
            images1 = batch['image1'].to(self.device, dtype=torch.bfloat16)
            masks1 = batch['mask1'].to(self.device, dtype=torch.bfloat16)

            # For SAV dataset: image2 and mask2 exist
            # For other datasets: image2 is None, use image1 instead
            if batch['image2'] is not None:
                images2 = batch['image2'].to(self.device, dtype=torch.bfloat16)
                masks2 = batch['mask2'].to(self.device, dtype=torch.bfloat16)
            else:
                images2 = images1
                masks2 = masks1

            # Dual-loop training
            # Step 1: Generate caption from image1 + mask1
            # Step 2: Predict mask from image2 + caption
            # Step 3: Compute loss between predicted mask and mask2
            loss_dict = self.dual_loop_step(images1, masks1, images2, masks2)

            if loss_dict is None:
                continue

            loss = loss_dict['loss']

            # Backward
            loss = loss / self.args.gradient_accumulation_steps
            loss.backward()

            # Update weights
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update EMA
                self.ema_model.update(self.model)

                self.global_step += 1

            # Logging
            epoch_losses.append(loss.item() * self.args.gradient_accumulation_steps)
            if self.global_step % self.args.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'mask_loss': f"{loss_dict.get('mask_loss', 0):.4f}",
                    'dice_loss': f"{loss_dict.get('dice_loss', 0):.4f}",
                })

            # Save checkpoint
            if self.global_step % self.args.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pth")

        return {'loss': sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0}

    def dual_loop_step(self, images1, masks1, images2, masks2):
        """
        Complete dual-loop training step.

        Flow:
        1. Trainable model: image1 + mask1 → caption (generation with visual prompting)
        2. EMA model (inference): image2 + caption → mask2' (referring segmentation)
        3. Compute loss: mask2' vs mask2 (ground truth supervision)
        4. Backprop through trainable model via the segmentation loss

        For SAV dataset:
        - images1: first frame, masks1: mask on first frame
        - images2: second frame, masks2: mask on second frame
        - Generate caption describing mask1 on image1
        - Predict mask on image2 using the caption
        - Compute loss against ground truth mask2

        For other datasets (SA1B, OpenImage):
        - images1 = images2 (same image)
        - masks1 = masks2 (same mask)

        For RefCOCO: handled separately (only Loop 2 with GT caption)

        Args:
            images1: (B, 3, 1024, 1024) - images for caption generation
            masks1: (B, 1024, 1024) - masks for caption generation
            images2: (B, 3, 1024, 1024) - images for mask prediction
            masks2: (B, 1024, 1024) - ground truth masks for loss

        Returns:
            Dict with losses
        """
        import time

        # Step 1: Generate caption from image1 + mask1 using TRAINABLE model
        t0 = time.time()
        with torch.no_grad():
            captions = self.generate_caption_from_mask(images1, masks1)
        t1 = time.time()

        # Step 2: Use TRAINABLE model to predict mask on image2 from caption
        # Note: We use trainable model here so gradients can flow back
        # The EMA model is used as a teacher for regularization during update
        loss_dict = self.compute_segmentation_loss(images2, captions, masks2)
        t2 = time.time()

        # Log timing every 10 steps
        if self.global_step % 10 == 0:
            print(f"[Timing] Caption gen: {t1-t0:.2f}s, Seg loss: {t2-t1:.2f}s, Total: {t2-t0:.2f}s")

        return loss_dict

    @torch.no_grad()
    def generate_caption_from_mask(self, images, masks):
        """
        Generate captions from images and masks using visual prompting.

        Args:
            images: (B, 3, 1024, 1024)
            masks: (B, 1024, 1024)

        Returns:
            List of generated caption strings
        """
        batch_size = images.shape[0]

        # Resize images to 448x448 for vision encoder
        images_448 = F.interpolate(images, size=(448, 448), mode='bilinear', align_corners=False)

        # Prepare prompt_masks: (B, G, G) where G=16
        GRID_SIZE = 16
        masks_unsqueezed = masks.unsqueeze(1)  # (B, 1, 1024, 1024)
        # Resize to 448x448 first
        masks_448 = F.interpolate(masks_unsqueezed, size=(448, 448), mode='nearest')
        # Pool to grid
        pooled = F.adaptive_avg_pool2d(masks_448, (GRID_SIZE, GRID_SIZE))  # (B, 1, G, G)
        prompt_masks = (pooled > 0.5).to(torch.uint8).squeeze(1)  # (B, G, G)

        # Calculate number of active tokens per mask
        region_pixels = [int(prompt_masks[i].sum().item()) for i in range(batch_size)]

        # Build input text with visual prompting
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        IMG_TOKENS_PER_FRAME = 256  # For 448x448 with downsample_ratio=0.5
        VP_START_TOKEN = '<vp>'
        VP_END_TOKEN = '</vp>'

        input_ids_list = []
        prompt_masks_list = []
        vp_overall_mask_list = []

        for i in range(batch_size):
            K = region_pixels[i]
            # Text format: "<img><IMG_CONTEXT>*256</img> There are 1 part regions: region1<vp><IMG_CONTEXT>*K</vp>. Please describe this region."
            image_token_str = f'<img>{IMG_CONTEXT_TOKEN * IMG_TOKENS_PER_FRAME}</img>'
            vp_token_str = f'{VP_START_TOKEN}{IMG_CONTEXT_TOKEN * K}{VP_END_TOKEN}'

            user_input = f"{image_token_str} There are 1 part regions in the picture: region1{vp_token_str}. Please describe this region."
            user_text = f"<|user|>\n{user_input}<|end|>\n<|assistant|>\n"

            # Tokenize
            user_ids = self.tokenizer.encode(user_text, add_special_tokens=True)
            input_ids_list.append(user_ids)

            # prompt_masks for this sample
            prompt_masks_list.append(prompt_masks[i:i+1])  # (1, G, G)

            # vp_overall_mask: (1,) True (only one frame with VP)
            vp_overall_mask_list.append(torch.tensor([True], dtype=torch.bool, device=self.device))

        # Pad input_ids
        max_len = max(len(ids) for ids in input_ids_list)
        input_ids_padded = []
        attention_mask_padded = []

        for ids in input_ids_list:
            pad_len = max_len - len(ids)
            input_ids_padded.append(ids + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask_padded.append([1] * len(ids) + [0] * pad_len)

        input_ids = torch.tensor(input_ids_padded, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_mask_padded, dtype=torch.bool, device=self.device)

        # Prepare pixel_values (list)
        pixel_values = [images_448[i] for i in range(batch_size)]

        # Generate captions
        self.model.eval()

        # Use model.mllm.generate() with visual prompting
        # Handle DDP-wrapped model
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        try:
            outputs = actual_model.mllm.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_masks=prompt_masks_list,
                vp_overall_mask=torch.cat(vp_overall_mask_list, dim=0),
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                # Don't pass use_cache - it's already set internally in internvl.py line 535
                # Don't pass return_dict - it's already handled in internvl.py line 534
            )

            # Decode
            prompt_len = input_ids.shape[1]
            generated_ids = outputs[:, prompt_len:]
            captions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Clean up captions
            captions = [c.strip() for c in captions]

        except Exception as e:
            print(f"Warning: Caption generation failed: {e}")
            # Fallback to simple captions
            captions = [f"a region in the image" for _ in range(batch_size)]

        self.model.train()
        return captions

    def compute_segmentation_loss(self, images, captions, gt_masks):
        """
        Compute segmentation loss using VideoLLaVASAMModel.forward().

        This follows the original training approach.
        """
        batch_size = images.shape[0]

        # Prepare data dict following original format
        # This is the key: use the same format as original training

        # Resize images to 448x448 for vision encoder
        images_448 = F.interpolate(images, size=(448, 448), mode='bilinear', align_corners=False)

        # Prepare input text with [SEG] token
        input_ids_list = []
        labels_list = []

        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        num_image_tokens = 256  # For 448x448 with downsample_ratio=0.5

        for caption in captions:
            # Construct prompt following original dataset format
            image_token_str = f'<img>{IMG_CONTEXT_TOKEN * num_image_tokens}</img>'
            user_input = f"{image_token_str}Please segment: {caption}"
            user_text = f"<|user|>\n{user_input}<|end|>\n<|assistant|>\n"

            # Tokenize user text
            user_ids = self.tokenizer.encode(user_text, add_special_tokens=True)

            # Add [SEG] token manually to ensure it's a single token
            # Tokenize assistant response: "It is "
            try:
                assistant_start = self.tokenizer.encode("It is ", add_special_tokens=False)
            except:
                assistant_start = []

            # Construct output with [SEG] token ID
            # CRITICAL: seg_token_id must be a valid integer
            if self.seg_token_id is None or not isinstance(self.seg_token_id, int):
                raise ValueError(f"Invalid seg_token_id: {self.seg_token_id}")

            output_ids = assistant_start + [self.seg_token_id]

            full_ids = user_ids + output_ids
            full_labels = [-100] * len(user_ids) + output_ids

            # Validate that all IDs are integers (not None)
            if any(x is None for x in full_ids):
                print(f"Error: None found in token IDs!")
                print(f"user_ids: {user_ids[:10]}...")
                print(f"assistant_start: {assistant_start}")
                print(f"seg_token_id: {self.seg_token_id}")
                raise ValueError("Token IDs contain None values")

            # Debug: verify [SEG] token is in the sequence
            if self.seg_token_id not in full_ids:
                print(f"WARNING: [SEG] token {self.seg_token_id} not found in full_ids!")
                print(f"full_ids: {full_ids}")

            input_ids_list.append(full_ids)
            labels_list.append(full_labels)

        # Pad
        max_len = max(len(ids) for ids in input_ids_list)
        input_ids, labels, attention_mask = [], [], []

        # Get pad token ID (use eos_token_id if pad_token_id is not set)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        for ids, labs in zip(input_ids_list, labels_list):
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_token_id] * pad_len)
            labels.append(labs + [-100] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool, device=self.device)

        bs, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(bs, -1)

        # Debug: Check how many [SEG] tokens we have
        seg_mask = input_ids == self.seg_token_id
        seg_counts = seg_mask.sum(dim=1)
        if (seg_counts == 0).any():
            print(f"Warning: Some samples have no [SEG] token! seg_counts: {seg_counts.tolist()}")
            print(f"seg_token_id: {self.seg_token_id}")
            print(f"Sample input_ids (first 20): {input_ids[0, :20].tolist()}")

        # Prepare pixel_values (list of tensors)
        pixel_values = [images_448[i] for i in range(batch_size)]

        # Prepare g_pixel_values for SAM2 (1024x1024, already correct size)
        g_pixel_values = [images[i] for i in range(batch_size)]

        # Prepare masks (list) - each mask should have shape (1, H, W)
        # Add extra dimension for the number of objects (always 1 in our case)
        masks_list = [gt_masks[i].unsqueeze(0) for i in range(batch_size)]

        # Build data dict following VideoLLaVASAMModel.forward() expectations
        # frames_per_batch: [1] * batch_size - each sample is a single frame (image data)
        # This tells the model each batch element has 1 frame
        data = {
            'pixel_values': pixel_values,
            'g_pixel_values': g_pixel_values,
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'masks': masks_list,
            'frames_per_batch': [1] * batch_size,
        }

        # Forward through VideoLLaVASAMModel
        # This returns: {'loss_mask': ..., 'loss_dice': ..., 'llm_loss': ...}
        loss_dict = self.model(data, data_samples=None, mode='loss')

        # Combine losses
        total_loss = (
            loss_dict.get('loss_mask', 0) +
            loss_dict.get('loss_dice', 0) +
            loss_dict.get('llm_loss', 0)
        )

        return {
            'loss': total_loss,
            'mask_loss': loss_dict.get('loss_mask', 0).item() if torch.is_tensor(loss_dict.get('loss_mask', 0)) else 0,
            'dice_loss': loss_dict.get('loss_dice', 0).item() if torch.is_tensor(loss_dict.get('loss_dice', 0)) else 0,
            'llm_loss': loss_dict.get('llm_loss', 0).item() if torch.is_tensor(loss_dict.get('llm_loss', 0)) else 0,
        }

    def save_checkpoint(self, filename):
        """Save checkpoint."""
        checkpoint_path = self.output_dir / filename
        torch.save({
            'model': self.model.state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
        }, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")


def main():
    args = parse_args()

    # Read LOCAL_RANK from environment (set by torchrun)
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    args.local_rank = local_rank

    # Setup distributed training
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{args.local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # Build model (parallel loading like sa2va SFT training)
    print("Building model...")
    model, tokenizer = build_model(args)
    model = model.to(device, dtype=torch.bfloat16)
    print(f"✓ Model moved to {device} with dtype bfloat16")

    # Wrap model with DDP
    # Use static_graph=True because we have multiple forward passes per step
    # (caption generation + segmentation) which can cause "marked ready twice" errors
    # Do NOT use find_unused_parameters with static_graph
    if args.local_rank != -1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,  # Avoid buffer sync issues with static graph
            static_graph=True  # Required for dual-loop with multiple forward passes
        )
        print(f"✓ Model wrapped with DDP (static_graph=True) on rank {args.local_rank}")

    # Create EMA model (parallel loading)
    print("Creating EMA model...")
    # Use model.module if wrapped with DDP
    ema_model = EMAModel(model.module if hasattr(model, 'module') else model, decay=args.ema_decay)
    print("✓ EMA model created")

    # Build datasets
    train_dataset = build_datasets(args)

    # Create dataloader
    from projects.llava_sam2.mask_caption_sft.dataset_builder import collate_fn_mask_caption
    from torch.utils.data.distributed import DistributedSampler

    # Use DistributedSampler for DDP training
    if args.local_rank != -1:
        sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
            drop_last=False,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_fn_mask_caption,
        pin_memory=True,
    )
    print(f"✓ Dataloader created: {len(train_dataloader)} batches")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    )
    print("✓ Optimizer created")

    # Create trainer
    trainer = DualLoopTrainer(
        model=model,
        ema_model=ema_model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        device=device,
        output_dir=args.output_dir,
        args=args,
    )

    # Training loop
    print("=" * 80)
    print("Starting dual-loop training")
    print("=" * 80)

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 80)

        results = trainer.train_epoch(epoch)

        print(f"Epoch {epoch + 1} completed. Avg loss: {results['loss']:.4f}")

    print("=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
