"""
Mask Captioning + Referring Segmentation SFT Trainer

Training loop:
1. Input: image + mask
2. Model generates caption
3. EMA model generates mask from caption
4. Compute IoU loss + segmentation losses

For SAV dataset: image1+mask1 -> caption -> image2+caption -> mask2'
For other datasets: image+mask -> caption -> image+caption -> mask'
For RefCOCO: Only participates in referring segmentation (has ground truth caption)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from projects.llava_sam2.rl_train.ema_model import EMAModel
from projects.llava_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel
from transformers import AutoTokenizer, StoppingCriteriaList
from projects.llava_sam2.hf.models.modeling_sa2va_chat import get_stop_criteria

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss


class MaskCaptionSFTTrainer:
    """
    SFT trainer for mask captioning + referring segmentation.
    """

    def __init__(
        self,
        model,
        ema_model,
        tokenizer,
        train_dataloader,
        val_dataloader=None,
        optimizer=None,
        scheduler=None,
        device='cuda',
        output_dir='./outputs',
        max_caption_length=128,
        ema_decay=0.999,
        ema_update_interval=1,
        log_interval=10,
        save_interval=1000,
        use_amp=True,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        # Loss weights
        caption_loss_weight=1.0,
        iou_loss_weight=1.0,
        mask_loss_weight=2.0,
        dice_loss_weight=0.5,
    ):
        """
        Args:
            model: Sa2VA main model (student)
            ema_model: EMA model (teacher)
            tokenizer: Tokenizer
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader (optional)
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to use
            output_dir: Output directory for checkpoints and logs
            max_caption_length: Maximum caption length
            ema_decay: EMA decay rate
            ema_update_interval: Update EMA model every N steps
            log_interval: Log every N steps
            save_interval: Save checkpoint every N steps
            use_amp: Use automatic mixed precision
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
            caption_loss_weight: Weight for caption generation loss (LM loss)
            iou_loss_weight: Weight for IoU-based loss
            mask_loss_weight: Weight for mask CrossEntropy loss
            dice_loss_weight: Weight for Dice loss
        """
        self.model = model
        self.ema_model = ema_model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_caption_length = max_caption_length
        self.ema_decay = ema_decay
        self.ema_update_interval = ema_update_interval
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # Loss weights
        self.caption_loss_weight = caption_loss_weight
        self.iou_loss_weight = iou_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.dice_loss_weight = dice_loss_weight

        # AMP scaler
        self.scaler = GradScaler() if use_amp else None

        # Tensorboard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))

        # Get special tokens
        self.seg_token_id = tokenizer.convert_tokens_to_ids('[SEG]')
        self.vp_start_id = tokenizer.convert_tokens_to_ids('<vp>')
        self.vp_end_id = tokenizer.convert_tokens_to_ids('</vp>')

        # Stopping criteria for generation
        stop_words = ['<|end|>']
        self.stop_criteria = get_stop_criteria(tokenizer=tokenizer, stop_words=stop_words)

        # Loss functions for segmentation
        self.loss_mask_fn = CrossEntropyLoss(
            use_sigmoid=True,
            reduction='mean',
            loss_weight=mask_loss_weight,
        )
        self.loss_dice_fn = DiceLoss(
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=dice_loss_weight,
        )

        # Step counter
        self.global_step = 0

        print(f"✓ MaskCaptionSFTTrainer initialized")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Use AMP: {self.use_amp}")
        print(f"  EMA decay: {self.ema_decay}")
        print(f"  Loss weights: caption={caption_loss_weight}, iou={iou_loss_weight}, "
              f"mask={mask_loss_weight}, dice={dice_loss_weight}")

    def prepare_visual_prompt_input(self, images, masks, tokenizer):
        """
        Prepare visual prompt input for mask captioning.

        Input: image + <vp> mask </vp>
        Output: caption

        Args:
            images: (B, 3, 448, 448) - ImageNet normalized for InternVL
            masks: (B, 16, 16) - Visual prompt masks [0, 1]
            tokenizer: Tokenizer

        Returns:
            Dict with pixel_values, input_ids, attention_mask, prompt_masks
        """
        batch_size = images.shape[0]

        # Calculate number of image tokens for InternVL
        # image_size=448, patch_size=14, downsample_ratio=0.5
        # num_image_tokens = (448/14)^2 * (0.5)^2 = 256
        num_image_tokens = 256

        # Create image token string with repeated IMG_CONTEXT tokens
        image_token_str = f'<img>{"<IMG_CONTEXT>" * num_image_tokens}</img>'

        # Create prompt with image tokens and visual prompt tags
        # Format: "<|user|>\n<img><IMG_CONTEXT>*256</img><vp></vp> Please describe this region.<|end|>\n<|assistant|>\n"
        prompts = []
        for i in range(batch_size):
            prompt = f"<|user|>\n{image_token_str}<vp></vp> Please describe this region in detail.<|end|>\n<|assistant|>\n"
            prompts.append(prompt)

        # Tokenize
        tokenized = tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )

        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)

        # Prepare prompt masks (visual prompt)
        # Masks are already 16×16 from dataset, just add batch dimension for model
        prompt_masks = masks.unsqueeze(1)  # (B, 1, 16, 16)

        return {
            'pixel_values': images.to(self.device),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'prompt_masks': prompt_masks.to(self.device),
        }

    def prepare_referring_segmentation_input(self, images, captions, tokenizer):
        """
        Prepare input for referring segmentation.

        Input: image + caption
        Output: [SEG] mask

        Args:
            images: (B, 3, H, W)
            captions: List of caption strings
            tokenizer: Tokenizer

        Returns:
            Dict with pixel_values, input_ids, attention_mask, position_ids, labels
        """
        batch_size = images.shape[0]

        # Calculate number of image tokens (same as prepare_visual_prompt_input)
        num_image_tokens = 256

        # Create image token string
        image_token_str = f'<img>{"<IMG_CONTEXT>" * num_image_tokens}</img>'

        # Create prompt with image tokens, caption, and [SEG] token
        # Format: <|user|>\n<img>..IMG_CONTEXT..</img>Please segment the {caption}[SEG]<|end|>\n<|assistant|>\nIt is [SEG].<|end|>
        prompts = []
        outputs_text = []
        for caption in captions:
            # User input with image and caption
            user_input = f"{image_token_str}Please segment the {caption}[SEG]"
            prompts.append(user_input)
            # Assistant output (contains [SEG] token for mask prediction)
            outputs_text.append("It is [SEG].")

        # Encode following original dataset style
        input_ids_list = []
        labels_list = []

        for i in range(batch_size):
            # Encode user input
            user_text = f"<|user|>\n{prompts[i]}<|end|>\n<|assistant|>\n"
            user_ids = tokenizer.encode(user_text, add_special_tokens=True)

            # Encode assistant output
            output_text = outputs_text[i] + "<|end|>"
            output_ids = tokenizer.encode(output_text, add_special_tokens=False)

            # Combine
            full_ids = user_ids + output_ids
            # Labels: ignore user input, only supervise assistant output
            full_labels = [-100] * len(user_ids) + output_ids

            input_ids_list.append(full_ids)
            labels_list.append(full_labels)

        # Pad sequences
        max_len = max(len(ids) for ids in input_ids_list)
        input_ids = []
        labels = []
        attention_mask = []

        for ids, labs in zip(input_ids_list, labels_list):
            pad_len = max_len - len(ids)
            input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
            labels.append(labs + [-100] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool, device=self.device)

        # Create position_ids
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)

        return {
            'pixel_values': images.to(self.device),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': labels,
        }

    @torch.no_grad()
    def generate_caption(self, images, masks):
        """
        Generate captions from images and masks using the main model.

        Args:
            images: (B, 3, 448, 448) - ImageNet normalized pixel_values
            masks: (B, 16, 16) - Visual prompt masks [0, 1]

        Returns:
            List of generated caption strings
        """
        inputs = self.prepare_visual_prompt_input(images, masks, self.tokenizer)

        # Get the base model (handle DDP wrapper)
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # Generate
        model.eval()
        outputs = model.generate(
            pixel_values=inputs['pixel_values'],
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            prompt_masks=inputs['prompt_masks'],
            max_new_tokens=self.max_caption_length,
            do_sample=False,  # Greedy decoding for SFT
            temperature=1.0,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            stopping_criteria=self.stop_criteria,
        )

        # Decode captions
        # Remove prompt tokens
        prompt_len = inputs['input_ids'].shape[1]
        generated_ids = outputs[:, prompt_len:]
        captions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        model.train()
        return captions

    def compute_iou_loss(self, pred_masks, gt_masks):
        """
        Compute IoU-based loss.

        IoU loss = 1 - IoU

        Args:
            pred_masks: (B, H, W) logits or probabilities
            gt_masks: (B, H, W) binary masks

        Returns:
            Scalar loss
        """
        # Convert to probabilities if needed
        if pred_masks.dtype == torch.float:
            pred_probs = torch.sigmoid(pred_masks)
        else:
            pred_probs = pred_masks.float()

        # Binarize
        pred_binary = (pred_probs > 0.5).float()
        gt_binary = (gt_masks > 0.5).float()

        # Compute IoU
        intersection = (pred_binary * gt_binary).sum(dim=(1, 2))
        union = (pred_binary + gt_binary - pred_binary * gt_binary).sum(dim=(1, 2))
        iou = intersection / (union + 1e-6)

        # IoU loss = 1 - IoU
        iou_loss = (1.0 - iou).mean()

        return iou_loss, iou.mean().item()

    def forward_referring_segmentation(self, pixel_values, g_pixel_values, captions, gt_masks):
        """
        Forward pass for referring segmentation.

        Implements mask loss computation manually since Sa2VAChatModel.forward()
        only returns CausalLMOutputWithPast (LLM loss only).

        Reference: VideoLLaVASAMModel.forward() in projects/llava_sam2/models/llava_sam2.py

        Args:
            pixel_values: (B, 3, 448, 448) - ImageNet normalized for InternVL
            g_pixel_values: (B, 3, 1024, 1024) - [0, 255] uint8 for SAM2
            captions: List of caption strings
            gt_masks: (B, 1024, 1024) ground truth masks [0, 1]

        Returns:
            Dict with losses and metrics
        """
        device = pixel_values.device
        batch_size = pixel_values.shape[0]

        # Ensure all inputs are on correct device
        pixel_values = pixel_values.to(device)
        g_pixel_values = g_pixel_values.to(device)
        gt_masks = gt_masks.to(device)

        # Prepare inputs using the original dataset format
        inputs = self.prepare_referring_segmentation_input(pixel_values, captions, self.tokenizer)

        # Handle DDP wrapper
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # Convert pixel_values to list for Sa2VAChatModel
        pixel_values_list = [pixel_values[i].unsqueeze(0) for i in range(batch_size)]

        # Prepare data dict for Sa2VAChatModel.forward()
        data = {
            'pixel_values': pixel_values_list,
            'input_ids': inputs['input_ids'],
            'position_ids': inputs['position_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': inputs['labels'],
        }

        # Forward through Sa2VAChatModel - returns CausalLMOutputWithPast
        output = model(data, data_samples=None, mode='loss')

        # Get LLM loss from output
        llm_loss = output.loss if output.loss is not None else torch.tensor(0.0, device=device)

        # Extract hidden states for [SEG] token mask prediction
        hidden_states = output.hidden_states
        if hidden_states is None:
            # If hidden states not returned, use LLM loss only
            return {
                'loss': llm_loss,
                'mask_loss': 0.0,
                'dice_loss': 0.0,
                'llm_loss': llm_loss.item() if torch.is_tensor(llm_loss) else 0.0,
                'mean_iou': 0.0,
            }

        # Get the last hidden state
        last_hidden_state = hidden_states[-1]  # (B, seq_len, hidden_dim)

        # Find [SEG] token positions
        seg_token_mask = inputs['input_ids'] == self.seg_token_id
        seg_token_counts = seg_token_mask.int().sum(-1)

        # Check if any [SEG] tokens exist
        if seg_token_counts.sum() == 0:
            # No [SEG] tokens, skip mask loss
            return {
                'loss': llm_loss,
                'mask_loss': 0.0,
                'dice_loss': 0.0,
                'llm_loss': llm_loss.item() if torch.is_tensor(llm_loss) else 0.0,
                'mean_iou': 0.0,
            }

        # Extract hidden states for [SEG] tokens
        # Apply text_hidden_fcs to convert to SAM2 embedding dimension
        hidden_states_transformed = model.text_hidden_fcs(last_hidden_state)

        # Get [SEG] token embeddings
        pred_embeddings = hidden_states_transformed[seg_token_mask]  # (num_seg_tokens, out_dim)

        # Split by batch
        pred_embeddings_list = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = [item for item in pred_embeddings_list if len(item) > 0]

        if len(pred_embeddings_list) == 0:
            return {
                'loss': llm_loss,
                'mask_loss': 0.0,
                'dice_loss': 0.0,
                'llm_loss': llm_loss.item() if torch.is_tensor(llm_loss) else 0.0,
                'mean_iou': 0.0,
            }

        # Get SAM2 embeddings (HF version: init_state)
        # get_sam2_embeddings expects [0, 255] uint8 and handles preprocessing internally
        sam_states = model.grounding_encoder.get_sam2_embeddings(g_pixel_values)

        # Prepare language embeddings in the format expected by HF SAM2:
        # language_embd[frame_idx][obj_idx] shape
        # Each pred_embeddings_list[i] has shape (num_objs, hidden_dim)
        num_objs_per_frame = [emb.shape[0] for emb in pred_embeddings_list]

        # Build language_embd as list of lists: language_embd[frame_idx][obj_idx]
        language_embd = []
        for i, emb in enumerate(pred_embeddings_list):
            # emb shape: (num_objs, hidden_dim)
            frame_embds = [emb[j] for j in range(emb.shape[0])]
            language_embd.append(frame_embds)

        # Inject language embeddings and get predicted masks using HF SAM2 API
        pred_masks = model.grounding_encoder.inject_language_embd(sam_states, language_embd)

        # Prepare ground truth masks
        # Resize gt_masks to match pred_masks size
        pred_mask_size = pred_masks.shape[-2:]
        gt_masks_resized = F.interpolate(
            gt_masks.unsqueeze(1).float(),
            size=pred_mask_size,
            mode='nearest'
        ).squeeze(1)

        # Flatten predictions for loss computation
        pred_masks_flat = pred_masks.flatten(0, 1)  # (B * num_objs, H, W)
        gt_masks_flat = gt_masks_resized  # (B, H, W)

        # Handle shape mismatch
        if len(pred_masks_flat) != len(gt_masks_flat):
            min_num = min(len(pred_masks_flat), len(gt_masks_flat))
            pred_masks_flat = pred_masks_flat[:min_num]
            gt_masks_flat = gt_masks_flat[:min_num]

        # Compute mask loss and dice loss
        loss_mask = self.loss_mask_fn(pred_masks_flat, gt_masks_flat)
        loss_dice = self.loss_dice_fn(pred_masks_flat, gt_masks_flat)

        # Compute IoU for metrics
        with torch.no_grad():
            pred_binary = (torch.sigmoid(pred_masks_flat) > 0.5).float()
            gt_binary = (gt_masks_flat > 0.5).float()
            intersection = (pred_binary * gt_binary).sum(dim=(1, 2))
            union = (pred_binary + gt_binary - pred_binary * gt_binary).sum(dim=(1, 2))
            iou = (intersection / (union + 1e-6)).mean().item()

        # Combine losses
        total_loss = (
            self.mask_loss_weight * loss_mask +
            self.dice_loss_weight * loss_dice +
            self.caption_loss_weight * llm_loss
        )

        return {
            'loss': total_loss,
            'mask_loss': loss_mask.item() if torch.is_tensor(loss_mask) else 0.0,
            'dice_loss': loss_dice.item() if torch.is_tensor(loss_dice) else 0.0,
            'llm_loss': llm_loss.item() if torch.is_tensor(llm_loss) else 0.0,
            'mean_iou': iou,
        }

    def training_step(self, batch):
        """
        Single training step.

        Handles different dataset types:
        - SAV: image1+mask1 -> caption -> image2+caption -> mask2'
        - SA-1B/OpenImage: image+mask -> caption -> image+caption -> mask'
        - RefCOCO: image+caption -> mask (ground truth caption)

        Args:
            batch: Batch from dataloader

        Returns:
            Dict with losses and metrics
        """
        if batch is None:
            return None

        # Unpack batch and move to device
        # New format: pixel_values (448 normalized), g_pixel_values (1024 [0,255]), masks (1024 GT)
        # SAV dataset: pixel_values1, g_pixel_values1, masks1 (and 2)
        # Other datasets: pixel_values, g_pixel_values, masks

        # Try SAV format first (has_paired_frame=True)
        if 'pixel_values1' in batch:
            pixel_values = batch['pixel_values1'].to(self.device)
            g_pixel_values = batch['g_pixel_values1'].to(self.device)
            gt_masks = batch['masks1'].to(self.device)
        else:
            pixel_values = batch['pixel_values'].to(self.device)
            g_pixel_values = batch['g_pixel_values'].to(self.device)
            gt_masks = batch['masks'].to(self.device)

        captions_gt = batch.get('caption', batch.get('captions', None))

        batch_size = pixel_values.shape[0]

        # Prepare referring segmentation inputs
        # For RefCOCO: use ground truth captions
        # For others: generate simple placeholder captions for now
        captions = []
        for i in range(batch_size):
            if captions_gt is not None and i < len(captions_gt):
                captions.append(captions_gt[i])
            else:
                # For non-RefCOCO data, use a simple prompt
                captions.append("Please segment the marked region in this image.")

        # Forward pass for referring segmentation
        result = self.forward_referring_segmentation(pixel_values, g_pixel_values, captions, gt_masks)

        return result

    def train_epoch(self, epoch):
        """
        Train for one epoch.
        """
        self.model.train()
        self.ema_model.eval()

        epoch_losses = []
        epoch_iou = []

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(pbar):
            # Training step
            # Use bfloat16 for autocast to match model dtype
            # NOTE: BFloat16 doesn't need GradScaler (sufficient dynamic range)
            with autocast(enabled=self.use_amp, dtype=torch.bfloat16):
                result = self.training_step(batch)

            # Handle None batch: skip this batch entirely
            if result is None or batch is None:
                continue

            loss = result['loss']

            # Backward pass
            # BFloat16 doesn't need loss scaling, so we don't use GradScaler
            (loss / self.gradient_accumulation_steps).backward()

            # Update weights
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update EMA model (use model.module if DDP wrapped)
                if self.global_step % self.ema_update_interval == 0:
                    student_model = self.model.module if hasattr(self.model, 'module') else self.model
                    self.ema_model.update(student_model)

                # Update scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

            # Logging
            epoch_losses.append(result['loss'].item())
            epoch_iou.append(result['mean_iou'])

            if self.global_step % self.log_interval == 0:
                self.writer.add_scalar('train/loss', result['loss'].item(), self.global_step)
                self.writer.add_scalar('train/mask_loss', result.get('mask_loss', 0), self.global_step)
                self.writer.add_scalar('train/dice_loss', result.get('dice_loss', 0), self.global_step)
                self.writer.add_scalar('train/llm_loss', result.get('llm_loss', 0), self.global_step)
                self.writer.add_scalar('train/mean_iou', result['mean_iou'], self.global_step)

                pbar.set_postfix({
                    'loss': f"{result['loss'].item():.4f}",
                    'mask_loss': f"{result.get('mask_loss', 0):.4f}",
                    'dice_loss': f"{result.get('dice_loss', 0):.4f}",
                    'iou': f"{result['mean_iou']:.4f}",
                })

            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pth")

        return {
            'avg_loss': np.mean(epoch_losses),
            'avg_iou': np.mean(epoch_iou),
        }

    def save_checkpoint(self, filename):
        """Save checkpoint."""
        import torch.distributed as dist

        # Only rank 0 saves checkpoint
        if hasattr(self.model, 'module'):
            # DDP wrapped model
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank != 0:
                return
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        checkpoint_path = self.output_dir / filename

        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': model_state_dict,
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")

        # Barrier to ensure all ranks wait for checkpoint save
        if dist.is_initialized():
            dist.barrier()

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle DDP wrapped model
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.global_step = checkpoint['global_step']

        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        print(f"  Resuming from step {self.global_step}")
