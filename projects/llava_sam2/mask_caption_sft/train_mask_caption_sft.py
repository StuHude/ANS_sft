"""
Main training script for Mask Captioning + Referring Segmentation SFT

Usage:
    python projects/llava_sam2/mask_caption_sft/train_mask_caption_sft.py \
        --config projects/llava_sam2/mask_caption_sft/config.py \
        --model_path /path/to/sa2va/checkpoint
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from projects.llava_sam2.mask_caption_sft.dataset_builder import (
    build_mask_caption_dataset,
    collate_fn_mask_caption,
)
from projects.llava_sam2.mask_caption_sft.trainer import MaskCaptionSFTTrainer
from projects.llava_sam2.rl_train.ema_model import EMAModel
from projects.llava_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser(description='Mask Captioning SFT Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to Sa2VA checkpoint')
    parser.add_argument('--output_dir', type=str, default='./work_dirs/mask_caption_sft',
                        help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--log_interval', type=int, default=10, help='Log every N steps')
    parser.add_argument('--save_interval', type=int, default=1000, help='Save every N steps')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA for LLM')
    parser.add_argument('--lora_r', type=int, default=128, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=256, help='LoRA alpha')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')

    # Dataset paths
    parser.add_argument('--sav_dir', type=str, default=None, help='Path to SAV NPZ directory')
    parser.add_argument('--sa1b_dir', type=str, default=None, help='Path to SA-1B dataset directory')
    parser.add_argument('--openimage_dir', type=str, default=None, help='Path to OpenImage directory')
    parser.add_argument('--refcoco_dir', type=str, default=None, help='Path to RefCOCO directory')

    # Dataset size control for testing
    parser.add_argument('--sa1b_max_samples', type=int, default=None,
                        help='Maximum number of SA-1B image files to process (default: None = all samples)')
    parser.add_argument('--sav_max_samples', type=int, default=None,
                        help='Maximum number of SAV samples to use (default: None = all samples). Set to 10 for testing.')

    args = parser.parse_args()
    return args


def setup_lora(model, r=128, lora_alpha=256, lora_dropout=0.05):
    """
    Setup LoRA for LLM part of Sa2VA model.

    Args:
        model: Sa2VA model
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout rate

    Returns:
        Model with LoRA applied
    """
    print("Setting up LoRA...")

    # LoRA config
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA to language model
    if hasattr(model, 'language_model'):
        model.language_model = get_peft_model(model.language_model, lora_config)
        print("✓ LoRA applied to language_model")
    elif hasattr(model.model, 'language_model'):
        model.model.language_model = get_peft_model(model.model.language_model, lora_config)
        print("✓ LoRA applied to model.language_model")
    else:
        print("⚠ Warning: Could not find language_model, LoRA not applied")

    return model


def freeze_model_parts(model):
    """
    Freeze parts of the model following Sa2VA training paradigm.

    Frozen:
    - Vision encoder
    - SAM2 image encoder

    Trainable:
    - LoRA parameters (if enabled)
    - MLP projector (mlp1)
    - SAM2 decoder
    - text_hidden_fcs
    """
    print("Freezing model parts...")

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze LoRA parameters
    if hasattr(model, 'language_model'):
        lm = model.language_model
    elif hasattr(model.model, 'language_model'):
        lm = model.model.language_model
    else:
        lm = None

    if lm is not None:
        for name, param in lm.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
        print("✓ LoRA parameters unfrozen")

    # Unfreeze MLP projector
    if hasattr(model, 'mlp1'):
        for param in model.mlp1.parameters():
            param.requires_grad = True
        print("✓ mlp1 unfrozen")
    elif hasattr(model.model, 'mlp1'):
        for param in model.model.mlp1.parameters():
            param.requires_grad = True
        print("✓ model.mlp1 unfrozen")

    # Unfreeze SAM2 decoder
    if hasattr(model, 'grounding_encoder'):
        ge = model.grounding_encoder
    elif hasattr(model.model, 'grounding_encoder'):
        ge = model.model.grounding_encoder
    else:
        ge = None

    if ge is not None and hasattr(ge, 'sam2_model'):
        if hasattr(ge.sam2_model, 'sam_mask_decoder'):
            for param in ge.sam2_model.sam_mask_decoder.parameters():
                param.requires_grad = True
            print("✓ SAM2 decoder unfrozen")

    # Unfreeze text_hidden_fcs
    if hasattr(model, 'text_hidden_fcs'):
        for param in model.text_hidden_fcs.parameters():
            param.requires_grad = True
        print("✓ text_hidden_fcs unfrozen")
    elif hasattr(model.model, 'text_hidden_fcs'):
        for param in model.model.text_hidden_fcs.parameters():
            param.requires_grad = True
        print("✓ model.text_hidden_fcs unfrozen")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    return model


def build_datasets(args):
    """Build training datasets."""
    print("Building datasets...")

    # OpenImage config
    openimage_config = None
    if args.openimage_dir:
        openimage_config = {
            'annotation_csv': os.path.join(args.openimage_dir, 'train-annotations-object-segmentation.csv'),
            'label_csv': os.path.join(args.openimage_dir, 'oidv7-class-descriptions.csv'),
            'image_dir': os.path.join(args.openimage_dir, 'images', 'train'),
            'mask_dir': os.path.join(args.openimage_dir, 'masks', 'train'),
        }

    # RefCOCO config
    refcoco_config = None
    if args.refcoco_dir:
        refcoco_config = {
            'data_root': args.refcoco_dir,
            'split': 'train',
            'dataset_name': 'refcoco',
        }

    # Build dataset
    # Note: Sa2VA model expects 448x448 images for vision encoder
    dataset = build_mask_caption_dataset(
        sav_dir=args.sav_dir,
        sa1b_dir=args.sa1b_dir,
        openimage_config=openimage_config,
        refcoco_config=refcoco_config,
        target_size=(448, 448),
        sa1b_max_samples=args.sa1b_max_samples,
        sav_max_samples=args.sav_max_samples,
    )

    print(f"✓ Dataset built: {len(dataset)} total samples")

    return dataset


def main():
    args = parse_args()

    # Setup distributed training
    import torch.distributed as dist
    import os

    # Get local_rank from environment (torchrun sets this)
    if args.local_rank == -1 and 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{args.local_rank}')
        print(f"[Rank {args.local_rank}] Using device: {device}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

    # Load model
    print(f"[Rank {args.local_rank}] Loading Sa2VA model from {args.model_path}...")

    # Load model directly without dtype specification
    # Let AMP handle dtype conversion during training
    model = Sa2VAChatModel.from_pretrained(
        args.model_path,
        low_cpu_mem_usage=True,
    )

    print(f"[Rank {args.local_rank}] ✓ Model loaded to CPU")

    # Move to device (keep original dtype)
    print(f"[Rank {args.local_rank}] Moving model to {device}...")
    model = model.to(device)
    print(f"[Rank {args.local_rank}] ✓ Model moved to {device}")

    # Load tokenizer
    print(f"[Rank {args.local_rank}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"[Rank {args.local_rank}] ✓ Tokenizer loaded")

    # Initialize model for generation (sets token IDs)
    print(f"[Rank {args.local_rank}] Preparing model for generation...")
    model.preparing_for_generation(tokenizer)
    print(f"[Rank {args.local_rank}] ✓ Model prepared for generation")

    # Setup LoRA if enabled
    if args.use_lora:
        print(f"[Rank {args.local_rank}] Setting up LoRA...")
        model = setup_lora(model, r=args.lora_r, lora_alpha=args.lora_alpha)

    # Freeze model parts
    print(f"[Rank {args.local_rank}] Freezing model parts...")
    model = freeze_model_parts(model)

    # Create EMA model BEFORE DDP wrapping
    print(f"[Rank {args.local_rank}] Creating EMA model...")
    ema_model = EMAModel(model, decay=args.ema_decay)
    print(f"[Rank {args.local_rank}] ✓ EMA model created")

    # Wrap model with DDP AFTER EMA creation
    if args.local_rank != -1:
        print(f"[Rank {args.local_rank}] Wrapping model with DDP...")
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        print(f"[Rank {args.local_rank}] ✓ DDP wrapper created")

    # Build datasets
    train_dataset = build_datasets(args)

    # Create distributed sampler if needed
    if args.local_rank != -1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn_mask_caption,
        pin_memory=True,
    )
    if args.local_rank <= 0:
        print(f"✓ Dataloader created: {len(train_dataloader)} batches")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    )
    print("✓ Optimizer created")

    # Create scheduler
    from transformers import get_cosine_schedule_with_warmup

    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(0.05 * total_steps)  # 5% warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"✓ Scheduler created: {total_steps} total steps, {warmup_steps} warmup steps")

    # Create trainer
    trainer = MaskCaptionSFTTrainer(
        model=model,
        ema_model=ema_model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        ema_decay=args.ema_decay,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
    )
    print("✓ Trainer created")

    # Resume from checkpoint if specified
    if args.resume_from is not None:
        trainer.load_checkpoint(args.resume_from)

    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 80)

        epoch_results = trainer.train_epoch(epoch)

        print(f"\nEpoch {epoch + 1} results:")
        print(f"  Average loss: {epoch_results['avg_loss']:.4f}")
        print(f"  Average IoU: {epoch_results['avg_iou']:.4f}")

    # Save final checkpoint
    trainer.save_checkpoint('final_checkpoint.pth')

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
