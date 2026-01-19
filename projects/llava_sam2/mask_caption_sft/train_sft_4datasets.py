"""
SFT Training with 4 Datasets (SAV, SA1B, OpenImage, RefCOCO)

Uses original Sa2VA dataset loaders and training loop.
Based on sa2va_4b.py config.
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import Sa2VA components
from projects.llava_sam2.models.llava_sam2 import VideoLLaVASAMModel
from projects.llava_sam2.models import SAM2TrainRunner
from projects.llava_sam2.models.internvl import InternVL_Slowfast
from projects.llava_sam2.models.preprocess.image_resize import DirectResize
from projects.llava_sam2.datasets import video_lisa_collate_fn

# Import Sa2VA dataset classes
from projects.llava_sam2.datasets import ReferSegmDataset
from projects.llava_sam2.datasets.describe_anything_referring_dataset import DescribeAnythingReferringDataset

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from transformers import AutoTokenizer
from peft import LoraConfig
from xtuner.registry import BUILDER
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import template_map_fn_factory
from xtuner.model.utils import guess_load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Sa2VA SFT Training with 4 Datasets')
    parser.add_argument('--pretrained_pth', type=str, required=True,
                        help='Path to pretrained checkpoint (HF format)')
    parser.add_argument('--model_path', type=str, default='./pretrained/InternVL2_5-4B',
                        help='Path to base model (InternVL)')
    parser.add_argument('--output_dir', type=str, default='./work_dirs/sft_4datasets',
                        help='Output directory')

    # Dataset paths
    parser.add_argument('--sav_dir', type=str, default='/data/xyc/DAM_data',
                        help='SAV dataset directory')
    parser.add_argument('--refcoco_dir', type=str, default='./data/ref_seg',
                        help='RefCOCO dataset directory')

    # Training params
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    # LoRA params
    parser.add_argument('--lora_r', type=int, default=128)
    parser.add_argument('--lora_alpha', type=int, default=256)

    # Other params
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--max_length', type=int, default=8192,
                        help='Maximum sequence length')

    args = parser.parse_args()
    return args


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)
    dist.barrier()
    return True, rank, world_size, gpu


def build_model(args):
    """Build VideoLLaVASAMModel following sa2va_4b.py config."""
    if int(os.environ.get('RANK', 0)) == 0:
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
    tokenizer.add_tokens(special_tokens, special_tokens=True)

    # Model config (following sa2va_4b.py)
    model_cfg = dict(
        type=VideoLLaVASAMModel,
        special_tokens=special_tokens,
        frozen_sam2_decoder=False,  # Train SAM2 decoder
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

    # Load pretrained weights from HuggingFace format
    if args.pretrained_pth and os.path.exists(args.pretrained_pth):
        if int(os.environ.get('RANK', 0)) == 0:
            print(f"Loading pretrained weights from {args.pretrained_pth}")

        from projects.llava_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel

        # Load HF model
        hf_model = Sa2VAChatModel.from_pretrained(
            args.pretrained_pth,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        # Transfer weights
        model.load_state_dict(hf_model.state_dict(), strict=False)
        if int(os.environ.get('RANK', 0)) == 0:
            print("✓ Pretrained weights loaded")
        del hf_model

    return model, tokenizer


def build_datasets(args, tokenizer):
    """Build 4 datasets: SAV, SA1B (via DAR), OpenImage (via DAR), RefCOCO."""
    prompt_template = PROMPT_TEMPLATE.phi3_chat

    # Extra image processor
    extra_image_processor = dict(
        type=DirectResize,
        target_length=1024,
    )
    extra_image_processor = BUILDER.build(extra_image_processor)

    # Template map function
    template_map_fn = dict(
        type=template_map_fn_factory,
        template=prompt_template
    )

    special_tokens = [
        '[SEG]', '<p>', '</p>',
        '<vp>', '</vp>',
        '<IMG_CONTEXT>',
        '<img>', '</img>'
    ]

    datasets = []

    # 1. SAV dataset via DescribeAnythingReferringDataset
    if args.sav_dir and os.path.exists(args.sav_dir):
        if int(os.environ.get('RANK', 0)) == 0:
            print(f"Loading SAV dataset from {args.sav_dir}")

        sav_dataset = DescribeAnythingReferringDataset(
            hf_dataset_name='nvidia/describe-anything-dataset',
            hf_dataset_config=['SAV'],
            dataset_repo_dir='/data/xyc/DAM_repo',
            local_data_root=args.sav_dir,
            cache_dir='./data/describe_anything_cache',
            tokenizer=dict(
                type=AutoTokenizer.from_pretrained,
                pretrained_model_name_or_path=args.model_path,
                trust_remote_code=True,
                padding_side='right'
            ),
            template_map_fn=template_map_fn,
            extra_image_processor=extra_image_processor,
            max_length=args.max_length,
            special_tokens=special_tokens,
            lazy=True,
            repeats=2,  # Repeat SAV 2x for higher sampling weight
        )
        datasets.append(sav_dataset)
        if int(os.environ.get('RANK', 0)) == 0:
            print(f"  SAV dataset: {len(sav_dataset)} samples (2x repeats)")

    # 2. RefCOCO datasets (refcoco, refcoco+, refcocog)
    if args.refcoco_dir and os.path.exists(args.refcoco_dir):
        for dataset_name in ['refcoco', 'refcoco+', 'refcocog']:
            split_by = 'unc' if dataset_name != 'refcocog' else 'umd'

            if int(os.environ.get('RANK', 0)) == 0:
                print(f"Loading {dataset_name} dataset")

            refcoco_dataset = ReferSegmDataset(
                tokenizer=dict(
                    type=AutoTokenizer.from_pretrained,
                    pretrained_model_name_or_path=args.model_path,
                    trust_remote_code=True,
                    padding_side='right'
                ),
                special_tokens=special_tokens,
                extra_image_processor=extra_image_processor,
                data_root=os.path.join(args.refcoco_dir, dataset_name),
                data_prefix=dict(img_path='coco2014/train2014/'),
                ann_file='instances.json',
                split_file=f'refs({split_by}).p',
                prompt_template=prompt_template,
                num_classes_per_sample=5,
                max_length=args.max_length,
            )

            # Repeat RefCOCO 4x for higher sampling weight (matching sa2va_4b.py)
            for _ in range(4):
                datasets.append(refcoco_dataset)

            if int(os.environ.get('RANK', 0)) == 0:
                print(f"  {dataset_name}: {len(refcoco_dataset)} samples (4x repeats)")

    if not datasets:
        raise ValueError("No datasets loaded! Check dataset paths.")

    # Combine datasets
    combined_dataset = ConcatDataset(datasets)

    if int(os.environ.get('RANK', 0)) == 0:
        print(f"\n✓ Total dataset size: {len(combined_dataset)} samples")
        print(f"  Number of sub-datasets: {len(datasets)}")

    return combined_dataset


def train_epoch(model, dataloader, optimizer, epoch, args, rank):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_mask_loss = 0
    total_dice_loss = 0

    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader

    for step, batch in enumerate(pbar):
        # Move batch to device
        data = batch['data']
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda(non_blocking=True)
            elif isinstance(data[key], list):
                # Handle list of tensors (e.g., g_pixel_values, masks)
                if len(data[key]) > 0 and isinstance(data[key][0], torch.Tensor):
                    data[key] = [item.cuda(non_blocking=True) for item in data[key]]

        # Forward pass
        try:
            loss_dict = model(data, data_samples=None, mode='loss')

            # Extract losses
            loss = loss_dict.get('loss', None)
            if loss is None:
                # Compute total loss from components
                loss_mask = loss_dict.get('loss_mask', 0)
                loss_dice = loss_dict.get('loss_dice', 0)
                loss = loss_mask + loss_dice
            else:
                loss_mask = loss_dict.get('loss_mask', 0)
                loss_dice = loss_dict.get('loss_dice', 0)

            # Backward pass
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            # Update weights
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            # Logging
            total_loss += loss.item() * args.gradient_accumulation_steps
            if isinstance(loss_mask, torch.Tensor):
                total_mask_loss += loss_mask.item()
            if isinstance(loss_dice, torch.Tensor):
                total_dice_loss += loss_dice.item()

            if rank == 0 and step % args.log_interval == 0:
                avg_loss = total_loss / (step + 1)
                avg_mask_loss = total_mask_loss / (step + 1)
                avg_dice_loss = total_dice_loss / (step + 1)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'mask': f'{avg_mask_loss:.4f}',
                    'dice': f'{avg_dice_loss:.4f}'
                })

        except Exception as e:
            if rank == 0:
                print(f"Error at step {step}: {e}")
                import traceback
                traceback.print_exc()
            continue

        # Save checkpoint
        if rank == 0 and (step + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_step_{step+1}.pth')
            torch.save({
                'epoch': epoch,
                'step': step + 1,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"\n✓ Saved checkpoint: {checkpoint_path}")

    return total_loss / len(dataloader)


def main():
    args = parse_args()

    # Setup distributed training
    is_distributed, rank, world_size, gpu = setup_distributed()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build model
    model, tokenizer = build_model(args)
    model = model.cuda()

    # Wrap with DDP
    if is_distributed:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=False)

    # Build datasets
    dataset = build_datasets(args, tokenizer)

    # Create dataloader
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=video_lisa_collate_fn,
        pin_memory=True,
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.05
    )

    # Training loop
    for epoch in range(args.num_epochs):
        if is_distributed:
            sampler.set_epoch(epoch)

        avg_loss = train_epoch(model, dataloader, optimizer, epoch, args, rank)

        if rank == 0:
            print(f"\nEpoch {epoch} completed. Average loss: {avg_loss:.4f}")

    if rank == 0:
        print("\n✓ Training completed!")


if __name__ == '__main__':
    main()
