#!/bin/bash
#
# Multi-GPU distributed training script for Mask Captioning SFT
#
# Usage:
#   bash projects/llava_sam2/mask_caption_sft/scripts/run_multi_gpu.sh 8  # for 8 GPUs

NUM_GPUS=${1:-8}

echo "Training with $NUM_GPUS GPUs"

torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    projects/llava_sam2/mask_caption_sft/train_mask_caption_sft.py \
    --config projects/llava_sam2/mask_caption_sft/config.py \
    --model_path /data/xyc/ANS/pretrain_hf \
    --sav_dir /data/xyc/formed_data/npz \
    --sa1b_dir /data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw \
    --openimage_dir /data/xyc/openv7/data \
    --refcoco_dir /data/xyc/ANS/data/ref_seg \
    --output_dir ./work_dirs/mask_caption_sft \
    --num_epochs 1 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --num_workers 4 \
    --log_interval 10 \
    --save_interval 1000 \
    --use_lora \
    --lora_r 128 \
    --lora_alpha 256
