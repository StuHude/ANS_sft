#!/bin/bash
#
# Single GPU training script for Mask Captioning SFT
#
# Usage:
#   bash projects/llava_sam2/mask_caption_sft/scripts/run_single_gpu.sh

python projects/llava_sam2/mask_caption_sft/train_mask_caption_sft.py \
    --config projects/llava_sam2/mask_caption_sft/config.py \
    --model_path /data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new \
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
