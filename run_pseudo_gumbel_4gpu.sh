#!/bin/bash
# Run Pseudo Token + ST Gumbel-Softmax training on GPU 4-7

export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_TIMEOUT=3600  # 1 hour timeout

# Change to project directory
cd /data/xyc/ANS

# Run training with torchrun
torchrun --nproc_per_node=4 --master_port=29600 \
    projects/llava_sam2/mask_caption_sft/train_pseudo_gumbel.py \
    --pretrained_pth /data/xyc/ANS/pretrained/sa2va_4b_from_hf.pth \
    --model_path /data/xyc/ANS/pretrained/InternVL2_5-4B \
    --output_dir ./work_dirs/pseudo_gumbel_4gpu \
    --sav_dir /data/xyc/formed_data/npz \
    --sa1b_dir /data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw \
    --openimage_dir /data/xyc/openv7/data \
    --refcoco_dir /data/xyc/ANS/data/ref_seg \
    --sav_max_samples 10 \
    --sa1b_max_samples 50 \
    --sav_repeats 1 \
    --refcoco_repeats 1 \
    --num_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --ema_decay 0.999 \
    --gumbel_tau 0.7 \
    --topk 128 \
    --mask_ratio 0.25 \
    --max_caption_len 64 \
    --num_workers 4 \
    --log_interval 10 \
    --save_interval 500 \
    2>&1 | tee work_dirs/pseudo_gumbel_4gpu.log
