#!/bin/bash
#
# Run Mask Captioning SFT Training on GPU 4-7 with batch_size=4
#

export CUDA_VISIBLE_DEVICES=4,5,6,7

echo "Training on GPU 4-7 with batch_size=4"

cd /data/xyc/ANS

torchrun --nproc_per_node=4 \
    --master_port=29600 \
    projects/llava_sam2/mask_caption_sft/train_mask_caption_sft.py \
    --config projects/llava_sam2/mask_caption_sft/config.py \
    --model_path /data/xyc/ANS/pretrain_hf \
    --sav_dir /data/xyc/formed_data/npz \
    --sa1b_dir /data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw \
    --openimage_dir /data/xyc/openv7/data \
    --refcoco_dir /data/xyc/ANS/data/ref_seg \
    --output_dir ./work_dirs/mask_caption_sft_gpu4_7 \
    --num_epochs 1 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2 \
    --num_workers 4 \
    --log_interval 10 \
    --save_interval 1000 \
    --use_lora \
    --lora_r 128 \
    --lora_alpha 256
