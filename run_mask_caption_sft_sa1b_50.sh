#!/bin/bash
#
# 4 datasets training: SA1B limited to 50 samples, SAV limited to 10 samples (for testing)
# GPU 4-7, batch_size=4
#

echo "Starting training with SA1B=50, SAV=10 samples on GPU 4-7"

docker exec -w /data/xyc/ANS vlm-env bash -c "
export CUDA_VISIBLE_DEVICES=4,5,6,7

nohup torchrun --nproc_per_node=4 \
    --master_port=29650 \
    projects/llava_sam2/mask_caption_sft/train_mask_caption_sft.py \
    --config projects/llava_sam2/mask_caption_sft/config.py \
    --model_path /data/xyc/ANS/pretrain_hf \
    --sav_dir /data/xyc/formed_data/npz \
    --sav_max_samples 10 \
    --sa1b_dir /data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw \
    --sa1b_max_samples 50 \
    --openimage_dir /data/xyc/openv7/data \
    --refcoco_dir /data/xyc/ANS/data/ref_seg \
    --output_dir ./work_dirs/mask_caption_sft_gpu4_7 \
    --num_epochs 1 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2 \
    --num_workers 2 \
    --log_interval 10 \
    --save_interval 500 \
    --use_lora \
    --lora_r 128 \
    --lora_alpha 256 \
    > /data/xyc/ANS/work_dirs/mask_caption_sa1b50.log 2>&1 &

echo 'Training started. Monitor with:'
echo 'tail -f /data/xyc/ANS/work_dirs/mask_caption_sa1b50.log'
"
