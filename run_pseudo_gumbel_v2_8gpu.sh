#!/bin/bash
# Run Pseudo Token + ST Gumbel-Softmax Training V2 on 8 GPUs (0-7)

set -e

# Check if Docker container is running
if ! docker ps | grep -q vlm-env; then
    echo "Starting vlm-env container..."
    docker start vlm-env || {
        echo "Error: Could not start vlm-env container"
        exit 1
    }
    sleep 2
fi

echo "===================================================================================="
echo "Pseudo Token + ST Gumbel-Softmax Training V2 (8 GPU)"
echo "===================================================================================="
echo ""
echo "Key settings:"
echo "  - caption_len: 256"
echo "  - OpenImage max: 100000"
echo "  - SA-1B max: 10000"
echo "  - SAV max: 200000"
echo "  - epochs: 2"
echo "  - batch_size(per GPU): 8"
echo "  - num_workers(per GPU): 4"
echo ""

docker exec -w /data/xyc/ANS vlm-env bash -lc "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_TIMEOUT=3600
export PYTHONPATH=/data/xyc/ANS:\$PYTHONPATH

torchrun --nproc_per_node=8 --master_port=29710 \
    projects/llava_sam2/mask_caption_sft/train_pseudo_gumbel_v2.py \
    --pretrained_pth /data/xyc/ANS/pretrained/sa2va_4b_from_hf.pth \
    --model_path /data/xyc/ANS/pretrained/InternVL2_5-4B \
    --output_dir ./work_dirs/pseudo_gumbel_v2_8gpu_len256_bs8 \
    --sav_dir /data/xyc/formed_data/npz \
    --sa1b_dir /data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw \
    --openimage_dir /data/xyc/openv7/data \
    --refcoco_dir /data/xyc/ANS/data/ref_seg \
    --sav_max_samples 200000 \
    --sa1b_max_samples 10000 \
    --openimage_max_samples 100000 \
    --sav_repeats 1 \
    --refcoco_repeats 1 \
    --num_epochs 2 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --ema_decay 0.999 \
    --gumbel_tau 0.7 \
    --topk 128 \
    --mask_ratio 0.25 \
    --max_caption_len 256 \
    --lora_r 128 \
    --lora_alpha 256 \
    --num_workers 4 \
    --log_interval 10 \
    --save_interval 500 \
    2>&1 | tee work_dirs/pseudo_gumbel_v2_8gpu_len256_bs8.log
"

echo ""
echo "===================================================================================="
echo "Training finished. Log: work_dirs/pseudo_gumbel_v2_8gpu_len256_bs8.log"
echo "===================================================================================="
