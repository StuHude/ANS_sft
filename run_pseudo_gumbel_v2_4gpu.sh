#!/bin/bash
# Run Pseudo Token + ST Gumbel-Softmax Training V2 (Corrected) on GPU 4-7

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
echo "Pseudo Token + ST Gumbel-Softmax Training V2 (Corrected Implementation)"
echo "===================================================================================="
echo ""
echo "Key Features:"
echo "  ✓ Uses correct sa2va_4b.py model configuration"
echo "  ✓ Implements complete training loop with Gumbel-Softmax"
echo "  ✓ Gradient flow through text_embeds verified with sanity checks"
echo "  ✓ Uses 4 datasets: SAV, SA1B, OpenImage, RefCOCO"
echo ""
echo "Training in Docker container vlm-env on GPU 4-7..."
echo "===================================================================================="
echo ""

# Run training inside Docker with GPU 4-7
docker exec -w /data/xyc/ANS vlm-env bash -c "
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_TIMEOUT=3600

# Add project root to Python path
export PYTHONPATH=/data/xyc/ANS:\$PYTHONPATH

torchrun --nproc_per_node=4 --master_port=29700 \
    projects/llava_sam2/mask_caption_sft/train_pseudo_gumbel_v2.py \
    --pretrained_pth /data/xyc/ANS/pretrained/sa2va_4b_from_hf.pth \
    --model_path /data/xyc/ANS/pretrained/InternVL2_5-4B \
    --output_dir ./work_dirs/pseudo_gumbel_v2_4gpu \
    --sav_dir /data/xyc/formed_data/npz \
    --sa1b_dir /data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw \
    --openimage_dir /data/xyc/openv7/data \
    --refcoco_dir /data/xyc/ANS/data/ref_seg \
    --sav_max_samples 100 \
    --sa1b_max_samples 500 \
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
    --lora_r 128 \
    --lora_alpha 256 \
    --num_workers 4 \
    --log_interval 10 \
    --save_interval 500 \
    2>&1 | tee work_dirs/pseudo_gumbel_v2_4gpu.log
"

echo ""
echo "===================================================================================="
echo "Training finished. Check work_dirs/pseudo_gumbel_v2_4gpu.log for details."
echo "Tensorboard logs in: work_dirs/pseudo_gumbel_v2_4gpu/logs/"
echo "===================================================================================="
