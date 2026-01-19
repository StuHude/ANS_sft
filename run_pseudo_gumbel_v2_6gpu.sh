#!/bin/bash
# Run Pseudo Token + ST Gumbel-Softmax Training V2 on 6 GPUs (0-5)
#
# Requested settings:
# - SAV: all samples (no --sav_max_samples)
# - SA-1B: 1/4 of previous (10000 -> 2500)
# - OpenImage: 200000
#
# NOTE: /data is currently 100% full; write outputs/logs to /home (has free space).

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

RUN_NAME=pseudo_gumbel_v2_6gpu_len256_bs16_topk512_lr4e-5_acc2_save500_savALL_sa1b2500_oi200k
# /data has free space again; keep outputs under repo work_dirs to avoid filling /home.
OUT_DIR=/data/xyc/ANS/work_dirs/${RUN_NAME}
LOG_FILE=/data/xyc/ANS/work_dirs/${RUN_NAME}.log

echo "===================================================================================="
echo "Pseudo Token + ST Gumbel-Softmax Training V2 (6 GPU)"
echo "===================================================================================="
echo "Run: ${RUN_NAME}"
echo "Output: ${OUT_DIR}"
echo "Log: ${LOG_FILE}"
echo ""
echo "Key settings:"
echo "  - GPUs: 0-5 (6 ranks)"
echo "  - caption_len: 256"
echo "  - topk: 512"
echo "  - SAV: ALL"
echo "  - SA-1B max: 2500"
echo "  - OpenImage max: 200000"
echo "  - epochs: 2"
echo "  - batch_size(per GPU): 16"
echo "  - grad_accum: 2"
echo "  - lr: 4e-5"
echo "  - save_interval(optimizer steps): 500"
echo "  - num_workers(per GPU): 4"
echo ""

docker exec -w /data/xyc/ANS vlm-env bash -lc "
mkdir -p /data/xyc/ANS/work_dirs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NCCL_TIMEOUT=3600
export PYTHONPATH=/data/xyc/ANS:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup torchrun --nproc_per_node=6 --master_port=29720 \
    projects/llava_sam2/mask_caption_sft/train_pseudo_gumbel_v2.py \
    --pretrained_pth /data/xyc/ANS/pretrained/sa2va_4b_from_hf.pth \
    --model_path /data/xyc/ANS/pretrained/InternVL2_5-4B \
    --output_dir ${OUT_DIR} \
    --sav_dir /data/xyc/formed_data/npz \
    --sa1b_dir /data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw \
    --openimage_dir /data/xyc/openv7/data \
    --refcoco_dir /data/xyc/ANS/data/ref_seg \
    --sa1b_max_samples 2500 \
    --openimage_max_samples 200000 \
    --sav_repeats 1 \
    --refcoco_repeats 1 \
    --num_epochs 2 \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 4e-5 \
    --ema_decay 0.999 \
    --gumbel_tau 0.7 \
    --topk 512 \
    --mask_ratio 0.25 \
    --max_caption_len 256 \
    --lora_r 128 \
    --lora_alpha 256 \
    --num_workers 4 \
    --ref_num_workers 0 \
    --log_interval 10 \
    --save_interval 500 \
    --seed 42 \
    > ${LOG_FILE} 2>&1 & echo \"PID: \$!\"
"

echo ""
echo "===================================================================================="
echo "Training finished. Log: ${LOG_FILE}"
echo "===================================================================================="
