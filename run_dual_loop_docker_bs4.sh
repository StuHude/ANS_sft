#!/bin/bash
#
# Dual-Loop Training in Docker - 6 GPU BS=4
# Uses GPUs 0-5, batch_size=4 per GPU
# Validates after 1 minute only
#

set -e

echo "========================================="
echo "Dual-Loop Training - 6 GPU BS=4 (Docker)"
echo "========================================="

# Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800

# Paths
PRETRAINED_PTH="/data/xyc/ANS/pretrain_hf"
MODEL_PATH="./pretrained/InternVL2_5-4B"
OUTPUT_DIR="./work_dirs/dual_loop_sft_docker_bs4"
LOG_FILE="/tmp/dual_loop_docker_bs4.log"

# Dataset paths
SAV_DIR="/data/xyc/formed_data/npz"
SA1B_DIR="/data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw"
REFCOCO_DIR="./data/ref_seg"
OPENIMAGE_DIR="/data/xyc/openv7/data"

# Dataset sampling
SAV_REPEATS=2
REFCOCO_REPEATS=4
SA1B_MAX_SAMPLES=2000

# Training params - 6 GPUs with BS=4
NUM_EPOCHS=1
BATCH_SIZE=4          # 4 per GPU
GRAD_ACC=3            # Gradient accumulation
LR=1e-5
NUM_WORKERS=6

# LoRA params
LORA_R=128
LORA_ALPHA=256

# Logging
LOG_INTERVAL=5
SAVE_INTERVAL=200

echo "GPUs: 6 (0-5)"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACC"
echo "Effective batch size: $((BATCH_SIZE * 6 * GRAD_ACC))"
echo "========================================="
echo "Datasets:"
echo "  - SAV: ${SAV_REPEATS}x repeats"
echo "  - SA1B: ${SA1B_MAX_SAMPLES} samples"
echo "  - OpenImage: 1x"
echo "  - RefCOCO: ${REFCOCO_REPEATS}x repeats"
echo "========================================="

# Clear log
mkdir -p ${OUTPUT_DIR}
> ${LOG_FILE}

echo "Starting training in Docker..."
echo "Log: ${LOG_FILE}"
echo ""

# Run in docker
docker run --rm -d \
    --name sa2va_dual_loop_bs4 \
    --gpus '"device=0,1,2,3,4,5"' \
    --shm-size=64g \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /data:/data \
    -v $(pwd):/workspace \
    -w /workspace \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
    -e NCCL_DEBUG=WARN \
    -e NCCL_TIMEOUT=1800 \
    pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel \
    bash -c "
        torchrun --nproc_per_node=6 \
            --master_port=29501 \
            projects/llava_sam2/mask_caption_sft/train_dual_loop.py \
            --pretrained_pth $PRETRAINED_PTH \
            --model_path $MODEL_PATH \
            --output_dir $OUTPUT_DIR \
            --sav_dir $SAV_DIR \
            --sa1b_dir $SA1B_DIR \
            --openimage_dir $OPENIMAGE_DIR \
            --refcoco_dir $REFCOCO_DIR \
            --sa1b_max_samples $SA1B_MAX_SAMPLES \
            --sav_repeats $SAV_REPEATS \
            --refcoco_repeats $REFCOCO_REPEATS \
            --num_epochs $NUM_EPOCHS \
            --batch_size $BATCH_SIZE \
            --gradient_accumulation_steps $GRAD_ACC \
            --learning_rate $LR \
            --lora_r $LORA_R \
            --lora_alpha $LORA_ALPHA \
            --num_workers $NUM_WORKERS \
            --log_interval $LOG_INTERVAL \
            --save_interval $SAVE_INTERVAL
    " > ${LOG_FILE} 2>&1

echo "Container started: sa2va_dual_loop_bs4"
echo ""
echo "Waiting 60 seconds for validation..."
sleep 60

# Check if container is still running
if docker ps | grep -q sa2va_dual_loop_bs4; then
    echo ""
    echo "========================================="
    echo "✓ Training is running successfully!"
    echo "========================================="
    echo ""
    echo "Last 25 lines of log:"
    docker logs sa2va_dual_loop_bs4 --tail 25
    echo ""
    echo "Container: sa2va_dual_loop_bs4"
    echo "To monitor: docker logs -f sa2va_dual_loop_bs4"
    echo "To stop: docker stop sa2va_dual_loop_bs4"
    echo ""

    # Quick status
    docker logs sa2va_dual_loop_bs4 2>&1 | grep -c "out of memory" > /tmp/oom_count.txt || echo "0" > /tmp/oom_count.txt
    docker logs sa2va_dual_loop_bs4 2>&1 | grep -c "loss" > /tmp/step_count.txt || echo "0" > /tmp/step_count.txt

    OOM_COUNT=$(cat /tmp/oom_count.txt)
    STEP_COUNT=$(cat /tmp/step_count.txt)

    echo "Quick Status:"
    echo "  OOM errors: ${OOM_COUNT}"
    echo "  Training steps: ${STEP_COUNT}"
    echo ""
else
    echo ""
    echo "========================================="
    echo "✗ Container exited within 60 seconds"
    echo "========================================="
    echo ""
    echo "Full logs:"
    docker logs sa2va_dual_loop_bs4
    exit 1
fi
