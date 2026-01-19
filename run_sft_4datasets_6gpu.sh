#!/bin/bash
#
# SFT Training with 4 Datasets - 6 GPU BS=4
# Uses original Sa2VA dataset loaders
# Runs in vlm-env container
#

set -e

echo "========================================="
echo "Sa2VA SFT Training - 4 Datasets (6 GPU BS=4)"
echo "========================================="

# Paths
PRETRAINED_PTH="/data/xyc/ANS/pretrain_hf"
MODEL_PATH="./pretrained/InternVL2_5-4B"
OUTPUT_DIR="./work_dirs/sft_4datasets_6gpu"

# Dataset paths
SAV_DIR="/data/xyc/DAM_data"
REFCOCO_DIR="./data/ref_seg"

# Training params
NUM_EPOCHS=1
BATCH_SIZE=4
GRAD_ACC=3
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
echo "Effective batch: $((BATCH_SIZE * 6 * GRAD_ACC))"
echo "========================================="

# Run training in vlm-env container
docker exec vlm-env bash -c "
    cd /data/xyc/ANS && \
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 && \
    export NCCL_DEBUG=WARN && \
    export NCCL_TIMEOUT=1800 && \
    torchrun --nproc_per_node=6 \
        --master_port=29502 \
        projects/llava_sam2/mask_caption_sft/train_sft_4datasets.py \
        --pretrained_pth $PRETRAINED_PTH \
        --model_path $MODEL_PATH \
        --output_dir $OUTPUT_DIR \
        --sav_dir $SAV_DIR \
        --refcoco_dir $REFCOCO_DIR \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACC \
        --learning_rate $LR \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --num_workers $NUM_WORKERS \
        --log_interval $LOG_INTERVAL \
        --save_interval $SAVE_INTERVAL \
        2>&1 | tee /tmp/sft_4datasets_6gpu.log &
"

echo ""
echo "Training started in vlm-env container"
echo "Waiting 60 seconds for validation..."
sleep 60

echo ""
echo "========================================="
echo "Training Status (after 60 seconds):"
echo "========================================="
echo ""

# Show last 30 lines of log
docker exec vlm-env tail -n 30 /tmp/sft_4datasets_6gpu.log 2>&1 || echo "Log not available yet"

echo ""
echo "To monitor: docker exec vlm-env tail -f /tmp/sft_4datasets_6gpu.log"
echo "To check GPU: docker exec vlm-env nvidia-smi"
echo ""

# Check if training is running
RUNNING=$(docker exec vlm-env bash -c "ps aux | grep train_sft_4datasets | grep -v grep | wc -l" 2>/dev/null || echo "0")

if [ "$RUNNING" -gt 0 ]; then
    echo "✓ Training process is running!"
    echo ""
    # Quick stats
    LOSS_COUNT=$(docker exec vlm-env bash -c "grep -c 'loss' /tmp/sft_4datasets_6gpu.log 2>/dev/null || echo 0")
    OOM_COUNT=$(docker exec vlm-env bash -c "grep -c 'out of memory' /tmp/sft_4datasets_6gpu.log 2>/dev/null || echo 0")
    echo "Quick Stats:"
    echo "  Training steps: $LOSS_COUNT"
    echo "  OOM errors: $OOM_COUNT"
else
    echo "✗ Training process not found"
    echo ""
    echo "Full log:"
    docker exec vlm-env cat /tmp/sft_4datasets_6gpu.log 2>&1 | tail -100
fi
