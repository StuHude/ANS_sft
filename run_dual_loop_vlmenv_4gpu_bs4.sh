#!/bin/bash
#
# Dual-Loop Training in vlm-env container - 4 GPU BS=4
# Uses GPUs 0-3, batch_size=4 per GPU
#

set -e

echo "========================================="
echo "Dual-Loop Training - 4 GPU BS=4 (vlm-env)"
echo "========================================="

# Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800

# Paths inside container
PRETRAINED_PTH="/data/xyc/ANS/pretrain_hf"
MODEL_PATH="/data/xyc/ANS/pretrained/InternVL2_5-4B"
OUTPUT_DIR="/data/xyc/ANS/work_dirs/dual_loop_sft_4gpu_bs4"

# Dataset paths
SAV_DIR="/data/xyc/formed_data/npz"
SA1B_DIR="/data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw"
REFCOCO_DIR="/data/xyc/ANS/data/ref_seg"
OPENIMAGE_DIR="/data/xyc/openv7/data"

# Dataset sampling
SAV_REPEATS=2
REFCOCO_REPEATS=4
SA1B_MAX_SAMPLES=2000

# Training params - 4 GPUs with BS=4
NUM_EPOCHS=1
BATCH_SIZE=4          # 4 per GPU
GRAD_ACC=2            # Gradient accumulation
LR=1e-5
NUM_WORKERS=6

# LoRA params
LORA_R=128
LORA_ALPHA=256

# Logging
LOG_INTERVAL=5
SAVE_INTERVAL=200

echo "GPUs: 4 (0-3)"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACC"
echo "Effective batch size: $((BATCH_SIZE * 4 * GRAD_ACC))"
echo "========================================="
echo "Datasets:"
echo "  - SAV: ${SAV_REPEATS}x repeats"
echo "  - SA1B: ${SA1B_MAX_SAMPLES} samples"
echo "  - OpenImage: 1x"
echo "  - RefCOCO: ${REFCOCO_REPEATS}x repeats"
echo "========================================="

# Clear and create output directory
mkdir -p ${OUTPUT_DIR}

echo "Starting training..."
echo "Output: ${OUTPUT_DIR}"
echo ""

# Run training with torchrun
torchrun --nproc_per_node=4 \
    --master_port=29501 \
    /data/xyc/ANS/projects/llava_sam2/mask_caption_sft/train_dual_loop.py \
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

echo ""
echo "========================================="
echo "Training completed!"
echo "========================================="
