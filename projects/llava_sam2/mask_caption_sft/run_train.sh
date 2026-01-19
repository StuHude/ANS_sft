#!/bin/bash

# Mask Captioning SFT Training Script

set -e

echo "=================================================================="
echo "Mask Captioning + Referring Segmentation SFT Training"
echo "=================================================================="

# Configuration
MODEL_PATH="/data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new"
OUTPUT_DIR="./work_dirs/mask_caption_sft"

# Dataset paths - CHANGE THESE to your actual paths
SAV_DIR="/path/to/sav/npz/data"  # CHANGE THIS
SA1B_DIR="/data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw"
OPENIMAGE_DIR="/data/xyc/openv7/data"
REFCOCO_DIR="/data/xyc/ANS/data/ref_seg"

# Training hyperparameters
NUM_EPOCHS=1
BATCH_SIZE=2
LEARNING_RATE=1e-5
GRADIENT_ACCUM_STEPS=4
MAX_GRAD_NORM=1.0
EMA_DECAY=0.999

# LoRA config
USE_LORA="--use_lora"
LORA_R=128
LORA_ALPHA=256

# Logging
LOG_INTERVAL=10
SAVE_INTERVAL=1000

# Number of workers
NUM_WORKERS=4

echo "Configuration:"
echo "  Model path: ${MODEL_PATH}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Num epochs: ${NUM_EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Use LoRA: ${USE_LORA}"
echo ""

# Activate environment
export PATH="/home/xiaoyicheng/miniconda3/bin:$PATH"
conda activate vlm
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Run training
python projects/llava_sam2/mask_caption_sft/train_mask_caption_sft.py \
    --model_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --sav_dir ${SAV_DIR} \
    --sa1b_dir ${SA1B_DIR} \
    --openimage_dir ${OPENIMAGE_DIR} \
    --refcoco_dir ${REFCOCO_DIR} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUM_STEPS} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --ema_decay ${EMA_DECAY} \
    ${USE_LORA} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --num_workers ${NUM_WORKERS} \
    --log_interval ${LOG_INTERVAL} \
    --save_interval ${SAVE_INTERVAL}

echo ""
echo "=================================================================="
echo "Training completed!"
echo "=================================================================="
