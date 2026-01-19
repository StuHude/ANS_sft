#!/bin/bash
#
# Single GPU Training - With EMA Model
#

# Environment
export CUDA_VISIBLE_DEVICES=0  # 单GPU训练
export NCCL_DEBUG=WARN

# Paths
PRETRAINED_PTH="/data/xyc/ANS/pretrain_hf"
MODEL_PATH="./pretrained/InternVL2_5-4B"
OUTPUT_DIR="./work_dirs/dual_loop_sft_single_gpu"

# Dataset paths (ALL 4 datasets)
SAV_DIR="/data/xyc/formed_data/npz"
SA1B_DIR="/data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw"
REFCOCO_DIR="./data/ref_seg"
OPENIMAGE_DIR="/data/xyc/openv7/data"

# Dataset sampling
SAV_REPEATS=2
REFCOCO_REPEATS=4
SA1B_MAX_SAMPLES=1000  # Limit for faster iteration

# Training params - SINGLE GPU
NUM_EPOCHS=1
BATCH_SIZE=1          # 1 per GPU
GRAD_ACC=64           # 梯度累积64步 (effective batch = 64)
LR=1e-5
NUM_WORKERS=4

# LoRA params
LORA_R=128
LORA_ALPHA=256

# Logging
LOG_INTERVAL=10
SAVE_INTERVAL=500

echo "========================================="
echo "Dual-Loop Training - SINGLE GPU"
echo "========================================="
echo "GPU: 1"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACC"
echo "Effective batch size: $GRAD_ACC"
echo "========================================="

# Run training (single GPU, no torchrun)
python projects/llava_sam2/mask_caption_sft/train_dual_loop.py \
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

echo "========================================="
echo "Training completed!"
echo "========================================="
