#!/bin/bash
#
# Dual-Loop Training - MAXIMIZE GPU UTILIZATION
# 使用所有8个GPU，增大batch size占满显存
#

# Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  # 使用前6张GPU
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800

# Paths
PRETRAINED_PTH="/data/xyc/ANS/pretrain_hf"
MODEL_PATH="./pretrained/InternVL2_5-4B"
OUTPUT_DIR="./work_dirs/dual_loop_sft_gpu_max"

# Dataset paths (ALL 4 datasets)
SAV_DIR="/data/xyc/formed_data/npz"
SA1B_DIR="/data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw"
REFCOCO_DIR="./data/ref_seg"
OPENIMAGE_DIR="/data/xyc/openv7/data"

# Dataset sampling (matching original Sa2VA config)
SAV_REPEATS=2         # SAV repeated 2x for higher weight
REFCOCO_REPEATS=4     # RefCOCO repeated 4x (matching original config)
SA1B_MAX_SAMPLES=2000 # Use 2000 SA1B samples for faster iteration

# Training params - 6 GPUs with BS=4
NUM_EPOCHS=1
BATCH_SIZE=4          # 4 per GPU (6 GPUs × 4 = 24 samples per step)
GRAD_ACC=3            # 梯度累积3步 (effective batch = 24 × 3 = 72)
LR=1e-5
NUM_WORKERS=6         # 数据加载线程

# LoRA params
LORA_R=128
LORA_ALPHA=256

# Logging
LOG_INTERVAL=5
SAVE_INTERVAL=200

echo "========================================="
echo "Dual-Loop Training - 6 GPU BS=4"
echo "========================================="
echo "GPUs: 6 (parallel loading)"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACC"
echo "Effective batch size: $((BATCH_SIZE * 6 * GRAD_ACC))"
echo "Samples per step: $((BATCH_SIZE * 6))"
echo "========================================="
echo "Datasets:"
echo "  - SAV: $SAV_DIR (${SAV_REPEATS}x repeats)"
echo "  - SA1B: $SA1B_DIR (${SA1B_MAX_SAMPLES} samples)"
echo "  - OpenImage: $OPENIMAGE_DIR (1x)"
echo "  - RefCOCO: $REFCOCO_DIR (${REFCOCO_REPEATS}x repeats)"
echo "========================================="

# Run training with torchrun (6 GPUs)
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

echo "========================================="
echo "Training completed!"
echo "========================================="
