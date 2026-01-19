#!/bin/bash
#
# Full Dual-Loop Mask Caption SFT Training
# Loop: image+mask → caption → mask' → segmentation loss
# Using ALL data from 4 datasets
#

# Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800

# Paths
PRETRAINED_PTH="/data/xyc/ANS/pretrain_hf"
MODEL_PATH="./pretrained/InternVL2_5-4B"
OUTPUT_DIR="./work_dirs/dual_loop_sft_full"

# Dataset paths (ALL 4 datasets)
SAV_DIR="/data/xyc/formed_data/npz"
REFCOCO_DIR="./data/ref_seg"
SA1B_DIR="/data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw"
OPENIMAGE_DIR="/data/xyc/openv7/data"  # Updated to correct OpenImage path

# Dataset repeat weights (increase sampling probability)
SAV_REPEATS=2         # SAV repeated 2x for higher weight
REFCOCO_REPEATS=4     # RefCOCO repeated 4x (matching original Sa2VA config)

# Training params
NUM_EPOCHS=1
BATCH_SIZE=2
GRAD_ACC=4
LR=1e-5

echo "========================================="
echo "Dual-Loop SFT Training (FULL DATA)"
echo "========================================="
echo "Pretrained: $PRETRAINED_PTH"
echo "Output: $OUTPUT_DIR"
echo "Datasets:"
echo "  - SAV: $SAV_DIR (${SAV_REPEATS}x repeats)"
echo "  - SA1B: $SA1B_DIR (ALL samples, 1x)"
echo "  - OpenImage: $OPENIMAGE_DIR (1x)"
echo "  - RefCOCO: $REFCOCO_DIR (${REFCOCO_REPEATS}x repeats)"
echo "========================================="

# Run training with torchrun (8 GPUs)
torchrun --nproc_per_node=8 \
    --master_port=29501 \
    projects/llava_sam2/mask_caption_sft/train_dual_loop.py \
    --pretrained_pth $PRETRAINED_PTH \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --sav_dir $SAV_DIR \
    --refcoco_dir $REFCOCO_DIR \
    --sa1b_dir $SA1B_DIR \
    --openimage_dir $OPENIMAGE_DIR \
    --sav_repeats $SAV_REPEATS \
    --refcoco_repeats $REFCOCO_REPEATS \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC \
    --learning_rate $LR \
    --lora_r 128 \
    --lora_alpha 256 \
    --num_workers 6 \
    --log_interval 10 \
    --save_interval 500 \
    --local_rank $LOCAL_RANK

echo "========================================="
echo "Full training completed!"
echo "========================================="
