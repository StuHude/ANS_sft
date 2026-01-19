#!/bin/bash
#
# Test Dual-Loop Mask Caption SFT Training
# Loop: image+mask → caption → mask' → segmentation loss
#

# Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN

# Paths
PRETRAINED_PTH="/data/xyc/ANS/pretrain_hf"
MODEL_PATH="./pretrained/InternVL2_5-4B"
OUTPUT_DIR="./work_dirs/dual_loop_sft_test"

# Dataset paths
SAV_DIR="/data/xyc/formed_data/npz"
REFCOCO_DIR="./data/ref_seg"
SA1B_DIR="/data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw"
OPENIMAGE_DIR="/data/xyc/openv7/data"  # Updated to correct OpenImage path

# Dataset sampling limits (for testing)
SA1B_MAX_SAMPLES=500  # Limit SA1B samples to 500 for quick testing

# Dataset repeat weights (increase sampling probability)
SAV_REPEATS=2         # SAV repeated 2x for higher weight
REFCOCO_REPEATS=4     # RefCOCO repeated 4x (matching original Sa2VA config)

# Training params
NUM_EPOCHS=1
BATCH_SIZE=1
GRAD_ACC=4
LR=1e-5

echo "========================================="
echo "Dual-Loop SFT Training (TEST)"
echo "========================================="
echo "Pretrained: $PRETRAINED_PTH"
echo "Output: $OUTPUT_DIR"
echo "SA1B samples: $SA1B_MAX_SAMPLES (limited for testing)"
echo "Dataset repeats: SAV=${SAV_REPEATS}x, RefCOCO=${REFCOCO_REPEATS}x"
echo "========================================="

# Run training
# Note: OpenImage dataset will only be loaded if the directory exists
python projects/llava_sam2/mask_caption_sft/train_dual_loop.py \
    --pretrained_pth $PRETRAINED_PTH \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --sav_dir $SAV_DIR \
    --refcoco_dir $REFCOCO_DIR \
    --sa1b_dir $SA1B_DIR \
    --openimage_dir $OPENIMAGE_DIR \
    --sa1b_max_samples $SA1B_MAX_SAMPLES \
    --sav_repeats $SAV_REPEATS \
    --refcoco_repeats $REFCOCO_REPEATS \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC \
    --learning_rate $LR \
    --lora_r 128 \
    --lora_alpha 256 \
    --num_workers 4 \
    --log_interval 5 \
    --save_interval 100

echo "========================================="
echo "Test completed!"
echo "========================================="
