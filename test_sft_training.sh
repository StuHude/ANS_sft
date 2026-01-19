#!/bin/bash
#
# Test script for Mask Caption SFT Training with real losses
# Uses limited data for quick testing
#

# Set environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN

# Training parameters
MODEL_PATH="/data/xyc/ANS/pretrain_hf"
OUTPUT_DIR="./work_dirs/mask_caption_sft_test"
NUM_EPOCHS=1
BATCH_SIZE=1
GRAD_ACC_STEPS=4
LR=1e-5

# Dataset paths
SAV_DIR="/data/xyc/formed_data/npz"
REFCOCO_DIR="./data/ref_seg"
SA1B_DIR="/data/xyc/mhx/SA1b/OpenDataLab___SA-1B/dataset"

# SA1B限制样本数（用于快速测试）
SA1B_MAX_SAMPLES=1000  # 只使用1000个SA1B样本进行测试

echo "========================================="
echo "Testing Mask Caption SFT Training"
echo "========================================="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACC_STEPS"
echo "SA1B max samples: $SA1B_MAX_SAMPLES"
echo "========================================="

# Run training
torchrun --nproc_per_node=4 \
    --master_port=29501 \
    projects/llava_sam2/mask_caption_sft/train_mask_caption_sft.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --learning_rate $LR \
    --sav_dir $SAV_DIR \
    --refcoco_dir $REFCOCO_DIR \
    --sa1b_dir $SA1B_DIR \
    --sa1b_max_samples $SA1B_MAX_SAMPLES \
    --use_lora \
    --lora_r 128 \
    --lora_alpha 256 \
    --num_workers 4 \
    --log_interval 5 \
    --save_interval 100

echo "========================================="
echo "Training completed!"
echo "========================================="
