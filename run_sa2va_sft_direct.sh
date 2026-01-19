#!/bin/bash
#
# Direct Sa2VA SFT Training using tools/train.py
# Uses original sa2va_4b.py config with selected datasets
# Runs in vlm-env container, 6 GPUs, BS=4
#

set -e

echo "========================================="
echo "Sa2VA Direct SFT Training - 6 GPU BS=4"
echo "========================================="

# Run in vlm-env container
docker exec vlm-env bash -c '
    cd /data/xyc/ANS && \
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 && \
    export NCCL_DEBUG=WARN && \
    export NCCL_TIMEOUT=1800 && \
    \
    # Use tools/dist.sh for distributed training
    bash tools/dist.sh train \
        projects/llava_sam2/configs/sa2va_4b.py \
        6 \
        2>&1 | tee /tmp/sa2va_sft_direct.log &
'

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
docker exec vlm-env tail -n 30 /tmp/sa2va_sft_direct.log 2>&1 || echo "Log not available yet"

echo ""
echo "To monitor: docker exec vlm-env tail -f /tmp/sa2va_sft_direct.log"
echo ""

# Check if training is running
RUNNING=$(docker exec vlm-env bash -c "ps aux | grep 'train.py' | grep -v grep | wc -l" 2>/dev/null || echo "0")

if [ "$RUNNING" -gt 0 ]; then
    echo "✓ Training process is running!"
else
    echo "✗ Training process not found"
    echo ""
    echo "Recent log:"
    docker exec vlm-env cat /tmp/sa2va_sft_direct.log 2>&1 | tail -100
fi
