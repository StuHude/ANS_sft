#!/bin/bash
# Smoke run: launch pseudo-gumbel v2 training on the last 6 GPUs (CUDA 2-7)
# and verify it can progress beyond 20 steps without crashing.

set -euo pipefail

if ! docker ps | grep -q vlm-env; then
  echo "Starting vlm-env container..."
  docker start vlm-env
  sleep 2
fi

RUN_NAME=_smoke_last6_pseudo_gumbel_v2_over20
OUT_DIR=/data/xyc/ANS/work_dirs/${RUN_NAME}
TRAIN_LOG=/data/xyc/ANS/work_dirs/${RUN_NAME}.train.log
PIPELINE_LOG=/data/xyc/ANS/work_dirs/${RUN_NAME}.pipeline.log

echo "[INFO] RUN=${RUN_NAME}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] TRAIN_LOG=${TRAIN_LOG}"
echo "[INFO] PIPELINE_LOG=${PIPELINE_LOG}"

docker exec -w /data/xyc/ANS vlm-env bash -lc "
set -euo pipefail

rm -rf ${OUT_DIR}
mkdir -p ${OUT_DIR}

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export NCCL_TIMEOUT=3600
export PYTHONPATH=/data/xyc/ANS:\${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PSEUDO_GUMBEL_DEBUG_GENERATE=0
export PSEUDO_GUMBEL_TOPK_CHUNK=8

TRAIN_PORT=\$(shuf -i 20000-39999 -n 1)
echo \"[INFO] master_port=\${TRAIN_PORT}\"

torchrun --nproc_per_node=6 --master_port=\${TRAIN_PORT} \
  projects/llava_sam2/mask_caption_sft/train_pseudo_gumbel_v2.py \
  --pretrained_pth /data/xyc/ANS/pretrained/sa2va_4b_from_hf.pth \
  --model_path /data/xyc/ANS/pretrained/InternVL2_5-4B \
  --output_dir ${OUT_DIR} \
  --sav_dir /data/xyc/formed_data/npz \
  --sa1b_dir /data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw \
  --openimage_dir /data/xyc/openv7/data \
  --refcoco_dir /data/xyc/ANS/data/ref_seg \
  --sav_max_samples 200000 \
  --sa1b_max_samples 10000 \
  --openimage_max_samples 100000 \
  --sav_repeats 1 \
  --refcoco_repeats 1 \
  --num_epochs 999 \
  --batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --ema_decay 0.999 \
  --gumbel_tau 0.7 \
  --topk 512 \
  --mask_ratio 0.25 \
  --max_caption_len 256 \
  --num_workers 4 \
  --ref_num_workers 0 \
  --log_interval 10 \
  --save_interval 1000000000 \
  --max_steps 30 \
  --ref_llm_loss_weight 1.0 \
  --pseudo_llm_loss_weight 1.0 \
  --save_full_model \
  --seed 42 \
  > ${TRAIN_LOG} 2>&1
" > ${PIPELINE_LOG} 2>&1 &

echo "[INFO] Launched. Tail with: tail -f ${TRAIN_LOG}"

