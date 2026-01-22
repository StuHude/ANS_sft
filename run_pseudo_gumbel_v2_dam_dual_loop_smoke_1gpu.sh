#!/bin/bash
set -euo pipefail

# Smoke run for the new DAM dual-loop mode (runs 1 optimizer step and exits).
# Run this inside an environment where `xtuner` is available (e.g. `vlm-env`).

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=/data/xyc/ANS:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

OUT_DIR=${OUT_DIR:-./work_dirs/pseudo_gumbel_v2_dam_dual_loop_smoke_1gpu}
mkdir -p "${OUT_DIR}"

python projects/llava_sam2/mask_caption_sft/train_pseudo_gumbel_v2.py \
  --pretrained_pth /data/xyc/ANS/pretrained/sa2va_4b_from_hf.pth \
  --model_path /data/xyc/ANS/pretrained/InternVL2_5-4B \
  --output_dir "${OUT_DIR}" \
  --sav_dir /data/xyc/formed_data/npz \
  --sa1b_dir /data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw \
  --openimage_dir /data/xyc/openv7/data \
  --refcoco_dir /data/xyc/ANS/data/ref_seg \
  --sav_max_samples 4 \
  --sa1b_max_samples 4 \
  --openimage_max_samples 4 \
  --num_epochs 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --ema_decay 0.999 \
  --topk 32 \
  --max_caption_len 32 \
  --num_workers 0 \
  --ref_num_workers 0 \
  --max_steps 1 \
  --save_interval 1 \
  --enable_dam_dual_loop \
  --dam_data_root /data/xyc/ANS/data/describe-anything-dataset/describe-anything-dataset \
  --dam_splits SAV,COCOStuff,LVIS,Mapillary,OpenImages,PACO \
  --dam_max_samples 8 \
  --dam_num_workers 0 \
  --dam_beta 0.5 \
  2>&1 | tee "${OUT_DIR}/run.log"

