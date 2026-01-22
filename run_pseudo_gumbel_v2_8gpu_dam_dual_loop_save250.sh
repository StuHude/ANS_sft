#!/bin/bash
# 8GPU training with DAM dual-loop enabled; saves every 250 optimizer steps.
set -euo pipefail

if ! docker ps | grep -q vlm-env; then
  echo "Starting vlm-env container..."
  docker start vlm-env
  sleep 2
fi

OUT_DIR=${OUT_DIR:-./work_dirs/pseudo_gumbel_v2_dam_dual_loop_8gpu_save250}
MASTER_PORT=${MASTER_PORT:-29721}

mkdir -p "${OUT_DIR}"

# Run in background so we can monitor with tail/grep.
docker exec -w /data/xyc/ANS vlm-env bash -lc "
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_TIMEOUT=3600
export PYTHONPATH=/data/xyc/ANS:\$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PSEUDO_GUMBEL_TOPK_CHUNK=8

# Detach cleanly from docker exec: avoid a nohup|tee pipeline (tee may die on exec exit).
nohup bash -lc \"torchrun --nproc_per_node=8 --master_port=${MASTER_PORT} \
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
  --lora_r 128 \
  --lora_alpha 256 \
  --num_workers 4 \
  --ref_num_workers 0 \
  --log_interval 10 \
  --save_interval 250 \
  --enable_dam_dual_loop \
  --dam_data_root /data/xyc/ANS/data/describe-anything-dataset/describe-anything-dataset \
  --dam_splits SAV,COCOStuff,LVIS,Mapillary,OpenImages,PACO \
  --dam_num_workers 0 \
  --dam_beta 0.5\" \
  > ${OUT_DIR}/train.log 2>&1 &

echo \$! > ${OUT_DIR}/train.pid
echo \"started pid=\$(cat ${OUT_DIR}/train.pid)\";
"

echo "Launched. Log: ${OUT_DIR}/train.log"
