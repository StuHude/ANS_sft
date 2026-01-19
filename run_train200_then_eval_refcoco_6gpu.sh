#!/bin/bash
# Train for exactly 200 optimizer steps, save a lightweight `.pth` (LoRA + trained heads), then run RefCOCO eval.
#
# Notes:
# - Uses DDP 6 GPUs (0-5) for training.
# - Training checkpoint at step 200 is adapter-centric (LoRA + a few modules), so eval loads:
#   `pretrained/sa2va_4b_from_hf.pth` as base, then applies the step-200 checkpoint on top.
# - Runs RefCOCO eval on 1 GPU using `projects/llava_sam2/evaluation/refcoco_eval.py`.

set -euo pipefail

if ! docker ps | grep -q vlm-env; then
  echo "Starting vlm-env container..."
  docker start vlm-env
  sleep 2
fi

RUN_NAME=pseudo_gumbel_v2_train200_bs16_topk512_lr4e-5_acc2_savALL_sa1b2500_oi200k
OUT_DIR=/data/xyc/ANS/work_dirs/${RUN_NAME}
LOG_FILE=/data/xyc/ANS/work_dirs/${RUN_NAME}.log

CKPT_STEP=200
CKPT_PATH=${OUT_DIR}/checkpoint_step_${CKPT_STEP}.pth

EVAL_LOG=/data/xyc/ANS/work_dirs/${RUN_NAME}.refcoco_eval.log
PIPELINE_LOG=/data/xyc/ANS/work_dirs/${RUN_NAME}.pipeline.log

echo "===================================================================================="
echo "Train 200 steps then RefCOCO eval (6 GPU train + 1 GPU eval)"
echo "Run: ${RUN_NAME}"
echo "Output: ${OUT_DIR}"
echo "Train log: ${LOG_FILE}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Eval log: ${EVAL_LOG}"
echo "Pipeline log: ${PIPELINE_LOG}"
echo "===================================================================================="

docker exec -w /data/xyc/ANS vlm-env bash -lc "
set -euo pipefail

mkdir -p ${OUT_DIR}

nohup bash -lc '
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
  export NCCL_TIMEOUT=3600
  export PYTHONPATH=/data/xyc/ANS:${PYTHONPATH:-}
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  echo \"[INFO] Starting training (max_steps=${CKPT_STEP})...\"
  torchrun --nproc_per_node=6 --master_port=29720 \
    projects/llava_sam2/mask_caption_sft/train_pseudo_gumbel_v2.py \
    --pretrained_pth /data/xyc/ANS/pretrained/sa2va_4b_from_hf.pth \
    --model_path /data/xyc/ANS/pretrained/InternVL2_5-4B \
    --output_dir ${OUT_DIR} \
    --sav_dir /data/xyc/formed_data/npz \
    --sa1b_dir /data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw \
    --openimage_dir /data/xyc/openv7/data \
    --refcoco_dir /data/xyc/ANS/data/ref_seg \
    --sa1b_max_samples 2500 \
    --openimage_max_samples 200000 \
    --sav_repeats 1 \
    --refcoco_repeats 1 \
    --num_epochs 999 \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 4e-5 \
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
    --save_interval 100000000 \
    --max_steps ${CKPT_STEP} \
    --seed 42 \
    > ${LOG_FILE} 2>&1

  echo \"[INFO] Training finished, starting RefCOCO eval...\" | tee -a ${LOG_FILE}

  export CUDA_VISIBLE_DEVICES=0
  python projects/llava_sam2/evaluation/refcoco_eval.py \
    /data/xyc/ANS/pretrain_hf \
    --base_state_dict_pth /data/xyc/ANS/pretrained/sa2va_4b_from_hf.pth \
    --state_dict_pth ${CKPT_PATH} \
    --dataset refcoco \
    --split val \
    --launcher none \
    --data_path ./data/ref_seg/ \
    --image_folder ./data/ref_seg/refcoco/coco2014/train2014/ \
    > ${EVAL_LOG} 2>&1

  echo \"[INFO] Eval done. See: ${EVAL_LOG}\"
' > ${PIPELINE_LOG} 2>&1 &
"

echo "Launched job inside container. Tail logs:"
echo "  tail -f ${LOG_FILE}"
echo "  tail -f ${EVAL_LOG}"
echo "  tail -f ${PIPELINE_LOG}"
