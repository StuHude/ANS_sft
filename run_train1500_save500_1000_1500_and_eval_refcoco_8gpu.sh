#!/bin/bash
# Train to 1500 optimizer steps (8 GPUs), save full-model checkpoints at steps 500/1000/1500,
# then run 8-GPU RefCOCO eval on each checkpoint using the *same* generate-based inference path.

set -euo pipefail

if ! docker ps | grep -q vlm-env; then
  echo "Starting vlm-env container..."
  docker start vlm-env
  sleep 2
fi

RUN_NAME=pseudo_gumbel_v2_exportfix_train1500_8gpu_bs16_topk512_lr2e-5
OUT_DIR=/data/xyc/ANS/work_dirs/${RUN_NAME}
TRAIN_LOG=/data/xyc/ANS/work_dirs/${RUN_NAME}.train.log
PIPELINE_LOG=/data/xyc/ANS/work_dirs/${RUN_NAME}.pipeline.log

CKPT_500=${OUT_DIR}/checkpoint_step_500.pth
CKPT_1000=${OUT_DIR}/checkpoint_step_1000.pth
CKPT_1500=${OUT_DIR}/checkpoint_step_1500.pth

EVAL_LOG_500=/data/xyc/ANS/work_dirs/${RUN_NAME}.refcoco_eval_step500_8gpu.log
EVAL_LOG_1000=/data/xyc/ANS/work_dirs/${RUN_NAME}.refcoco_eval_step1000_8gpu.log
EVAL_LOG_1500=/data/xyc/ANS/work_dirs/${RUN_NAME}.refcoco_eval_step1500_8gpu.log

echo "===================================================================================="
echo "Train 1500 steps (save @500/@1000/@1500) + RefCOCO eval (8 GPUs)"
echo "Run:        ${RUN_NAME}"
echo "Output:     ${OUT_DIR}"
echo "Train log:  ${TRAIN_LOG}"
echo "Pipeline:   ${PIPELINE_LOG}"
echo "CKPT 500:   ${CKPT_500}"
echo "CKPT 1000:  ${CKPT_1000}"
echo "CKPT 1500:  ${CKPT_1500}"
echo "Eval500:    ${EVAL_LOG_500}"
echo "Eval1000:   ${EVAL_LOG_1000}"
echo "Eval1500:   ${EVAL_LOG_1500}"
echo "===================================================================================="

docker exec -w /data/xyc/ANS vlm-env bash -lc "
set -euo pipefail

rm -rf ${OUT_DIR}
mkdir -p ${OUT_DIR}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_TIMEOUT=3600
export PYTHONPATH=/data/xyc/ANS:\${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PSEUDO_GUMBEL_DEBUG_GENERATE=0
export PSEUDO_GUMBEL_TOPK_CHUNK=8

nohup bash -lc '
  set -euo pipefail
  TRAIN_PORT=\$(shuf -i 20000-39999 -n 1)
  EVAL500_PORT=\$(shuf -i 20000-39999 -n 1)
  EVAL1000_PORT=\$(shuf -i 20000-39999 -n 1)
  EVAL1500_PORT=\$(shuf -i 20000-39999 -n 1)
  echo \"[INFO] Starting training (max_steps=1500, save_interval=500, master_port=\${TRAIN_PORT})...\"
  torchrun --nproc_per_node=8 --master_port=\${TRAIN_PORT} \
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
    --save_interval 500 \
    --max_steps 1500 \
    --ref_llm_loss_weight 1.0 \
    --pseudo_llm_loss_weight 1.0 \
    --save_full_model \
    --seed 42 \
    > ${TRAIN_LOG} 2>&1

  echo \"[INFO] Training finished. Running RefCOCO eval on step 500...\" | tee -a ${TRAIN_LOG}
  torchrun --nproc_per_node=8 --master_port=\${EVAL500_PORT} \
    projects/llava_sam2/evaluation/refcoco_eval.py \
    /data/xyc/ANS/pretrain_hf \
    --base_state_dict_pth /data/xyc/ANS/pretrained/sa2va_4b_from_hf.pth \
    --state_dict_pth ${CKPT_500} \
    --dataset refcoco \
    --split val \
    --launcher pytorch \
    --data_path ./data/ref_seg/ \
    --image_folder ./data/ref_seg/refcoco/coco2014/train2014/ \
    > ${EVAL_LOG_500} 2>&1

  echo \"[INFO] Running RefCOCO eval on step 1000...\" | tee -a ${TRAIN_LOG}
  torchrun --nproc_per_node=8 --master_port=\${EVAL1000_PORT} \
    projects/llava_sam2/evaluation/refcoco_eval.py \
    /data/xyc/ANS/pretrain_hf \
    --base_state_dict_pth /data/xyc/ANS/pretrained/sa2va_4b_from_hf.pth \
    --state_dict_pth ${CKPT_1000} \
    --dataset refcoco \
    --split val \
    --launcher pytorch \
    --data_path ./data/ref_seg/ \
    --image_folder ./data/ref_seg/refcoco/coco2014/train2014/ \
    > ${EVAL_LOG_1000} 2>&1

  echo \"[INFO] Running RefCOCO eval on step 1500...\" | tee -a ${TRAIN_LOG}
  torchrun --nproc_per_node=8 --master_port=\${EVAL1500_PORT} \
    projects/llava_sam2/evaluation/refcoco_eval.py \
    /data/xyc/ANS/pretrain_hf \
    --base_state_dict_pth /data/xyc/ANS/pretrained/sa2va_4b_from_hf.pth \
    --state_dict_pth ${CKPT_1500} \
    --dataset refcoco \
    --split val \
    --launcher pytorch \
    --data_path ./data/ref_seg/ \
    --image_folder ./data/ref_seg/refcoco/coco2014/train2014/ \
    > ${EVAL_LOG_1500} 2>&1

  echo \"[INFO] Done.\"
' > ${PIPELINE_LOG} 2>&1 &"

echo "Done. Logs:"
echo "  ${TRAIN_LOG}"
echo "  ${PIPELINE_LOG}"
echo "  ${EVAL_LOG_500}"
echo "  ${EVAL_LOG_1000}"
echo "  ${EVAL_LOG_1500}"
