import argparse
import copy
import math
import os
import torch
import tqdm
from pycocotools import mask as _mask
import numpy as np
import random

from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)

from utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu
from dataset import RESDataset
import sys
from pathlib import Path
import inspect

# 把仓库根目录加进 sys.path，和训练脚本写法一样
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--base_state_dict_pth',
        default=None,
        help='Optional base `.pth` (e.g. `pretrained/sa2va_4b_from_hf.pth`) to load before `--state_dict_pth`.')
    parser.add_argument(
        '--state_dict_pth',
        default=None,
        help='Optional `.pth` to load on top of `model_path` / `--base_state_dict_pth` (supports full state_dict or LoRA-only).')
    parser.add_argument('--lora_r', type=int, default=128, help='LoRA rank (for merging LoRA-only checkpoints).')
    parser.add_argument('--lora_alpha', type=float, default=256, help='LoRA alpha (for merging LoRA-only checkpoints).')
    parser.add_argument(
        '--image_folder',
        default=None,
        help='COCO image folder (defaults to the ref_seg bundled path in this repo).')
    parser.add_argument(
        '--data_path',
        default=None,
        help='Referring dataset root (defaults to `./data/ref_seg/`).')
    parser.add_argument(
        '--dataset',
        choices=DATASETS_ATTRIBUTES.keys(),
        default='refcoco',
        help='Specify a ref dataset')
    parser.add_argument(
        '--split',
        default='val',
        help='Specify a split')
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Optional cap on number of dataset samples to evaluate (for quick sanity checks).')
    parser.add_argument(
        '--max_texts_per_image',
        type=int,
        default=None,
        help='Optional cap on number of referring expressions per image (for quick sanity checks).')
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=None,
        help='Override generation `max_new_tokens` for eval (useful if the model does not hit stop criteria).')
    parser.add_argument(
        '--force_seg',
        action='store_true',
        help='Teacher-force a fixed assistant answer containing `[SEG]` and decode masks from its hidden state '
             '(avoids relying on free-form generation of `[SEG]`).')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


DATASETS_ATTRIBUTES = {
    'refcoco': {'splitBy': "unc", 'dataset_name': 'refcoco'},
    'refcoco_plus': {'splitBy': "unc", 'dataset_name': 'refcoco_plus'},
    'refcocog': {'splitBy': "umd", 'dataset_name': 'refcocog'},
}

DEFAULT_IMAGE_FOLDER = './data/ref_seg/refcoco/coco2014/train2014/'
DEFAULT_DATA_PATH = './data/ref_seg/'


def to_numpy_mask(x) -> np.ndarray:
    """兼容 torch.Tensor / numpy.ndarray，统一成 uint8 的 HxW numpy 掩码（阈值0.5）"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        pass
    else:
        raise TypeError(f"Unknown mask type: {type(x)}")
    # 挤掉可能的单通道维度
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"Expect 2D mask after squeeze, got shape {x.shape}")
    if x.dtype != np.uint8:
        x = (x > 0.5).astype(np.uint8)
    return x


def to_list_of_masks(arr) -> list:
    """把输入统一为 [H×W, H×W, ...] 的列表（numpy.uint8）"""
    if arr is None:
        return []
    if isinstance(arr, (list, tuple)):
        return [to_numpy_mask(m) for m in arr]
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    else:
        arr = np.asarray(arr)
    if arr.ndim == 2:
        return [to_numpy_mask(arr)]
    elif arr.ndim == 3:
        return [to_numpy_mask(arr[i]) for i in range(arr.shape[0])]
    else:
        raise ValueError(f"Unsupported mask shape: {arr.shape}")


def mask_to_rle(mask_list):
    """mask_list: [H×W, H×W, ...]；返回 COCO RLE 列表"""
    rle = []
    for m in mask_list:
        m = np.asfortranarray(m.astype(np.uint8))
        enc = _mask.encode(m)  # 对 2D 数组返回 dict
        enc['counts'] = enc['counts'].decode()
        rle.append(enc)
    return rle

from projects.llava_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel
from transformers import AutoTokenizer

def _unwrap_state_dict(obj):
    """Unwrap common checkpoint containers to a flat `state_dict`."""
    if not isinstance(obj, dict):
        raise TypeError(f'Unsupported checkpoint type: {type(obj)}')
    if 'state_dict' in obj and isinstance(obj['state_dict'], dict):
        return obj['state_dict']
    if 'model' in obj and isinstance(obj['model'], dict):
        return obj['model']
    return obj


def _map_checkpoint_key_to_sa2va_chat(k: str) -> str:
    # DDP prefix
    if k.startswith('module.'):
        k = k[len('module.'):]
    # Training code wraps InternVL/VideoLLaVA under `mllm.model.*`
    if k.startswith('mllm.model.'):
        k = k[len('mllm.model.'):]

    # PEFT LoRA checkpoints may save base weights under `language_model.base_model.*`
    k = k.replace('language_model.base_model.model.model.', 'language_model.model.')
    k = k.replace('language_model.base_model.model.lm_head.', 'language_model.lm_head.')
    # NOTE: keep `grounding_encoder.sam2_model.*` unchanged (Sa2VAChatModel uses this prefix).
    return k


def _map_checkpoint_to_sa2va_chat(sd: dict) -> dict:
    """
    Map common training checkpoints to Sa2VAChatModel key space.
    """
    return {_map_checkpoint_key_to_sa2va_chat(k): v for k, v in sd.items()}


def _merge_lora_into_model_(model: torch.nn.Module, mapped_sd: dict, *, lora_r: int, lora_alpha: float):
    """
    Merge LoRA A/B weights into base Linear weights in-place.

    Expects keys like:
      language_model.model.layers.{i}.self_attn.q_proj.lora_A.default.weight
      language_model.model.layers.{i}.self_attn.q_proj.lora_B.default.weight
    """
    named_params = dict(model.named_parameters())

    def _base_from_lora_key(k: str, which: str) -> str:
        suffix = f'.lora_{which}.default.weight'
        if not k.endswith(suffix):
            raise ValueError(f'Unexpected LoRA key: {k}')
        return k[:-len(suffix)]

    lora_A = {}
    lora_B = {}
    for k, v in mapped_sd.items():
        if '.lora_A.default.weight' in k:
            lora_A[_base_from_lora_key(k, 'A')] = v
        elif '.lora_B.default.weight' in k:
            lora_B[_base_from_lora_key(k, 'B')] = v

    bases = sorted(set(lora_A.keys()) & set(lora_B.keys()))
    if len(bases) == 0:
        return 0

    scale = float(lora_alpha) / float(lora_r)
    merged = 0
    with torch.no_grad():
        for base in bases:
            target_key = base + '.weight'
            param = named_params.get(target_key, None)
            if param is None:
                continue

            A = lora_A[base]
            B = lora_B[base]
            # Match device/dtype of the target weight for efficient matmul/add.
            A = A.to(device=param.device, dtype=param.dtype)
            B = B.to(device=param.device, dtype=param.dtype)
            delta = (B @ A) * scale
            param.add_(delta)
            merged += 1
    return merged


def _load_checkpoint_into_model(model: torch.nn.Module, pth: str, *, lora_r: int, lora_alpha: float):
    ckpt = torch.load(pth, map_location='cpu')
    sd = _unwrap_state_dict(ckpt)
    if not isinstance(sd, dict):
        raise TypeError(f'Checkpoint `{pth}` does not contain a dict state_dict.')
    mapped = _map_checkpoint_to_sa2va_chat(sd)

    # If it's a LoRA-only ckpt (or mostly LoRA), merge first then load the remaining non-LoRA keys.
    has_lora = any('.lora_A.default.weight' in k or '.lora_B.default.weight' in k for k in mapped.keys())
    if has_lora:
        merged = _merge_lora_into_model_(model, mapped, lora_r=lora_r, lora_alpha=lora_alpha)
        # Drop LoRA keys to avoid `unexpected` spam.
        mapped = {k: v for k, v in mapped.items() if '.lora_A.default.weight' not in k and '.lora_B.default.weight' not in k}
        print(f"[INFO] LoRA merge: merged {merged} matrices (scale={lora_alpha}/{lora_r}={float(lora_alpha)/float(lora_r):g}).")

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print(f"[INFO] Loaded `{pth}`. missing={len(missing)} unexpected={len(unexpected)}")
    if len(unexpected) > 0:
        print("[WARN] Unexpected keys (first 20):", unexpected[:20])
    if len(missing) > 0:
        print("[WARN] Missing keys (first 20):", missing[:20])
    return missing, unexpected

def main():
    args = parse_args()

    if args.launcher != 'none':
        _init_dist_pytorch('nccl')
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # build model
    model = Sa2VAChatModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda().eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    # Prepare generation config once so we can override generation params from CLI.
    if hasattr(model, 'preparing_for_generation'):
        model.preparing_for_generation(tokenizer=tokenizer)
        if args.max_new_tokens is not None and hasattr(model, 'gen_config') and model.gen_config is not None:
            model.gen_config.max_new_tokens = int(args.max_new_tokens)

    if args.base_state_dict_pth is not None:
        print(f"[INFO] Loading base state_dict from: {args.base_state_dict_pth}")
        _load_checkpoint_into_model(model, args.base_state_dict_pth, lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    if args.state_dict_pth is not None:
        print(f"[INFO] Loading extra state_dict from: {args.state_dict_pth}")
        _load_checkpoint_into_model(model, args.state_dict_pth, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    dataset_info = DATASETS_ATTRIBUTES[args.dataset]

    image_folder = args.image_folder or DEFAULT_IMAGE_FOLDER
    data_path = args.data_path or DEFAULT_DATA_PATH

    dataset = RESDataset(
        image_folder=image_folder,
        dataset_name=dataset_info['dataset_name'],
        data_path=data_path,
        split=args.split,
    )
    if args.max_samples is not None:
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))

    debug_printed = False  # 只在 rank==0 的第一条样本上打印一次

    for idx in tqdm.tqdm(per_rank_ids):
        data_batch = dataset[idx]

        # ---- GT 处理：统一成 RLE 列表 ----
        gt_masks_list = to_list_of_masks(data_batch['gt_masks'])
        gt_rle = mask_to_rle(gt_masks_list)

        prediction = {
            'img_id': data_batch['img_id'],
            'gt_masks': gt_rle
        }

        texts = data_batch['text']
        if args.max_texts_per_image is not None:
            texts = texts[:args.max_texts_per_image]
        # 清理出给模型的 batch
        del data_batch['img_id'], data_batch['gt_masks'], data_batch['text']

        pred_masks = []
        for ti, text in enumerate(texts):
            _data_batch = copy.deepcopy(data_batch)
            _data_batch['text'] = text

            # Backward/forward compatibility:
            # Some Sa2VAChatModel versions do not accept `force_seg`.
            sig = inspect.signature(model.predict_forward)
            if 'force_seg' in sig.parameters:
                out = model.predict_forward(**_data_batch, tokenizer=tokenizer, force_seg=bool(args.force_seg))
            else:
                out = model.predict_forward(**_data_batch, tokenizer=tokenizer)
            pred_mask = out.get('prediction_masks', None)

            # ---- 自检：只打印一次（rank==0 & 第一条样本的第一个文本）----
            if (not debug_printed) and rank == 0:
                print("[DEBUG] prediction_masks type:", type(pred_mask))
                if isinstance(pred_mask, (list, tuple)) and len(pred_mask) > 0:
                    print("[DEBUG] prediction_masks[0] type:", type(pred_mask[0]),
                          "shape:", getattr(pred_mask[0], "shape", None))
                else:
                    print("[DEBUG] prediction_masks empty or not list/tuple.")
                debug_printed = True

            if pred_mask is None or len(pred_mask) == 0:
                print("No seg pred !!!")
                # Make eval robust: treat "no prediction" as an empty mask (all zeros),
                # so metrics can still be computed (IoU=0 for that expression).
                if len(gt_masks_list) > 0:
                    empty = np.zeros_like(gt_masks_list[0], dtype=np.uint8)
                    pred_masks.append(mask_to_rle([empty]))
                else:
                    pred_masks.append([])
            else:
                # 只用第一张（保持与你原逻辑一致）；若想融合多张可改为 OR
                first_mask = to_numpy_mask(pred_mask[0])
                rle_list = mask_to_rle([first_mask])  # 注意包成列表
                pred_masks.append(rle_list)

        prediction.update({'prediction_masks': pred_masks})
        results.append(prediction)

    tmpdir = './dist_test_temp_res_' + args.dataset + args.split + args.model_path.replace('/', '').replace('.', '')
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    if get_rank() == 0:
        # `Subset` does not expose `.evaluate`; delegate to the underlying dataset.
        eval_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
        metric = eval_dataset.evaluate(results, './work_dirs')
        print(metric)


if __name__ == '__main__':
    main()
