import argparse
from pathlib import Path
import shutil
import torch

from projects.llava_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Convert training .pth to HF format (simple)')
    parser.add_argument('--model_path', required=True, help='Base HF model path (e.g. pretrain_hf)')
    parser.add_argument('--base_state_dict_pth', default=None, help='Optional base .pth to load')
    parser.add_argument('--state_dict_pth', required=True, help='Training .pth to load (full or LoRA-only)')
    parser.add_argument('--lora_r', type=int, default=128)
    parser.add_argument('--lora_alpha', type=float, default=256)
    parser.add_argument('--output_dir', required=True, help='Output HF dir')
    return parser.parse_args()


def _unwrap_state_dict(obj):
    if not isinstance(obj, dict):
        raise TypeError(f'Unsupported checkpoint type: {type(obj)}')
    if 'state_dict' in obj and isinstance(obj['state_dict'], dict):
        return obj['state_dict']
    if 'model' in obj and isinstance(obj['model'], dict):
        return obj['model']
    return obj


def _map_checkpoint_key_to_sa2va_chat(k: str) -> str:
    if k.startswith('module.'):
        k = k[len('module.'):]
    if k.startswith('mllm.model.'):
        k = k[len('mllm.model.'):]
    k = k.replace('language_model.base_model.model.model.', 'language_model.model.')
    k = k.replace('language_model.base_model.model.lm_head.', 'language_model.lm_head.')
    return k


def _map_checkpoint_to_sa2va_chat(sd: dict) -> dict:
    return {_map_checkpoint_key_to_sa2va_chat(k): v for k, v in sd.items()}


def _merge_lora_into_model_(model: torch.nn.Module, mapped_sd: dict, *, lora_r: int, lora_alpha: float):
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

            A = lora_A[base].to(device=param.device, dtype=param.dtype)
            B = lora_B[base].to(device=param.device, dtype=param.dtype)
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

    has_lora = any('.lora_A.default.weight' in k or '.lora_B.default.weight' in k for k in mapped.keys())
    if has_lora:
        merged = _merge_lora_into_model_(model, mapped, lora_r=lora_r, lora_alpha=lora_alpha)
        mapped = {k: v for k, v in mapped.items() if '.lora_A.default.weight' not in k and '.lora_B.default.weight' not in k}
        print(f"[INFO] LoRA merge: merged {merged} matrices (scale={lora_alpha}/{lora_r}={float(lora_alpha)/float(lora_r):g}).")

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print(f"[INFO] Loaded `{pth}`. missing={len(missing)} unexpected={len(unexpected)}")
    if len(unexpected) > 0:
        print("[WARN] Unexpected keys (first 20):", unexpected[:20])
    if len(missing) > 0:
        print("[WARN] Missing keys (first 20):", missing[:20])
    return missing, unexpected


def _copy_hf_skeleton(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        name = item.name
        if name.startswith('model-') and name.endswith('.safetensors'):
            continue
        if name == 'model.safetensors.index.json':
            continue
        if name == 'pytorch_model.bin':
            continue
        if name.endswith('.safetensors'):
            continue
        if item.is_dir():
            shutil.copytree(item, dst_dir / name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst_dir / name)


def main():
    args = parse_args()

    model = Sa2VAChatModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda().eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    if hasattr(model, 'preparing_for_generation'):
        model.preparing_for_generation(tokenizer=tokenizer)

    if args.base_state_dict_pth:
        print(f"[INFO] Loading base state_dict from: {args.base_state_dict_pth}")
        _load_checkpoint_into_model(model, args.base_state_dict_pth,
                                    lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    print(f"[INFO] Loading training state_dict from: {args.state_dict_pth}")
    _load_checkpoint_into_model(model, args.state_dict_pth,
                                lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    output_dir = Path(args.output_dir)
    _copy_hf_skeleton(Path(args.model_path), output_dir)
    torch.save(model.state_dict(), output_dir / 'pytorch_model.bin')
    tokenizer.save_pretrained(output_dir)
    print(f"[INFO] Saved HF model to {output_dir}")


if __name__ == '__main__':
    main()
