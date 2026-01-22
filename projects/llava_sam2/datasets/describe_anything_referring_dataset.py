# projects/llava_sam2/datasets/describe_anything_referring_dataset.py
# -*- coding: utf-8 -*-

import os
import io
import json
import pickle as pkl
import logging
import tarfile
from typing import List, Tuple, Union, Optional, Dict, Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

from datasets import (
    load_dataset,
    get_dataset_config_names,
    DownloadConfig,
)

from xtuner.registry import BUILDER
from xtuner.utils import DEFAULT_IMAGE_TOKEN
from .encode_fn import video_lisa_encode_fn

logger = logging.getLogger(__name__)

# ------- 视觉前处理（与 InternVL 设定一致） -------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INTERNVL_IMAGE_SIZE = 448
PATCH_SIZE = 14
DOWNSAMPLE_RATIO = 0.5

# token 网格边长 G 与每帧 token 数（需与模型前向一致）
GRID_SIZE = int((INTERNVL_IMAGE_SIZE // PATCH_SIZE) * DOWNSAMPLE_RATIO)  # 16
IMG_TOKENS_PER_FRAME = GRID_SIZE * GRID_SIZE                             # 256

# 文本占位 token（需加入 tokenizer 的 special_tokens）
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
VP_START_TOKEN = '<vp>'
VP_END_TOKEN = '</vp>'


def _ensure_rel_data_dir(module_dir: str, abs_data_root: Optional[str]) -> Optional[str]:
    if abs_data_root is None:
        return None
    rel = "_data"
    target = os.path.join(module_dir, rel)
    try:
        if not (os.path.islink(target) or os.path.exists(target)):
            os.symlink(abs_data_root, target)
    except Exception as e:
        logger.warning(f"Create symlink failed: {target} -> {abs_data_root}: {e}")
    return rel


def _decode_rle_to_mask(mask_rle: Dict[str, Any]) -> torch.Tensor:
    """HF/COCO RLE -> (1,H,W) float {0,1}"""
    try:
        from pycocotools import mask as cocomask
    except Exception as e:
        raise RuntimeError("需要 pycocotools 才能解码 RLE：pip install pycocotools") from e

    rle = dict(mask_rle)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")
    m = cocomask.decode(rle)  # (H,W) 或 (H,W,N)
    if m.ndim == 3:
        m = m[..., 0]
    m = torch.from_numpy(m.astype("float32"))  # (H,W)
    m = (m > 0.5).float().unsqueeze(0)  # (1,H,W)
    return m


def _is_valid_region_like(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    cap = item.get("caption", None)
    mrl = item.get("mask_rle", None)
    if not (isinstance(cap, str) and cap.strip()):
        return False
    if not (isinstance(mrl, dict) and ("size" in mrl) and ("counts" in mrl)):
        return False
    return True


def _normalize_dam_annotations(ann: Any) -> List[Dict[str, Any]]:
    """
    Normalize one DAM annotations.json entry into a list of region dicts.
    - Some splits store {id: {caption, mask_rle, ...}}.
    - Some splits store {id: [{caption, mask_rle, ...}, ...]}.
    """
    if isinstance(ann, list):
        return [it for it in ann if _is_valid_region_like(it)]
    if isinstance(ann, dict):
        if _is_valid_region_like(ann):
            return [ann]
        region_list = _as_region_list(ann)
        return [it for it in region_list if _is_valid_region_like(it)]
    return []


def _default_dam_root() -> str:
    # Repo-relative default (works for /data/xyc/ANS layout).
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(repo_root, "data", "describe-anything-dataset", "describe-anything-dataset")


def _build_dam_tar_index(images_dir: str, cache_path: Optional[str] = None) -> Dict[str, str]:
    """
    Build a mapping: sample_id -> shard_tar_path by scanning `*.pickle` entries in each tar shard.
    This keeps runtime random access O(1) per sample without brute-forcing all shards.
    """
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pkl.load(f)
        except Exception as e:
            logger.warning(f"Failed to load DAM tar index cache: {cache_path}: {e}")

    tar_paths = sorted(
        [os.path.join(images_dir, n) for n in os.listdir(images_dir) if n.endswith(".tar")]
    )
    index: Dict[str, str] = {}
    for tp in tar_paths:
        try:
            with tarfile.open(tp, "r") as tf:
                for name in tf.getnames():
                    if not name.endswith(".pickle"):
                        continue
                    sample_id = os.path.splitext(os.path.basename(name))[0]
                    index[sample_id] = tp
        except Exception as e:
            logger.warning(f"Failed to scan DAM shard: {tp}: {e}")

    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pkl.dump(index, f)
        except Exception as e:
            logger.warning(f"Failed to write DAM tar index cache: {cache_path}: {e}")
    return index


class _TarShardReader:
    def __init__(self, max_open: int = 8):
        self.max_open = int(max_open)
        self._open: Dict[str, tarfile.TarFile] = {}
        self._lru: List[str] = []

    def _touch(self, path: str) -> None:
        if path in self._lru:
            self._lru.remove(path)
        self._lru.append(path)

    def get(self, path: str) -> tarfile.TarFile:
        tf = self._open.get(path, None)
        if tf is not None:
            self._touch(path)
            return tf
        tf = tarfile.open(path, "r")
        self._open[path] = tf
        self._touch(path)
        while len(self._lru) > self.max_open:
            old = self._lru.pop(0)
            old_tf = self._open.pop(old, None)
            if old_tf is not None:
                try:
                    old_tf.close()
                except Exception:
                    pass
        return tf

    def close(self) -> None:
        for tf in self._open.values():
            try:
                tf.close()
            except Exception:
                pass
        self._open.clear()
        self._lru.clear()


@BUILDER.register_module()
class DescribeAnythingLocalRegionDataset(Dataset):
    """
    Local loader for the *extracted* Describe Anything Dataset folder layout:
      {data_root}/{SPLIT}/annotations.json
      {data_root}/{SPLIT}/images/*.tar  (each tar contains *.jpg + *.pickle)

    Returns raw tensors suitable for Sa2VA training loops:
      - pixel_values: (3,448,448) ImageNet normalized (InternVL)
      - g_pixel_values: (3,1024,1024) uint8 [0,255] (SAM2)
      - prompt_masks: (16,16) uint8 {0,1} (visual prompt grid)
      - masks_1024: (1024,1024) float {0,1} (GT mask)
      - caption: str
      - dataset_type: str
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        splits: Union[str, List[str], Tuple[str, ...]] = ("SAV",),
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        tar_cache_max_open: int = 8,
        **_ignore_kwargs,
    ) -> None:
        self.data_root = data_root or _default_dam_root()
        if isinstance(splits, str):
            self.splits = [splits]
        else:
            self.splits = list(splits)
        self.max_samples = int(max_samples) if (max_samples is not None) else None
        self.cache_dir = cache_dir
        self._tar_reader = _TarShardReader(max_open=tar_cache_max_open)
        self._ann_by_split: Dict[str, Dict[str, Any]] = {}

        # Transforms
        self._img_448 = transforms.Compose([
            transforms.Resize((INTERNVL_IMAGE_SIZE, INTERNVL_IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self._img_1024 = transforms.Compose([
            transforms.Resize((1024, 1024), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),  # [0,1]
        ])
        self._mask_1024 = transforms.Compose([
            transforms.Resize((1024, 1024), interpolation=InterpolationMode.NEAREST),
        ])

        # Build (sample_id, split, region_idx) list and shard indexes.
        self._samples: List[Tuple[str, str, int]] = []
        self._tar_index_by_split: Dict[str, Dict[str, str]] = {}
        for sp in self.splits:
            split_dir = os.path.join(self.data_root, sp)
            ann_path = os.path.join(split_dir, "annotations.json")
            img_dir = os.path.join(split_dir, "images")
            if not os.path.isfile(ann_path):
                raise FileNotFoundError(f"DAM annotations not found: {ann_path}")
            if not os.path.isdir(img_dir):
                raise FileNotFoundError(f"DAM images dir not found: {img_dir}")

            with open(ann_path, "r") as f:
                ann = json.load(f)
            if not isinstance(ann, dict):
                raise RuntimeError(f"Unexpected DAM annotations.json format: {ann_path} type={type(ann)}")
            self._ann_by_split[sp] = ann

            for sample_id, entry in ann.items():
                regions = _normalize_dam_annotations(entry)
                for ridx in range(len(regions)):
                    self._samples.append((str(sample_id), sp, ridx))

            cache_path = None
            if cache_dir is not None:
                cache_path = os.path.join(cache_dir, f"dam_tar_index_{sp}.pkl")
            self._tar_index_by_split[sp] = _build_dam_tar_index(img_dir, cache_path=cache_path)

        if self.max_samples is not None:
            self._samples = self._samples[: self.max_samples]

        logger.info(
            "DescribeAnythingLocalRegionDataset | "
            f"data_root={self.data_root}, splits={self.splits}, samples={len(self._samples)}, "
            f"cache_dir={self.cache_dir}"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def _load_image_from_tar(self, split: str, sample_id: str) -> Image.Image:
        tp = self._tar_index_by_split[split].get(sample_id, None)
        if tp is None:
            raise KeyError(f"Sample id not found in tar index: split={split} id={sample_id}")
        tf = self._tar_reader.get(tp)

        # SAV stores multiple frames as "{id}.{t}.jpg"; others commonly store "{id}.jpg".
        candidates = [f"{sample_id}.0.jpg", f"{sample_id}.jpg"]
        member = None
        for cand in candidates:
            try:
                member = tf.getmember(cand)
                break
            except KeyError:
                continue
        if member is None:
            # Fallback: find any jpg starting with sample_id (best-effort).
            prefix = f"{sample_id}."
            for n in tf.getnames():
                if n.startswith(prefix) and n.endswith(".jpg"):
                    member = tf.getmember(n)
                    break
        if member is None:
            raise KeyError(f"JPG not found in shard: split={split} id={sample_id}")

        f = tf.extractfile(member)
        if f is None:
            raise RuntimeError(f"Failed to extract {member.name} from {tp}")
        return Image.open(io.BytesIO(f.read())).convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_id, split, ridx = self._samples[idx]
        ann = self._ann_by_split[split]
        entry = ann.get(sample_id, None)
        if entry is None:
            return self.__getitem__((idx + 1) % len(self))
        regions = _normalize_dam_annotations(entry)
        if not regions:
            return self.__getitem__((idx + 1) % len(self))
        region = regions[ridx % len(regions)]

        caption = str(region["caption"]).strip()
        mask_rle = region["mask_rle"]

        img = self._load_image_from_tar(split, sample_id)
        img_448 = self._img_448(img)  # (3,448,448) normalized
        img_1024 = self._img_1024(img)  # (3,1024,1024) float [0,1]
        g_pixel_values = (img_1024 * 255).byte()

        mask_1hw = _decode_rle_to_mask(mask_rle)  # (1,H,W)
        # Resize mask to 1024 for GT and to 16x16 for prompt grid.
        mask_1024 = self._mask_1024(mask_1hw)  # (1,1024,1024)
        prompt_masks = F.adaptive_avg_pool2d(mask_1024, (GRID_SIZE, GRID_SIZE))  # (1,16,16)
        prompt_masks = (prompt_masks > 0.5).to(torch.uint8).squeeze(0)  # (16,16)

        return {
            "pixel_values": img_448,
            "g_pixel_values": g_pixel_values,
            "prompt_masks": prompt_masks,
            "masks_1024": mask_1024.squeeze(0),
            "caption": caption,
            "dataset_type": f"dam_{split.lower()}",
        }

def _wrap_image_processor(proc: Any) -> Callable[[Image.Image], torch.Tensor]:
    if callable(proc):
        def _call(img: Image.Image) -> torch.Tensor:
            out = proc(img)
            if isinstance(out, Image.Image):
                out = transforms.ToTensor()(out)
            elif not isinstance(out, torch.Tensor):
                out = transforms.ToTensor()(out)
            return out
        return _call

    for m in ("process", "forward", "transform", "resize"):
        if hasattr(proc, m):
            fn = getattr(proc, m)
            def _call(img: Image.Image, _fn=fn) -> torch.Tensor:
                out = _fn(img)
                if isinstance(out, Image.Image):
                    out = transforms.ToTensor()(out)
                elif not isinstance(out, torch.Tensor):
                    out = transforms.ToTensor()(out)
                return out
            return _call

    tfm = transforms.Compose([
        transforms.Resize((INTERNVL_IMAGE_SIZE, INTERNVL_IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    def _call(img: Image.Image, _tfm=tfm) -> torch.Tensor:
        return _tfm(img)

    return _call


# --------- meta 中 region 列表抽取 & 有效性判定 ---------
def _as_region_list(meta_obj: Any) -> List[dict]:
    if isinstance(meta_obj, list):
        return meta_obj
    if isinstance(meta_obj, dict):
        for k in ("regions", "items", "objects", "annotations", "list"):
            v = meta_obj.get(k, None)
            if isinstance(v, list):
                return v
        vals = list(meta_obj.values())
        if vals and all(isinstance(v, dict) for v in vals):
            return vals
    return []


def _is_valid_region(it: Any) -> bool:
    if not isinstance(it, dict):
        return False
    cap = it.get("caption", None)
    mrl = it.get("mask_rle", None)
    if not (isinstance(cap, str) and cap.strip()):
        return False
    if not (isinstance(mrl, dict) and ("size" in mrl) and ("counts" in mrl)):
        return False
    return True


@BUILDER.register_module()
class DescribeAnythingReferringDataset(Dataset):
    """
    与 Osprey 系列保持一致的“视觉提示做描述”数据集实现：
      - Dataset 只负责：图像、prompt_masks(二值网格)、vp_overall_mask、以及带 <vp>/<img> 占位的文本（对话为成对格式）。
      - 模型前向负责：将 prompt_masks → 视觉提示 token，并替换文本里对应的 <IMG_CONTEXT>*K 段。
    """

    def __init__(self,
                 hf_dataset_name: str,
                 hf_dataset_config: Union[str, List[str], Tuple[str, ...]],
                 tokenizer,
                 template_map_fn,
                 extra_image_processor,
                 max_length: int,
                 instruction_template: str,
                 local_data_root: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 repeats: int = 1,
                 lazy: bool = True,
                 hf_split: str = "train",
                 dataset_repo_dir: Optional[str] = None,
                 fast_pair_sampling: bool = True,
                 precomputed_total_pairs: Optional[int] = None,
                 **_ignore_kwargs) -> None:

        # 组件
        self.tokenizer = BUILDER.build(tokenizer) if isinstance(tokenizer, dict) else tokenizer
        self.template_map_fn = template_map_fn
        self._raw_image_processor = BUILDER.build(extra_image_processor) \
            if isinstance(extra_image_processor, dict) else extra_image_processor
        self._image_processor = _wrap_image_processor(self._raw_image_processor)

        self.max_length = int(max_length)
        self.instruction_template = str(instruction_template).strip()
        self.repeats = int(repeats)
        self.lazy = bool(lazy)

        # config 规范化
        if isinstance(hf_dataset_config, (list, tuple)):
            cfgs = list(dict.fromkeys(hf_dataset_config))
        else:
            cfgs = [hf_dataset_config]

        # 路径解析
        resolve_paths: List[Tuple[str, str, bool]] = []
        if dataset_repo_dir is not None:
            for c in cfgs:
                subdir = os.path.join(dataset_repo_dir, c)
                resolve_paths.append((c, subdir, True))
        else:
            for c in cfgs:
                resolve_paths.append((c, hf_dataset_name, False))

        # 校验
        if dataset_repo_dir is not None:
            bad = [c for (c, p, _) in resolve_paths if not os.path.isdir(p)]
            if bad:
                raise ValueError(f"以下子配置在本地脚本仓库中未找到：{bad}；请检查 dataset_repo_dir='{dataset_repo_dir}'。")
        else:
            available = set(get_dataset_config_names(hf_dataset_name))
            bad = [c for c in cfgs if c not in available]
            if bad:
                raise ValueError(f"无效的 hf_dataset_config: {bad}；可用配置：{sorted(available)}")

        logger.info(
            "DescribeAnythingReferringDataset | "
            f"dataset_repo_dir={dataset_repo_dir}, "
            f"hf_dataset_name={hf_dataset_name}, "
            f"configs={cfgs}, split='{hf_split}', data_dir='{local_data_root}', cache_dir='{cache_dir}', "
            f"lazy={self.lazy}, fast_pair_sampling={fast_pair_sampling}, precomputed_total_pairs={precomputed_total_pairs}"
        )

        # 加载 split（不解码）
        self.subsets: List[object] = []
        for (name, path, is_local_script) in [(n, p, l) for (n, p, l) in resolve_paths]:
            if is_local_script:
                rel_data_dir = _ensure_rel_data_dir(path, local_data_root)
                kwargs = dict(split=hf_split, cache_dir=cache_dir,
                              download_config=DownloadConfig(local_files_only=True))
                if rel_data_dir is not None:
                    kwargs["data_dir"] = rel_data_dir
                ds_split = load_dataset(path, **kwargs)
            else:
                ds_split = load_dataset(path, name=name, split=hf_split,
                                        data_dir=local_data_root, cache_dir=cache_dir)
            self.subsets.append(ds_split)

        # 两种模式：fast（惰性） vs full（逐 region）
        self.fast_pair_sampling = bool(fast_pair_sampling)
        if self.fast_pair_sampling:
            self.records: List[Tuple[int, int]] = []  # (subset_id, raw_idx)
            for subset_id, ds_split in enumerate(self.subsets):
                self.records.extend((subset_id, i) for i in range(len(ds_split)))

            if precomputed_total_pairs is not None:
                self._fast_total_pairs = int(precomputed_total_pairs)
            else:
                self._fast_total_pairs = max(1, len(self.records) * 2)

            self._mode = "fast"
            self.total_len = self._fast_total_pairs
        else:
            self.samples: List[Tuple[int, int, int]] = []
            for subset_id, ds_split in enumerate(self.subsets):
                for i in range(len(ds_split)):
                    rec = ds_split[i]
                    pb = rec.get('pickle', None)
                    if pb is None:
                        continue
                    try:
                        meta = pkl.loads(pb)
                    except Exception:
                        continue
                    region_list = _as_region_list(meta)
                    for j, it in enumerate(region_list):
                        if _is_valid_region(it):
                            self.samples.append((subset_id, i, j))
            self._mode = "full"
            self.total_len = len(self.samples)

        # 给 LengthGroupedSampler：固定长度数组
        self._modality_length_value = self.max_length
        self.modality_length = [self._modality_length_value] * max(1, self.total_len)

        # 汇总
        effective_len = self.total_len * self.repeats
        msg_header = (f"Dataset ready => mode={self._mode}, subsets={len(self.subsets)}, "
                      f"samples(pairs)={self.total_len}, repeats={self.repeats}, "
                      f"effective_len={effective_len}, split='{hf_split}'")
        logger.info(msg_header)
        print(msg_header, flush=True)

    def __len__(self) -> int:
        return self.total_len * self.repeats

    # ---- 工具 ----
    def _load_pil_from_record(self, rec: Dict[str, Any]) -> Image.Image:
        jpg_obj = rec['jpg']
        if isinstance(jpg_obj, (bytes, bytearray, memoryview)):
            return Image.open(io.BytesIO(jpg_obj)).convert("RGB")
        if isinstance(jpg_obj, Image.Image):
            return jpg_obj.convert("RGB") if jpg_obj.mode != "RGB" else jpg_obj
        try:
            from torchvision.transforms.functional import to_pil_image
            return to_pil_image(jpg_obj).convert("RGB")
        except Exception as e:
            raise TypeError(f"'jpg' 字段既不是 bytes 也不是 PIL.Image，且无法 to_pil：type={type(jpg_obj)}") from e

    def _apply_image_processor(self, img: Image.Image) -> torch.Tensor:
        out = self._image_processor(img)
        if not isinstance(out, torch.Tensor):
            out = transforms.ToTensor()(out)
        if out.ndim != 3:
            raise RuntimeError(f"extra_image_processor 应输出 (C,H,W) 张量，当前 shape={getattr(out, 'shape', None)}")
        return out

    def _map_fast_index_to_record_and_region(self, idx: int) -> Tuple[int, int, int]:
        base_len = len(self.records)
        base_i = idx % base_len
        round_i = idx // base_len
        subset_id, raw_idx = self.records[base_i]

        rec = self.subsets[subset_id][raw_idx]
        meta_obj = pkl.loads(rec['pickle'])
        region_list = _as_region_list(meta_obj)
        valid = [j for j, it in enumerate(region_list) if _is_valid_region(it)]
        if not valid:
            for step in range(1, base_len + 1):
                subset_id2, raw_idx2 = self.records[(base_i + step) % base_len]
                rec2 = self.subsets[subset_id2][raw_idx2]
                meta2 = pkl.loads(rec2['pickle'])
                region_list2 = _as_region_list(meta2)
                valid2 = [j for j, it in enumerate(region_list2) if _is_valid_region(it)]
                if valid2:
                    return subset_id2, raw_idx2, valid2[round_i % len(valid2)]
            return subset_id, raw_idx, 0

        region_idx = valid[round_i % len(valid)]
        return subset_id, raw_idx, region_idx

    # ---- 文本构造（Osprey 同构：成对格式） ----
    def _build_conversation_pairs(self, n_regions: int, region_pixels: List[int], answer: str) -> List[Dict[str, str]]:
        """
        构造一个 pair：
          input: "<image> There are N part regions ... regioni <vp> <IMG_CONTEXT>*Ki </vp> ...\\n" + instruction
          output: caption
        """
        start_region = f"<image> There are {n_regions} part regions in the picture: "
        parts = []
        for i in range(n_regions):
            K = int(region_pixels[i])
            parts.append(f"region{i+1}{VP_START_TOKEN}{IMG_CONTEXT_TOKEN * K}{VP_END_TOKEN}")
        start_region += (", ".join(parts) + ".\n")
        human_text = start_region + self.instruction_template

        return [{'input': human_text, 'output': answer}]

    @staticmethod
    def _replace_image_str(data_dict: Dict[str, Any], image_str: str) -> Dict[str, Any]:
        """
        把 DEFAULT_IMAGE_TOKEN 替换为 <img> + <IMG_CONTEXT>*num_image_tokens + </img>
        仅作用在第一条 pair 的 'input' 上（与 Osprey 保持一致）
        """
        if 'conversation' in data_dict and data_dict['conversation']:
            s = data_dict['conversation'][0]['input']
            s = s.replace(DEFAULT_IMAGE_TOKEN, '').strip()
            s = (DEFAULT_IMAGE_TOKEN + '\n' + s).strip()
            s = s.replace(DEFAULT_IMAGE_TOKEN, image_str)
            data_dict['conversation'][0]['input'] = s
        return data_dict

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 1) 选样本与 region
        if self.fast_pair_sampling:
            real_idx = idx % self.total_len
            subset_id, raw_idx, region_idx = self._map_fast_index_to_record_and_region(real_idx)
            rec = self.subsets[subset_id][raw_idx]
            meta_obj = pkl.loads(rec['pickle'])
            region_list = _as_region_list(meta_obj)
            item = region_list[region_idx]
            if not _is_valid_region(item):
                return self.__getitem__(idx + 1)
        else:
            real_idx = idx % self.total_len
            subset_id, raw_idx, region_idx = self.samples[real_idx]
            rec = self.subsets[subset_id][raw_idx]
            meta_obj = pkl.loads(rec['pickle'])
            region_list = _as_region_list(meta_obj)
            if not (0 <= region_idx < len(region_list)) or not _is_valid_region(region_list[region_idx]):
                return self.__getitem__(idx + 1)
            item = region_list[region_idx]

        # 2) 图像（给 MLLM）
        img = self._load_pil_from_record(rec)
        pixel_1 = self._apply_image_processor(img)                 # (3,448,448)
        pixel_values = torch.stack([pixel_1.contiguous()], dim=0)  # (1,3,H,W)
        n_tiles = pixel_values.shape[0]                            # 当前为 1

        # 3) region mask → G×G 二值网格（仅作“视觉提示占位”）
        caption: str = str(item.get('caption', '')).strip()
        mask_rle: Dict[str, Any] = item['mask_rle']
        mask_1hw = _decode_rle_to_mask(mask_rle)                   # (1,H0,W0)

        # 对齐到视觉输入尺寸
        mask_resizer = transforms.Resize(pixel_1.shape[-2:], interpolation=InterpolationMode.NEAREST)
        mask_1hw = mask_resizer(mask_1hw)                          # (1,H,W)

        # 聚合到 token 网格 G×G → 二值（与模型前向对齐）
        pooled_1gg = F.adaptive_avg_pool2d(mask_1hw, (GRID_SIZE, GRID_SIZE))  # (1,G,G)
        prompt_masks = (pooled_1gg > 0.5).to(torch.uint8)                     # (1,G,G)
        region_pixels = [int(prompt_masks[0].sum().item())]                   # 单 region 的 K

        # 4) 哪些帧承载这些 prompt_masks（本数据集单帧 → 该帧 True）
        vp_overall_mask = torch.tensor([True] + [False] * (n_tiles - 1), dtype=torch.bool)

        # 5) 文本对话（Osprey 成对格式）
        conversation = self._build_conversation_pairs(n_regions=1, region_pixels=region_pixels, answer=caption)

        # 6) 将 <image> 默认占位替换为 <img> + <IMG_CONTEXT>*num_image_tokens + </img>
        num_image_tokens = n_tiles * IMG_TOKENS_PER_FRAME
        image_token_str = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_tokens}{IMG_END_TOKEN}"

        data_dict: Dict[str, Any] = {
            'pixel_values': pixel_values,        # (T,3,H,W)
            'prompt_masks': prompt_masks,        # (N_regions,G,G) 这里是 (1,G,G)
            'vp_overall_mask': vp_overall_mask,  # (T,) bool
            'conversation': conversation,        # 成对格式
        }
        data_dict = self._replace_image_str(data_dict, image_token_str)

        # 7) 模板与编码（与 Osprey 完全一致）
        template_map_fn = self.template_map_fn
        if isinstance(template_map_fn, dict) and self.lazy:
            _type = template_map_fn['type']
            kwargs = dict(template_map_fn)
            del kwargs['type']
            template_map_fn = _type(**kwargs)

        result = template_map_fn(data_dict)
        data_dict.update(result)

        result = video_lisa_encode_fn(
            data_dict,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            with_image_token=True
        )
        data_dict.update(result)

        # Sampler 需要的长度提示
        data_dict['modality_length'] = self._modality_length_value

        return data_dict
