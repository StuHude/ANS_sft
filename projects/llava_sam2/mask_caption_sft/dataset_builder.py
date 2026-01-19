"""
Unified dataset wrappers for mask captioning SFT training.

All datasets return multiple formats for different stages:
1. pixel_values_448: 448x448 ImageNet normalized (for InternVL vision encoder)
2. g_pixel_values_1024: 1024x1024 [0, 255] uint8 (for SAM2 encoder)
3. prompt_masks_16: 16x16 [0, 1] (for visual prompt indexing)
4. masks_1024: 1024x1024 [0, 1] (for GT loss computation)

IMPORTANT: Following Sa2VA's original implementation (RefCOCO_Dataset.py):
- pixel_values: ImageNet normalized (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- g_pixel_values: [0, 255] uint8 (DirectResize, no normalization)
- Masks: [0, 1] range (no normalization)
"""

import os
import sys
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import torchvision.transforms as T
from pathlib import Path

# Use project-internal dataset loaders
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from projects.llava_sam2.datasets.external_loaders import (
    MaskletNPZDataset,
    SA1BDataset,
)

# ImageNet normalization (same as Osprey_Dataset.py and RefCOCO_Dataset.py)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SAVDatasetWrapper(Dataset):
    """
    Wrapper for SAV dataset.

    Returns multiple formats following Sa2VA RefCOCO implementation:
    - pixel_values_448: 448x448 ImageNet normalized (for InternVL)
    - g_pixel_values_1024: 1024x1024 [0, 255] uint8 (for SAM2)
    - prompt_masks_16: 16x16 [0, 1] (for visual prompt)
    - masks_1024: 1024x1024 [0, 1] (for GT loss)
    """

    def __init__(
        self,
        npz_dir: str,
        target_size=(1024, 1024),
        prefix: str = "masklet_data",
        shuffle: bool = False,
        max_samples: int = None,
    ):
        self.dataset = MaskletNPZDataset(
            npz_dir=npz_dir,
            prefix=prefix,
            shuffle=shuffle,
        )
        self.target_size = target_size
        self.max_samples = max_samples

        # Image transform for InternVL: 448 ImageNet normalized
        self.image_transform_448 = T.Compose([
            T.ToPILImage(),
            T.Resize((448, 448), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),  # [0, 1]
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)  # ImageNet normalize
        ])

        # Image transform for SAM2: 1024 [0, 255] uint8
        self.image_transform_1024 = T.Compose([
            T.ToPILImage(),
            T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),  # [0, 1]
            # Do NOT normalize, will convert to [0, 255] uint8
        ])

        # Mask transform for visual prompt: 16x16
        self.mask_transform_16 = T.Compose([
            T.ToPILImage(mode='L'),
            T.Resize((16, 16), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),  # [0, 1]
        ])

        # Mask transform for GT loss: 1024x1024
        self.mask_transform_1024 = T.Compose([
            T.ToPILImage(mode='L'),
            T.Resize(target_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),  # [0, 1]
        ])

    def __len__(self):
        if self.max_samples is not None:
            return min(len(self.dataset), self.max_samples)
        return len(self.dataset)

    def __getitem__(self, idx):
        # Check if idx is within max_samples limit
        if self.max_samples is not None and idx >= self.max_samples:
            idx = idx % self.max_samples

        try:
            result = self.dataset[idx]
            if result is None:
                # Skip corrupted data, return next valid sample
                return self.__getitem__((idx + 1) % len(self))

            (frame1, mask1), (frame2, mask2) = result

            if frame1 is None:
                return self.__getitem__((idx + 1) % len(self))
        except Exception as e:
            print(f"Warning: Failed to load sample {idx}: {e}")
            # Skip to next sample
            return self.__getitem__((idx + 1) % len(self))

        # Normalize frames to [0, 1] if needed
        if frame1.max() > 1.0:
            frame1 = frame1 / 255.0
        if frame2.max() > 1.0:
            frame2 = frame2 / 255.0

        # Normalize masks to [0, 1] if needed
        if mask1.max() > 1.0:
            mask1 = mask1 / 255.0
        if mask2.max() > 1.0:
            mask2 = mask2 / 255.0

        # Convert frames to HWC format if they're in CHW format
        # NPZ files return (H, W, C), so we don't need to permute
        # Just convert to numpy array
        frame1_np = frame1.numpy() if frame1.dim() == 3 and frame1.shape[-1] in [1, 3, 4] else frame1.permute(1, 2, 0).numpy()
        frame2_np = frame2.numpy() if frame2.dim() == 3 and frame2.shape[-1] in [1, 3, 4] else frame2.permute(1, 2, 0).numpy()

        # Apply transforms
        # 1. For InternVL (448 normalized)
        image1_448 = self.image_transform_448(frame1_np)
        image2_448 = self.image_transform_448(frame2_np)

        # 2. For SAM2 (1024 [0, 255] uint8)
        image1_1024 = self.image_transform_1024(frame1_np)
        image2_1024 = self.image_transform_1024(frame2_np)
        g_pixel_values1 = (image1_1024 * 255).byte()  # Convert to uint8 [0, 255]
        g_pixel_values2 = (image2_1024 * 255).byte()

        # 3. For visual prompt (16x16)
        mask1_np = (mask1 * 255).byte().numpy()
        mask2_np = (mask2 * 255).byte().numpy()
        prompt_mask1 = self.mask_transform_16(mask1_np)
        prompt_mask2 = self.mask_transform_16(mask2_np)

        # 4. For GT loss (1024x1024)
        gt_mask1 = self.mask_transform_1024(mask1_np)
        gt_mask2 = self.mask_transform_1024(mask2_np)

        return {
            # For InternVL vision encoder
            'pixel_values1': image1_448,              # (3, 448, 448) normalized
            'pixel_values2': image2_448,              # (3, 448, 448) normalized

            # For SAM2 grounding encoder
            'g_pixel_values1': g_pixel_values1,       # (3, 1024, 1024) uint8 [0, 255]
            'g_pixel_values2': g_pixel_values2,       # (3, 1024, 1024) uint8 [0, 255]

            # For visual prompt (boolean indexing)
            'prompt_masks1': prompt_mask1.squeeze(0), # (16, 16) [0, 1]
            'prompt_masks2': prompt_mask2.squeeze(0), # (16, 16) [0, 1]

            # For GT loss
            'masks1': gt_mask1.squeeze(0),            # (1024, 1024) [0, 1]
            'masks2': gt_mask2.squeeze(0),            # (1024, 1024) [0, 1]

            'dataset_type': 'sav',
            'has_paired_frame': True,
        }


class SA1BDatasetWrapper(Dataset):
    """
    Wrapper for SA-1B dataset.

    Returns multiple formats following Sa2VA RefCOCO implementation:
    - pixel_values: 448x448 ImageNet normalized (for InternVL)
    - g_pixel_values: 1024x1024 [0, 255] uint8 (for SAM2)
    - prompt_masks: 16x16 [0, 1] (for visual prompt)
    - masks: 1024x1024 [0, 1] (for GT loss)
    """

    def __init__(
        self,
        dataset_dir: str,
        target_size=(1024, 1024),
        annotation_dir='js',
        image_dir='img',
        min_object=0,
        max_samples=None,
    ):
        self.dataset = SA1BDataset(
            dataset_dir=dataset_dir,
            annotation_dir=annotation_dir,
            image_dir=image_dir,
            min_object=min_object,
            target_size=target_size,
            max_samples=max_samples,
            cache_images=False,
            max_cache_images=0,
        )
        self.target_size = target_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask, class_id = self.dataset[idx]
        # SA1BDataset returns: image [C, H, W] [0, 1], mask [H, W] [0, 1]

        # 1. For InternVL (448 normalized)
        image_448 = F.interpolate(image.unsqueeze(0), size=(448, 448), mode='bilinear', align_corners=False).squeeze(0)
        image_448 = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(image_448)

        # 2. For SAM2 (1024 [0, 255] uint8) - already 1024 from dataset
        g_pixel_values = (image * 255).byte()

        # 3. For visual prompt (16x16)
        prompt_masks = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(16, 16), mode='nearest').squeeze()

        # 4. For GT loss (1024) - already 1024 from dataset
        gt_masks = mask

        return {
            'pixel_values': image_448,          # (3, 448, 448) normalized
            'g_pixel_values': g_pixel_values,   # (3, 1024, 1024) uint8 [0, 255]
            'prompt_masks': prompt_masks,       # (16, 16) [0, 1]
            'masks': gt_masks,                  # (1024, 1024) [0, 1]
            'dataset_type': 'sa1b',
            'has_paired_frame': False,
        }


class OpenImageDatasetWrapper(Dataset):
    """
    Wrapper for OpenImage dataset.

    Returns multiple formats following Sa2VA RefCOCO implementation:
    - pixel_values: 448x448 ImageNet normalized (for InternVL)
    - g_pixel_values: 1024x1024 [0, 255] uint8 (for SAM2)
    - prompt_masks: 16x16 [0, 1] (for visual prompt)
    - masks: 1024x1024 [0, 1] (for GT loss)

    IMPORTANT: Loads from original images/masks to avoid double-resize.
    """

    def __init__(
        self,
        annotation_csv: str,
        label_csv: str,
        image_dir: str,
        mask_dir: str,
        target_size=(1024, 1024),
        max_samples: int | None = None,
    ):
        # Keep memory usage low: store per-row file offsets instead of loading the full CSV.
        # annotation_csv columns: ImageID, MaskPath, LabelName
        self.annotation_csv = annotation_csv
        self._fp = None  # opened lazily in each worker process

        with open(annotation_csv, "rb") as f:
            header = f.readline().decode("utf-8")
            header_cols = next(csv.reader([header]))
            self._col_image_id = header_cols.index("ImageID")
            self._col_mask_path = header_cols.index("MaskPath")
            self._col_label = header_cols.index("LabelName")

            self._offsets = []
            while True:
                if max_samples is not None and len(self._offsets) >= max_samples:
                    break
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                self._offsets.append(offset)

        # Optional label map (not required for training, but useful for debugging)
        self.label_map = {}
        try:
            with open(label_csv, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = row.get("LabelName")
                    if not label:
                        continue
                    self.label_map[label] = row.get("DisplayName", label)
        except FileNotFoundError:
            self.label_map = {}

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size

        # Image transform for InternVL: 448 ImageNet normalized
        self.image_transform_448 = T.Compose([
            T.Resize((448, 448), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        # Image transform for SAM2: 1024 [0, 255] uint8
        self.image_transform_1024 = T.Compose([
            T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),  # [0, 1]
        ])

        # Mask transform for visual prompt: 16x16
        self.mask_transform_16 = T.Compose([
            T.Resize((16, 16), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

        # Mask transform for GT loss: 1024
        self.mask_transform_1024 = T.Compose([
            T.Resize(target_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self._offsets)

    def __getitem__(self, idx):
        if self._fp is None:
            self._fp = open(self.annotation_csv, "rb")

        # Robust refetch loop: some rows can point to missing/corrupted/oversized files.
        # Oversized files can spike worker RSS and get SIGKILL'ed by the OS; skip them early.
        max_refetch = 50
        max_image_bytes = 50 * 1024 * 1024
        max_mask_bytes = 50 * 1024 * 1024

        import io
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        cur = int(idx) % len(self)
        for _ in range(max_refetch):
            self._fp.seek(self._offsets[cur])
            line = self._fp.readline().decode("utf-8")
            row = next(csv.reader([line]))
            image_id = row[self._col_image_id]
            mask_path = row[self._col_mask_path]
            label_name = row[self._col_label]

            image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
            full_mask_path = os.path.join(self.mask_dir, mask_path)

            try:
                if os.path.getsize(image_path) > max_image_bytes or os.path.getsize(full_mask_path) > max_mask_bytes:
                    cur = (cur + 1) % len(self)
                    continue

                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                with open(full_mask_path, "rb") as f:
                    mask_bytes = f.read()
                mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
                break
            except Exception:
                cur = (cur + 1) % len(self)
                continue
        else:
            raise RuntimeError("OpenImageDatasetWrapper: failed to fetch a valid sample after refetch attempts")

        # Apply transforms
        # 1. For InternVL (448 normalized)
        image_448 = self.image_transform_448(image)

        # 2. For SAM2 (1024 [0, 255] uint8)
        image_1024 = self.image_transform_1024(image)
        g_pixel_values = (image_1024 * 255).byte()

        # 3. For visual prompt (16x16)
        prompt_masks = self.mask_transform_16(mask).squeeze(0)

        # 4. For GT loss (1024)
        gt_masks = self.mask_transform_1024(mask).squeeze(0)

        return {
            'pixel_values': image_448,          # (3, 448, 448) normalized
            'g_pixel_values': g_pixel_values,   # (3, 1024, 1024) uint8 [0, 255]
            'prompt_masks': prompt_masks,       # (16, 16) [0, 1]
            'masks': gt_masks,                  # (1024, 1024) [0, 1]
            'dataset_type': 'openimage',
            'has_paired_frame': False,
        }


class RefCOCODatasetWrapper(Dataset):
    """
    Wrapper for RefCOCO dataset.
    Returns: (image, mask, caption, dataset_type='refcoco')

    This dataset only participates in the referring segmentation part,
    NOT in the mask captioning part.

    Following Sa2VA original implementation:
    - Images: ImageNet normalized
    - Masks: [0, 1] range
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        dataset_name: str = 'refcoco',
        target_size=(1024, 1024),
    ):
        """
        Args:
            data_root: Path to data directory (e.g., './data/ref_seg/')
            split: 'train', 'val', or 'test'
            dataset_name: 'refcoco', 'refcoco+', or 'refcocog'
            target_size: Target image size
        """
        from third_parts.mmdet.datasets.refcoco import RefCocoDataset

        # Map dataset names to split files
        split_by_map = {
            'refcoco': 'unc',
            'refcoco+': 'unc',
            'refcocog': 'umd',
        }

        split_by = split_by_map.get(dataset_name, 'unc')

        self.dataset = RefCocoDataset(
            data_root=os.path.join(data_root, dataset_name),
            data_prefix=dict(img_path='coco2014/train2014/'),
            ann_file='instances.json',
            split_file=f'refs({split_by}).p',
        )

        self.data_root = data_root
        self.target_size = target_size

        # Image transform for InternVL: 448 ImageNet normalized
        self.image_transform_448 = T.Compose([
            T.Resize((448, 448), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        # Image transform for SAM2: 1024 [0, 255] uint8
        self.image_transform_1024 = T.Compose([
            T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),  # [0, 1]
        ])

        # Mask transform for visual prompt: 16x16
        self.mask_transform_16 = T.Compose([
            T.ToPILImage(mode='L'),
            T.Resize((16, 16), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

        # Mask transform for GT loss: 1024
        self.mask_transform_1024 = T.Compose([
            T.ToPILImage(mode='L'),
            T.Resize(target_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_info = self.dataset[idx]

        # Load image
        # RefCocoDataset returns img_path directly
        if 'img_path' in data_info:
            image_path = data_info['img_path']
        elif 'img_prefix' in data_info:
            image_path = os.path.join(
                self.data_root,
                data_info['img_prefix'],
                data_info['img_info']['file_name']
            )
        else:
            # Fallback: construct from data_root and img_info
            image_path = os.path.join(
                self.data_root,
                data_info.get('dataset_name', 'refcoco'),
                'coco2014/train2014',
                data_info['img_info']['file_name']
            )

        image_pil = Image.open(image_path).convert('RGB')

        # Apply image transforms
        # 1. For InternVL (448 normalized)
        image_448 = self.image_transform_448(image_pil)

        # 2. For SAM2 (1024 [0, 255] uint8)
        image_1024 = self.image_transform_1024(image_pil)
        g_pixel_values = (image_1024 * 255).byte()

        # Get mask from instances
        # RefCocoDataset returns: instances = [{'mask': segmentation, 'ignore_flag': 0}, ...]
        if 'instances' in data_info and len(data_info['instances']) > 0:
            # Get first instance's mask (segmentation in COCO format)
            mask_data = data_info['instances'][0]['mask']
            # Convert COCO segmentation to binary mask
            from pycocotools import mask as mask_util
            if isinstance(mask_data, list):
                # Polygon format: list of polygons, each is [x1,y1,x2,y2,...]
                # Convert polygon to RLE then decode
                from pycocotools import mask as maskUtils
                # Get original image size (before transform)
                w_orig, h_orig = image_pil.size

                # Convert polygon to RLE
                rles = maskUtils.frPyObjects(mask_data, h_orig, w_orig)
                rle = maskUtils.merge(rles)
                mask = maskUtils.decode(rle)
            elif isinstance(mask_data, dict) and 'counts' in mask_data:
                # RLE format
                mask = mask_util.decode(mask_data)
            else:
                # Unknown format, use zero mask
                mask = np.zeros((self.target_size[0], self.target_size[1]), dtype=np.uint8)

            mask = torch.from_numpy(mask.astype(np.float32))
        elif 'ann_info' in data_info:
            ann = data_info['ann_info']
            mask = ann.get('mask', ann.get('masks', None))
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float()
            elif not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask).float()
        elif 'gt_masks' in data_info:
            mask = data_info['gt_masks']
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float()
            elif not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask).float()
        elif 'mask' in data_info:
            mask = data_info['mask']
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float()
            elif not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask).float()
        else:
            # Fallback: create zero mask
            print(f"Warning: No mask found in data_info keys: {data_info.keys()}")
            mask = torch.zeros((self.target_size[0], self.target_size[1]), dtype=torch.float32)

        # Ensure mask is 2D
        if mask.dim() > 2:
            mask = mask.squeeze()

        # Apply mask transforms
        # 3. For visual prompt (16x16)
        if mask.shape[0] != self.target_size[0] or mask.shape[1] != self.target_size[1]:
            # Resize to 1024 first if needed
            mask_1024 = self.mask_transform_1024((mask * 255).byte().numpy()).squeeze(0)
        else:
            mask_1024 = mask.squeeze()

        prompt_masks = self.mask_transform_16((mask_1024 * 255).byte().numpy()).squeeze(0)

        # Get caption
        # RefCocoDataset returns: text = ['caption1', 'caption2', ...]
        if 'text' in data_info:
            caption = data_info['text'][0] if isinstance(data_info['text'], list) else data_info['text']
        else:
            caption = data_info.get('caption', data_info.get('sentence', ''))

        return {
            'pixel_values': image_448,          # (3, 448, 448) normalized
            'g_pixel_values': g_pixel_values,   # (3, 1024, 1024) uint8 [0, 255]
            'prompt_masks': prompt_masks,       # (16, 16) [0, 1]
            'masks': mask_1024,                 # (1024, 1024) [0, 1]
            'caption': caption,
            'dataset_type': 'refcoco',
            'has_paired_frame': False,
        }


def build_mask_caption_dataset(
    sav_dir=None,
    sa1b_dir=None,
    openimage_config=None,
    refcoco_config=None,
    target_size=(1024, 1024),
    sa1b_max_samples=None,
    sav_max_samples=None,
    openimage_max_samples=None,
    sav_repeats=1,
    refcoco_repeats=1,
):
    """
    Build concatenated dataset from all sources.

    Args:
        sav_dir: Path to SAV NPZ directory
        sa1b_dir: Path to SA-1B dataset directory
        openimage_config: Dict with keys: annotation_csv, label_csv, image_dir, mask_dir
        refcoco_config: Dict with keys: data_root, split, dataset_name
        target_size: Target image size
        sa1b_max_samples: Maximum number of SA-1B image files to process (default: None = all)
                          Set to a number to limit samples for testing
        sav_max_samples: Maximum number of SAV samples to use (default: None = all)
                         Set to a small number (e.g., 10) for testing
        sav_repeats: Number of times to repeat SAV dataset (default: 1)
        refcoco_repeats: Number of times to repeat RefCOCO dataset (default: 1)

    Returns:
        ConcatDataset of all enabled datasets
    """
    datasets = []

    # SAV dataset with repeats
    if sav_dir is not None:
        sav_info = "all samples" if sav_max_samples is None else f"max_samples={sav_max_samples}"
        print(f"Loading SAV dataset from {sav_dir} ({sav_info}, repeats={sav_repeats})")
        sav_dataset = SAVDatasetWrapper(
            npz_dir=sav_dir,
            target_size=target_size,
            max_samples=sav_max_samples,
        )
        # Add SAV dataset multiple times for higher sampling weight
        for _ in range(sav_repeats):
            datasets.append(sav_dataset)

    # SA1B dataset (no repeats, used once)
    if sa1b_dir is not None:
        samples_info = "all samples" if sa1b_max_samples is None else f"max_samples={sa1b_max_samples}"
        print(f"Loading SA-1B dataset from {sa1b_dir} ({samples_info})")
        datasets.append(SA1BDatasetWrapper(
            dataset_dir=sa1b_dir,
            target_size=target_size,
            max_samples=sa1b_max_samples,
        ))

    # OpenImage dataset (no repeats, used once)
    if openimage_config is not None:
        print(f"Loading OpenImage dataset")
        datasets.append(OpenImageDatasetWrapper(
            target_size=target_size,
            max_samples=openimage_max_samples,
            **openimage_config,
        ))

    # RefCOCO dataset with repeats
    if refcoco_config is not None:
        print(f"Loading RefCOCO dataset (repeats={refcoco_repeats})")
        refcoco_dataset = RefCOCODatasetWrapper(
            target_size=target_size,
            **refcoco_config,
        )
        # Add RefCOCO dataset multiple times for higher sampling weight
        for _ in range(refcoco_repeats):
            datasets.append(refcoco_dataset)

    if not datasets:
        raise ValueError("At least one dataset must be specified")

    print(f"Total datasets: {len(datasets)}")
    for i, ds in enumerate(datasets):
        print(f"  Dataset {i}: {len(ds)} samples")

    return ConcatDataset(datasets)


def collate_fn_mask_caption(batch):
    """
    Collate samples from unified wrappers into a consistent batch format.

    The returned batch always contains paired keys `*1` and `*2`:
    - SAV (paired-frame): `*1` and `*2` are frame1/frame2.
    - Single-frame datasets (SA1B/OpenImage/RefCOCO): `*2 == *1`.

    Keys:
    - pixel_values{1,2}: (B, 3, 448, 448) ImageNet normalized
    - prompt_masks{1,2}: (B, 16, 16) in [0, 1]
    - g_pixel_values{1,2}: (B, 3, 1024, 1024) uint8 in [0, 255]
    - masks{1,2}: (B, 1024, 1024) in [0, 1]
    - captions: list[str|None] (RefCOCO provides text; others None)
    - dataset_types: list[str]
    """
    # Filter out None samples
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None

    pv1_list, pv2_list = [], []
    pm1_list, pm2_list = [], []
    gp1_list, gp2_list = [], []
    m1_list, m2_list = [], []
    captions = []
    dataset_types = []

    for sample in batch:
        dataset_type = sample.get('dataset_type', 'unknown')
        dataset_types.append(dataset_type)

        if dataset_type == 'sav' and sample.get('has_paired_frame', False):
            pv1_list.append(sample['pixel_values1'])
            pv2_list.append(sample['pixel_values2'])
            pm1_list.append(sample['prompt_masks1'])
            pm2_list.append(sample['prompt_masks2'])
            gp1_list.append(sample['g_pixel_values1'])
            gp2_list.append(sample['g_pixel_values2'])
            m1_list.append(sample['masks1'])
            m2_list.append(sample['masks2'])
        else:
            pv = sample['pixel_values']
            pm = sample['prompt_masks']
            gp = sample['g_pixel_values']
            m = sample['masks']

            pv1_list.append(pv)
            pv2_list.append(pv)
            pm1_list.append(pm)
            pm2_list.append(pm)
            gp1_list.append(gp)
            gp2_list.append(gp)
            m1_list.append(m)
            m2_list.append(m)

        captions.append(sample.get('caption', None))

    return {
        'pixel_values1': torch.stack(pv1_list, dim=0),
        'pixel_values2': torch.stack(pv2_list, dim=0),
        'prompt_masks1': torch.stack(pm1_list, dim=0),
        'prompt_masks2': torch.stack(pm2_list, dim=0),
        'g_pixel_values1': torch.stack(gp1_list, dim=0),
        'g_pixel_values2': torch.stack(gp2_list, dim=0),
        'masks1': torch.stack(m1_list, dim=0),
        'masks2': torch.stack(m2_list, dim=0),
        'captions': captions,
        'dataset_types': dataset_types,
    }
