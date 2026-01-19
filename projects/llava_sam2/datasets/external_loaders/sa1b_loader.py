from pycocotools import mask as maskUtils
import os
from PIL import Image
import numpy as np
import json
import torch
import cv2
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import OrderedDict

class SA1BDataset:
    def __init__(self, dataset_dir, ids=None, annotation_dir='js', image_dir='img',
                 min_object=0, target_size=(1024, 1024), transform=None, max_samples=None,
                 cache_images: bool = False, max_cache_images: int = 0):
        """
        参数说明：
        target_size: 统一输出尺寸 (height, width)
        transform: 可自定义，但必须包含Resize操作
        """
        # 初始化核心参数
        self.dataset_dir = dataset_dir
        self.min_object = min_object
        self.target_size = target_size
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir

        # 自动构建包含Resize的transform
        base_transform = [
            transforms.Resize(target_size),
            transforms.ToTensor()
        ]
        self.transform = transform or transforms.Compose(base_transform)

        # 加载文件列表
        if ids is None:
            all_files = sorted(os.listdir(os.path.join(dataset_dir, annotation_dir)))
            if max_samples is not None:
                all_files = all_files[:max_samples]
            ids = [f.replace(".json", "") for f in all_files]

        # 初始化样本路径
        self.samples = [
            (os.path.join(dataset_dir, image_dir, f"{id}.jpg"),
             os.path.join(dataset_dir, annotation_dir, f"{id}.json"))
            for id in ids
        ]

        # 延迟加载参数
        self.indices = None
        self.cache_images = bool(cache_images) and int(max_cache_images) > 0
        self.max_cache_images = int(max_cache_images) if int(max_cache_images) > 0 else 0
        self.img_cache = OrderedDict() if self.cache_images else None
        self.processed = False

    def _precompute_indices(self):
        """优化的延迟索引计算 - 跳过预统计直接处理"""
        self.indices = []
        print(f"🚚 开始预处理SA-1B数据集 ({len(self.samples)} 图像文件)...")
        print("⏳ 正在处理标注... (这可能需要一些时间，但训练将在后台继续)")

        # 直接处理，不预先统计（节省一半时间）
        # 使用文件数而不是annotation数作为进度指示
        processed_anns = 0
        with tqdm(total=len(self.samples), desc="🔧 处理图像", unit="img") as pbar:
            for img_idx, (img_path, ann_path) in enumerate(self.samples):
                try:
                    # 加载图像原始尺寸
                    with Image.open(img_path) as img:
                        orig_h, orig_w = img.size[1], img.size[0]

                    # 处理标注
                    with open(ann_path) as f:
                        annotations = json.load(f)['annotations']

                    for ann_idx, ann in enumerate(annotations):
                        try:
                            rle = self._ann_to_rle(ann, orig_h, orig_w)
                            area = maskUtils.area(rle).sum().item()
                            if area >= self.min_object:
                                self.indices.append((img_idx, ann_idx))
                                processed_anns += 1
                        except Exception as e:
                            # 静默跳过单个标注错误，避免日志过多
                            pass

                    # 每1000个图像打印一次进度
                    if (img_idx + 1) % 1000 == 0:
                        print(f"  已处理: {img_idx + 1}/{len(self.samples)} 图像, {processed_anns} 有效标注")

                except Exception as e:
                    print(f"⚠️ 图像错误: {img_path} - {str(e)}")
                finally:
                    pbar.update(1)

        self.processed = True
        print(f"✅ 预处理完成: {len(self.samples)} 图像, {len(self.indices):,} 有效标注")

    def __len__(self):
        if not self.processed:
            self._precompute_indices()
        return len(self.indices)

    def __getitem__(self, index):
        if not self.processed:
            self._precompute_indices()

        img_idx, ann_idx = self.indices[index]
        img_path, ann_path = self.samples[img_idx]

        # 加载图像（默认不缓存；缓存会导致RAM不断增长直至OOM）
        if self.cache_images:
            if img_idx in self.img_cache:
                img_tensor, (orig_w, orig_h) = self.img_cache.pop(img_idx)
                self.img_cache[img_idx] = (img_tensor, (orig_w, orig_h))
            else:
                with Image.open(img_path) as img_pil:
                    img_pil = img_pil.convert("RGB")
                    orig_w, orig_h = img_pil.size
                    img_tensor = self.transform(img_pil)
                self.img_cache[img_idx] = (img_tensor, (orig_w, orig_h))
                if len(self.img_cache) > self.max_cache_images:
                    self.img_cache.popitem(last=False)
        else:
            with Image.open(img_path) as img_pil:
                img_pil = img_pil.convert("RGB")
                orig_w, orig_h = img_pil.size
                img_tensor = self.transform(img_pil)  # 应用尺寸变换

        # 加载标注并处理mask
        with open(ann_path) as f:
            ann = json.load(f)['annotations'][ann_idx]

        # 生成原始mask
        orig_mask = self._ann_to_mask(ann, orig_h, orig_w)  # (h, w)

        # 调整mask尺寸（保持二值特性）
        mask = cv2.resize(
            orig_mask.astype(np.uint8),
            (self.target_size[1], self.target_size[0]),  # (width, height)
            interpolation=cv2.INTER_NEAREST
        )
        mask = torch.from_numpy(mask).float()

        # 获取类别ID
        class_id = torch.tensor(ann['id'], dtype=torch.long)

        return img_tensor, mask, class_id

    def _ann_to_rle(self, ann, height, width):
        """将标注转换为RLE格式"""
        segm = ann['segmentation']
        if isinstance(segm, list):
            rles = maskUtils.frPyObjects(segm, height, width)
            return maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            return maskUtils.frPyObjects(segm, height, width)
        return ann['segmentation']

    def _ann_to_mask(self, ann, height, width):
        """从RLE生成二值mask"""
        rle = self._ann_to_rle(ann, height, width)
        return maskUtils.decode(rle)
