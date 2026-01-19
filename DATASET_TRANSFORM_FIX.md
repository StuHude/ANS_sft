# Dataset Transform修复总结

## 问题分析

### 原始问题
1. **OpenImage双重resize**: 原始→224→1024，损失精度
2. **缺少ImageNet normalization**: 所有数据集返回[0,1]图像，但Sa2VA原始实现使用ImageNet normalization
3. **Mask resize不一致**: 部分数据集mask resize流程与Osprey不一致

### Sa2VA原始实现参考 (Osprey_Dataset.py, RefCOCO_Dataset.py)

**图像处理**:
```python
# 原始尺寸 -> resize到448 -> ToTensor -> ImageNet Normalize
transformer = T.Compose([
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
```

**Mask处理** (作为visual prompt):
```python
# Osprey_Dataset.py line 213-214
masks = self.decode_mask(masks, height, width)  # 原始尺寸
masks, region_pixels = self._get_region_infos(masks)  # resize到16x16
# 16x16 = 448 // 14 * 0.5 (image_size // patch_size * downsample_ratio)
```

---

## 修复方案

### 所有数据集统一处理流程

#### 1. SAVDatasetWrapper
**修改前**:
- Image: 原始 → 1024 → ToTensor → [0, 1]
- Mask: 原始 → 1024 → ToTensor → [0, 1]

**修改后**:
- Image: 原始 → 1024 → ToTensor → **ImageNet Normalize** ✅
- Mask: 原始 → 1024 → ToTensor → [0, 1] (no normalize)

#### 2. SA1BDatasetWrapper
**修改前**:
- Image: 原始 → 1024 (SA1BDataset) → [0, 1]
- Mask: 原始 → 1024 → [0, 1]

**修改后**:
- Image: 原始 → 1024 (SA1BDataset) → **ImageNet Normalize** ✅
- Mask: 原始 → 1024 → [0, 1] (no normalize)

#### 3. OpenImageDatasetWrapper
**修改前** ⚠️:
- Image: 原始 → 224 (SegmentationDataset) → 1024 (Wrapper) ❌ 双重resize！
- Mask: 原始 → 224 → 1024 ❌ 双重resize！

**修改后** ✅:
- Image: **直接从原始加载** → 1024 → ToTensor → **ImageNet Normalize**
- Mask: **直接从原始加载** → 1024 → ToTensor → [0, 1]
- **移除底层SegmentationDataset**，直接读取原始文件，避免双重resize

#### 4. RefCOCODatasetWrapper
**修改前**:
- Image: 原始 → 1024 → ToTensor → [0, 1]
- Mask: 原始 → 1024 → ToTensor → [0, 1]

**修改后**:
- Image: 原始 → 1024 → ToTensor → **ImageNet Normalize** ✅
- Mask: 原始 → 1024 → ToTensor → [0, 1] (no normalize)

---

## 训练代码中的处理

### pseudo_gumbel_core.py

**图像流程**:
```python
# Dataset返回: 1024x1024 ImageNet normalized
images (B, 3, 1024, 1024) [normalized]
  ↓
# Resize到448 (保持normalized状态)
images_448 = F.interpolate(images, size=(448, 448), mode='bilinear')
  ↓
# 送入InternVL vision encoder (期望normalized输入)
pixel_values = [images_448[i] for i in range(batch_size)]
```

**Mask流程**:
```python
# Dataset返回: 1024x1024 [0, 1]
masks (B, 1024, 1024) [0, 1]
  ↓
# Resize到16x16 for visual prompt (following Osprey)
prompt_masks = F.interpolate(masks.unsqueeze(1), size=(16, 16), mode='nearest')
  ↓
# 计算region pixels (following Osprey line 214)
K = int(prompt_masks[i].bool().to(torch.int64).sum().item())
```

---

## 关键改进

### 1. 一致的ImageNet Normalization
- **所有数据集**图像都使用ImageNet normalization
- 与Sa2VA原始实现 (Osprey, RefCOCO) 保持一致
- InternVL vision encoder期望normalized输入

### 2. 避免双重Resize
- **OpenImage**: 从原始文件加载，一次resize到1024
- **其他数据集**: 原始→1024，一次resize
- Mask在训练代码中从1024→16，总共两次resize (与Osprey原始→16类似)

### 3. Image和Mask尺寸对应
- Dataset中: Image和Mask都resize到**1024x1024**
- 保持相同的空间对应关系
- 训练代码中统一处理resize (448 for vision, 16 for prompt_masks)

---

## 验证清单

✅ SAV: Image ImageNet normalized, Mask [0,1], 都是1024
✅ SA1B: Image ImageNet normalized, Mask [0,1], 都是1024
✅ OpenImage: 从原始加载，避免双重resize，Image ImageNet normalized, Mask [0,1]
✅ RefCOCO: Image ImageNet normalized, Mask [0,1], 都是1024

✅ 所有数据集transform实现与Sa2VA原始实现一致
✅ Image和Mask的resize比例对应，确保空间位置准确
✅ 训练代码中的处理流程正确 (448 for vision, 16 for prompt_masks)

---

## 参考文件

- `/data/xyc/ANS/projects/llava_sam2/datasets/Osprey_Dataset.py` - Osprey实现，mask作为visual prompt
- `/data/xyc/ANS/projects/llava_sam2/datasets/RefCOCO_Dataset.py` - RefCOCO实现，ImageNet normalization
- `/data/xyc/ANS/projects/llava_sam2/mask_caption_sft/dataset_builder.py` - 修复后的数据集wrapper
- `/data/xyc/ANS/projects/llava_sam2/mask_caption_sft/pseudo_gumbel_core.py` - 训练代码中的处理

---

**修复完成时间**: 2026-01-06
**修复验证**: 所有数据集transform与Sa2VA原始实现保持一致
