# Dataset修改完成说明

## 已完成修改

### SAVDatasetWrapper ✅

现在返回以下格式：
```python
{
    # For InternVL vision encoder
    'pixel_values1': (3, 448, 448) ImageNet normalized,
    'pixel_values2': (3, 448, 448) ImageNet normalized,

    # For SAM2 grounding encoder
    'g_pixel_values1': (3, 1024, 1024) uint8 [0, 255],
    'g_pixel_values2': (3, 1024, 1024) uint8 [0, 255],

    # For visual prompt
    'prompt_masks1': (16, 16) [0, 1],
    'prompt_masks2': (16, 16) [0, 1],

    # For GT loss
    'masks1': (1024, 1024) [0, 1],
    'masks2': (1024, 1024) [0, 1],

    'dataset_type': 'sav',
    'has_paired_frame': True,
}
```

## 需要修改的其他数据集

### SA1BDatasetWrapper

需要类似修改，返回：
```python
{
    'pixel_values': (3, 448, 448) normalized,
    'g_pixel_values': (3, 1024, 1024) uint8 [0, 255],
    'prompt_masks': (16, 16) [0, 1],
    'masks': (1024, 1024) [0, 1],
    'dataset_type': 'sa1b',
}
```

### OpenImageDatasetWrapper

需要类似修改，返回：
```python
{
    'pixel_values': (3, 448, 448) normalized,
    'g_pixel_values': (3, 1024, 1024) uint8 [0, 255],
    'prompt_masks': (16, 16) [0, 1],
    'masks': (1024, 1024) [0, 1],
    'dataset_type': 'openimage',
}
```

### RefCOCODatasetWrapper

需要类似修改，返回：
```python
{
    'pixel_values': (3, 448, 448) normalized,
    'g_pixel_values': (3, 1024, 1024) uint8 [0, 255],
    'prompt_masks': (16, 16) [0, 1],
    'masks': (1024, 1024) [0, 1],
    'dataset_type': 'refcoco',
}
```

## 训练代码需要修改的地方

### pseudo_gumbel_core.py

**修改前**:
```python
# Line 30: 从1024 resize到448
images_448 = F.interpolate(images, size=(448, 448), mode='bilinear')

# Line 36: 从1024 pool到16×16
prompt_masks = F.interpolate(masks_unsqueezed, size=(16, 16), mode='nearest')
```

**修改后**:
```python
# 直接使用dataset提供的格式
images_448 = batch['pixel_values']  # 已经是448 normalized
prompt_masks = batch['prompt_masks']  # 已经是16×16
g_pixel_values = batch['g_pixel_values']  # 已经是1024 [0, 255]
gt_masks = batch['masks']  # 已经是1024 [0, 1]
```

### trainer.py

**删除错误的代码** (line 398-405):
```python
# ❌ 删除这段
images_1024 = F.interpolate(images, size=(1024, 1024), mode='bilinear')
images_1024 = (images_1024 * 255.0).clamp(0, 255)  # 错误！
```

**修改为**:
```python
# ✅ 直接使用dataset提供的g_pixel_values
g_pixel_values = batch['g_pixel_values']  # 已经是1024 [0, 255]
```

## 优势总结

### 1. 修复严重bug
- ❌ 之前：直接对normalized图像乘255，破坏图像信息
- ✅ 现在：从原始图像正确生成[0, 255]格式

### 2. 避免多次resize
- ❌ 之前：原始 → 1024 → 448 → 1024 (多次resize)
- ✅ 现在：原始 → 448 (一次), 原始 → 1024 (一次)

### 3. 避免反normalize
- ❌ 之前：normalize → 反normalize (精度损失)
- ✅ 现在：分别生成normalized和非normalized版本

### 4. 与Sa2VA一致
- ✅ 完全遵循RefCOCO_Dataset.py的实现
- ✅ pixel_values和g_pixel_values分别生成

## 下一步

1. 完成其他三个dataset wrapper的修改
2. 修改训练代码使用新的格式
3. 测试验证所有格式正确
4. 运行训练确认bug已修复
