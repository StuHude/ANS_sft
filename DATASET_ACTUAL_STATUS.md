# 数据集实际使用情况 - 准确分析

## ❌ 问题确认

### 1. 实际使用的数据集：**只有3个，不是4个**

```
✓ Dataset 1: SAV      - /data/xyc/formed_data/npz
✓ Dataset 2: SA1B     - /data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw
✓ Dataset 3: RefCOCO  - ./data/ref_seg
✗ Dataset 4: OpenImage - ./data/openimages (不存在)
```

### 2. LengthGroupedSampler：**原始训练使用了，但我们没有**

**原始Sa2VA配置** (sa2va_4b.py):
```python
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',  # ← 关键！
        per_device_batch_size=batch_size * accumulative_counts
    ),
    collate_fn=dict(type=video_lisa_collate_fn)
)
```

**我们当前的配置**:
```python
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,  # ← 只是简单随机！
    num_workers=args.num_workers,
    collate_fn=collate_fn_mask_caption,
    pin_memory=True,
)
```

### 3. 数据集采样参数：**我们没有使用repeats和权重**

**原始配置的数据集重复**:
```python
train_dataset=dict(
    type=ConcatDataset, datasets=[
        # RefCOCO系列重复4次！
        refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # GranDf重复10次
        glamm_grandf_dataset,  # repeats=10
        # 其他数据集...
    ]
)
```

**我们当前的配置**:
- SAV: 1次
- SA1B: 1次（测试时限制500样本）
- RefCOCO: 1次
- 没有任何repeats或权重调整

---

## ✅ 需要的修复

### 修复1: 添加LengthGroupedSampler

**为什么需要**:
1. **原始训练使用了** - 应该保持一致
2. **提高训练效率** - 相似长度的样本在同一batch，减少padding浪费
3. **稳定内存使用** - 避免batch间内存差异过大

**如何实现**:

#### Step 1: 在dataset中添加modality_length属性

```python
# 在 dataset_builder.py 中的 SAVDatasetWrapper, SA1BDatasetWrapper, RefCOCODatasetWrapper

class SAVDatasetWrapper(Dataset):
    # ... 现有代码 ...

    @property
    def modality_length(self):
        """Return list of modality lengths for LengthGroupedSampler"""
        # 简单实现：固定长度（因为图像都是1024x1024）
        # 但可以根据caption长度来计算
        return [1024] * len(self)  # 或者计算实际的序列长度

    def __getitem__(self, idx):
        # ... 现有代码 ...
        result = {
            'image1': image1,
            'mask1': mask1.squeeze(0),
            'image2': image2,
            'mask2': mask2.squeeze(0),
            'dataset_type': 'sav',
            'has_paired_frame': True,
            'modality_length': 1024,  # ← 添加这个
        }
        return result
```

#### Step 2: 在dataloader中使用LengthGroupedSampler

```python
from xtuner.dataset.samplers import LengthGroupedSampler

# 在 train_dual_loop.py 的 main() 函数中
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    # shuffle=True,  # ← 移除这个
    sampler=LengthGroupedSampler(  # ← 添加这个
        train_dataset,
        batch_size=args.batch_size,
        world_size=1 if args.local_rank == -1 else torch.distributed.get_world_size(),
        rank=0 if args.local_rank == -1 else args.local_rank,
        seed=42,
    ),
    num_workers=args.num_workers,
    collate_fn=collate_fn_mask_caption,
    pin_memory=True,
)
```

### 修复2: 添加数据集采样权重（可选）

如果想让某些数据集被采样更多次：

```python
# 方案A: 简单重复（像原始配置）
datasets = []
if sav_dir:
    datasets.append(SAVDatasetWrapper(...))
if sa1b_dir:
    datasets.append(SA1BDatasetWrapper(...))
if refcoco_dir:
    # RefCOCO重复4次（像原始配置）
    for _ in range(4):
        datasets.append(RefCOCODatasetWrapper(...))

# 方案B: 使用WeightedRandomSampler（更灵活）
from torch.utils.data import WeightedRandomSampler

dataset_weights = {
    'sav': 1.0,
    'sa1b': 1.0,
    'refcoco': 4.0,  # RefCOCO权重4倍
}

sample_weights = []
for i in range(len(train_dataset)):
    sample = train_dataset[i]
    weight = dataset_weights.get(sample['dataset_type'], 1.0)
    sample_weights.append(weight)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True
)
```

### 修复3: 添加第4个数据集 - OpenImage

**选项A: 跳过OpenImage（如果数据不可用）**
- 当前状态：已实现自动跳过
- 训练会正常进行，只用3个数据集

**选项B: 获取OpenImage数据**

1. **下载OpenImage v7数据**:
```bash
# 创建目录
mkdir -p ./data/openimages/{images,masks}/train

# 下载数据（需要确认具体数据源）
# 这部分需要用户提供OpenImage数据的下载方式
```

2. **配置数据集路径**:
```bash
# 确保以下文件存在
./data/openimages/
├── train-annotations-object-segmentation.csv
├── oidv7-class-descriptions.csv
├── images/train/
└── masks/train/
```

---

## 📊 推荐配置

### 配置1: 最小修改（3个数据集 + LengthGroupedSampler）

**适用场景**: OpenImage数据不可用，快速开始训练

**修改**:
1. ✅ 添加LengthGroupedSampler
2. ✅ 保持3个数据集（SAV, SA1B, RefCOCO）
3. ✅ 不使用数据集重复

**预期效果**:
- 更稳定的训练
- 更少的padding浪费
- 与原始配置更接近

### 配置2: 完整配置（4个数据集 + LengthGroupedSampler + 权重）

**适用场景**: OpenImage数据可用，追求最佳效果

**修改**:
1. ✅ 添加LengthGroupedSampler
2. ✅ 添加OpenImage数据集
3. ✅ 使用数据集权重（RefCOCO×4）
4. ✅ 使用更大的batch size

**预期效果**:
- 最接近原始训练配置
- 更好的性能
- 更平衡的数据分布

---

## 🔧 具体实施步骤

### 立即执行（推荐）

**Step 1: 添加modality_length属性**
```bash
# 修改 dataset_builder.py
# 在每个Dataset wrapper的__getitem__中添加 'modality_length' 字段
```

**Step 2: 修改dataloader配置**
```bash
# 修改 train_dual_loop.py
# 将 shuffle=True 改为使用 LengthGroupedSampler
```

**Step 3: 重新测试**
```bash
docker exec -w /data/xyc/ANS vlm-env bash test_dual_loop.sh
```

### 可选执行

**添加数据集重复**（如果想模仿原始配置）:
- RefCOCO重复4次
- 或使用WeightedRandomSampler设置权重

---

## 📝 对比总结

| 项目 | 原始Sa2VA配置 | 当前我们的配置 | 差异 |
|------|--------------|--------------|------|
| 数据集数量 | 15+ datasets | 3 datasets | ❌ 少很多 |
| LengthGroupedSampler | ✅ 使用 | ❌ 未使用 | ❌ 缺失 |
| 数据集重复 | ✅ RefCOCO×4 | ❌ 无重复 | ❌ 缺失 |
| modality_length | ✅ 有 | ❌ 无 | ❌ 缺失 |
| OpenImage | N/A | ❌ 不存在 | ⚠️ 数据问题 |

**结论**: 我们的配置与原始训练差异较大，**应该添加LengthGroupedSampler**。

---

## ⚡ 快速修复代码

我可以立即为您实现：
1. 添加modality_length到所有dataset
2. 修改dataloader使用LengthGroupedSampler
3. （可选）添加数据集权重配置

是否需要我现在就实施这些修复？
