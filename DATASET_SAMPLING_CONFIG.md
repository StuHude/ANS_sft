# 数据集采样配置说明

## 当前数据集配置

### 1. SAV Dataset
- **路径**: `/data/xyc/formed_data/npz`
- **样本数**: 735,577 个NPZ文件
- **采样策略**: 全部使用，无限制
- **数据格式**: 两帧配对 (image1, mask1) + (image2, mask2)
- **用途**: Dual-loop训练的核心数据集
  - image1 + mask1 → 生成caption
  - image2 + caption → 预测mask2'
  - Loss = segmentation_loss(mask2', mask2)

### 2. SA-1B Dataset
- **路径**: `/data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw`
- **测试采样**: `--sa1b_max_samples 500` (限制500个图像文件用于快速测试)
- **完整训练**: 不设置max_samples参数，使用全部数据
- **数据格式**: 单帧 (image, mask)
- **用途**:
  - image + mask → 生成caption
  - image + caption → 预测mask'
  - Loss = segmentation_loss(mask', mask)

### 3. RefCOCO Dataset
- **路径**: `./data/ref_seg`
- **样本数**: ~16,994 samples (refcoco训练集)
- **采样策略**: 全部使用
- **数据格式**: 单帧 + referring expression (image, mask, caption)
- **用途**:
  - 使用ground truth caption进行指代分割训练
  - image + caption → 预测mask'
  - Loss = segmentation_loss(mask', mask_GT)

### 4. OpenImage Dataset (可选)
- **路径**: `./data/openimages` (如果存在)
- **状态**: 目前未配置，路径不存在时会自动跳过
- **数据格式**: 单帧 (image, mask, label)
- **配置要求**:
  ```
  openimages/
  ├── train-annotations-object-segmentation.csv
  ├── oidv7-class-descriptions.csv
  ├── images/train/
  └── masks/train/
  ```

## 采样参数配置

### 测试配置 (test_dual_loop.sh)
```bash
SA1B_MAX_SAMPLES=500        # SA1B限制500个样本
BATCH_SIZE=1                # 每GPU批次大小
GRADIENT_ACCUMULATION=4     # 梯度累积步数
EFFECTIVE_BATCH_SIZE=4      # 实际批次大小 = 1 × 4
```

### 完整训练配置 (run_dual_loop_full.sh)
```bash
SA1B_MAX_SAMPLES=None       # SA1B使用全部数据
BATCH_SIZE=2                # 每GPU批次大小
GRADIENT_ACCUMULATION=4     # 梯度累积步数
NUM_GPUS=8                  # GPU数量
EFFECTIVE_BATCH_SIZE=64     # 实际批次大小 = 2 × 4 × 8
```

## LengthGroupedSampler 使用建议

### 当前实现
目前使用标准的 `shuffle=True` 随机采样。

### 是否需要LengthGroupedSampler？

**不推荐使用**，原因如下：

1. **固定图像尺寸**: 所有数据集都resize到1024×1024
   - SAV: resize到1024×1024
   - SA1B: resize到1024×1024
   - RefCOCO: resize到1024×1024
   - OpenImage: resize到1024×1024

2. **固定序列长度**:
   - 图像tokens: 256 (448×448 with downsample_ratio=0.5)
   - 文本长度: 虽然caption长度不同，但padding后batch内长度一致

3. **内存使用稳定**: 由于图像和序列长度固定，每个batch的内存使用基本一致

4. **数据多样性**: 随机采样确保不同数据集的样本充分混合

### 如果要使用LengthGroupedSampler

如果确实需要按长度分组（例如未来支持动态分辨率），可以这样实现：

```python
from torch.utils.data import Sampler
import numpy as np

class LengthGroupedSampler(Sampler):
    def __init__(self, dataset, batch_size, lengths=None):
        self.dataset = dataset
        self.batch_size = batch_size

        # Get lengths (e.g., text length, image size, etc.)
        if lengths is None:
            lengths = [self._get_length(i) for i in range(len(dataset))]

        # Sort by length
        self.indices = np.argsort(lengths)

    def _get_length(self, idx):
        # Define how to compute length for each sample
        sample = self.dataset[idx]
        if 'caption' in sample:
            return len(sample['caption'])
        return 0

    def __iter__(self):
        # Group by similar lengths and shuffle within groups
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i+self.batch_size]
            np.random.shuffle(batch_indices)
            yield from batch_indices

    def __len__(self):
        return len(self.dataset)

# 使用示例
# sampler = LengthGroupedSampler(train_dataset, args.batch_size)
# train_dataloader = DataLoader(
#     train_dataset,
#     batch_size=args.batch_size,
#     sampler=sampler,  # 使用sampler而不是shuffle=True
#     ...
# )
```

## 数据集权重配置（未来优化）

如果需要调整不同数据集的采样权重，可以使用 `WeightedRandomSampler`:

```python
from torch.utils.data import WeightedRandomSampler

# 示例：SAV权重2倍，其他数据集权重1倍
dataset_weights = {
    'sav': 2.0,
    'sa1b': 1.0,
    'refcoco': 1.0,
    'openimage': 1.0,
}

# 为每个样本分配权重
sample_weights = []
for sample in train_dataset:
    dataset_type = sample['dataset_type']
    sample_weights.append(dataset_weights.get(dataset_type, 1.0))

# 创建采样器
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True
)
```

## 总结

**当前配置** (推荐):
- ✅ 四个数据集：SAV (全部), SA1B (测试500/完整全部), RefCOCO (全部), OpenImage (可选)
- ✅ 固定分辨率：1024×1024
- ✅ 随机采样：`shuffle=True`
- ✅ 批次大小：测试1/完整2 per GPU
- ✅ 梯度累积：4 steps

**不需要**:
- ❌ LengthGroupedSampler (固定长度，不需要分组)
- ❌ 数据集权重调整 (目前均匀采样效果好)

**未来可选优化**:
- 动态分辨率支持 → 需要LengthGroupedSampler
- 数据集采样权重调整 → 需要WeightedRandomSampler
- 课程学习 (Curriculum Learning) → 逐步增加难度
