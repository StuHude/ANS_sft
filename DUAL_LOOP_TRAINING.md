# Dual-Loop Mask Captioning SFT Training

## 训练流程

**Dual-Loop自监督训练**：

```
image + mask ──→ Sa2VA模型 ──→ caption
                                  ↓
                          EMA Sa2VA模型
                                  ↓
                          predicted mask'
                                  ↓
                      Segmentation Loss
                      (mask' vs mask)
```

### 关键特点

1. **使用4个数据集**：
   - SAV: 735,577个样本
   - SA1B: 全部样本（或限制样本用于测试）
   - OpenImage: 分割数据
   - RefCOCO: Referring segmentation数据

2. **使用原始VideoLLaVASAMModel架构**：
   - 完全兼容原始训练代码
   - 自动计算mask loss和dice loss
   - 支持LoRA微调

3. **载入预训练权重**：
   - 从sa2va_4b_iter152k_fp32.pth初始化
   - 继续训练而不是从头开始

## 快速开始

### 测试运行（使用有限数据）

```bash
bash test_dual_loop.sh
```

这将：
- 使用500个SA1B样本（快速测试）
- 单机4卡训练
- Batch size 1, gradient accumulation 4
- 验证代码能正常运行

### 完整训练（使用全部数据）

```bash
bash run_dual_loop_full.sh
```

这将：
- 使用全部4个数据集的所有样本
- 8卡分布式训练
- Batch size 2, gradient accumulation 4 (effective batch size = 64)
- 正式开始训练

## 文件结构

```
projects/llava_sam2/mask_caption_sft/
├── train_dual_loop.py          # 新的dual-loop训练脚本
├── dataset_builder.py          # 数据集构建器（已有）
├── trainer.py                  # 旧的trainer（不再使用）
└── train_mask_caption_sft.py   # 旧的训练脚本（不再使用）

scripts/
├── test_dual_loop.sh           # 测试脚本（有限数据）
└── run_dual_loop_full.sh       # 完整训练脚本
```

## 实现细节

### 1. 模型架构

使用**VideoLLaVASAMModel**（而不是Sa2VAChatModel）：

```python
model = VideoLLaVASAMModel(
    mllm=InternVL_Slowfast(...),  # Vision-Language模型
    grounding_encoder=SAM2TrainRunner(...),  # SAM2分割
    loss_mask=CrossEntropyLoss(...),  # 像素级loss
    loss_dice=DiceLoss(...),  # 区域overlap loss
    pretrained_pth="...",  # 预训练权重
)
```

### 2. 数据格式

完全遵循原始训练的数据格式：

```python
data = {
    'pixel_values': [...],  # 448x448 for vision encoder
    'g_pixel_values': [...],  # 1024x1024 for SAM2
    'input_ids': tensor,  # Tokenized text with [SEG]
    'position_ids': tensor,
    'attention_mask': tensor,
    'labels': tensor,  # For LLM loss
    'masks': [...],  # Ground truth masks
    'frames_per_batch': [1, 1, ...],  # Each sample = 1 frame
}

# Forward
loss_dict = model(data, mode='loss')
# Returns: {'loss_mask': ..., 'loss_dice': ..., 'llm_loss': ...}
```

### 3. Dual-Loop实现

当前实现（**第一版，简化版**）：

```python
def dual_loop_step(images, masks):
    # Step 1: Generate caption (currently placeholder)
    captions = generate_caption_from_mask(images, masks)

    # Step 2: Predict mask from caption
    loss_dict = model.forward(
        image + caption → mask'
    )

    # Step 3: Compute loss (mask' vs mask)
    return loss_dict
```

**TODO**（完整dual-loop）：
- [ ] 实现真正的mask→caption生成（使用模型生成而不是placeholder）
- [ ] 使用EMA模型生成mask'作为teacher
- [ ] 添加caption quality reward（METEOR/LLM judge）

### 4. 训练参数

```python
# Model
LoRA: r=128, alpha=256
Frozen: Vision encoder, LLM backbone
Trainable: LoRA adapters, SAM2 decoder, text_hidden_fcs, mlp1

# Training
Learning rate: 1e-5
Weight decay: 0.05
Max grad norm: 1.0
Batch size: 1 per GPU (test), 2 per GPU (full)
Gradient accumulation: 4
EMA decay: 0.999

# Loss weights (from sa2va_4b.py)
loss_mask: 2.0
loss_dice: 0.5
llm_loss: 1.0
```

## 监控训练

### 关键指标

```
loss: 总损失（应该下降）
mask_loss: 像素级交叉熵（应该下降）
dice_loss: Dice系数损失（应该下降）
llm_loss: 语言模型损失（应该下降）
```

### 预期行为

**正常训练**：
- 初始loss: ~2-3
- mask_loss: 从~1.5降到~0.5
- dice_loss: 从~0.8降到~0.2
- 训练稳定，loss持续下降

**异常情况**：
- Loss不变或上升 → 检查学习率、梯度
- Loss=NaN → 检查数据、梯度裁剪
- OOM → 减小batch size或增加gradient accumulation

## 下一步完善

当前实现是**简化版本**，可以训练但还未完全实现dual-loop。

### 完整dual-loop实现需要

1. **Caption生成**（在`generate_caption_from_mask`中）：
   ```python
   @torch.no_grad()
   def generate_caption_from_mask(images, masks):
       # 使用visual prompting
       # 调用model.generate()生成caption
       # 返回实际生成的caption
   ```

2. **EMA teacher**：
   ```python
   # 使用EMA模型生成mask'作为更稳定的监督
   with torch.no_grad():
       teacher_loss_dict = ema_model(...)
       teacher_masks = extract_predicted_masks(...)
   # 使用teacher_masks计算loss
   ```

3. **Reward-based training**（可选）：
   - 添加caption quality reward（METEOR/LLM judge）
   - 结合mask IOU reward
   - 使用GRPO/PPO优化

## 故障排除

### 模型加载失败

```bash
# 检查pretrained_pth路径
ls -lh /data/xiaoyicheng/Sa2VA/pretrained/4B_checkpoint/sa2va_4b_iter152k_fp32.pth/pytorch_model.bin

# 检查model_path
ls -lh ./pretrained/InternVL2_5-4B/
```

### 数据集加载失败

```bash
# 检查数据集路径
ls /data/xyc/formed_data/npz/ | head
ls /data/xyc/mhx/SA1b/OpenDataLab___SA-1B/dataset/ | head
ls ./data/ref_seg/refcoco/
```

### OOM (Out of Memory)

```bash
# 选项1: 减小batch size
--batch_size 1 --gradient_accumulation_steps 8

# 选项2: 限制SA1B样本
--sa1b_max_samples 1000

# 选项3: 减少workers
--num_workers 2
```

### 训练太慢

```bash
# 检查dataloader workers
--num_workers 6  # 增加到6-8

# 检查是否使用了distributed training
torchrun --nproc_per_node=8 ...

# 检查gradient accumulation
--gradient_accumulation_steps 4  # 不要太大
```

## 参考代码

本实现完全基于原始Sa2VA训练代码：

- **配置文件**: `projects/llava_sam2/configs/sa2va_4b.py`
- **模型代码**: `projects/llava_sam2/models/llava_sam2.py`
- **数据集代码**: `projects/llava_sam2/datasets/RefCOCO_Dataset.py`
- **Collate函数**: `projects/llava_sam2/datasets/collect_fns.py`

## 联系与支持

如果遇到问题：
1. 检查日志输出中的错误信息
2. 参考上面的故障排除部分
3. 查看原始训练配置`sa2va_4b.py`对比参数
