# Sa2VA Training Analysis

## 发现的问题

通过查看原始代码，我发现了关键架构差异：

### 1. 原始训练使用 VideoLLaVASAMModel
位置：`projects/llava_sam2/models/llava_sam2.py`

**特点**：
- 继承自LisaModel
- forward方法返回loss_dict: `{'loss_mask': ..., 'loss_dice': ..., 'llm_loss': ...}`
- 内置mask loss计算
- 使用xtuner框架训练（`tools/train.py`）

**配置**：`projects/llava_sam2/configs/sa2va_4b.py`

```python
model = dict(
    type=VideoLLaVASAMModel,
    mllm=dict(type=InternVL_Slowfast, ...),
    grounding_encoder=dict(type=SAM2TrainRunner),
    loss_mask=dict(type=CrossEntropyLoss, ...),
    loss_dice=dict(type=DiceLoss, ...),
)
```

### 2. 当前mask_caption_sft使用 Sa2VAChatModel
位置：`pretrain_hf/modeling_sa2va_chat.py`

**特点**：
- HuggingFace格式模型（用于推理）
- forward方法返回CausalLMOutputWithPast（只有LLM loss）
- **没有内置mask loss计算**
- 有grounding_encoder和text_hidden_fcs（用于推理生成mask）
- 不是为训练设计的

## 解决方案

你有两个选择：

### 方案1：使用原始训练框架（**推荐**）

直接使用xtuner框架和VideoLLaVASAMModel：

```bash
# 使用原始训练配置
python tools/train.py projects/llava_sam2/configs/sa2va_4b.py

# 或者创建自定义配置，只训练RefCOCO等数据集
# 可以参考sa2va_4b.py，注释掉不需要的数据集
```

**优点**：
- 已经过验证，稳定可靠
- 自动计算mask loss和dice loss
- 完整的训练管道

**缺点**：
- 需要适应xtuner框架
- 配置文件较复杂

### 方案2：修改mask_caption_sft使用VideoLLaVASAMModel

修改`train_mask_caption_sft.py`：

```python
# 替换模型加载
from projects.llava_sam2.models.llava_sam2 import VideoLLaVASAMModel from xtuner.registry import BUILDER

# 构建模型（参考sa2va_4b.py的model配置）
model_cfg = dict(
    type=VideoLLaVASAMModel,
    mllm=...,  # 需要从pretrain_hf构建
    grounding_encoder=dict(type=SAM2TrainRunner),
    loss_mask=dict(...),
    loss_dice=dict(...),
    ...
)
model = BUILDER.build(model_cfg)
```

然后trainer中直接调用model.forward()即可。

### 方案3：在Sa2VAChatModel上手动添加mask loss计算

保持当前架构，但在trainer中：
1. 调用model.forward获取hidden_states
2. 手动提取[SEG] token embeddings
3. 手动调用grounding_encoder生成masks
4. 手动计算mask loss和dice loss

**这就是我刚才修改的代码要做的事情，但需要进一步完善。**

## 推荐的行动方案

**我强烈建议使用方案1（原始训练框架）**：

1. 创建一个简化的配置文件（基于`sa2va_4b.py`）：

```python
# projects/llava_sam2/configs/sa2va_refcoco_only.py
# 复制sa2va_4b.py，然后只保留RefCOCO数据集

train_dataset = dict(
    type=ConcatDataset,
    datasets=[
        refcoco_segm_dataset,
        refcoco_plus_segm_dataset,
        refcocog_segm_dataset,
    ]
)
```

2. 使用原始训练脚本：

```bash
bash tools/dist.sh train projects/llava_sam2/configs/sa2va_refcoco_only.py 4
```

这样可以直接利用已经验证过的训练管道，不需要重新实现所有逻辑。

## 当前mask_caption_sft的问题

当前的`mask_caption_sft`实现尝试：
- 使用HuggingFace格式的Sa2VAChatModel（推理模型）
- 手动构建训练循环
- 手动计算损失

**主要问题**：
1. Sa2VAChatModel.forward不返回mask loss
2. 需要手动提取hidden states并计算loss
3. 重复实现了VideoLLaVASAMModel已有的功能

## 你想要的结果

如果你想训练mask captioning（mask→caption）+ referring segmentation（caption→mask）：

**最简单的方法**是使用原始框架，添加mask captioning数据集到配置文件中。原始的VideoLLaVASAMModel已经支持这两种任务。

你需要我帮你：
1. 创建简化的配置文件使用原始框架？
2. 还是继续修复当前的mask_caption_sft实现？

请告诉我你的偏好！
