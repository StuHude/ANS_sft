# Pseudo Gumbel Training Fix Plan

## 核心问题
1. [SEG] token未在input_ids中正确出现 → seg_counts=0 → loss=0
2. 未使用Sa2VA的标准数据格式（prompt_masks, vp_overall_mask）
3. Template系统不匹配（应使用phi3_chat）

## 快速修复（推荐）

### 方案1：基于train_dual_loop.py修改

参考 `/data/xyc/ANS/projects/llava_sam2/mask_caption_sft/train_dual_loop.py`，该文件已经正确实现了：
- 数据集加载（使用dataset_builder）
- 模型调用
- 双循环训练

只需要在其基础上添加：
1. Step 3: 在生成caption后，添加pseudo token masking
2. Step 4: 添加Gumbel-Softmax替换argmax
3. Step 5: 确保第二次forward使用inputs_embeds

### 方案2：修复当前train_pseudo_gumbel.py

关键修复点：

1. **修复[SEG] token问题**:
   - 检查tokenizer.convert_tokens_to_ids('[SEG]')是否正确
   - 确保在构造suffix时[SEG]被正确编码
   - 添加调试输出确认token ID

2. **使用正确的forward接口**:
   参考`describe_anything_referring_dataset.py`的数据格式：
   ```python
   data_dict = {
       'pixel_values': [image_448],  # List of tensors
       'prompt_masks': [prompt_mask],  # List of [G, G] masks
       'vp_overall_mask': torch.tensor([True]),
       'input_ids': input_ids,
       'labels': labels,
       'g_pixel_values': [image_1024],  # For SAM2
       'masks': [gt_mask],
       'frames_per_batch': [1],
   }
   ```

3. **使用phi3_chat template**:
   参考xtuner的template系统，不要自己构造prompt

## 详细修复步骤

### Step 1: 修复[SEG] token (train_pseudo_gumbel.py:640-660)

```python
# 原代码：
suffix = " [SEG]<|end|>\n<|assistant|>\nIt is [SEG]."

# 问题：[SEG]可能没有被正确编码

# 修复：明确检查[SEG] token
seg_token_str = '[SEG]'
seg_token_id = self.tokenizer.convert_tokens_to_ids(seg_token_str)
if seg_token_id == self.tokenizer.unk_token_id:
    # [SEG] not in vocabulary, use model's seg_token_idx
    seg_token_id = self.actual_model.seg_token_idx

# 构造时确保包含SEG token
suffix = f" {seg_token_str}<|end|>\n<|assistant|>\nIt is {seg_token_str}."
```

### Step 2: 使用标准数据格式

参考dual_loop或describe_anything的数据格式，确保传给模型的数据包含所有必需字段。

### Step 3: 简化训练循环

对于非RefCOCO数据：
1. EMA生成pseudo tokens（greedy）
2. Random mask 25%
3. Trainable model输出logits（不用forward，直接调用mllm）
4. Gumbel-Softmax得到text_embeds
5. 用text_embeds调用模型做referring segmentation

## 推荐行动

**最快的方式**：
1. 停止当前训练
2. 参考`train_dual_loop.py`，在其基础上添加Gumbel-Softmax逻辑
3. 先让一个简化版本跑起来（不使用masked pseudo tokens）
4. 逐步添加完整功能

**如果坚持修复当前代码**：
1. 添加大量调试输出
2. 确认seg_token_id正确
3. 确认input_ids中包含[SEG]
4. 修复数据格式以匹配VideoLLaVASAMModel的期望

## 参考文件

- `/data/xyc/ANS/projects/llava_sam2/configs/sa2va_4b.py` - 标准配置
- `/data/xyc/ANS/projects/llava_sam2/datasets/describe_anything_referring_dataset.py` - 数据格式
- `/data/xyc/ANS/projects/llava_sam2/datasets/encode_fn.py` - 编码逻辑
- `/data/xyc/ANS/projects/llava_sam2/mask_caption_sft/train_dual_loop.py` - 双循环训练参考
