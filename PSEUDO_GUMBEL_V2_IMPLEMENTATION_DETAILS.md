# Pseudo Gumbel V2训练详细实现说明

## 问题3：显存占用分析

### 当前显存使用（需验证）：
让我先检查实际显存使用情况...

### 预期显存占用：

**模型组件：**
1. **Trainable Model (主模型)**:
   - Vision encoder (frozen): ~2GB (InternVL2.5-4B vision部分)
   - LLM with LoRA (部分trainable): ~8GB
   - SAM2 encoder (frozen): ~3GB
   - SAM2 decoder (trainable): ~1GB
   - Projectors (trainable): <1GB
   - **Total**: ~14-16GB

2. **EMA Model (完整副本)**:
   - 完整模型参数副本: ~14-16GB
   - **Total**: ~14-16GB

3. **训练中间状态**:
   - 梯度 (只存trainable参数): ~4GB
   - Optimizer states (AdamW for trainable params): ~8GB
   - Activation缓存: ~3-5GB
   - **Total**: ~15-17GB

**理论总显存**: 14 + 14 + 15 = **43-47GB**

**如果实际只用25GB，说明有问题！**可能原因：
- EMA model没有真正创建/加载
- 梯度没有正确反向传播
- Optimizer没有正确初始化

### 需要检查的代码位置：
1. `train_pseudo_gumbel_v2.py:666-669`: EMA模型创建
2. `train_pseudo_gumbel_v2.py:376-381`: EMA更新
3. Backward是否真的穿过了text_embeds

---

## 问题4：25% Mask和第一次Trainable Forward的详细说明

### 完整训练循环（Pseudo-Gumbel 6步）：

```
输入: image1+mask1 (from dataset), image2+mask2
```

#### **Step 1: EMA生成pseudo tokens (stop-grad)**

位置: `pseudo_gumbel_core.py:generate_pseudo_tokens_with_ema()`

**输入**:
- `images1`: [B, 3, 1024, 1024] - 原始图像
- `masks1`: [B, 1024, 1024] - 原始mask

**处理**:
1. Resize images到448×448 (InternVL输入尺寸)
2. 构建visual prompt masks:
   ```python
   # Osprey实现：直接resize到16×16
   prompt_masks = F.interpolate(masks1.unsqueeze(1), size=(16, 16), mode='nearest').squeeze(1)
   # Shape: [B, 16, 16]
   ```
3. 计算K (每个sample):
   ```python
   K = int(prompt_masks[i].bool().to(torch.int64).sum().item())
   # K: 16×16网格中被mask覆盖的cell数量 (1到256)
   ```
4. 构建prompt:
   ```python
   prompt = "<|user|>\n<img><IMG_CONTEXT>*256</img> Region: <vp><IMG_CONTEXT>*K</vp>. Describe this region briefly.<|end|>\n<|assistant|>\n"
   ```
   - 256个IMG_CONTEXT用于完整图像
   - K个IMG_CONTEXT用于visual prompt区域

5. **EMA模型forward** (no gradient):
   ```python
   with torch.no_grad():
       outputs = ema_model.mllm.generate(
           pixel_values=[img_448],
           input_ids=input_ids,
           prompt_masks=[prompt_masks],  # [1, 16, 16]
           vp_overall_mask=[True],        # 启用visual prompt
           generation_config=gen_config,
       )
   ```

**输出**:
- `pseudo_toks`: [B, max_caption_len=64] - EMA生成的伪token序列

---

#### **Step 2: 随机mask 25%的tokens**

位置: `pseudo_gumbel_core.py:random_mask_tokens()`

**输入**:
- `pseudo_toks`: [B, 64] - Step 1生成的tokens

**处理**:
```python
def random_mask_tokens(tokens, mask_ratio=0.25, vocab_size, pad_token_id, device):
    masked = tokens.clone()
    B, T = tokens.shape  # [B, 64]

    # 为每个token生成随机数，决定是否mask
    mask = torch.rand(B, T, device=device) < mask_ratio  # [B, 64], 约25%为True

    # 不mask padding tokens
    mask = mask & (tokens != pad_token_id)

    # 将被mask的位置替换为随机token
    random_tokens = torch.randint(0, vocab_size, (B, T), device=device)
    masked[mask] = random_tokens[mask]

    return masked  # [B, 64]
```

**举例**:
```
原始: [1234, 5678, 9012, 3456, 7890, ...]  (64 tokens)
Mask: [False, True, False, True, False, ...]  (约16个True)
替换: [1234, RAND, 9012, RAND, 7890, ...]  (16个位置被随机token替换)
```

**输出**:
- `masked_pseudo_toks`: [B, 64] - 25%位置被随机替换的token序列

---

#### **Step 3: Trainable模型第一次forward → 输出logits**

位置: `pseudo_gumbel_core.py:forward_for_logits()`

**目标**: 让trainable模型学习"去噪"，从masked tokens恢复原始tokens

**输入**:
- `images1`: [B, 3, 1024, 1024]
- `masks1`: [B, 1024, 1024]
- `masked_pseudo_toks`: [B, 64] - Step 2的masked tokens

**处理**:
1. 构建完整input序列:
   ```python
   user_prompt = "<|user|>\n<img><IMG_CONTEXT>*256</img> Region: <vp><IMG_CONTEXT>*K</vp>. Describe this region briefly.<|end|>\n<|assistant|>\n"

   full_sequence = user_prompt_ids + masked_pseudo_toks_ids + [EOS]
   # 完整序列长度 ≈ 280 + 64 + 1 = 345 tokens
   ```

2. **Trainable模型forward** (with gradient):
   ```python
   data = {
       'pixel_values': [img_448],
       'input_ids': full_input_ids,  # user_prompt + masked_tokens + EOS
       'prompt_masks': [prompt_masks],
       'vp_overall_mask': [True],
       'labels': None,  # 不计算LM loss，只要logits
   }

   output = trainable_model.mllm(data, mode='loss')
   logits = output.logits  # [B, seq_len, vocab_size]
   ```

3. 提取caption位置的logits:
   ```python
   # 只要对应masked_pseudo_toks位置的logits
   caption_logits = logits[:, -(max_caption_len+1):-1, :]  # [B, 64, vocab_size]
   ```

**输出**:
- `logits`: [B, 64, vocab_size] - 对每个token位置的预测logits

---

#### **Step 4: ST Gumbel-Softmax → 可微的text_embeds**

位置: `pseudo_gumbel_core.py:topk_gumbel_softmax()`

**输入**:
- `logits`: [B, 64, vocab_size]

**处理**:
```python
def topk_gumbel_softmax(logits, tau=0.7, topk=128, embedding_layer):
    B, T, V = logits.shape  # [B, 64, 151679]

    # 1. 选择top-k logits (稀疏化)
    topk_vals, topk_idx = logits.topk(topk, dim=-1)  # [B, 64, 128]

    # 2. Gumbel-Softmax采样
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(topk_vals) + 1e-10) + 1e-10)
    y_soft = F.softmax((topk_vals + gumbel_noise) / tau, dim=-1)  # [B, 64, 128]

    # 3. Straight-Through trick (硬前向，软后向)
    index = y_soft.argmax(dim=-1, keepdim=True)  # [B, 64, 1]
    y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)  # one-hot
    y = y_hard - y_soft.detach() + y_soft  # [B, 64, 128]
    # 梯度会通过y_soft流动!

    # 4. 加权求和得到embeddings
    E = embedding_layer.weight  # [vocab_size, embed_dim]
    E_topk = E[topk_idx]  # [B, 64, 128, embed_dim]

    text_embeds = (y.unsqueeze(-1) * E_topk).sum(dim=-2)  # [B, 64, embed_dim]

    return text_embeds
```

**关键**: `text_embeds`是**可微的**！梯度可以反向传播回logits。

**输出**:
- `text_embeds`: [B, 64, embed_dim] - 可微的文本embeddings

---

#### **Step 5: Trainable模型第二次forward → 预测mask**

位置: `pseudo_gumbel_core.py:forward_mask_with_text_embeds()`

**目标**: 用text_embeds作为referring expression，预测mask

**输入**:
- `images2`: [B, 3, 1024, 1024]
- `text_embeds`: [B, 64, embed_dim] - Step 4的可微embeddings
- `gt_masks2`: [B, 1024, 1024] - Ground truth mask

**处理**:
1. 构建prompt (用embeddings替代tokens):
   ```python
   prefix = "<|user|>\n<img><IMG_CONTEXT>*256</img>Please segment: "
   suffix = " [SEG]<|end|>\n<|assistant|>\nIt is [SEG]."

   # 构建inputs_embeds而不是input_ids:
   input_embeds = [
       embed(prefix_ids),      # embedding of prefix
       text_embeds,            # [B, 64, D] - 可微的caption embeddings!
       embed(suffix_ids),      # embedding of suffix
   ]
   # Shape: [B, total_len, embed_dim]
   ```

2. 替换IMG_CONTEXT位置为vision embeddings:
   ```python
   vit_embeds = model.mllm.model.extract_feature(images_448)
   input_embeds[IMG_CONTEXT位置] = vit_embeds  # 替换256个位置
   ```

3. **LLM forward** (用inputs_embeds):
   ```python
   outputs = model.mllm.model.language_model(
       inputs_embeds=input_embeds,  # 直接用embeddings!
       attention_mask=attention_mask,
   )

   hidden_states = outputs.hidden_states[-1]
   ```

4. 提取[SEG] token位置的hidden states:
   ```python
   seg_mask = (input_ids == seg_token_id)
   pred_embeddings = model.text_hidden_fcs(hidden_states[seg_mask])
   ```

5. **SAM2 decoder预测mask**:
   ```python
   sam_states = model.grounding_encoder.get_sam2_embeddings(images_1024)
   pred_masks = model.grounding_encoder.inject_language_embd(
       sam_states,
       pred_embeddings
   )
   ```

**输出**:
- `pred_masks`: [B, H, W] - 预测的mask

---

#### **Step 6: 计算loss并反向传播**

**Loss计算**:
```python
loss_mask = F.binary_cross_entropy_with_logits(pred_masks, gt_masks2) * 2.0
loss_dice = dice_loss(pred_masks, gt_masks2) * 0.5
total_loss = loss_mask + loss_dice
```

**反向传播路径** (关键!):
```
total_loss
  ↓ backward
SAM2 decoder (trainable)
  ↓
text_hidden_fcs (trainable)
  ↓
LLM hidden states
  ↓
text_embeds (可微!)  ← ST Gumbel-Softmax的梯度传递点
  ↓
logits (第一次forward的输出)
  ↓
LLM LoRA layers (trainable)
  ↓
Vision-Language projector (trainable)
```

**梯度不会流向EMA模型** (因为Step 1用了`torch.no_grad()`)

---

## 总结：25% Mask的作用

**训练目标**:
1. EMA生成pseudo tokens (作为"软标签")
2. Mask掉25%，让trainable模型学习去噪/补全
3. 去噪后的caption(通过Gumbel转成embeddings)用于预测mask
4. Mask预测的loss反向传播，优化整个pipeline

**关键创新**:
- **第一次forward**: 学习去噪masked caption
- **Gumbel-Softmax**: 让离散token变成可微embeddings
- **第二次forward**: 用可微caption预测mask
- **联合训练**: 同时优化caption生成和mask预测

**和标准BERT mask不同**:
- BERT: mask → predict masked tokens (监督学习)
- 这里: mask → predict caption → predict mask (通过mask loss优化caption)
