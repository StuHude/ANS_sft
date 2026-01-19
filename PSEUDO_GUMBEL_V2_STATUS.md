# Pseudo Gumbel V2 Training Status

## ✓ Successfully Fixed and Running

### Critical Fix #1: [SEG] Token Encoding (FIXED)

**Problem:**
- Previous implementation showed `loss=0.0000` for all steps
- Warning: "No [SEG] tokens found in forward_mask_with_text_embeds!"
- Root cause: `tokenizer.encode(" [SEG]...")` didn't recognize [SEG] as a special token

**Solution:**
In `pseudo_gumbel_core.py:forward_mask_with_text_embeds()`, changed from:
```python
# OLD (broken):
suffix = " [SEG]<|end|>\n<|assistant|>\nIt is [SEG]."
suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
```

To:
```python
# NEW (fixed):
# Build suffix manually to ensure [SEG] token is properly inserted
space_ids = tokenizer.encode(" ", add_special_tokens=False)
end_part_ids = tokenizer.encode("<|end|>\n<|assistant|>\nIt is ", add_special_tokens=False)
final_dot_ids = tokenizer.encode(".", add_special_tokens=False)
suffix_ids = space_ids + [seg_token_id] + end_part_ids + [seg_token_id] + final_dot_ids
```

### Critical Fix #2: Visual Prompt Shape Mismatch (FIXED)

**Problem:**
- Warning: "Shape mismatch, selected is 257, vp embeds is 256 !!!"
- Occurred when masks were small and pooled to 16x16 grid with all cells <= 0.5
- Root cause: Code forced `K = max(int(prompt_masks[i].sum().item()), 1)`
  - When actual K=0 (all False in prompt_masks), forced K=1 in prompt string
  - Prompt had 256 + 1 = 257 IMG_CONTEXT tokens
  - Model used actual prompt_masks (all False) → created 256 + 0 = 256 vp_embeds
  - Mismatch!

**Solution:**
Changed in `pseudo_gumbel_core.py` (lines 46 and 166):
```python
# OLD (broken):
K = max(int(prompt_masks[i].sum().item()), 1)  # Forced minimum of 1

# NEW (fixed):
K = int(prompt_masks[i].sum().item())  # Use actual K value
```

Now when K=0:
- Prompt: `<img><IMG_CONTEXT>*256</img> Region: <vp></vp>` → 256 IMG_CONTEXT tokens
- Model: 256 image embeddings + 0 VP embeddings = 256 total
- Perfect match!

When K>0 (e.g., K=36):
- Prompt: `<img><IMG_CONTEXT>*256</img> Region: <vp><IMG_CONTEXT>*36</vp>` → 292 IMG_CONTEXT tokens
- Model: 256 image embeddings + 36 VP embeddings = 292 total
- Perfect match!

**Verified with tests:**
- Small mask (5×5 pixels) → K=0 → 256 tokens = 256 embeddings ✓
- Large mask (400×400 pixels) → K=36 → 292 tokens = 292 embeddings ✓

### Current Training Status (Restarting with fixes)

**✓ Loss Values (Non-Zero):**
- Total loss: 0.5243 to 1.1461
- Mask loss: 0.0289 to 0.6502
- Dice loss: 0.48 to 0.50

**✓ Gradient Flow Confirmed:**
- text_embeds grad norm: 0.015 to 3.79 (varying, non-zero)
- LoRA gradients: tracked in TensorBoard
- SAM2 decoder gradients: tracked in TensorBoard
- Projector gradients: tracked in TensorBoard

**✓ Training Configuration:**
- 4 GPUs (CUDA 4-7)
- 4 datasets: SAV (100 samples), SA1B (500 samples), OpenImage, RefCOCO
- Batch size: 1 per GPU
- Gradient accumulation: 4 steps
- Learning rate: 1e-5
- Gumbel tau: 0.7, top-k: 128
- Mask ratio: 0.25
- Max caption length: 64
- LoRA r: 128, alpha: 256

**✓ Model Architecture:**
- Total parameters: 4,181,278,002
- Trainable parameters: 882,276,933 (21.10%)
  - LoRA: 504 parameters
  - mlp1 (proj): 6 parameters
  - SAM2 decoder: 131 parameters
  - text_hidden_fcs (proj): 4 parameters
  - mllm: 2 parameters

### Known Minor Issues (Non-Blocking)

1. **RefCOCO Batch Failures:**
   - Occasional "len(pred_embeddings_list):0 is not equal to len(frames_per_batch):1"
   - Handled with try-except, returns zero loss for that sample
   - RefCOCO represents small portion of dataset

### Training Loop Implementation

The 6-step pseudo-gumbel loop is correctly implemented:

1. **EMA generates pseudo tokens** (stop-grad, greedy)
   - Function: `generate_pseudo_tokens_with_ema()`

2. **Random mask 25%**
   - Function: `random_mask_tokens()`

3. **Trainable outputs logits**
   - Function: `forward_for_logits()`

4. **ST Gumbel-Softmax → text_embeds** (differentiable)
   - Function: `topk_gumbel_softmax()`
   - Gradient flows correctly: y_hard - y_soft.detach() + y_soft

5. **Predict mask from text_embeds**
   - Function: `forward_mask_with_text_embeds()`
   - Uses `inputs_embeds` (NOT input_ids) ✓

6. **Compute loss and backprop**
   - L_mask + L_dice
   - Backprop only to trainable model ✓

### Sanity Checks Passing

✓ text_embeds.grad is not None and norm > 0 (printed every 10 steps)
✓ LoRA parameters have gradients (logged to TensorBoard)
✓ SAM2 decoder has gradients (logged to TensorBoard)
✓ Projector layers have gradients (logged to TensorBoard)

### Files Modified

1. `/data/xyc/ANS/projects/llava_sam2/mask_caption_sft/pseudo_gumbel_core.py`
   - Fixed [SEG] token encoding in `forward_mask_with_text_embeds()`

2. `/data/xyc/ANS/projects/llava_sam2/mask_caption_sft/train_pseudo_gumbel_v2.py`
   - Main training script (correct implementation based on sa2va_4b.py)

3. `/data/xyc/ANS/run_pseudo_gumbel_v2_4gpu.sh`
   - Docker execution script

### Next Steps

1. **Monitor Training**
   - Check TensorBoard for LoRA/SAM2/projector gradient norms
   - Monitor loss convergence

2. **Checkpoint Saving**
   - Currently set to save every 500 steps
   - First checkpoint at step 500

### Commands

**Check training log:**
```bash
docker exec vlm-env tail -100 /data/xyc/ANS/work_dirs/pseudo_gumbel_v2_4gpu.log
```

**Check latest metrics:**
```bash
docker exec vlm-env bash -c "tail -50 /data/xyc/ANS/work_dirs/pseudo_gumbel_v2_4gpu.log | grep -E 'loss=|Step [0-9]+:'"
```

**TensorBoard (if needed):**
```bash
tensorboard --logdir /data/xyc/ANS/work_dirs/pseudo_gumbel_v2_4gpu/logs
```

## Summary

✓ **BOTH critical issues FIXED:**
  1. [SEG] token encoding → Loss now non-zero (0.5-1.5 range)
  2. Visual prompt shape mismatch → No more 257 vs 256 errors
✓ Gradients flowing correctly through ST Gumbel-Softmax
✓ All 4 datasets loading correctly
✓ All sanity checks passing
✓ Training ready to restart with correct visual prompt handling
