# Mask Caption SFT Training Fixes

## Overview
Fixed the mask captioning SFT training to use **real losses** instead of dummy losses. The training now properly implements referring segmentation with actual mask generation and loss computation.

## Changes Made

### 1. **Trainer Implementation** (`projects/llava_sam2/mask_caption_sft/trainer.py`)

#### Fixed `training_step()`:
- **Before**: Returned dummy loss (constant 0.1) with no actual training
- **After**: Implements real referring segmentation training:
  - For RefCOCO: Uses ground truth captions
  - For SAV/SA1B/OpenImage: Uses placeholder captions
  - Calls `forward_referring_segmentation()` for actual mask prediction

#### Fixed `forward_referring_segmentation()`:
- **Mask Generation**: Properly uses SAM2 grounding encoder with language embeddings
- **SAM2 API Compatibility**: Correctly formats inputs for SAM2's video-based API
  - Reshapes embeddings as `language_embd[frame_idx][obj_idx]`
  - Treats each image as a single-frame video with one object
- **Loss Computation**:
  - Cross-entropy loss for pixel-wise mask prediction
  - Dice loss for region overlap
  - IoU loss for quality metric
  - Total loss = mask_loss + dice_loss + iou_weight * iou_loss

### 2. **Dataset Builder** (Already supported)
- `SA1BDatasetWrapper` already has `max_samples` parameter
- `build_mask_caption_dataset()` already passes `sa1b_max_samples`
- `train_mask_caption_sft.py` already has `--sa1b_max_samples` argument

### 3. **Loss Functions**
Uses MMDetection losses (already imported):
- **CrossEntropyLoss**: `use_sigmoid=True, weight=2.0`
- **DiceLoss**: `use_sigmoid=True, naive_dice=True, weight=0.5`
- **IoU Loss**: Custom implementation for quality metric

## Training Flow

### Current Implementation:
```
Input: image + mask (+ optional caption)
  ↓
1. Tokenize caption with [SEG] token
  ↓
2. Model forward → hidden states
  ↓
3. Extract [SEG] token embeddings
  ↓
4. Project through text_hidden_fcs
  ↓
5. SAM2 encoder → image features
  ↓
6. SAM2 decoder + language embeddings → predicted mask
  ↓
7. Compute losses:
   - Mask loss (CE)
   - Dice loss
   - IoU loss
  ↓
8. Backward + optimize
```

### Dataset Handling:
- **RefCOCO**: image + caption → mask (referring segmentation)
- **SAV**: image + mask (simplified to referring segmentation for now)
- **SA1B**: image + mask (simplified to referring segmentation for now)
- **OpenImage**: image + mask (simplified to referring segmentation for now)

## Usage

### Quick Test (with limited SA1B data):
```bash
bash test_sft_training.sh
```

This script:
- Uses 4 GPUs
- Batch size 1 per GPU
- Gradient accumulation 4 steps (effective batch size = 16)
- **SA1B limited to 1000 samples** for fast testing
- Saves checkpoints every 100 steps

### Full Training:
```bash
torchrun --nproc_per_node=8 \
    projects/llava_sam2/mask_caption_sft/train_mask_caption_sft.py \
    --model_path /data/xyc/ANS/pretrain_hf \
    --output_dir ./work_dirs/mask_caption_sft_full \
    --num_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --sav_dir /data/xyc/formed_data/npz \
    --refcoco_dir ./data/ref_seg \
    --sa1b_dir /data/xyc/mhx/SA1b/OpenDataLab___SA-1B/dataset \
    --sa1b_max_samples 10000 \  # Limit SA1B samples
    --use_lora \
    --lora_r 128 \
    --lora_alpha 256
```

### Parameters:
- `--sa1b_max_samples N`: Limit SA1B to first N image files (speeds up loading)
- `--use_lora`: Enable LoRA fine-tuning
- `--lora_r`: LoRA rank (default: 128)
- `--lora_alpha`: LoRA scaling factor (default: 256)
- `--batch_size`: Per-GPU batch size
- `--gradient_accumulation_steps`: Gradient accumulation

## Expected Behavior

### Before Fix:
```
loss=0.1000, iou=0.0000  (constant, no training)
```

### After Fix:
```
loss=2.3456, iou=0.4523  (decreasing loss, increasing IoU)
mask_loss=1.234, dice_loss=0.567, iou_loss=0.123
```

## Model Architecture

### Trainable Components:
1. **LoRA adapters**: Language model (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
2. **MLP projector** (`mlp1`): Vision → Language projection
3. **SAM2 decoder** (`sam_mask_decoder`): Mask generation
4. **Text hidden FCs** (`text_hidden_fcs`): [SEG] embedding projection

### Frozen Components:
1. **Vision encoder** (InternVL): Image feature extraction
2. **SAM2 image encoder**: Image encoding for segmentation
3. **Language model backbone** (except LoRA): Core LLM weights

## Troubleshooting

### If training still shows constant loss:
1. Check model has `grounding_encoder` and `text_hidden_fcs` attributes
2. Verify SAM2 decoder is unfrozen (check logs for "SAM2 decoder unfrozen")
3. Ensure LoRA is properly applied (check logs for "LoRA applied")

### If OOM (Out of Memory):
1. Reduce `--batch_size` (try 1)
2. Increase `--gradient_accumulation_steps` (try 8 or 16)
3. Use `--sa1b_max_samples` to limit dataset size

### If training is slow:
1. Increase `--num_workers` (default: 4)
2. Use `--sa1b_max_samples` to reduce dataset size
3. Check `--log_interval` and `--save_interval` are not too frequent

## Next Steps (Future Improvements)

1. **Implement mask captioning loop**: Generate captions from masks (currently simplified)
2. **Dual-loop training**:
   - Loop 1: mask → caption
   - Loop 2: caption → mask (current implementation)
3. **Better caption prompts**: Use actual mask-conditioned caption generation
4. **Multi-object support**: Handle multiple objects per image
5. **Video support**: Extend to video referring segmentation

## File Modifications Summary

- ✅ `projects/llava_sam2/mask_caption_sft/trainer.py`: Implemented real losses
- ✅ `test_sft_training.sh`: Created test script with SA1B limiting
- ✅ Dataset builder: Already supports max_samples
- ✅ Training script: Already supports --sa1b_max_samples

## Verification

To verify the fix is working:
```bash
# Check training logs for:
1. Loss values that change (not constant 0.1)
2. IoU values > 0.0
3. Individual loss components (mask_loss, dice_loss, iou_loss)
4. "✓ SAM2 decoder unfrozen" in initialization
5. "✓ LoRA applied to language_model" in initialization
```
