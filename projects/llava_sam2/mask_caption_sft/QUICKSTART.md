# Quick Start Guide - Mask Captioning SFT Training

This is a quick start guide to get you training as fast as possible.

## 1. Prepare SAV Dataset Path

Edit `config.py` and update the SAV dataset path:

```python
# In config.py, line 36
sav_dir = '/path/to/your/sav/npz/data'  # UPDATE THIS PATH
```

Replace `/path/to/your/sav/npz/data` with the actual path to your SAV NPZ files.

## 2. Verify Other Dataset Paths (Optional)

The following datasets are already configured, but verify paths exist:

```bash
# SA-1B dataset
ls /data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw

# OpenImage dataset
ls /data/xyc/openv7/data

# RefCOCO dataset (optional, can skip if not using)
ls /data/xyc/ANS/data/ref_seg/refcoco
```

## 3. Activate Environment

```bash
export PATH="/home/xiaoyicheng/miniconda3/bin:$PATH"
conda activate vlm
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
cd /data/xyc/ANS
```

## 4. Start Training

### Option A: Single GPU (Quick Test)

```bash
bash projects/llava_sam2/mask_caption_sft/scripts/run_single_gpu.sh
```

### Option B: Multi-GPU (8 GPUs, Recommended)

```bash
bash projects/llava_sam2/mask_caption_sft/scripts/run_multi_gpu.sh 8
```

## 5. Monitor Training

Watch the training logs:

```bash
tail -f work_dirs/mask_caption_sft/logs/events.out.tfevents.*
```

Or use TensorBoard:

```bash
tensorboard --logdir work_dirs/mask_caption_sft/logs --port 6006
```

## 6. Check Outputs

Checkpoints are saved to:
```
work_dirs/mask_caption_sft/
├── checkpoint_step_1000.pth
├── checkpoint_step_2000.pth
├── ...
└── final_model.pth
```

## What's Happening During Training?

For each batch, the model:

1. **Stage 1 - Mask Captioning:**
   - Takes `image1 + mask1` from SAV/SA-1B/OpenImage
   - Generates caption describing the masked region

2. **Stage 2 - Referring Segmentation (via EMA):**
   - Takes `image2 + caption` (or same image for non-SAV datasets)
   - EMA model generates `mask'`
   - Computes loss: IoU + Dice + CrossEntropy between `mask'` and ground truth `mask2`

3. **Backpropagation:**
   - Gradients flow through the student model
   - EMA model is updated via exponential moving average (no gradients)

4. **RefCOCO (if enabled):**
   - Skips caption generation (uses ground truth caption)
   - Directly performs referring segmentation
   - Adds to the total loss

## Troubleshooting Quick Fixes

### "CUDA Out of Memory"
```python
# In config.py, reduce batch size:
batch_size = 1  # Instead of 2
gradient_accumulation_steps = 8  # Instead of 4
```

### "SAV dataset not found"
```python
# In config.py, update the path:
sav_dir = '/actual/path/to/sav/npz/files'
```

### "RefCOCO dataset not found"
```bash
# Disable RefCOCO in training script
# Remove --use_refcoco flag from run_single_gpu.sh or run_multi_gpu.sh
# Or download RefCOCO data (see README.md)
```

### Training is slow
```python
# In config.py, limit dataset samples for quick testing:
max_sa1b_samples = 1000  # Instead of 10000
max_openimage_samples = 1000  # Instead of 10000
```

## Expected Training Time

With 8 GPUs (batch_size=2 per GPU, gradient_accumulation=4):
- Effective batch size: 64
- Training speed: ~5-10 iterations/second (depends on dataset size)
- Full epoch: Depends on total dataset size

Example:
- SAV: 10,000 samples
- SA-1B: 10,000 samples
- OpenImage: 10,000 samples
- Total: 30,000 samples
- Batches: 30,000 / 16 (per-gpu batch size × num_gpus) ≈ 1,875 batches
- Time: ~3-6 hours for 1 epoch (rough estimate)

## Next Steps After Training

1. **Evaluate the model:**
   ```bash
   # Use existing evaluation scripts in projects/llava_sam2/evaluation/
   bash projects/llava_sam2/evaluation/dist_test.sh \
       projects/llava_sam2/evaluation/refcoco_eval.py \
       work_dirs/mask_caption_sft/final_model.pth \
       8 --dataset refcoco --split val
   ```

2. **Convert to HuggingFace format:**
   ```bash
   python tools/convert_to_hf_new.py \
       --model_path work_dirs/mask_caption_sft/final_model.pth \
       --output_dir work_dirs/mask_caption_sft_hf
   ```

3. **Fine-tune further with RL training:**
   ```bash
   # Use the SFT-trained model as initialization for RL training
   python projects/llava_sam2/rl_train/train_sa2va_rl.py \
       --model_path work_dirs/mask_caption_sft_hf \
       ...
   ```

## Key Configuration Parameters

In `config.py`:

```python
# Training
num_epochs = 1              # Number of epochs
batch_size = 2              # Per-GPU batch size
learning_rate = 1e-5        # Learning rate
gradient_accumulation_steps = 4  # Gradient accumulation

# LoRA
lora_r = 128                # LoRA rank
lora_alpha = 256            # LoRA alpha

# EMA
ema_decay = 0.999           # EMA decay rate

# Loss weights
caption_loss_weight = 1.0   # Caption generation loss
iou_loss_weight = 1.0       # IoU loss
mask_loss_weight = 2.0      # Mask CE loss
dice_loss_weight = 0.5      # Dice loss
```

Adjust these based on your needs and hardware constraints.

## Questions?

See the full README.md for detailed documentation, or check existing code in:
- `trainer.py` - Training loop implementation
- `datasets/` - Dataset wrappers
- `config.py` - Configuration settings

Happy training! 🚀
