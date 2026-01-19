"""
Configuration file for Mask Captioning SFT Training

This file defines dataset paths and training hyperparameters.
"""

# Model
model_path = '/data/xyc/ANS/pretrain_hf'

# Output
output_dir = './work_dirs/mask_caption_sft'

# Training hyperparameters
num_epochs = 1
batch_size = 2  # Per GPU
learning_rate = 1e-5
gradient_accumulation_steps = 4
max_grad_norm = 1.0
ema_decay = 0.999

# LoRA config
use_lora = True
lora_r = 128
lora_alpha = 256
lora_dropout = 0.05

# Dataloader
num_workers = 4

# Logging
log_interval = 10
save_interval = 1000

# Dataset paths
# SAV Dataset
sav_dir = '/data/xyc/formed_data/npz'

# SA-1B Dataset
sa1b_dir = '/data/xyc/mhx/SA1b/OpenDataLab___SA-1B/raw'

# OpenImage Dataset
openimage_dir = '/data/xyc/openv7/data'

# RefCOCO Dataset
refcoco_dir = '/data/xyc/ANS/data/ref_seg'

# Loss weights
caption_loss_weight = 1.0
iou_loss_weight = 1.0
mask_loss_weight = 2.0
dice_loss_weight = 0.5
