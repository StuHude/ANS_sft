# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ANS (Autoregressive Nested Scaling)** is a pixel-level multi-modal large language model that combines vision understanding with segmentation capabilities. The core implementation is **Sa2VA** (SAM2 + Vision-LLM), which integrates:
- **Vision encoder**: InternVL2.5-4B for image/video understanding
- **Language model**: InternLM2 or Phi3 with LoRA fine-tuning
- **Grounding encoder**: SAM2 (Segment Anything Model 2) for pixel-level segmentation
- **Special tokens**: `[SEG]`, `<p>`, `</p>`, `<vp>`, `</vp>`, `<IMG_CONTEXT>`, `<img>`, `</img>` for mask-text interaction

The project supports both supervised fine-tuning (SFT) and reinforcement learning (RL) training pipelines.

## Directory Structure

```
projects/
├── llava_sam2/           # Main Sa2VA implementation
│   ├── models/           # Core model architectures
│   │   ├── llava_sam2.py      # Main VideoLLaVASAMModel
│   │   ├── internvl.py        # InternVL vision encoder
│   │   ├── sam2.py            # SAM2 integration
│   │   └── sam2_train.py      # SAM2TrainRunner
│   ├── datasets/         # Dataset loaders for training
│   ├── evaluation/       # Evaluation scripts
│   ├── rl_train/         # RL training pipeline
│   │   ├── train_sa2va_rl.py       # Main RL training script
│   │   ├── sa2va_dual_loop_trainer.py  # Dual-loop GRPO trainer
│   │   ├── dataset_gar.py          # GAR dataset loader
│   │   ├── reward_functions.py     # IOU, METEOR, LLM rewards
│   │   └── logits_processor.py     # Numerical stability fixes
│   ├── hf/               # HuggingFace model conversion
│   └── configs/          # Training configurations
├── glamm/                # GLAMM model components (region encoder)
third_parts/
├── sam2/                 # SAM2 implementation
└── mmdet/                # MMDetection losses (Dice, CrossEntropy)
vlm/
├── engine/               # Training runners and loops
└── utils/                # Utility functions
tools/
├── train.py              # Main training entry (uses xtuner)
├── dist.sh               # Distributed training launcher
└── convert_to_hf_new.py  # Checkpoint conversion
```

## Common Commands

### Training (Supervised Fine-tuning)

**Basic training** (single GPU):
```bash
python tools/train.py projects/llava_sam2/configs/sa2va_4b.py
```

**Distributed training** (multi-GPU):
```bash
bash tools/dist.sh train <config_file> <num_gpus>

# Example: Train with 8 GPUs
bash tools/dist.sh train projects/llava_sam2/configs/sa2va_4b.py 8
```

The `dist.sh` script uses `torchrun` for distributed training with DeepSpeed ZeRO-2 optimization.

### RL Training (GRPO-based)

**Dual-loop RL training** with R1-V framework:
```bash
# Basic RL training (loop 1: mask→caption)
python projects/llava_sam2/rl_train/train_sa2va_rl.py \
    --model_path /path/to/sa2va/checkpoint \
    --data_dir /path/to/GAR/dataset \
    --output_dir ./work_dirs/sa2va_rl_training \
    --batch_size 4 \
    --num_generations 4 \
    --learning_rate 1e-5 \
    --num_epochs 1 \
    --use_llm_judge

# Multi-GPU training (8 GPUs)
bash projects/llava_sam2/rl_train/run_rl_8gpu.sh
```

**Key RL parameters**:
- `--use_llm_judge`: Enable LLM judge for loop 1 caption rewards (0.25×METEOR + 0.75×LLM)
- `--use_llm_judge_loop2`: Enable LLM judge for loop 2 (default: METEOR only)
- `--num_generations`: Number of generations per prompt (G in GRPO paper)

### Evaluation

**RefCOCO/RefCOCO+/RefCOCOg evaluation**:
```bash
bash projects/llava_sam2/evaluation/dist_test.sh \
    projects/llava_sam2/evaluation/refcoco_eval.py \
    /path/to/model \
    <num_gpus> \
    --dataset refcoco \
    --split val
```

**Video referring segmentation (ReVOS/MeVIS)**:
```bash
python projects/llava_sam2/evaluation/ref_vos_eval.py /path/to/model
```

**Grounded conversation generation (GCG)**:
```bash
python projects/llava_sam2/evaluation/gcg_eval.py /path/to/model
```

### Model Conversion

**Convert to HuggingFace format**:
```bash
python tools/convert_to_hf_new.py \
    --model_path /path/to/checkpoint.pth \
    --output_dir /path/to/hf_output
```

**Merge HuggingFace safetensors**:
```bash
python scripts/merge_hf_safetensors.py --model_path /path/to/hf/dir
```

## Architecture Details

### Model Components

**VideoLLaVASAMModel** (`projects/llava_sam2/models/llava_sam2.py`):
- Inherits from `LisaModel`
- Three main components:
  1. `mllm`: Multi-modal LLM (InternVL or Qwen or LLaVA)
  2. `grounding_encoder`: SAM2TrainRunner with trainable decoder
  3. `tokenizer`: Special token handling for mask-text interaction

**Training paradigm**:
- **Frozen**: Vision encoder, SAM2 image encoder
- **LoRA-tuned**: LLM layers (r=128, alpha=256)
- **Fully trainable**: MLP projector (`mlp1`), SAM2 decoder, `text_hidden_fcs`

### Loss Functions

Defined in `projects/llava_sam2/configs/sa2va_4b.py`:
- **Mask loss**: CrossEntropyLoss with sigmoid (weight=2.0)
- **Dice loss**: DiceLoss with sigmoid (weight=0.5)
- **Point sampling**: Sample 12,544 points from masks for loss computation

### Data Format

All datasets return samples with:
- `pixel_values`: Image/video frames (tensor)
- `prompt_masks`: Input masks for visual prompting
- `input_ids`: Tokenized text with special tokens
- `labels`: Ground truth masks (for training)
- `modality_length`: Used by LengthGroupedSampler

**Special token encoding**:
- `[SEG]`: Marks segmentation output location
- `<p>`, `</p>`: Bounding box coordinates
- `<vp>`, `</vp>`: Visual prompt markers
- `<IMG_CONTEXT>`: Image context token

### RL Training Pipeline

**Dual-loop training** (`projects/llava_sam2/rl_train/sa2va_dual_loop_trainer.py`):
1. **Loop 1** (mask→caption): Given mask, generate caption. Reward: IOU + METEOR/LLM judge
2. **Loop 2** (caption→mask): Given caption, generate mask + caption'. Reward: METEOR/LLM judge

**Reward functions** (`reward_functions.py`):
- `iou_reward_batch()`: Pixel-level IOU for mask quality
- `combined_caption_reward()`: METEOR (linguistic) or 0.25×METEOR + 0.75×LLM judge
- `LLMJudge`: OpenAI-compatible API for semantic quality

**Numerical stability** (`logits_processor.py`):
- `NumericalStabilityLogitsProcessor`: Clips logits to [-30, 30], prevents NaN/inf
- `TemperatureLogitsWarper`: Safe temperature scaling
- See `projects/llava_sam2/rl_train/NANFIX_README.md` for details

## Configuration System

Uses **MMEngine config system** (Python-based configs):

**Key config sections** (`projects/llava_sam2/configs/sa2va_4b.py`):
1. **Model**: Architecture, LoRA config, loss weights
2. **Dataset**: ConcatDataset with multiple sources (RefCOCO, GCG, Osprey, video datasets)
3. **Scheduler**: CosineAnnealingLR with LinearLR warmup
4. **DeepSpeed**: ZeRO-2 with BF16 (`deepspeed_zero2_sam2.json`)
5. **Runtime**: Checkpointing, logging, distributed settings

**Common parameters to modify**:
- `batch_size`: Per-device batch size (default: 1)
- `accumulative_counts`: Gradient accumulation steps (default: 4)
- `lr`: Learning rate (default: 1e-5)
- `max_epochs`: Training epochs (default: 1)
- `save_steps`: Checkpoint frequency (default: 2000)
- `work_dir`: Output directory for checkpoints

## Dataset Paths

**Configured in** `projects/llava_sam2/configs/sa2va_4b.py`:

```python
DATA_ROOT = './data/'
VIDEO_DATA_ROOT = DATA_ROOT + 'video_datas/'

# Image datasets
RES_ROOT = DATA_ROOT + 'ref_seg/'           # RefCOCO/+/g
LLAVA_ROOT = DATA_ROOT + 'llava_data/'      # LLaVA-Instruct-150K
OSPREY_ROOT = DATA_ROOT + "osprey-724k/"    # Osprey-724K
glamm_data_root = DATA_ROOT + 'glamm_data/' # GCG datasets

# Video datasets
data_root_revos = VIDEO_DATA_ROOT + 'revos/'
data_root_mevis = VIDEO_DATA_ROOT + 'mevis/train/'
data_root_refytvos = VIDEO_DATA_ROOT + 'rvos/'

# RL dataset
GAR_DATA = '/data/xiaoyicheng/Sa2VA/data/GAR'  # Grasp-Any-Region dataset
```

## Development Notes

### Working with Models

**Loading Sa2VA model**:
```python
from projects.llava_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel

model = Sa2VAChatModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
```

**Model has three main attributes**:
- `model.vision_model`: InternVL vision encoder
- `model.language_model`: InternLM2/Phi3 with LoRA
- `model.grounding_encoder`: SAM2 encoder/decoder

### Adding New Datasets

1. Create dataset class in `projects/llava_sam2/datasets/`
2. Inherit from base dataset or implement `__getitem__` returning:
   - `data_dict` with keys: `input_ids`, `pixel_values`, `prompt_masks`, `labels`
3. Register in config: Add to `train_dataset` ConcatDataset
4. Use `video_lisa_collate_fn` for batching

### Monitoring Training

**Important logs to watch**:
- `loss_mask`: Mask prediction loss (should decrease)
- `loss_dice`: Dice loss for overlap (should decrease)
- `grad_norm`: Gradient norm (should be < max_norm=1.0)
- `NaN/inf warnings`: From logits processor (should be rare after warmup)

**Gradient monitoring** (RL training):
```python
# In train_sa2va_rl.py
from projects.llava_sam2.rl_train.training_callbacks import GradientMonitorCallback

callbacks = [
    GradientMonitorCallback(
        check_every_n_steps=1,
        log_every_n_steps=10,
        halt_on_nan=False,  # Set to True for debugging
    ),
]
```

### Debugging Tips

**For NaN/inf in RL training**:
1. Check `projects/llava_sam2/rl_train/NANFIX_README.md`
2. Reduce learning rate: `--learning_rate 5e-6`
3. Tighten gradient clipping: `max_grad_norm=0.5`
4. Enable halt on NaN: `halt_on_nan=True` in callback

**For OOM (out of memory)**:
1. Reduce `batch_size` in config
2. Increase `accumulative_counts` to maintain effective batch size
3. Use gradient checkpointing (if available)
4. Reduce `max_length` (default: 8192)

**For slow training**:
1. Check `dataloader_num_workers` (default: 6)
2. Enable `lazy=True` for datasets
3. Use `LengthGroupedSampler` to batch similar lengths
4. Profile with `torch.profiler`

## Framework Dependencies

- **xtuner**: Training framework (v0.1.23)
- **transformers**: HuggingFace models (v4.42.3)
- **mmdet**: Detection losses (v3.3.0)
- **DeepSpeed**: Distributed training optimization
- **SAM2**: Segment Anything Model 2 (in `third_parts/sam2/`)
- **R1-V**: GRPO framework for RL training (external dependency)

## Template System

Uses **Phi3 chat template** (defined in `xtuner.utils.PROMPT_TEMPLATE`):
```
<|user|>
{instruction}<|end|>
<|assistant|>
{output}<|end|>
```

**For tokenization**:
```python
from xtuner.dataset.map_fns import template_map_fn_factory
from xtuner.utils import PROMPT_TEMPLATE

template_map_fn = template_map_fn_factory(template=PROMPT_TEMPLATE.phi3_chat)
```

## Important Environment Setup

```bash
# Activate environment
export PATH="/home/xiaoyicheng/miniconda3/bin:$PATH"
conda activate vlm
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# For distributed training
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800
```
