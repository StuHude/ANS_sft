"""
Pseudo Token + ST Gumbel-Softmax Training for Sa2VA (Corrected Version)

Based on correct sa2va_4b.py configuration and dataset implementations.

Training Flow:
1. EMA generates pseudo_toks from image1+mask1 (stop-grad)
2. Random mask 25% of pseudo_toks
3. Trainable model outputs logits from image1+mask1+masked_pseudo_toks
4. ST Gumbel-Softmax: logits -> text_embeds (differentiable)
5. Trainable model predicts mask2' from image2+text_embeds
6. Loss = mask_loss + dice_loss, backprop to trainable model

Datasets:
- SAV: image1+mask1 -> image2+mask2 (dual-frame)
- SA1B/OpenImage: image1+mask1 -> image1+mask1 (single-frame, reuse)
- RefCOCO: Direct referring segmentation (GT caption, no gumbel loop)

Trainable: proj, LLM LoRA, SAM2 decoder
Frozen: vision encoder, LLM base
"""

import os
import sys
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import torch.distributed as dist

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from projects.llava_sam2.models.llava_sam2 import VideoLLaVASAMModel
from projects.llava_sam2.models import SAM2TrainRunner
from projects.llava_sam2.models.internvl import InternVL_Slowfast
from projects.llava_sam2.rl_train.ema_model import EMAModel
from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from transformers import AutoTokenizer
from peft import LoraConfig
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint
from xtuner.utils import PROMPT_TEMPLATE
from projects.glamm.datasets.utils.utils import SEG_QUESTIONS, ANSWER_LIST


def parse_args():
    parser = argparse.ArgumentParser(description='Pseudo Token + ST Gumbel-Softmax Training')

    # Model paths
    parser.add_argument('--pretrained_pth', type=str, required=True, help='Pretrained Sa2VA checkpoint')
    parser.add_argument('--model_path', type=str, default='./pretrained/InternVL2_5-4B')
    parser.add_argument('--output_dir', type=str, default='./work_dirs/pseudo_gumbel_v2')

    # Dataset paths
    parser.add_argument('--sav_dir', type=str, default=None, help='SAV npz directory')
    parser.add_argument('--sa1b_dir', type=str, default=None, help='SA-1B dataset directory')
    parser.add_argument('--openimage_dir', type=str, default=None, help='OpenImage dataset directory')
    parser.add_argument('--refcoco_dir', type=str, default=None, help='RefCOCO dataset directory')

    # Dataset sampling
    parser.add_argument('--sav_max_samples', type=int, default=None)
    parser.add_argument('--sa1b_max_samples', type=int, default=None)
    parser.add_argument('--openimage_max_samples', type=int, default=None)
    parser.add_argument('--sav_repeats', type=int, default=1)
    parser.add_argument('--refcoco_repeats', type=int, default=1)

    # Training params
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument(
        '--ref_llm_loss_weight',
        type=float,
        default=1.0,
        help='Weight for RefCOCO language modeling loss (keeps `[SEG]` generation behavior).')
    parser.add_argument(
        '--pseudo_llm_loss_weight',
        type=float,
        default=1.0,
        help='Weight for pseudo-loop LM loss on the fixed segmentation answer (keeps `[SEG]` generation behavior).')
    parser.add_argument(
        '--masked_lm_loss_weight',
        type=float,
        default=0.0,
        help='Optional denoising LM loss weight for Step3: predict original EMA pseudo tokens on masked positions. '
             'This anchors LLM/LoRA to language behavior and helps preserve `[SEG]` generation during inference.')

    # Gumbel-Softmax params
    parser.add_argument('--gumbel_tau', type=float, default=0.7, help='Temperature for Gumbel-Softmax')
    parser.add_argument('--topk', type=int, default=128, help='Top-k for sparse Gumbel-Softmax')
    parser.add_argument('--mask_ratio', type=float, default=0.25, help='Ratio of tokens to mask')
    parser.add_argument('--max_caption_len', type=int, default=64, help='Fixed caption length')

    # LoRA params (same as sa2va_4b.py)
    parser.add_argument('--lora_r', type=int, default=128)
    parser.add_argument('--lora_alpha', type=int, default=256)

    # Other
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--ref_num_workers', type=int, default=0,
                        help='RefCOCO dataloader workers (keep low to reduce shm/CPU mem).')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument(
        '--save_full_model',
        action='store_true',
        help='Save a full-model checkpoint (same format/keyspace as `--pretrained_pth`, i.e. `{meta, state_dict}`).')
    parser.add_argument(
        '--export_embed_lm_head',
        action='store_true',
        help='When `--save_full_model` is set, also export `embed_tokens` / `lm_head` from the current model. '
             'Default is OFF to avoid accidentally overwriting the base checkpoint vocabulary expansion / special-token '
             'rows with newly-initialized weights.')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Stop after N optimizer steps (debug/smoke runs).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_rank', type=int, default=-1)

    return parser.parse_args()


def build_model(args):
    """Build VideoLLaVASAMModel following sa2va_4b.py configuration."""
    print(f"Building VideoLLaVASAMModel from {args.model_path}...")

    # Special tokens (from sa2va_4b.py)
    special_tokens = [
        '[SEG]', '<p>', '</p>',
        '<vp>', '</vp>',
        '<IMG_CONTEXT>',
        '<img>', '</img>'
    ]

    # Tokenizer config (model will build its own tokenizer internally)
    tokenizer_cfg = dict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=args.model_path,
        trust_remote_code=True,
        padding_side='right',
    )

    # Model config (following sa2va_4b.py)
    model_cfg = dict(
        type=VideoLLaVASAMModel,
        special_tokens=special_tokens,
        frozen_sam2_decoder=False,  # SAM2 decoder trainable
        mllm=dict(
            type=InternVL_Slowfast,
            model_path=args.model_path,
            freeze_llm=True,  # LLM base frozen
            freeze_visual_encoder=True,  # Vision encoder frozen
            llm_lora=dict(
                type=LoraConfig,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.05,
                bias='none',
                task_type='CAUSAL_LM'
            ),
            special_tokens=special_tokens,
        ),
        tokenizer=tokenizer_cfg,
        grounding_encoder=dict(type=SAM2TrainRunner),
        loss_mask=dict(
            type=CrossEntropyLoss,
            use_sigmoid=True,
            reduction='mean',
            loss_weight=2.0
        ),
        loss_dice=dict(
            type=DiceLoss,
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=0.5
        ),
        pretrained_pth=None,
        loss_sample_points=True,
        bs=args.batch_size,
    )

    model = BUILDER.build(model_cfg)
    print("✓ Model built")

    # Load pretrained weights (prefer safetensors shard streaming to avoid per-rank OOM).
    if args.pretrained_pth and os.path.exists(args.pretrained_pth):
        index_path = os.path.join(args.pretrained_pth, 'model.safetensors.index.json')
        if os.path.exists(index_path):
            print(f"Loading from HuggingFace safetensors (sharded): {args.pretrained_pth}")
            try:
                import json
                from safetensors.torch import load_file

                with open(index_path, "r") as f:
                    index = json.load(f)

                shard_files = sorted(set(index["weight_map"].values()))

                # Reduce peak memory/I/O: load shards sequentially across ranks if distributed.
                if dist.is_available() and dist.is_initialized():
                    world_size = dist.get_world_size()
                    rank = dist.get_rank()
                    for r in range(world_size):
                        if rank == r:
                            for shard_file in shard_files:
                                shard_path = os.path.join(args.pretrained_pth, shard_file)
                                state = load_file(shard_path, device="cpu")
                                model.load_state_dict(state, strict=False)
                                del state
                        dist.barrier()
                else:
                    for shard_file in shard_files:
                        shard_path = os.path.join(args.pretrained_pth, shard_file)
                        state = load_file(shard_path, device="cpu")
                        model.load_state_dict(state, strict=False)
                        del state

                print("✓ Pretrained weights loaded from safetensors shards")
            except Exception as e:
                print(f"Warning: Failed to load safetensors shards: {e}")
        else:
            try:
                print(f"Loading from checkpoint: {args.pretrained_pth}")
                checkpoint = guess_load_checkpoint(args.pretrained_pth)
                model.load_state_dict(checkpoint, strict=False)
                print("✓ Pretrained weights loaded from checkpoint")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint: {e}")

    tokenizer = model.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def freeze_for_pseudo_gumbel_training(model):
    """
    Freeze everything except:
    - Sa2VA projector: InternVL `mlp1` + Sa2VA `text_hidden_fcs`
    - LLM LoRA params
    - SAM2 mask decoder
    """
    for _, p in model.named_parameters():
        p.requires_grad = False

    trainable_patterns = (
        'mlp1',
        'text_hidden_fcs',
        'lora',
        'sam_mask_decoder',
    )

    for name, p in model.named_parameters():
        if any(pattern in name.lower() for pattern in trainable_patterns):
            p.requires_grad = True


def print_trainable_parameters(model, name="Model"):
    """Print trainable parameters to verify correct freezing."""
    trainable_params = 0
    all_params = 0
    trainable_names = []

    for n, p in model.named_parameters():
        all_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
            trainable_names.append(n)

    print(f"\n{'='*60}")
    print(f"{name} Trainable Parameters:")
    print(f"{'='*60}")
    print(f"Total params: {all_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / all_params:.2f}%")
    print(f"\nTrainable parameter groups:")

    # Group by module
    groups = {}
    for n in trainable_names:
        if 'lora' in n.lower():
            group = 'LoRA'
        elif 'text_hidden_fcs' in n:
            group = 'text_hidden_fcs (proj)'
        elif 'sam_mask_decoder' in n or 'grounding_encoder' in n:
            group = 'SAM2 decoder'
        elif 'mlp1' in n:
            group = 'mlp1 (proj)'
        else:
            parts = n.split('.')
            group = parts[0] if len(parts) > 0 else 'other'

        if group not in groups:
            groups[group] = []
        groups[group].append(n)

    for g, params in groups.items():
        print(f"  {g}: {len(params)} parameters")

    # Spot-check unexpected trainables
    allowed_substrings = ('mlp1', 'text_hidden_fcs', 'lora', 'sam_mask_decoder')
    unexpected = [n for n in trainable_names if not any(s in n.lower() for s in allowed_substrings)]
    if unexpected:
        print("\n⚠ Unexpected trainable params (should be frozen):")
        for n in unexpected[:50]:
            print(f"  - {n}")
        if len(unexpected) > 50:
            print(f"  ... and {len(unexpected) - 50} more")
    print(f"{'='*60}\n")


def build_datasets(args):
    """Build (pseudo_dataset, refcoco_dataset) using dataset_builder."""
    from projects.llava_sam2.mask_caption_sft.dataset_builder import build_mask_caption_dataset

    # OpenImage config
    openimage_config = None
    if args.openimage_dir and os.path.exists(args.openimage_dir):
        openimage_config = {
            'annotation_csv': os.path.join(args.openimage_dir, 'train-annotations-object-segmentation.csv'),
            'label_csv': os.path.join(args.openimage_dir, 'oidv7-class-descriptions.csv'),
            'image_dir': os.path.join(args.openimage_dir, 'images', 'train'),
            'mask_dir': os.path.join(args.openimage_dir, 'masks', 'train'),
        }
        print(f"✓ OpenImage: {args.openimage_dir}")

    # RefCOCO config
    refcoco_config = None
    if args.refcoco_dir:
        refcoco_config = {
            'data_root': args.refcoco_dir,
            'split': 'train',
            'dataset_name': 'refcoco',
        }
        print(f"✓ RefCOCO: {args.refcoco_dir}")

    pseudo_dataset = build_mask_caption_dataset(
        sav_dir=args.sav_dir,
        sa1b_dir=args.sa1b_dir,
        openimage_config=openimage_config,
        refcoco_config=None,
        target_size=(1024, 1024),
        sa1b_max_samples=args.sa1b_max_samples,
        sav_max_samples=args.sav_max_samples,
        openimage_max_samples=args.openimage_max_samples,
        sav_repeats=args.sav_repeats,
        refcoco_repeats=0,
    )

    refcoco_dataset = None
    if refcoco_config is not None:
        refcoco_dataset = build_mask_caption_dataset(
            sav_dir=None,
            sa1b_dir=None,
            openimage_config=None,
            refcoco_config=refcoco_config,
            target_size=(1024, 1024),
            sa1b_max_samples=None,
            sav_max_samples=None,
            openimage_max_samples=None,
            sav_repeats=0,
            refcoco_repeats=args.refcoco_repeats,
        )

    print(f"✓ Pseudo dataset: {len(pseudo_dataset)} samples")
    if refcoco_dataset is not None:
        print(f"✓ RefCOCO dataset: {len(refcoco_dataset)} samples")
    return pseudo_dataset, refcoco_dataset


class PseudoGumbelTrainerV2:
    """
    Corrected Pseudo Token + Gumbel-Softmax Trainer.

    Follows Sa2VA's standard data format and model interfaces.
    """

    def __init__(self, model, ema_model, tokenizer, train_dataloader_pseudo, train_dataloader_refcoco,
                 optimizer, device, output_dir, args):
        self.model = model
        self.ema_model = ema_model
        self.tokenizer = tokenizer
        self.train_dataloader_pseudo = train_dataloader_pseudo
        self.train_dataloader_refcoco = train_dataloader_refcoco
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.args = args

        self.global_step = 0
        self.gumbel_tau = args.gumbel_tau
        self.topk = args.topk
        self.mask_ratio = args.mask_ratio
        self.max_caption_len = args.max_caption_len
        self.ref_llm_loss_weight = float(getattr(args, 'ref_llm_loss_weight', 1.0))
        self.pseudo_llm_loss_weight = float(getattr(args, 'pseudo_llm_loss_weight', 1.0))
        self.masked_lm_loss_weight = float(getattr(args, 'masked_lm_loss_weight', 0.0))

        # Get actual model (handle DDP)
        self.actual_model = model.module if hasattr(model, 'module') else model
        self.ema_actual = ema_model.model

        # Get special token IDs
        self.seg_token_id = self.actual_model.seg_token_idx
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.img_context_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')

        # Embedding layer for Gumbel-Softmax
        self.embedding_layer = self.actual_model.mllm.model.language_model.get_input_embeddings()
        self.vocab_size = self.actual_model.mllm.model.language_model.config.vocab_size
        self.hidden_size = self.embedding_layer.embedding_dim

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))

        # Template (phi3_chat from sa2va_4b.py)
        self.prompt_template = PROMPT_TEMPLATE.phi3_chat
        self.reached_max_steps = False
        self._last_teacher_token_stat_step = None

    def _log_teacher_token_stats(self, pseudo_toks: torch.Tensor):
        """
        Log simple statistics of EMA-generated pseudo tokens.

        We do NOT change training behavior here; this is only monitoring to understand how often the
        generated fixed-length sequence contains an EOS and how early it appears.

        Notes:
        - `pseudo_toks` is shape [B, T] where T == max_caption_len.
        - EMA generation can stop early and be right-padded; however in this pipeline padding is not
          guaranteed to be a distinct PAD token (some tokenizers use EOS as PAD).
        """
        if pseudo_toks is None or pseudo_toks.numel() == 0:
            return
        if not torch.is_tensor(pseudo_toks) or pseudo_toks.dim() != 2:
            return
        if self._last_teacher_token_stat_step == self.global_step:
            return
        if self.global_step % self.args.log_interval != 0:
            return

        self._last_teacher_token_stat_step = self.global_step

        eos_id = int(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id is not None else None
        pad_id = int(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id is not None else None
        T = int(pseudo_toks.shape[1])

        def first_pos(ids: torch.Tensor, token_id: int) -> torch.Tensor:
            mask = (ids == token_id)
            has = mask.any(dim=1)
            pos = mask.to(torch.int64).argmax(dim=1)
            pos = torch.where(has, pos, torch.full_like(pos, -1))
            return pos

        # EOS stats
        eos_found = torch.tensor(0, device=self.device, dtype=torch.long)
        eos_pos_sum = torch.tensor(0, device=self.device, dtype=torch.long)
        eos_pos_min = torch.tensor(T, device=self.device, dtype=torch.long)
        eos_pos_max = torch.tensor(-1, device=self.device, dtype=torch.long)

        if eos_id is not None:
            eos_pos = first_pos(pseudo_toks, eos_id)  # [-1 or 0..T-1]
            has_eos = eos_pos >= 0
            eos_found = has_eos.sum()
            if int(eos_found.item()) > 0:
                eos_pos_valid = eos_pos[has_eos]
                eos_pos_sum = eos_pos_valid.sum()
                eos_pos_min = eos_pos_valid.min()
                eos_pos_max = eos_pos_valid.max()

        # PAD stats (only meaningful if PAD differs from EOS)
        pad_found = torch.tensor(0, device=self.device, dtype=torch.long)
        pad_pos_sum = torch.tensor(0, device=self.device, dtype=torch.long)
        if pad_id is not None and (eos_id is None or pad_id != eos_id):
            pad_pos = first_pos(pseudo_toks, pad_id)
            has_pad = pad_pos >= 0
            pad_found = has_pad.sum()
            if int(pad_found.item()) > 0:
                pad_pos_sum = pad_pos[has_pad].sum()

        # Non-pad content ratio (helps catch "all pad" catastrophic bugs)
        non_pad_frac = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if pad_id is not None:
            non_pad_frac = (pseudo_toks != pad_id).to(torch.float32).mean()

        B = torch.tensor(int(pseudo_toks.shape[0]), device=self.device, dtype=torch.long)

        if dist.is_available() and dist.is_initialized():
            packed_int = torch.stack(
                [B, eos_found, eos_pos_sum, eos_pos_min, eos_pos_max, pad_found, pad_pos_sum],
                dim=0,
            )
            dist.all_reduce(packed_int, op=dist.ReduceOp.SUM)
            B, eos_found, eos_pos_sum, eos_pos_min, eos_pos_max, pad_found, pad_pos_sum = packed_int.tolist()

            non_pad_sum = non_pad_frac * float(int(pseudo_toks.shape[0]))
            dist.all_reduce(non_pad_sum, op=dist.ReduceOp.SUM)
            non_pad_frac = non_pad_sum / float(max(B, 1))
        else:
            B = int(B.item())
            eos_found = int(eos_found.item())
            eos_pos_sum = int(eos_pos_sum.item())
            eos_pos_min = int(eos_pos_min.item())
            eos_pos_max = int(eos_pos_max.item())
            pad_found = int(pad_found.item())
            pad_pos_sum = int(pad_pos_sum.item())
            non_pad_frac = float(non_pad_frac.item())

        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        if B <= 0:
            return

        if eos_id is not None:
            eos_found_frac = float(eos_found) / float(B)
            eos_pos_mean = float(eos_pos_sum) / float(max(eos_found, 1))
            self.writer.add_scalar('teacher/eos_found_frac', eos_found_frac, self.global_step)
            self.writer.add_scalar('teacher/eos_pos_mean', eos_pos_mean, self.global_step)
            if eos_found > 0:
                self.writer.add_scalar('teacher/eos_pos_min', float(eos_pos_min), self.global_step)
                self.writer.add_scalar('teacher/eos_pos_max', float(eos_pos_max), self.global_step)

        if pad_id is not None and (eos_id is None or pad_id != eos_id):
            pad_found_frac = float(pad_found) / float(B)
            pad_pos_mean = float(pad_pos_sum) / float(max(pad_found, 1))
            self.writer.add_scalar('teacher/pad_found_frac', pad_found_frac, self.global_step)
            self.writer.add_scalar('teacher/pad_pos_mean', pad_pos_mean, self.global_step)

        self.writer.add_scalar('teacher/non_pad_frac', float(non_pad_frac), self.global_step)

        print(f"✓ Trainer initialized")
        print(f"  [SEG] token ID: {self.seg_token_id}")
        print(f"  Gumbel tau: {self.gumbel_tau}, Top-k: {self.topk}")
        print(f"  Mask ratio: {self.mask_ratio}, Max caption len: {self.max_caption_len}")

    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        self.ema_model.eval()

        # Ensure DistributedSampler shuffles deterministically per-epoch.
        for dl in (self.train_dataloader_pseudo, self.train_dataloader_refcoco):
            if dl is None:
                continue
            sampler = getattr(dl, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

        epoch_losses = []
        if self.train_dataloader_refcoco is None:
            raise ValueError("RefCOCO dataloader is required")

        pseudo_steps = len(self.train_dataloader_pseudo)
        ref_steps = len(self.train_dataloader_refcoco)

        # Sample RefCOCO as part of the overall data mixture.
        #
        # IMPORTANT: A pure shuffle can result in the early part of training seeing *no* RefCOCO
        # batches for a long time (especially when `ref_steps << pseudo_steps`), which can cause
        # the model to drift away from the `[SEG]` generation behavior expected at inference.
        #
        # We therefore build a deterministic, approximately proportional interleaving schedule
        # and ensure at least one RefCOCO batch appears early in the epoch (if available).
        schedule = []
        warmup_ref = 1 if ref_steps > 0 else 0
        if warmup_ref:
            schedule.append('refcoco')
        remaining_ref = max(ref_steps - warmup_ref, 0)
        if remaining_ref > 0:
            # Insert remaining RefCOCO batches roughly uniformly over the pseudo stream.
            insert_pos = set(
                int((j + 1) * pseudo_steps / (remaining_ref + 1))
                for j in range(remaining_ref)
            )
        else:
            insert_pos = set()

        inserted = 0
        for i in range(pseudo_steps):
            if i in insert_pos:
                schedule.append('refcoco')
                inserted += 1
            schedule.append('pseudo')
        # In rare cases due to rounding, we might insert fewer/more; fix up deterministically.
        while inserted < remaining_ref:
            schedule.append('refcoco')
            inserted += 1
        while inserted > remaining_ref:
            schedule.remove('refcoco')
            inserted -= 1

        pseudo_iter = iter(self.train_dataloader_pseudo)
        ref_iter = iter(self.train_dataloader_refcoco)

        pbar = tqdm(total=len(schedule), desc=f"Epoch {epoch}")
        dataset_type_counts = {'sav': 0, 'sa1b': 0, 'openimage': 0, 'other': 0, 'refcoco': 0}

        for step, which in enumerate(schedule):
            if self.args.max_steps is not None and self.global_step >= self.args.max_steps:
                self.reached_max_steps = True
                break

            if which == 'pseudo':
                try:
                    batch = next(pseudo_iter)
                except StopIteration:
                    pseudo_iter = iter(self.train_dataloader_pseudo)
                    batch = next(pseudo_iter)
                if batch is None:
                    raise RuntimeError("Pseudo batch is None (unexpected; dataset wrapper should refetch).")

                pixel_values1 = batch['pixel_values1'].to(self.device, dtype=torch.bfloat16)
                prompt_masks1 = batch['prompt_masks1'].to(self.device)
                pixel_values2 = batch['pixel_values2'].to(self.device, dtype=torch.bfloat16)
                g_pixel_values2 = batch['g_pixel_values2'].to(self.device)
                gt_masks2 = batch['masks2'].to(self.device, dtype=torch.float32)

                dataset_types = batch.get('dataset_types', None) or (['unknown'] * pixel_values1.shape[0])
                for dt in dataset_types:
                    if dt in dataset_type_counts:
                        dataset_type_counts[dt] += 1
                    else:
                        dataset_type_counts['other'] += 1

                try:
                    loss_dict = self.pseudo_gumbel_step(
                        pixel_values1, prompt_masks1, pixel_values2, g_pixel_values2, gt_masks2
                    )
                except Exception:
                    import traceback
                    if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
                        print("❌ Pseudo step crashed with exception:", flush=True)
                        traceback.print_exc()
                    raise

            elif which == 'refcoco':
                try:
                    batch = next(ref_iter)
                except StopIteration:
                    ref_iter = iter(self.train_dataloader_refcoco)
                    batch = next(ref_iter)
                if batch is None:
                    raise RuntimeError("RefCOCO batch is None (unexpected; dataset wrapper should refetch).")

                pixel_values = batch['pixel_values2'].to(self.device, dtype=torch.bfloat16)
                g_pixel_values = batch['g_pixel_values2'].to(self.device)
                gt_masks = batch['masks2'].to(self.device, dtype=torch.float32)
                captions = batch.get('captions', None)
                if captions is None:
                    raise RuntimeError("RefCOCO batch missing captions")

                dataset_type_counts['refcoco'] += pixel_values.shape[0]
                try:
                    loss_dict = self.refcoco_step(pixel_values, g_pixel_values, captions, gt_masks)
                except Exception:
                    import traceback
                    if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
                        print("❌ RefCOCO step crashed with exception:", flush=True)
                        traceback.print_exc()
                    raise

            else:
                raise ValueError(f"Unknown schedule entry: {which}")

            loss = loss_dict['loss']

            try:
                # Backward
                loss_scaled = loss / self.args.gradient_accumulation_steps
                loss_scaled.backward()

                # Sanity checks (每log_interval步检查一次)
                if self.global_step % self.args.log_interval == 0:
                    self.check_gradients()

                # Update weights
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Update EMA
                    student = self.model.module if hasattr(self.model, 'module') else self.model
                    self.ema_model.update(student)

                    self.global_step += 1
                    if self.args.max_steps is not None and self.global_step >= self.args.max_steps:
                        # Save once at the exact max step, then stop training.
                        self.save_checkpoint(f"checkpoint_step_{self.global_step}.pth")
                        self.reached_max_steps = True
                        break
            except Exception:
                import traceback
                if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
                    print("❌ Backward/step crashed with exception:", flush=True)
                    traceback.print_exc()
                raise

            # Logging
            epoch_losses.append(loss.item())
            if self.global_step % self.args.log_interval == 0 and self.global_step > 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/mask_loss', loss_dict.get('mask_loss', 0), self.global_step)
                self.writer.add_scalar('train/dice_loss', loss_dict.get('dice_loss', 0), self.global_step)
                if 'llm_loss' in loss_dict:
                    self.writer.add_scalar('train/llm_loss', loss_dict.get('llm_loss', 0), self.global_step)

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'mask': f"{loss_dict.get('mask_loss', 0):.4f}",
                    'dice': f"{loss_dict.get('dice_loss', 0):.4f}",
                    'llm': f"{loss_dict.get('llm_loss', 0):.4f}",
                })
            pbar.update(1)

            # Save checkpoint
            if self.global_step > 0 and self.global_step % self.args.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pth")

            if self.reached_max_steps:
                break

        # Aggregate dataset type counts across ranks for logging/debugging.
        if dist.is_available() and dist.is_initialized():
            keys = list(dataset_type_counts.keys())
            local = torch.tensor([dataset_type_counts[k] for k in keys], device=self.device, dtype=torch.long)
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
            if dist.get_rank() == 0:
                merged = {k: int(v) for k, v in zip(keys, local.tolist())}
                print(f"Dataset mix (all ranks): {merged}")
        else:
            print(f"Dataset mix: {dataset_type_counts}")

        return {'loss': sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0}

    def pseudo_gumbel_step(
        self,
        pixel_values1,
        prompt_masks1,
        pixel_values2,
        g_pixel_values2,
        gt_masks2,
    ):
        """
        Full pseudo-token + Gumbel-Softmax training step.

        Implements the complete training loop:
        1. EMA generates pseudo_toks (stop-grad)
        2. Random mask 25%
        3. Trainable outputs logits
        4. Gumbel-Softmax -> text_embeds (differentiable)
        5. Trainable predicts mask from text_embeds
        6. Compute loss and backprop
        """
        from projects.llava_sam2.mask_caption_sft.pseudo_gumbel_core import (
            generate_pseudo_tokens_with_ema,
            random_mask_tokens,
            forward_for_logits,
            topk_gumbel_softmax,
            forward_mask_with_text_embeds,
        )

        # Step 1: EMA generates pseudo tokens (stop-grad)
        pseudo_toks = generate_pseudo_tokens_with_ema(
            ema_model=self.ema_actual,
            pixel_values=pixel_values1,
            prompt_masks=prompt_masks1,
            tokenizer=self.tokenizer,
            max_caption_len=self.max_caption_len,
            device=self.device
        )
        self._log_teacher_token_stats(pseudo_toks)

        # Step 2: Random mask
        pseudo_toks_masked = random_mask_tokens(
            tokens=pseudo_toks,
            mask_ratio=self.mask_ratio,
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            device=self.device,
            forbidden_token_ids=[self.img_context_id],
        )

        # Step 3: Trainable model outputs logits
        logits = forward_for_logits(
            model=self.actual_model,
            pixel_values=pixel_values1,
            prompt_masks=prompt_masks1,
            masked_pseudo_toks=pseudo_toks_masked,
            tokenizer=self.tokenizer,
            max_caption_len=self.max_caption_len,
            device=self.device
        )

        # Optional anchor: denoising LM loss on masked positions (predict original `pseudo_toks`).
        masked_lm_loss_t = torch.tensor(0.0, device=self.device)
        if self.masked_lm_loss_weight > 0:
            mask_pos = (pseudo_toks_masked != pseudo_toks) & (pseudo_toks != self.pad_token_id)
            if mask_pos.any():
                logits_masked = logits[mask_pos].float()
                targets = pseudo_toks[mask_pos]
                masked_lm_loss_t = F.cross_entropy(logits_masked, targets, reduction='mean')

        # Step 4: Top-k Gumbel-Softmax -> text_embeds (differentiable!)
        text_embeds = topk_gumbel_softmax(
            logits=logits,
            tau=self.gumbel_tau,
            topk=self.topk,
            embedding_layer=self.embedding_layer
        )

        # Register gradient hook for sanity check
        text_embeds.retain_grad()
        self.text_embeds_ref = text_embeds

        # Step 5: Trainable predicts mask from text_embeds (+ LM loss on fixed answer to preserve `[SEG]` generation)
        loss_dict = forward_mask_with_text_embeds(
            model=self.actual_model,
            pixel_values=pixel_values2,
            g_pixel_values=g_pixel_values2,
            text_embeds=text_embeds,
            gt_masks=gt_masks2,
            seg_token_id=self.seg_token_id,
            img_context_id=self.img_context_id,
            tokenizer=self.tokenizer,
            device=self.device
        )

        # Combine losses: segmentation loss (bce+dice) + LM loss on the fixed assistant output.
        llm_loss_t = loss_dict.get('llm_loss_t', None)
        if llm_loss_t is None:
            llm_loss = loss_dict.get('llm_loss', 0.0)
            llm_loss_t = llm_loss if torch.is_tensor(llm_loss) else torch.tensor(float(llm_loss), device=self.device)
        total = loss_dict['loss'] + (llm_loss_t * self.pseudo_llm_loss_weight)
        if self.masked_lm_loss_weight > 0:
            total = total + (masked_lm_loss_t * self.masked_lm_loss_weight)
            loss_dict['masked_lm_loss'] = masked_lm_loss_t.item()
        loss_dict['loss'] = total
        return loss_dict

    def refcoco_step(self, pixel_values, g_pixel_values, captions, gt_masks):
        """
        RefCOCO: Direct referring segmentation with GT caption.

        Uses standard tokenization (not Gumbel-Softmax).
        """
        batch_size = pixel_values.shape[0]

        IMG_CONTEXT = '<IMG_CONTEXT>'
        NUM_IMG_TOKENS = 256

        input_ids_list = []
        labels_list = []

        question_template = random.choice(SEG_QUESTIONS)
        answer_template = random.choice(ANSWER_LIST)
        if answer_template.count("[SEG]") != 1:
            raise ValueError(f"Unexpected ANSWER_LIST template: {answer_template}")
        a_prefix, a_suffix = answer_template.split("[SEG]")

        for i, caption in enumerate(captions):
            # Match ReferSegmDataset:
            # - Human text begins with "<image>\n"
            # - Then replaced with "<img>{IMG_CONTEXT * num_image_tokens}</img>"
            img_str = f'<img>{IMG_CONTEXT * NUM_IMG_TOKENS}</img>'
            question = question_template.format(class_name=str(caption))
            human_text = f"<image>\n{question}"
            human_text = human_text.replace("<image>", img_str)
            # Match ReferSegmDataset override:
            #   "<|user|>\n{input}<|end|><|assistant|>\n"
            user_text = f"<|user|>\n{human_text}<|end|><|assistant|>\n"
            output_ids = (
                self.tokenizer.encode(a_prefix, add_special_tokens=False)
                + [self.seg_token_id]
                + self.tokenizer.encode(a_suffix, add_special_tokens=False)
                + self.tokenizer.encode("<|end|>", add_special_tokens=False)
            )

            user_ids = self.tokenizer.encode(user_text, add_special_tokens=True)

            full_ids = user_ids + output_ids
            full_labels = [-100] * len(user_ids) + output_ids

            input_ids_list.append(full_ids)
            labels_list.append(full_labels)

        # Pad
        max_len = max(len(ids) for ids in input_ids_list)
        if dist.is_available() and dist.is_initialized():
            t = torch.tensor([int(max_len)], device=self.device, dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            max_len = int(t.item())
        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)

        for i, (ids, labs) in enumerate(zip(input_ids_list, labels_list)):
            input_ids[i, :len(ids)] = torch.tensor(ids, device=self.device)
            labels[i, :len(labs)] = torch.tensor(labs, device=self.device)
            attention_mask[i, :len(ids)] = True

        position_ids = torch.arange(max_len, device=self.device).unsqueeze(0).expand(batch_size, -1)

        # Prepare data
        # IMPORTANT: we must include LM loss here, otherwise LoRA/embeddings can drift and the model may
        # stop *generating* `[SEG]` during inference (even though it can still segment under teacher forcing).
        pixel_values_list = [pixel_values[i] for i in range(batch_size)]
        g_pixel_values_list = [g_pixel_values[i] for i in range(batch_size)]
        masks_list = [gt_masks[i:i+1] for i in range(batch_size)]

        data = {
            'pixel_values': pixel_values_list,
            'g_pixel_values': g_pixel_values_list,
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'masks': masks_list,
            'frames_per_batch': [1] * batch_size,
        }

        try:
            loss_dict = self.actual_model(data, data_samples=None, mode='loss')

            loss_mask = loss_dict.get('loss_mask', 0)
            loss_dice = loss_dict.get('loss_dice', 0)
            llm_loss = loss_dict.get('llm_loss', 0)
            if llm_loss is None:
                llm_loss = 0

            total_loss = loss_mask + loss_dice + (llm_loss * self.ref_llm_loss_weight)

            return {
                'loss': total_loss,
                'mask_loss': loss_mask.item() if torch.is_tensor(loss_mask) else float(loss_mask),
                'dice_loss': loss_dice.item() if torch.is_tensor(loss_dice) else float(loss_dice),
                'llm_loss': llm_loss.item() if torch.is_tensor(llm_loss) else float(llm_loss),
            }
        except Exception as e:
            print(f"⚠ RefCOCO step failed: {e}, returning zero loss")
            return {
                'loss': torch.tensor(0.0, device=self.device, requires_grad=True),
                'mask_loss': 0.0,
                'dice_loss': 0.0,
                'llm_loss': 0.0,
            }

    def check_gradients(self):
        """Sanity check: verify gradients flow correctly."""
        # Check 1: text_embeds gradient
        if hasattr(self, 'text_embeds_ref') and self.text_embeds_ref is not None:
            if self.text_embeds_ref.grad is not None:
                grad_norm = self.text_embeds_ref.grad.norm().item()
                self.writer.add_scalar('grad/text_embeds_norm', grad_norm, self.global_step)
                if grad_norm > 0:
                    print(f"✓ Step {self.global_step}: text_embeds grad norm = {grad_norm:.6f}")
                else:
                    print(f"⚠ Step {self.global_step}: text_embeds grad is ZERO!")
            else:
                print(f"⚠ Step {self.global_step}: text_embeds.grad is None!")

        # Check 2: LoRA gradients
        lora_grad_norm = 0
        lora_count = 0
        for n, p in self.actual_model.named_parameters():
            if 'lora' in n.lower() and p.grad is not None:
                lora_grad_norm += p.grad.norm().item()
                lora_count += 1

        if lora_count > 0:
            avg_lora_grad = lora_grad_norm / lora_count
            self.writer.add_scalar('grad/lora_avg_norm', avg_lora_grad, self.global_step)
            if avg_lora_grad == 0:
                print(f"⚠ Step {self.global_step}: LoRA grad is ZERO!")

        # Check 3: SAM2 decoder gradients
        sam_grad_norm = 0
        sam_count = 0
        for n, p in self.actual_model.grounding_encoder.named_parameters():
            if 'sam_mask_decoder' in n and p.grad is not None:
                sam_grad_norm += p.grad.norm().item()
                sam_count += 1

        if sam_count > 0:
            avg_sam_grad = sam_grad_norm / sam_count
            self.writer.add_scalar('grad/sam2_decoder_avg_norm', avg_sam_grad, self.global_step)

        # Check 4: Projector gradients
        proj_grad_norm = 0
        proj_count = 0
        for n, p in self.actual_model.named_parameters():
            if ('mlp1' in n or 'text_hidden_fcs' in n) and p.grad is not None:
                proj_grad_norm += p.grad.norm().item()
                proj_count += 1

        if proj_count > 0:
            avg_proj_grad = proj_grad_norm / proj_count
            self.writer.add_scalar('grad/proj_avg_norm', avg_proj_grad, self.global_step)

    def save_checkpoint(self, filename):
        """
        Save checkpoint (weights only).

        - Default: model.state_dict() (adapter-centric; LoRA + a few trained heads).
        - If `--save_full_model`: export a full-model checkpoint matching the original
          `--pretrained_pth` format and keyspace: `{meta, state_dict}`.
        """
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            if rank != 0:
                return

        model_obj = self.model.module if hasattr(self.model, 'module') else self.model

        def _unwrap_pretrained(pth: str):
            obj = torch.load(pth, map_location='cpu')
            if isinstance(obj, dict) and 'state_dict' in obj and isinstance(obj['state_dict'], dict):
                return obj.get('meta', {}), obj['state_dict']
            if isinstance(obj, dict):
                return {}, obj
            raise TypeError(f'Unsupported checkpoint type: {type(obj)}')

        def _export_full_state_dict(base_sd: dict, delta_sd: dict, *, lora_r: int, lora_alpha: float) -> dict:
            # Start from base weights, then apply trainable deltas.
            out = {k: v.clone() for k, v in base_sd.items()}

            # 1) Merge LoRA into base LLM weights.
            scale = float(lora_alpha) / float(lora_r)
            lora_A = {}
            lora_B = {}
            for k, v in delta_sd.items():
                if k.endswith('.lora_A.default.weight'):
                    lora_A[k[:-len('.lora_A.default.weight')]] = v
                elif k.endswith('.lora_B.default.weight'):
                    lora_B[k[:-len('.lora_B.default.weight')]] = v

            common = set(lora_A.keys()) & set(lora_B.keys())
            for base in common:
                # Example:
                #   base = mllm.model.language_model.base_model.model.model.layers.0.self_attn.q_proj
                # ->  W key = mllm.model.language_model.model.layers.0.self_attn.q_proj.weight
                w_key = base
                w_key = w_key.replace('mllm.model.language_model.base_model.model.model.', 'mllm.model.language_model.model.')
                w_key = w_key + '.weight'
                if w_key not in out:
                    continue
                W = out[w_key]
                A = lora_A[base].to(dtype=W.dtype)
                B = lora_B[base].to(dtype=W.dtype)
                # Ensure on CPU for saving.
                if A.device.type != 'cpu':
                    A = A.cpu()
                if B.device.type != 'cpu':
                    B = B.cpu()
                if W.device.type != 'cpu':
                    W = W.cpu()
                out[w_key] = W + (B @ A) * scale

            # 2) Optionally copy embedding / lm_head.
            #
            # NOTE: In our setup, the pretrained `.pth` already contains the correct special-token expansion
            # (including `[SEG]`). Copying `embed_tokens` / `lm_head` from the current PEFT-wrapped model can
            # accidentally overwrite those rows with newly-initialized weights, breaking `[SEG]` generation.
            if bool(getattr(self.args, 'export_embed_lm_head', False)):
                copy_map = {
                    'mllm.model.language_model.base_model.model.model.embed_tokens.weight': 'mllm.model.language_model.model.embed_tokens.weight',
                    'mllm.model.language_model.base_model.model.lm_head.weight': 'mllm.model.language_model.lm_head.weight',
                }
                for src, dst in copy_map.items():
                    if src in delta_sd and dst in out:
                        out[dst] = delta_sd[src].detach().cpu().to(dtype=out[dst].dtype)

            # 3) Projector (mlp1) keys differ between formats.
            for k, v in delta_sd.items():
                if k.startswith('mllm.model.mlp1.'):
                    nk = 'mlp1.' + k[len('mllm.model.mlp1.'):]
                    if nk in out:
                        out[nk] = v.detach().cpu().to(dtype=out[nk].dtype)

            # 4) text_hidden_fcs + SAM2 decoder weights: keyspace matches.
            for k, v in delta_sd.items():
                if k.startswith('text_hidden_fcs.'):
                    if k in out:
                        out[k] = v.detach().cpu().to(dtype=out[k].dtype)
                if k.startswith('grounding_encoder.sam2_model.sam_mask_decoder.'):
                    if k in out:
                        out[k] = v.detach().cpu().to(dtype=out[k].dtype)

            return out

        path = self.output_dir / filename

        if getattr(self.args, 'save_full_model', False):
            base_meta, base_sd = _unwrap_pretrained(self.args.pretrained_pth)
            delta_sd = {k: v.detach().cpu() for k, v in model_obj.state_dict().items()}
            full_sd = _export_full_state_dict(
                base_sd,
                delta_sd,
                lora_r=int(self.args.lora_r),
                lora_alpha=float(self.args.lora_alpha),
            )
            meta = dict(base_meta) if isinstance(base_meta, dict) else {}
            meta.update(
                dict(
                    exported_from='train_pseudo_gumbel_v2.py',
                    global_step=int(self.global_step),
                ))
            torch.save({'meta': meta, 'state_dict': full_sd}, path)
            print(f"✓ Saved full-model ckpt: {path}")
            return

        # Default: adapter-centric checkpoint.
        model_state = model_obj.state_dict()

        cpu_state = {k: v.detach().cpu() for k, v in model_state.items()}
        torch.save(cpu_state, path)
        print(f"✓ Saved: {path}")


def main():
    args = parse_args()

    # Setup distributed training
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    args.local_rank = local_rank

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{args.local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}, Local rank: {args.local_rank}")

    # Build model (sequential per-rank init to reduce peak host RAM during multi-process startup).
    # IMPORTANT: Move the model to GPU immediately after init on each rank to avoid holding full
    # CPU weights across all ranks at the same time.
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        model = None
        tokenizer = None
        for r in range(world_size):
            if rank == r:
                model, tokenizer = build_model(args)
                model = model.to(device, dtype=torch.bfloat16)
                torch.cuda.empty_cache()
            dist.barrier()
        assert model is not None and tokenizer is not None
    else:
        model, tokenizer = build_model(args)
        model = model.to(device, dtype=torch.bfloat16)

    # Freeze/unfreeze exactly the intended modules
    freeze_for_pseudo_gumbel_training(model)

    # Print trainable parameters
    print_trainable_parameters(model, "VideoLLaVASAMModel")

    # DDP
    if args.local_rank != -1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,
                    broadcast_buffers=False, static_graph=False)
        print(f"✓ DDP on rank {args.local_rank}")

    # EMA
    print("Creating EMA model...")
    ema_model = EMAModel(model.module if hasattr(model, 'module') else model, decay=args.ema_decay)
    ema_model = ema_model.to(device)
    print(f"✓ EMA model created (decay={args.ema_decay})")

    # Dataset
    pseudo_dataset, refcoco_dataset = build_datasets(args)

    from projects.llava_sam2.mask_caption_sft.dataset_builder import collate_fn_mask_caption

    if args.local_rank != -1:
        pseudo_sampler = DistributedSampler(pseudo_dataset, shuffle=True)
        ref_sampler = DistributedSampler(refcoco_dataset, shuffle=True) if refcoco_dataset is not None else None
        shuffle = False
    else:
        pseudo_sampler = None
        ref_sampler = None
        shuffle = True

    # Reduce /dev/shm pressure from DataLoader tensor sharing inside Docker cgroup.
    try:
        mp.set_sharing_strategy('file_system')
    except Exception:
        pass

    pseudo_loader = DataLoader(
        pseudo_dataset,
        batch_size=args.batch_size,
        sampler=pseudo_sampler,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_fn_mask_caption,
        pin_memory=False,
        prefetch_factor=1 if args.num_workers and args.num_workers > 0 else None,
        persistent_workers=bool(args.num_workers and args.num_workers > 0),
        drop_last=True,
    )
    print(f"✓ Pseudo dataloader: {len(pseudo_loader)} batches")

    ref_loader = None
    if refcoco_dataset is not None:
        ref_loader = DataLoader(
            refcoco_dataset,
            batch_size=args.batch_size,
            sampler=ref_sampler,
            shuffle=False if ref_sampler is not None else shuffle,
            num_workers=args.ref_num_workers,
            collate_fn=collate_fn_mask_caption,
            pin_memory=False,
            prefetch_factor=1 if args.ref_num_workers and args.ref_num_workers > 0 else None,
            persistent_workers=bool(args.ref_num_workers and args.ref_num_workers > 0),
            drop_last=True,
        )
        print(f"✓ RefCOCO dataloader: {len(ref_loader)} batches")

    # Optimizer (only trainable parameters)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.05
    )
    print("✓ Optimizer created")

    # Trainer
    trainer = PseudoGumbelTrainerV2(
        model=model,
        ema_model=ema_model,
        tokenizer=tokenizer,
        train_dataloader_pseudo=pseudo_loader,
        train_dataloader_refcoco=ref_loader,
        optimizer=optimizer,
        device=device,
        output_dir=args.output_dir,
        args=args
    )

    # Train
    print("=" * 80)
    print("Starting Pseudo Token + ST Gumbel-Softmax Training (V2 - Corrected)")
    print("=" * 80)

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        results = trainer.train_epoch(epoch)
        print(f"Epoch {epoch + 1} done. Loss: {results['loss']:.4f}")
        if args.max_steps is not None and trainer.reached_max_steps:
            print(f"Reached max_steps={args.max_steps}, stopping training.")
            break

    print("=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
