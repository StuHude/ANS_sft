# Repository Guidelines

## Project Structure & Module Organization

- `projects/llava_sam2/`: primary research code (models, datasets, training, evaluation).
  - `mask_caption_sft/`: dual-loop SFT training (mask→caption→mask).
  - `rl_train/`: RL finetuning and stability tooling.
  - `configs/`: experiment configs (e.g., `projects/llava_sam2/configs/sa2va_4b.py`).
  - `evaluation/`: evaluation entrypoints and helpers.
- `vlm/`: shared engine utilities (runner loops, hooks, checkpoint/video IO).
- `tools/`: launchers/utilities (e.g., distributed `dist.sh`, `train.py`, conversion scripts).
- `scripts/`: small one-off utilities (e.g., HF safetensors merge).
- `assets/`: static assets used by docs/visuals.
- `work_dirs/`: training outputs and logs (keep out of git).
- `data/`, `pretrained/`: local datasets/checkpoints (paths are environment-specific; avoid committing large artifacts).

## Build, Test, and Development Commands

- `pip install -r requirements.txt`: install core dependencies (GPU/DeepSpeed setups may require additional system packages).
- `bash projects/llava_sam2/mask_caption_sft/scripts/run_single_gpu.sh`: quick local SFT smoke run.
- `bash projects/llava_sam2/mask_caption_sft/scripts/run_multi_gpu.sh 8`: multi-GPU SFT run.
- `bash tools/dist.sh train projects/llava_sam2/configs/sa2va_4b.py 8`: distributed training via the `xtuner` entrypoint.
- `python tools/convert_to_hf_new.py --help`: convert a training checkpoint to HuggingFace format.
- `bash run_pseudo_gumbel_v2_8gpu.sh`: 8×GPU “EMA→ST-Gumbel→Seg” loop training (docker `vlm-env`).

## Coding Style & Naming Conventions

- Python: 4-space indentation, keep code PEP8-ish; prefer `snake_case` for functions/vars and `CamelCase` for classes.
- CLIs typically use `--snake_case` flags (e.g., `--model_path`, `--output_dir`).
- Prefer configurable paths (args/config files) over hard-coded `/data/...` paths; keep experiment-specific constants in `config.py`/`configs/*.py`.
- Treat `third_parts/` as vendored code: avoid non-essential edits.

## Testing Guidelines

- Tests are primarily script-based smoke/integration checks:
  - `python test_dataset_transform.py`: validates dataset transforms and mask ranges.
  - `bash test_sft_training.sh` / `bash test_dual_loop.sh`: short training runs (require datasets + GPUs).
- When adding coverage, prefer small, deterministic tests that run without full training; place them near the feature (or introduce a `tests/` folder if a suite emerges).

## Commit & Pull Request Guidelines

- Commit messages follow a simple imperative style (e.g., “Add …”, “Fix …”); keep the subject ≤72 chars and put rationale/notes in the body.
- PRs should include: what changed, how to reproduce (exact command + config), and any dataset/checkpoint assumptions; attach key logs/metrics if training behavior changes.
- Do not commit large weights, dataset dumps, or `work_dirs/` outputs.
