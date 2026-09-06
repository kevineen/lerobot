# ROCm training (this fork)

kevineen/lerobot, branch `feat/rocm-train`. Do **not** open PRs against huggingface/lerobot for these changes.

Tested on **AMD Radeon AI PRO R9700** (gfx1201) with conda env `lerobot-rocm` (Python 3.12) and PyTorch `2.11.0+rocm7.14.1`.

Operational notes for the training PC also live in [note.md](../note.md) and in [kevineen/leisaac note.txt](https://github.com/kevineen/leisaac/blob/feat/training-pc/note.txt).

## Device names

ROCm PyTorch still uses the `cuda` device type (`torch.cuda.is_available()` is True). Detection is `torch.version.hip`.

- `--policy.device=cuda` is correct on Radeon.
- `--policy.device=rocm` is accepted and canonicalized to `cuda`.
- Logs say `ROCm/HIP backend detected`.

See `lerobot.utils.device_utils` (`is_rocm`, `canonical_torch_device`, `select_attn_implementation`).

## Video: do not install torchcodec

PyPI `torchcodec` wheels are CUDA ABI and can crash ROCm PyTorch. This fork:

- Removes torchcodec from `lerobot[dataset]`.
- Defaults to native **PyAV** (`decode_video_frames_pyav`). ROCm torchvision has no `VideoReader`.
- If someone still passes `--dataset.video_backend=torchcodec` on ROCm, it falls back to PyAV.

NVIDIA CUDA hosts may install torchcodec themselves:

```bash
pip install "torchcodec>=0.3.0,<0.11.0"
```

## GR00T / flash-attn

FlashAttention2 is NVIDIA CUDA only. `lerobot[groot]` does **not** install `flash-attn`.

On ROCm, Eagle/Qwen2 use **SDPA**. CUDA hosts that want flash-attn:

```bash
pip install --no-build-isolation "flash-attn>=2.5.9,<3.0.0"
```

## Install (this PC)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot-rocm

pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  "torch[device-gfx1201]==2.11.0+rocm7.14.1" \
  "torchvision[device-gfx1201]==0.26.0+rocm7.14.1"
pip install "numpy>=2.0.0,<2.3.0"
pip install -e "/home/kevin/robot/lerobot[training,smolvla]" \
  --extra-index-url https://repo.amd.com/rocm/whl-multi-arch/
```

Always pass the AMD extra index so pip does not replace torch with a CUDA wheel.

```bash
export HF_LEROBOT_HOME=/mnt/shared_hdd/robot/datasets/lerobot
export WANDB_MODE=disabled
export TORCH_BLAS_PREFER_HIPBLASLT=0   # gfx1201 hipblasLt warning
# do not run llama.cpp on the same GPU
```

## Smoke (2026-09-06)

| Policy | Result |
|--------|--------|
| ACT 50 step | OK (`act-rocm-smoke`) |
| SmolVLA 10 step | OK (`smolvla-rocm-smoke`) |
| diffusion 10 step | OK (`diffusion-rocm-smoke`) |
| pi0 | extra installs; training needs HF access to gated `google/paligemma-3b-pt-224` |

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht \
  --dataset.video_backend=pyav \
  --output_dir=/mnt/shared_hdd/robot/models/checkpoints/act-rocm-smoke \
  --steps=50 --batch_size=4 --num_workers=2 \
  --eval_freq=0 --wandb.enable=false
```

Isaac Sim / Isaac Lab stay on an NVIDIA machine (or after a GPU swap). This repo does not replace them with MuJoCo.
