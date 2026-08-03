# NanoCord

## Installation

### Base install (dataset preparation only, no GPU required)

```bash
pip install nanocord
```

### Training install

The `[train]` extra requires torch, unsloth, and unsloth_zoo to be installed manually first, matched to your hardware, before installing the extra itself. Do not skip this order — installing `nanocord[train]` before torch/unsloth are present will cause pip's dependency resolver to fail or hang.

**Step 1: Identify your GPU vendor and follow the matching subsection below, then continue to Step 2 (all vendors).**

#### NVIDIA (Linux, WSL2, or Windows)

1. Check your CUDA version:
```bash
   nvidia-smi
```
   Note the CUDA version in the top-right of the output (e.g. 12.1, 12.4, 12.6).
2. Install torch matching that CUDA version (replace `cu121` to match step 1's result):
```bash
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```
3. Windows only — Triton: Unsloth depends on `triton`, which has no official Windows wheels on PyPI. Unsloth versions 2025.6.1+ (which is what nanocord requires) declare a platform-conditional dependency on `triton-windows` automatically — no manual triton install step is needed. Linux/WSL2 users can skip this note; official `triton` wheels work there natively.
4. Minimum GPU requirement: CUDA Compute Capability 7.0+ (V100, T4, RTX 20-series and newer, A100, H100, etc.). Older cards (e.g. GTX 10-series) may work but will be slow.

#### AMD (Linux native, or Windows/WSL2 — Linux recommended for broadest support)

1. Check your ROCm version:
```bash
   amd-smi version
```
   Look for the line reporting the ROCm version (e.g. 6.4, 7.1, 7.2).
2. Install the matching ROCm torch build (ROCm 6.0+ required; use the closest available index tag — rocm6.0/6.1/6.2/6.3/6.4/7.0/7.1/7.2; ROCm 6.5–6.9 should use the rocm6.4 tag, ROCm 7.3+ should use rocm7.2):
```bash
   pip install "torch>=2.4,<2.11.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.1
```
3. AMD distributed/multi-GPU training is Linux-only as of this writing; single-GPU training works on Windows/WSL2 but with less mature tooling than Linux.

#### Intel GPU

Unsloth has official Intel GPU support, but the install path is less standardized than NVIDIA/AMD as of this writing. Follow Unsloth's official Intel install guide at https://docs.unsloth.ai for the current recommended torch/XPU backend install command for your specific Intel GPU generation, since this changes frequently.

#### No dedicated GPU (CPU-only) or Apple Silicon / macOS

Training is not practical without a GPU — CPU training is technically possible but far too slow for iterating on a model of this size, and the core `unsloth` library nanocord depends on does not yet support macOS/Apple Silicon/MLX for training (this is listed as "in the works" upstream, distinct from the separate Unsloth Studio product, which nanocord does not use). If you're on a Mac or don't have a supported GPU, use a cloud GPU instead — e.g. a rented NVIDIA instance (Lambda, RunPod, Vast.ai, a cloud provider's GPU VM) or Google Colab — and follow the NVIDIA instructions above inside that environment.

**Step 2 (all vendors): verify torch sees your accelerator before proceeding.**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
(For AMD/ROCm, `torch.cuda.is_available()` is the correct check — ROCm builds of torch report through the same CUDA-named API.) This must print `True`. If it prints `False`, stop here and fix your driver/ROCm/toolkit install before continuing — installing unsloth on top of a broken accelerator setup will not fix it.

**Step 3: Install unsloth and unsloth_zoo (same command regardless of vendor):**
```bash
pip install unsloth==2026.7.5 unsloth_zoo==2026.7.6
```

**Step 4: Install nanocord's train extra:**
```bash
pip install -e .[train]
```
(or `pip install nanocord[train]` for a non-editable install)

**Step 5: Sanity check before running any training commands:**
```bash
python -c "import unsloth; print('unsloth OK')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Both should complete without error and the second should print `True`.

### Notes

- Windows NVIDIA users hitting persistent CUDA/build-tooling errors during Step 3 or 4 should consider WSL2 instead of native Windows — Unsloth's official guidance recommends it, since triton/xformers/bitsandbytes all have first-class Linux support and only patched-in Windows support.
- The torch/unsloth/unsloth_zoo version pins above (2.5.1 / 2026.7.5 / 2026.7.6) are the versions nanocord's train/cpt.py and train/sft.py were built and tested against, and assume an NVIDIA GPU; AMD users should use the ROCm-specific torch install shown above instead of the pinned 2.5.1+cuXXX build. If you install newer unsloth/torch versions than these pins, be aware Unsloth's internal APIs (chat template handling, train_on_responses_only's signature) may have changed since — if training raises an error originating from inside unsloth's own code rather than nanocord's, a version mismatch is the first thing to check.

### Prerequisites

NanoCord uses DiscordChatExporter to download Discord channel exports before it can build a dataset. It is now treated as an external prerequisite rather than a bundled dependency.

1. Download the latest DiscordChatExporter release from https://github.com/Tyrrrz/DiscordChatExporter/releases.
2. Install it and make sure the executable is available on your machine.
3. If the tool is not detected automatically, set the DISCORD_CHAT_EXPORTER_PATH environment variable to the full path of the executable, or enter the path when prompted by the CLI.

Example:

- PowerShell: $env:DISCORD_CHAT_EXPORTER_PATH = "C:\Path\To\DiscordChatExporter.Cli.exe"
- Bash: export DISCORD_CHAT_EXPORTER_PATH="/path/to/DiscordChatExporter.Cli"

### Commands

NanoCord provides a complete pipeline for building Discord persona bots:

```bash
# Initialize configuration
nanocord init

# Build CPT dataset from Discord logs (voice/tone)
nanocord dataset cpt -c <channel_id> -u <user_id> -d <discord_bot_token>

# Build SFT dataset from CPT dataset (conversational behavior)
nanocord dataset sft

# Run CPT training on the dataset
nanocord train cpt

# Run SFT training on the CPT checkpoint
nanocord train sft

# Register Discord bot and serve the fine-tuned model
nanocord bot register

# Run the complete pipeline (all 5 stages)
nanocord pipeline
```

### Configuration

NanoCord uses a YAML configuration file. You can create a default config with:

```bash
nanocord init
```

You can also set individual configuration values:

```bash
nanocord config set dataset.channel_id "123456789"
nanocord config set dataset.user_id "987654321"
nanocord config set dataset.discord_token "your-bot-token-here"
```

### Pipeline Stages

The complete pipeline consists of 5 stages:

1. `dataset cpt` - scrape Discord channel logs, group messages into "thoughts," write CPT dataset
2. `train cpt` - LoRA continued pretraining (unsupervised) on a small base model
3. `dataset sft` - extract (context → reply) pairs from DiscordChatExporter's reply-reference metadata
4. `train sft` - LoRA fine-tune on top of the CPT checkpoint using the SFT dataset
5. `bot register` - export to GGUF, serve via Ollama, register as a Discord slash command

Use `--skip-*` flags with the pipeline command to run partial stages:

```bash
# Skip training stages and only build datasets
nanocord pipeline --skip-cpt-train --skip-sft-train --skip-bot

# Skip dataset building and only train
nanocord pipeline --skip-cpt-dataset --skip-sft-dataset
```