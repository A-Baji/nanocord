# NanoCord

## Installation

```bash
pip install nanocord
```

## Usage

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