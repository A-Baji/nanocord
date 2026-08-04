import os
from pathlib import Path
from typing import Annotated, Optional, List

import typer
import yaml

from nanocord import global_logger
from nanocord.bot.register import run_bot, build_command_tree, compute_command_fingerprint, _fingerprint_path
from nanocord.config import load_and_merge_config
from nanocord.dataset.cpt import build_cpt_dataset
from nanocord.dataset.sft import build_sft_dataset
from nanocord.dataset.sft import DEFAULT_SYSTEM_PROMPT
from nanocord.dataset.sft import MissingPersonaNameError
from nanocord.paths import CONFIG_PATH
from nanocord.train.cpt import run_cpt_training
from nanocord.train.sft import run_sft_training
from nanocord.infer import resolve_checkpoint_path, resolve_preset, load_bot_config_section, generate_response

def _yaml_single_quote(s: str) -> str:
    """Quote a string for use in YAML as a single-quoted scalar."""
    return "'" + s.replace("'", "''") + "'"

# Use the global logger
logger = global_logger

# Create Typer application
app = typer.Typer(
    name="nanocord",
    help="Discord persona SLM CLI: export chat data, train CPT/SFT models, and register a Discord bot",
    no_args_is_help=True,
)

# Sub-app for dataset commands
dataset_app = typer.Typer(
    name="dataset",
    help="Export and prepare Discord chat data"
)
app.add_typer(dataset_app, name="dataset")


# Sub-app for train commands
train_app = typer.Typer(
    name="train",
    help="Train models using different techniques"
)
app.add_typer(train_app, name="train")


# Sub-app for bot commands
bot_app = typer.Typer(
    name="bot",
    help="Manage Discord bot functionality"
)
app.add_typer(bot_app, name="bot")


# Sub-app for inference commands
infer_app = typer.Typer(
    name="infer",
    help="Test model inference before registering a bot command"
)
app.add_typer(infer_app, name="infer")


@app.callback(invoke_without_command=True)
def main(
    version_flag: Optional[bool] = typer.Option(None, "-V", "--version", help="Show version and exit"),
):
    """Main CLI entry point."""
    if version_flag:
        # Try to get version from package metadata
        try:
            import importlib.metadata
            version = importlib.metadata.version("nanocord")
        except Exception:
            # Fallback to pyproject.toml version
            version = "unknown"

        typer.echo(f"nanocord {version}")
        raise typer.Exit()


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Force re-initialization even if config exists")
):
    """
    Initialize NanoCord by creating a default configuration file.

    This command creates a config.yaml file in the user's data directory
    and prompts for the Discord bot token and DiscordChatExporter path.
    """
    # Create config directory if it doesn't exist
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Check if config already exists
    if CONFIG_PATH.exists() and not force:
        typer.echo(f"Configuration file already exists at: {CONFIG_PATH}")
        typer.echo("Skipping initialization.")
        raise typer.Exit()

    # Prompt for Discord token
    typer.echo("Initializing NanoCord configuration...")
    discord_token = typer.prompt(
        "Enter your Discord bot token (optional - press Enter to skip)",
        default="",
        show_default=False
    )

    # Prompt for DiscordChatExporter path
    discord_chat_exporter_path = typer.prompt(
        "Enter the path to DiscordChatExporter.Cli executable (optional - press Enter to skip)",
        default="",
        show_default=False
    )

    # Escape paths for YAML using single quotes to handle Windows backslashes properly
    discord_token_yaml = _yaml_single_quote(discord_token)
    discord_chat_exporter_path_yaml = _yaml_single_quote(discord_chat_exporter_path)

    # Create default config content
    default_config = f"""# NanoCord Configuration File
# This file contains default settings for the nanocord CLI tool.
# Values can be overridden by command-line arguments.

# Shared across every section below (dataset.cpt, dataset.sft, train.cpt,
# train.sft) via config inheritance - identifies which Discord user/
# channel this persona is built from.
channel_id: ""
user_id: ""

dataset:
    # Params here are shared by both dataset.cpt and dataset.sft below -
    # redeclare a key inside either one to override it just for that mode.
    discord_token: {discord_token_yaml} # Optional here, falls back to DISCORD_BOT_TOKEN env var if left empty
    discord_chat_exporter_path: {discord_chat_exporter_path_yaml} # Path to DiscordChatExporter.Cli executable
    thought_time: 5
    thought_min: 6
    thought_max: null
    max_entries: 1000
    offset: 0
    distributed: false
    reverse: false
    redownload: false
    cpt:
        # Overrides for CPT dataset config
    sft:
        persona_name: "" # Required. The display name of the persona (target user) to embed in the SFT system prompt.
        system_prompt: "You are {{persona_name}}." # Overridable. Must contain a {{persona_name}} placeholder.

train:
    # Params here are shared by both train.cpt and train.sft below -
    # redeclare a key inside either one to override it just for that mode.
    base_model: "qwen2.5-7b" # One of: smollm3-3b, qwen3-4b, qwen3-1.7b, llama-3.2-3b, qwen2.5-7b
    load_in_4bit: true
    max_seq_length: 2048
    lora_r: 32
    lora_alpha: 64
    lora_dropout: 0
    learning_rate: 0.0002
    effective_batch_size: 16
    per_device_train_batch_size: 2
    num_train_epochs: 3
    eval_split: 0.05
    early_stopping_patience: 3
    weight_decay: 0.01
    warmup_ratio: 0.05
    seed: 3407
    vram_memory_fraction: 0.9  # Fraction of total GPU memory this process is allowed to use (0-1]; lower this if training crashes the system/driver rather than raising a clean OOM error
    cpt:
        # Overrides for CPT training config
        packing: true
        max_seq_length: 1024
    sft:
        num_train_epochs: 5 # Override - more epochs to firmly cement conversational voice on top of the shared default of 3
        lora_dropout: 0.05 # Override - light regularization against overfitting on a small SFT dataset over 5 epochs, unlike CPT's 0 dropout
        eval_split: 0.1 # Override - a 5% eval slice is noisy for early-stopping decisions on a small personal-chat SFT dataset
        neftune_noise_alpha: 5 # NEFTune - adds noise to embeddings during SFT training, improves output quality/diversity on small instruction datasets; not used for CPT

bot:
    # Named generation-parameter presets, referenced by name from bot.commands below.
    presets:
        # example_preset:
        #   temperature: 0.5
        #   repetition_penalty: 1.18
        #   no_repeat_ngram_size: 3
        #   max_new_tokens: 256
    # Registered Discord slash commands. Each entry's model_path may be omitted to
    # auto-resolve to {output_dir}/models/{user_id}_{channel_id}_{stage}_lora.
    commands:
        # - name: "ask"
        #   description: "Ask the persona something"
        #   model_path: null
        #   stage: "sft"  # cpt or sft, used only when model_path is omitted
        #   preset: "example_preset"  # a name from bot.presets, or "random"
        #   preset_pool: []  # list of preset names; only used when preset is "random". Empty/omitted = all presets.
output_dir: null  # Optional - overrides the base directory for ALL generated output (datasets, model checkpoints, raw Discord exports, Unsloth cache/temp dirs); config.yaml's own location is NOT affected
"""

    # Write the config file
    with open(CONFIG_PATH, 'w') as f:
        f.write(default_config)

    typer.echo(f"Configuration file created at: {CONFIG_PATH}")
    if discord_token:
        typer.echo("Discord bot token saved to configuration.")
    else:
        typer.echo("No Discord bot token provided. You can set it later via:")
        typer.echo(f"  - Environment variable: DISCORD_BOT_TOKEN")
        typer.echo(f"  - Configuration file: {CONFIG_PATH}")

    if discord_chat_exporter_path:
        typer.echo("DiscordChatExporter path saved to configuration.")
    else:
        typer.echo("No DiscordChatExporter path provided. You can set it later via:")
        typer.echo(f"  - Environment variable: DISCORD_CHAT_EXPORTER_PATH")
        typer.echo(f"  - Configuration file: {CONFIG_PATH}")


@dataset_app.command("cpt")
def dataset_cpt(
    discord_token: Optional[str] = typer.Option(
        None,
        "-d",
        "--discord-token",
        help="The Discord token for your bot. Must either be provided as an argument or set as the DISCORD_BOT_TOKEN environment variable"
    ),
    channel_id: Optional[str] = typer.Option(
        None,
        "-c",
        "--channel_id",
        help="The ID of the Discord channel you want to use"
    ),
    user_id: Optional[str] = typer.Option(
        None,
        "-u",
        "--user_id",
        help="The ID of the Discord user you want to use"
    ),
    exporter_path: Optional[str] = typer.Option(
        None,
        "--exporter-path",
        help="Path to the DiscordChatExporter.Cli executable"
    ),
    thought_time: Optional[int] = typer.Option(
        None,
        "--ttime",
        "--thought-time",
        help='The maximum amount of time in seconds to consider two individual messages to be part of the same "thought"'
    ),
    thought_max: Optional[int] = typer.Option(
        None,
        "--tmax",
        "--thought-max",
        help="The maximum length in words of each thought"
    ),
    thought_min: Optional[int] = typer.Option(
        None,
        "--tmin",
        "--thought-min",
        help="The minimum length in words of each thought"
    ),
    max_entries: Optional[int] = typer.Option(
        None,
        "-m",
        "--max-entries",
        help="The max amount of entries (by lines) that may exist in the dataset"
    ),
    offset: Optional[int] = typer.Option(
        None,
        "--os",
        "--offset",
        help="The offset by line index starting at 0 for where to start selecting lines for the dataset"
    ),
    distributed: Optional[bool] = typer.Option(
        None,
        "--distributed",
        help="Select lines as an even distribution instead of sequentially (omit to use the config.yaml value)"
    ),
    reverse_lines: Optional[bool] = typer.Option(
        None,
        "--reverse-lines",
        help="Reverse the order in which to select lines for the dataset (omit to use the config.yaml value)"
    ),
    redownload: Optional[bool] = typer.Option(
        None,
        "--redownload",
        help="Redownload the Discord chat logs (omit to use the config.yaml value)"
    ),
    config_file: Optional[str] = typer.Option(
        str(CONFIG_PATH),
        "--config",
        help="Path to YAML configuration file (default: user data directory)"
    ),
):
    """
    Download Discord channel logs, parse them into a CPT dataset, and save the results locally
    """

    # Prepare CLI arguments for merging
    cli_args = {
        "discord_token": discord_token,
        "channel_id": channel_id,
        "user_id": user_id,
        "discord_chat_exporter_path": exporter_path,
        "thought_time": thought_time,
        "thought_max": thought_max,
        "thought_min": thought_min,
        "max_entries": max_entries,
        "offset": offset,
        "distributed": distributed,
        "reverse": reverse_lines,
        "redownload": redownload
    }

    # Load and merge configuration - pass "dataset.cpt" as the section to load
    merged_config = load_and_merge_config(config_file, cli_args, "dataset.cpt")

    # Handle Discord token fallback to environment variable
    if not merged_config["discord_token"]:
        merged_config["discord_token"] = os.getenv("DISCORD_BOT_TOKEN")

    # Validate that required parameters are present
    if not merged_config["channel_id"]:
        typer.secho("Error: Channel ID must be provided via --channel_id or config.yaml", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if not merged_config["user_id"]:
        typer.secho("Error: User ID must be provided via --user_id or config.yaml", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Unpack the merged configuration into build_cpt_dataset function
    build_cpt_dataset(
        channel_id=merged_config["channel_id"],
        user_id=merged_config["user_id"],
        bot_token=merged_config["discord_token"],
        discord_chat_exporter_path=merged_config["discord_chat_exporter_path"],
        thought_time=merged_config["thought_time"],
        thought_max=merged_config["thought_max"],
        thought_min=merged_config["thought_min"],
        max_entry_count=merged_config["max_entries"],
        offset=merged_config["offset"],
        distributed=merged_config["distributed"],
        reverse=merged_config["reverse"],
        redownload=merged_config["redownload"],
        output_dir=merged_config.get("output_dir")
    )


@dataset_app.command("sft")
def dataset_sft(
    discord_token: Optional[str] = typer.Option(
        None,
        "-d",
        "--discord-token",
        help="The Discord token for your bot. Must either be provided as an argument or set as the DISCORD_BOT_TOKEN environment variable"
    ),
    channel_id: Optional[str] = typer.Option(
        None,
        "-c",
        "--channel_id",
        help="The ID of the Discord channel you want to use"
    ),
    user_id: Optional[str] = typer.Option(
        None,
        "-u",
        "--user_id",
        help="The ID of the Discord user you want to use"
    ),
    exporter_path: Optional[str] = typer.Option(
        None,
        "--exporter-path",
        help="Path to the DiscordChatExporter.Cli executable"
    ),
    thought_time: Optional[int] = typer.Option(
        None,
        "--ttime",
        "--thought-time",
        help='The maximum amount of time in seconds to consider two individual messages to be part of the same "thought", for the response (target-user) side'
    ),
    thought_max: Optional[int] = typer.Option(
        None,
        "--tmax",
        "--thought-max",
        help="The maximum length in words of each thought, for the response side"
    ),
    thought_min: Optional[int] = typer.Option(
        None,
        "--tmin",
        "--thought-min",
        help="The minimum length in words of each thought, for the response side"
    ),
    context_thought_time: Optional[int] = typer.Option(
        None,
        "--cttime",
        "--context-thought-time",
        help="Same as --thought-time but for the context side. Defaults to --thought-time if not given"
    ),
    context_thought_max: Optional[int] = typer.Option(
        None,
        "--ctmax",
        "--context-thought-max",
        help="Same as --thought-max but for the context side. Defaults to --thought-max if not given"
    ),
    context_thought_min: Optional[int] = typer.Option(
        None,
        "--ctmin",
        "--context-thought-min",
        help="Same as --thought-min but for the context side. Defaults to --thought-min if not given"
    ),
    persona_name: Optional[str] = typer.Option(
        None,
        "-p",
        "--persona-name",
        help="The display name of the persona (target user) to embed in the SFT system prompt. Required."
    ),
    system_prompt: Optional[str] = typer.Option(
        None,
        "--system-prompt",
        help="System prompt content embedded in every output record"
    ),
    max_entries: Optional[int] = typer.Option(
        None,
        "-m",
        "--max-entries",
        help="The max amount of entries (by lines) that may exist in the dataset"
    ),
    offset: Optional[int] = typer.Option(
        None,
        "--os",
        "--offset",
        help="The offset by line index starting at 0 for where to start selecting lines for the dataset"
    ),
    distributed: Optional[bool] = typer.Option(
        None,
        "--distributed",
        help="Select lines as an even distribution instead of sequentially (omit to use the config.yaml value)"
    ),
    reverse_lines: Optional[bool] = typer.Option(
        None,
        "--reverse-lines",
        help="Reverse the order in which to select lines for the dataset (omit to use the config.yaml value)"
    ),
    redownload: Optional[bool] = typer.Option(
        None,
        "--redownload",
        help="Redownload the Discord chat logs (omit to use the config.yaml value)"
    ),
    config_file: Optional[str] = typer.Option(
        str(CONFIG_PATH),
        "--config",
        help="Path to YAML configuration file (default: user data directory)"
    ),
):
    """
    Download Discord channel logs (if needed) and parse them into an SFT dataset
    """

    # Prepare CLI arguments for merging
    cli_args = {
        "discord_token": discord_token,
        "channel_id": channel_id,
        "user_id": user_id,
        "discord_chat_exporter_path": exporter_path,
        "thought_time": thought_time,
        "thought_max": thought_max,
        "thought_min": thought_min,
        "persona_name": persona_name,
        "system_prompt": system_prompt,
        "max_entries": max_entries,
        "offset": offset,
        "distributed": distributed,
        "reverse": reverse_lines,
        "redownload": redownload
    }

    # Load and merge configuration - pass "dataset.sft" as the section to load
    merged_config = load_and_merge_config(config_file, cli_args, "dataset.sft")

    # Load the context-specific overrides, if any. CLI flags for the context
    # side (--context-thought-*) are merged specifically at the
    # "dataset.sft.context" level, so a CLI flag there takes priority only
    # for the context side. If dataset.sft.context: is not set in config.yaml
    # and no --context-* flags are passed, this resolves to the exact same
    # values as merged_config above.
    context_cli_args = {
        "thought_time": context_thought_time,
        "thought_max": context_thought_max,
        "thought_min": context_thought_min,
    }
    context_config = load_and_merge_config(config_file, context_cli_args, "dataset.sft.context")
    merged_config["context_thought_time"] = context_config["thought_time"]
    merged_config["context_thought_max"] = context_config["thought_max"]
    merged_config["context_thought_min"] = context_config["thought_min"]

    # Handle Discord token fallback to environment variable
    if not merged_config["discord_token"]:
        merged_config["discord_token"] = os.getenv("DISCORD_BOT_TOKEN")

    # Validate that required parameters are present
    if not merged_config["channel_id"]:
        typer.secho("Error: Channel ID must be provided via --channel_id or config.yaml", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if not merged_config["user_id"]:
        typer.secho("Error: User ID must be provided via --user_id or config.yaml", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        dataset_path = build_sft_dataset(merged_config)
        typer.echo(f"SFT dataset created at: {dataset_path}")
    except (MissingPersonaNameError, ValueError) as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@train_app.command("cpt")
def train_cpt(
    config_file: Optional[str] = typer.Option(
        str(CONFIG_PATH),
        "--config",
        help="Path to YAML configuration file (default: user data directory)"
    ),
    base_model: Optional[str] = typer.Option(
        None,
        "--base-model",
        help="Base model to use for training. One of: smollm3-3b, qwen3-4b, qwen3-4b-instruct, qwen3-1.7b, llama-3.2-3b, qwen2.5-7b"
    ),
    load_in_4bit: Optional[bool] = typer.Option(
        None,
        "--load-in-4bit",
        help="Load model in 4-bit precision"
    ),
    max_seq_length: Optional[int] = typer.Option(
        None,
        "--max-seq-length",
        help="Maximum sequence length for training"
    ),
    lora_r: Optional[int] = typer.Option(
        None,
        "--lora-r",
        help="LoRA rank"
    ),
    lora_alpha: Optional[int] = typer.Option(
        None,
        "--lora-alpha",
        help="LoRA alpha parameter"
    ),
    lora_dropout: Optional[float] = typer.Option(
        None,
        "--lora-dropout",
        help="LoRA dropout rate"
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "--learning-rate",
        help="Learning rate for training"
    ),
    effective_batch_size: Optional[int] = typer.Option(
        None,
        "--effective-batch-size",
        help="Effective batch size for training"
    ),
    per_device_train_batch_size: Optional[int] = typer.Option(
        None,
        "--per-device-train-batch-size",
        help="Per device train batch size"
    ),
    num_train_epochs: Optional[int] = typer.Option(
        None,
        "--num-train-epochs",
        help="Number of training epochs"
    ),
    eval_split: Optional[float] = typer.Option(
        None,
        "--eval-split",
        help="Fraction of data to use for evaluation"
    ),
    early_stopping_patience: Optional[int] = typer.Option(
        None,
        "--early-stopping-patience",
        help="Early stopping patience"
    ),
    weight_decay: Optional[float] = typer.Option(
        None,
        "--weight-decay",
        help="Weight decay for training"
    ),
    warmup_ratio: Optional[float] = typer.Option(
        None,
        "--warmup-ratio",
        help="Warmup ratio for learning rate scheduler"
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for training"
    ),
    vram_memory_fraction: Optional[float] = typer.Option(
        None,
        "--vram-memory-fraction",
        help="Fraction of total GPU memory this process is allowed to use (0-1]; lower this if training crashes the system/driver rather than raising a clean OOM error"
    ),
    packing: Optional[bool] = typer.Option(
        None,
        "--packing",
        help="Use packing for training"
    ),
):
    """
    Run CPT training on the dataset
    """

    # Prepare CLI arguments for merging
    cli_args = {
        "base_model": base_model,
        "load_in_4bit": load_in_4bit,
        "max_seq_length": max_seq_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "learning_rate": learning_rate,
        "effective_batch_size": effective_batch_size,
        "per_device_train_batch_size": per_device_train_batch_size,
        "num_train_epochs": num_train_epochs,
        "eval_split": eval_split,
        "early_stopping_patience": early_stopping_patience,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "seed": seed,
        "vram_memory_fraction": vram_memory_fraction,
        "packing": packing,
    }

    # Load and merge configuration - pass "train.cpt" as the section to load
    merged_config = load_and_merge_config(config_file, cli_args, "train.cpt")

    # Check for CUDA availability before attempting to run training
    import torch
    if not torch.cuda.is_available():
        typer.secho("Error: No CUDA-capable GPU detected. Training requires an NVIDIA GPU.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        # Call the run_cpt_training function
        checkpoint_path = run_cpt_training(merged_config)
        typer.echo(f"CPT training completed. Checkpoint saved at: {checkpoint_path}")
    except Exception as e:
        # Re-raise any other exceptions for better debugging
        raise e


@train_app.command("sft")
def train_sft(
    config_file: Optional[str] = typer.Option(
        str(CONFIG_PATH),
        "--config",
        help="Path to YAML configuration file (default: user data directory)"
    ),
    base_model: Optional[str] = typer.Option(
        None,
        "--base-model",
        help="Base model to use for training. One of: smollm3-3b, qwen3-4b, qwen3-4b-instruct, qwen3-1.7b, llama-3.2-3b, qwen2.5-7b"
    ),
    load_in_4bit: Optional[bool] = typer.Option(
        None,
        "--load-in-4bit",
        help="Load model in 4-bit precision"
    ),
    max_seq_length: Optional[int] = typer.Option(
        None,
        "--max-seq-length",
        help="Maximum sequence length for training"
    ),
    lora_r: Optional[int] = typer.Option(
        None,
        "--lora-r",
        help="LoRA rank"
    ),
    lora_alpha: Optional[int] = typer.Option(
        None,
        "--lora-alpha",
        help="LoRA alpha parameter"
    ),
    lora_dropout: Optional[float] = typer.Option(
        None,
        "--lora-dropout",
        help="LoRA dropout rate"
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "--learning-rate",
        help="Learning rate for training"
    ),
    effective_batch_size: Optional[int] = typer.Option(
        None,
        "--effective-batch-size",
        help="Effective batch size for training"
    ),
    per_device_train_batch_size: Optional[int] = typer.Option(
        None,
        "--per-device-train-batch-size",
        help="Per device train batch size"
    ),
    num_train_epochs: Optional[int] = typer.Option(
        None,
        "--num-train-epochs",
        help="Number of training epochs"
    ),
    eval_split: Optional[float] = typer.Option(
        None,
        "--eval-split",
        help="Fraction of data to use for evaluation"
    ),
    early_stopping_patience: Optional[int] = typer.Option(
        None,
        "--early-stopping-patience",
        help="Early stopping patience"
    ),
    weight_decay: Optional[float] = typer.Option(
        None,
        "--weight-decay",
        help="Weight decay for training"
    ),
    warmup_ratio: Optional[float] = typer.Option(
        None,
        "--warmup-ratio",
        help="Warmup ratio for learning rate scheduler"
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for training"
    ),
    vram_memory_fraction: Optional[float] = typer.Option(
        None,
        "--vram-memory-fraction",
        help="Fraction of total GPU memory this process is allowed to use (0-1]; lower this if training crashes the system/driver rather than raising a clean OOM error"
    ),
    neftune_noise_alpha: Optional[float] = typer.Option(
        None,
        "--neftune-noise-alpha",
        help="NEFTune - adds noise to embeddings during SFT training, improves output quality/diversity on small instruction datasets; not used for CPT"
    ),
):
    """
    Run SFT training on the CPT checkpoint
    """

    # Prepare CLI arguments for merging
    cli_args = {
        "base_model": base_model,
        "load_in_4bit": load_in_4bit,
        "max_seq_length": max_seq_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "learning_rate": learning_rate,
        "effective_batch_size": effective_batch_size,
        "per_device_train_batch_size": per_device_train_batch_size,
        "num_train_epochs": num_train_epochs,
        "eval_split": eval_split,
        "early_stopping_patience": early_stopping_patience,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "seed": seed,
        "vram_memory_fraction": vram_memory_fraction,
        "neftune_noise_alpha": neftune_noise_alpha,
    }

    # Load and merge configuration - pass "train.sft" as the section to load
    merged_config = load_and_merge_config(config_file, cli_args, "train.sft")

    # Check for CUDA availability before attempting to run training
    import torch
    if not torch.cuda.is_available():
        typer.secho("Error: No CUDA-capable GPU detected. Training requires an NVIDIA GPU.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        # Call the run_sft_training function
        model_path = run_sft_training(merged_config)
        typer.echo(f"SFT training completed. Model saved at: {model_path}")
    except Exception as e:
        # Re-raise any other exceptions for better debugging
        raise e




@bot_app.command("add-preset")
def bot_add_preset(
    name: Optional[str] = typer.Option(None, "--name", help="Name of the preset"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="Temperature for generation"),
    repetition_penalty: Optional[float] = typer.Option(None, "--repetition-penalty", help="Repetition penalty"),
    no_repeat_ngram_size: Optional[int] = typer.Option(None, "--no-repeat-ngram-size", help="No repeat ngram size"),
    max_new_tokens: Optional[int] = typer.Option(None, "--max-new-tokens", help="Maximum new tokens"),
    config_file: str = typer.Option(str(CONFIG_PATH), "--config", help="Path to YAML configuration file (default: user data directory)"),
):
    """
    Add a new generation parameter preset to the configuration
    """
    # Load existing config
    if not CONFIG_PATH.exists():
        typer.secho("Error: No configuration file found. Create one first with 'nanocord init'.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f) or {}

    # Prompt for name if not provided
    if name is None:
        name = typer.prompt("Enter preset name")
        if not name:
            typer.secho("Error: Preset name cannot be empty", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    # Prompt for other parameters if not provided
    if temperature is None:
        temperature = typer.prompt("Enter temperature (default 0.5)", default=0.5, type=float)

    if repetition_penalty is None:
        repetition_penalty = typer.prompt("Enter repetition penalty (default 1.18)", default=1.18, type=float)

    if no_repeat_ngram_size is None:
        no_repeat_ngram_size = typer.prompt("Enter no repeat ngram size (default 3)", default=3, type=int)

    if max_new_tokens is None:
        max_new_tokens = typer.prompt("Enter maximum new tokens (default 256)", default=256, type=int)

    # Check if preset already exists
    bot_section = config.setdefault("bot", {})
    presets = bot_section.setdefault("presets", {})

    if name in presets:
        typer.secho(f"Error: preset '{name}' already exists. Choose a different name or edit config.yaml directly.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Add the new preset
    presets[name] = {
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "max_new_tokens": max_new_tokens
    }

    # Write back to file
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)

    typer.echo(f"Added preset '{name}' to configuration")


@bot_app.command("add-command")
def bot_add_command(
    name: Optional[str] = typer.Option(None, "--name", help="Name of the command"),
    description: Optional[str] = typer.Option(None, "--description", help="Description of the command"),
    model_path: Optional[str] = typer.Option(None, "--model-path", help="Path to the model checkpoint"),
    stage: Optional[str] = typer.Option(None, "--stage", help="Training stage ('cpt' or 'sft') - used only when --model-path is omitted"),
    preset: Optional[str] = typer.Option(None, "--preset", help="Name of the preset to use (from config) or 'random'"),
    preset_pool: Optional[List[str]] = typer.Option(None, "--preset-pool", help="List of preset names to use when --preset is 'random'"),
    config_file: str = typer.Option(str(CONFIG_PATH), "--config", help="Path to YAML configuration file (default: user data directory)"),
):
    """
    Add a new Discord slash command to the configuration
    """
    # Load existing config
    if not CONFIG_PATH.exists():
        typer.secho("Error: No configuration file found. Create one first with 'nanocord init'.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f) or {}

    # Prompt for name if not provided
    if name is None:
        name = typer.prompt("Enter command name")
        if not name:
            typer.secho("Error: Command name cannot be empty", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    # Prompt for description if not provided
    if description is None:
        description = typer.prompt("Enter command description")
        if not description:
            typer.secho("Error: Command description cannot be empty", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    # Validate that either model_path or stage is provided (or both can be None to prompt for them)
    if model_path is None and stage is None:
        # Prompt for model path
        model_path = typer.prompt(
            "Model path (leave blank to auto-resolve from output_dir/channel_id/user_id)",
            default="",
            show_default=False
        )

        # If user left it blank, prompt for stage
        if not model_path:
            while True:
                stage_input = typer.prompt("Stage ('cpt' or 'sft')")
                if stage_input in ['cpt', 'sft']:
                    stage = stage_input
                    break
                else:
                    typer.secho("Error: Stage must be either 'cpt' or 'sft'", fg=typer.colors.RED)
    elif model_path is not None and stage is not None:
        # Both provided, which is fine - but validate stage value
        if stage not in ['cpt', 'sft']:
            typer.secho("Error: Stage must be either 'cpt' or 'sft'", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    # Validate preset if provided
    if preset is not None:
        # Check if preset is "random" or an existing key in config.get("bot", {}).get("presets", {})
        bot_section = config.get("bot", {})
        presets = bot_section.get("presets", {})

        if preset != "random" and preset not in presets:
            available_presets = list(presets.keys())
            typer.secho(
                f"Error: Preset '{preset}' not found. Available presets are: {', '.join(available_presets)}",
                fg=typer.colors.RED
            )
            raise typer.Exit(code=1)

        # If preset is "random", validate the preset_pool if provided
        if preset == "random" and preset_pool is not None:
            for pool_preset in preset_pool:
                if pool_preset not in presets:
                    available_presets = list(presets.keys())
                    typer.secho(
                        f"Error: Preset '{pool_preset}' in --preset-pool not found. Available presets are: {', '.join(available_presets)}",
                        fg=typer.colors.RED
                    )
                    raise typer.Exit(code=1)

    # Check if command name already exists
    bot_section = config.setdefault("bot", {})
    commands = bot_section.setdefault("commands", [])

    for cmd in commands:
        if cmd.get("name") == name:
            typer.secho(
                f"Error: Command '{name}' already exists. Choose a different name or edit config.yaml directly.",
                fg=typer.colors.RED
            )
            raise typer.Exit(code=1)

    # Build the command dict
    command = {
        "name": name,
        "description": description,
        "model_path": model_path if model_path else None,
        "stage": stage,
        "preset": preset
    }

    # Add preset_pool only if it's non-empty
    if preset_pool is not None and len(preset_pool) > 0:
        command["preset_pool"] = preset_pool

    # Append the new command
    commands.append(command)

    # Write back to file
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)

    typer.echo(f"Added command '{name}' to configuration")

    # Show a hint about testing the command
    if model_path:
        typer.echo(f"To test this command, run: nanocord infer interactive --model-path {model_path}")
    else:
        stage_hint = stage or "sft"
        typer.echo(f"To test this command, run: nanocord infer interactive --stage {stage_hint}")


@infer_app.command("interactive")
def infer_interactive(
    model_path: Optional[str] = typer.Option(
        None,
        "-m",
        "--model-path",
        help="Path to the model checkpoint directory (overrides stage)"
    ),
    stage: Optional[str] = typer.Option(
        None,
        "--stage",
        help="Training stage ('cpt' or 'sft') - used only when model-path is not provided"
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help="Name of the preset to use (from config) or 'random'"
    ),
    config_file: Optional[str] = typer.Option(
        str(CONFIG_PATH),
        "--config",
        help="Path to YAML configuration file (default: user data directory)"
    ),
):
    """
    Run interactive inference with the model
    """
    # Load and merge configuration for scalars (output_dir, channel_id, user_id)
    merged_config = load_and_merge_config(config_file, {}, "bot")

    # Load bot config section for presets
    bot_config = load_bot_config_section(config_file)

    # Validate that either model_path or stage is provided, but not both
    if model_path is None and stage is None:
        typer.secho(
            "Error: Either --model-path or --stage must be provided",
            fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    if model_path is not None and stage is not None:
        typer.secho(
            "Error: Cannot specify both --model-path and --stage",
            fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    # Resolve checkpoint path
    try:
        checkpoint_path = resolve_checkpoint_path(merged_config, model_path, stage)
    except ValueError as e:
        typer.secho(f"Error resolving checkpoint: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Get available presets
    presets = bot_config.get("presets", {})

    # Loop for interactive input
    while True:
        try:
            user_input = typer.prompt("You")
            if user_input.lower() in ("exit", "quit"):
                break

            # Resolve preset (or use default if not provided)
            if preset is None or preset == "":
                # Use a default preset if none specified
                if presets:
                    preset_name = list(presets.keys())[0]  # Use first preset as default
                    selected_preset = presets[preset_name]
                else:
                    selected_preset = {}
            else:
                try:
                    selected_preset = resolve_preset(presets, preset)
                except ValueError as e:
                    typer.secho(f"Error resolving preset: {e}", fg=typer.colors.RED)
                    raise typer.Exit(code=1)

            # Generate response
            response = generate_response(checkpoint_path, user_input, selected_preset)
            typer.echo(response)

        except KeyboardInterrupt:
            typer.echo("\nExiting interactive mode...")
            break
        except Exception as e:
            typer.secho(f"Error during inference: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1)


@infer_app.command("batch")
def infer_batch(
    model_path: Optional[str] = typer.Option(
        None,
        "-m",
        "--model-path",
        help="Path to the model checkpoint directory (overrides stage)"
    ),
    stage: Optional[str] = typer.Option(
        None,
        "--stage",
        help="Training stage ('cpt' or 'sft') - used only when model-path is not provided"
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help="Name of the preset to use (from config) or 'random'"
    ),
    config_file: Optional[str] = typer.Option(
        str(CONFIG_PATH),
        "--config",
        help="Path to YAML configuration file (default: user data directory)"
    ),
    prompt: str = typer.Option(
        ...,
        "--prompt",
        help="Prompt to use for all trials"
    ),
    trials: int = typer.Option(
        5,
        "--trials",
        help="Number of trials to run"
    ),
):
    """
    Run batch inference with the model
    """
    # Load and merge configuration for scalars (output_dir, channel_id, user_id)
    merged_config = load_and_merge_config(config_file, {}, "bot")

    # Load bot config section for presets
    bot_config = load_bot_config_section(config_file)

    # Validate that either model_path or stage is provided, but not both
    if model_path is None and stage is None:
        typer.secho(
            "Error: Either --model-path or --stage must be provided",
            fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    if model_path is not None and stage is not None:
        typer.secho(
            "Error: Cannot specify both --model-path and --stage",
            fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    # Resolve checkpoint path
    try:
        checkpoint_path = resolve_checkpoint_path(merged_config, model_path, stage)
    except ValueError as e:
        typer.secho(f"Error resolving checkpoint: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Get available presets
    presets = bot_config.get("presets", {})

    # Resolve preset once (if it's random, we'll resolve it each trial)
    if preset is not None and preset != "":
        try:
            resolved_preset = resolve_preset(presets, preset)
        except ValueError as e:
            typer.secho(f"Error resolving preset: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
    else:
        resolved_preset = None

    # Run trials
    for i in range(trials):
        try:
            # Resolve preset for this trial if it's "random"
            if preset == "random":
                trial_preset = resolve_preset(presets, "random")
            elif resolved_preset is not None:
                trial_preset = resolved_preset
            else:
                # Use default if no preset specified
                if presets:
                    preset_name = list(presets.keys())[0]  # Use first preset as default
                    trial_preset = presets[preset_name]
                else:
                    trial_preset = {}

            # Generate response
            response = generate_response(checkpoint_path, prompt, trial_preset)
            typer.echo(f"[{i+1}/{trials}] {response}")

        except Exception as e:
            typer.secho(f"Error during trial {i+1}: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1)


@app.command("pipeline")
def pipeline_run(
    skip_cpt_dataset: bool = typer.Option(False, "--skip-cpt-dataset", help="Skip building the CPT dataset"),
    skip_cpt_train: bool = typer.Option(False, "--skip-cpt-train", help="Skip training on the CPT dataset"),
    skip_sft_dataset: bool = typer.Option(False, "--skip-sft-dataset", help="Skip building the SFT dataset"),
    skip_sft_train: bool = typer.Option(False, "--skip-sft-train", help="Skip training on the SFT dataset"),
    skip_bot: bool = typer.Option(False, "--skip-bot", help="Skip bot registration"),
    config_file: Optional[str] = typer.Option(
        str(CONFIG_PATH),
        "--config",
        help="Path to YAML configuration file (default: user data directory)"
    ),
):
    """
    Run the complete pipeline from dataset creation to model serving
    """

    # Load full configuration once
    merged_config = load_and_merge_config(config_file, {}, "dataset.cpt")

    try:
        if not skip_cpt_dataset:
            typer.echo("Building CPT dataset...")
            # Call the existing build_cpt_dataset function (this is from dataset/cpt.py)
            build_cpt_dataset(
                channel_id=merged_config["channel_id"],
                user_id=merged_config["user_id"],
                bot_token=merged_config["discord_token"],
                discord_chat_exporter_path=merged_config["discord_chat_exporter_path"],
                thought_time=merged_config["thought_time"],
                thought_max=merged_config["thought_max"],
                thought_min=merged_config["thought_min"],
                max_entry_count=merged_config["max_entries"],
                offset=merged_config["offset"],
                distributed=merged_config["distributed"],
                reverse=merged_config["reverse"],
                redownload=merged_config["redownload"],
                output_dir=merged_config.get("output_dir")
            )
            typer.echo("CPT dataset built successfully")

        if not skip_cpt_train:
            typer.echo("Running CPT training...")
            # Load the CPT config section
            cpt_config = load_and_merge_config(config_file, {}, "train.cpt")
            run_cpt_training(cpt_config)
            typer.echo("CPT training completed successfully")

        if not skip_sft_dataset:
            typer.echo("Building SFT dataset...")
            # Load the SFT config section
            sft_config = load_and_merge_config(config_file, {}, "dataset.sft")
            # Load the context-specific overrides, if any - same fallback
            # behavior as the standalone `dataset sft` command: if
            # dataset.sft.context: isn't set, this resolves to the same
            # values already in sft_config.
            context_config = load_and_merge_config(config_file, {}, "dataset.sft.context")
            sft_config["context_thought_time"] = context_config["thought_time"]
            sft_config["context_thought_max"] = context_config["thought_max"]
            sft_config["context_thought_min"] = context_config["thought_min"]
            build_sft_dataset(sft_config)
            typer.echo("SFT dataset built successfully")

        if not skip_sft_train:
            typer.echo("Running SFT training...")
            # Load the SFT config section
            sft_train_config = load_and_merge_config(config_file, {}, "train.sft")
            run_sft_training(sft_train_config)
            typer.echo("SFT training completed successfully")

        if not skip_bot:
            typer.echo("Running bot...")
            # Load the bot config section
            bot_config = load_and_merge_config(config_file, {}, "bot")
            run_bot(bot_config)
            typer.echo("Bot run completed successfully")

    except (NotImplementedError, MissingPersonaNameError, ValueError) as e:
        typer.secho(f"Pipeline step failed: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@bot_app.command("run")
def bot_run(
    config_file: Optional[str] = typer.Option(
        str(CONFIG_PATH),
        "--config",
        help="Path to YAML configuration file (default: user data directory)"
    ),
    force_sync: bool = typer.Option(False, "--force-sync", help="Sync commands to Discord even if the command set hasn't changed"),
    guild_id: Optional[int] = typer.Option(None, "--guild", help="Discord guild (server) ID to sync commands to instead of globally — syncs instantly, useful while iterating"),
):
    """
    Run the Discord bot and serve the fine-tuned model
    """

    # Load and merge configuration - pass "bot" as the section to load
    merged_config = load_and_merge_config(config_file, {}, "bot")

    try:
        # Call the run_bot function
        run_bot(merged_config, force_sync, guild_id)
        typer.echo("Bot run completed successfully")
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@bot_app.command("sync")
def bot_sync(
    config_file: Optional[str] = typer.Option(
        str(CONFIG_PATH),
        "--config",
        help="Path to YAML configuration file (default: user data directory)"
    ),
    guild_id: Optional[int] = typer.Option(None, "--guild", help="Discord guild (server) ID to sync commands to instead of globally — syncs instantly, useful while iterating"),
    force: bool = typer.Option(False, "--force", help="Force sync even if command set hasn't changed"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show whether a sync would happen without performing it"),
):
    """
    Sync Discord bot commands to Discord
    """

    # Load and merge configuration - pass "bot" as the section to load
    merged_config = load_and_merge_config(config_file, {}, "bot")

    # Load bot configuration (presets and commands)
    from nanocord.bot.register import load_bot_config_section
    bot_config = load_bot_config_section(merged_config)
    commands = bot_config.get("commands", [])

    if not commands:
        typer.secho("Error: No bot commands registered. Run 'nanocord bot add-command' first.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Compute current fingerprint
    current_fingerprint = compute_command_fingerprint(commands)

    # Read cached fingerprint if it exists
    fingerprint_file = _fingerprint_path(merged_config)
    try:
        import json
        with open(fingerprint_file, 'r') as f:
            cached_data = json.load(f)
            cached_fingerprint = cached_data.get("fingerprint")
            cached_guild_id = cached_data.get("guild_id")
    except (FileNotFoundError, json.JSONDecodeError):
        cached_fingerprint = None
        cached_guild_id = None

    # Check if sync is needed
    should_sync = force or (cached_fingerprint != current_fingerprint)

    if dry_run:
        if should_sync:
            typer.echo("Sync would occur: command set has changed")
        else:
            typer.echo("Sync would be skipped: command set unchanged")
        raise typer.Exit(code=0)
    elif should_sync:
        # Build command tree to get the commands
        from nanocord.bot.register import build_command_tree
        tree = build_command_tree(commands, merged_config)

        # Create a minimal Discord client for syncing (without starting event loop)
        import discord
        intents = discord.Intents.default()
        client = discord.Client(intents=intents)
        tree._client = client

        try:
            # Sync commands with Discord - this will be run in an asyncio context by the function itself
            # We need to avoid importing asyncio here as it's already handled within run_bot
            import asyncio
            synced = asyncio.run(tree.sync(guild=discord.Object(id=guild_id) if guild_id else None))

            # Save new fingerprint
            sync_data = {
                "fingerprint": current_fingerprint,
                "guild_id": guild_id
            }
            with open(fingerprint_file, 'w') as f:
                json.dump(sync_data, f)

            typer.echo(f"Synced {len(synced)} commands to {'guild' if guild_id else 'global'}")
        except Exception as e:
            typer.secho(f"Error syncing commands: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
    else:
        typer.echo("Command set unchanged, skipping sync")


# Create a sub-app for config commands
config_app = typer.Typer(
    name="config",
    help="Manage configuration settings"
)
app.add_typer(config_app, name="config")


@config_app.callback(invoke_without_command=True)
def config_callback(ctx: typer.Context):
    """
    Manage configuration settings.

    If called without subcommands, display the current configuration.
    """
    if ctx.invoked_subcommand is None:
        if not CONFIG_PATH.exists():
            typer.echo("No configuration file found.")
            raise typer.Exit(code=1)

        with open(CONFIG_PATH, 'r') as f:
            config_content = f.read()

        typer.echo(config_content)


@config_app.command("set")
def config_set_cmd(
    key: str = typer.Argument(..., help="Configuration key (supports dot notation like 'dataset.channel_id')"),
    value: str = typer.Argument(..., help="Value to set for the configuration key")
):
    """
    Set a configuration value.

    Supports dot notation for nested keys (e.g., 'dataset.channel_id').
    Keys not already present in the YAML are created automatically.
    Valid top-level sections: channel_id, user_id, dataset (with cpt/sft subsections),
    train (with cpt/sft subsections), bot, output_dir.
    """
    # Load existing config
    if not CONFIG_PATH.exists():
        typer.echo("No configuration file found. Create one first with 'nanocord init'.")
        raise typer.Exit(code=1)

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f) or {}

    # Parse the key path (supporting dot notation)
    keys = key.split('.')

    # Navigate to the parent of the target key
    current = config
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    # Set the final key value
    try:
        # Try to parse as YAML to support numbers, booleans, etc.
        parsed_value = yaml.safe_load(value)
    except yaml.YAMLError:
        # If parsing fails, treat as string
        parsed_value = value

    current[keys[-1]] = parsed_value

    # Write back to file
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)

    typer.echo(f"Set {key} = {value}")


def nanocord():
    """Entry point function for the CLI."""
    app()

if __name__ == "__main__":
    nanocord()