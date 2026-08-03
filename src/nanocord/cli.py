import os
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml

from nanocord import global_logger
from nanocord.bot.register import register_bot
from nanocord.config import load_and_merge_config
from nanocord.dataset.cpt import build_cpt_dataset
from nanocord.dataset.sft import build_sft_dataset
from nanocord.dataset.sft import DEFAULT_SYSTEM_PROMPT
from nanocord.dataset.sft import MissingPersonaNameError
from nanocord.paths import CONFIG_PATH
from nanocord.train.cpt import run_cpt_training
from nanocord.train.sft import run_sft_training

# Use the global logger
logger = global_logger

# Create Typer application
app = typer.Typer(
    name="nanocord",
    help="Discord chat export and dataset preparation CLI",
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
    discord_token: "{discord_token}" # Optional here, falls back to DISCORD_BOT_TOKEN env var if left empty
    discord_chat_exporter_path: "{discord_chat_exporter_path}" # Path to DiscordChatExporter.Cli executable
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
    # Placeholder for bot config
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
        help="Select lines as an even distribution instead of sequentially"
    ),
    reverse_lines: Optional[bool] = typer.Option(
        None,
        "--reverse-lines",
        help="Reverse the order in which to select lines for the dataset"
    ),
    redownload: Optional[bool] = typer.Option(
        None,
        "--redownload",
        help="Redownload the Discord chat logs"
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
        help="Select lines as an even distribution instead of sequentially"
    ),
    reverse_lines: Optional[bool] = typer.Option(
        None,
        "--reverse-lines",
        help="Reverse the order in which to select lines for the dataset"
    ),
    redownload: Optional[bool] = typer.Option(
        None,
        "--redownload",
        help="Redownload the Discord chat logs"
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
):
    """
    Run CPT training on the dataset
    """

    # Load and merge configuration - pass "train.cpt" as the section to load
    merged_config = load_and_merge_config(config_file, {}, "train.cpt")

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
):
    """
    Run SFT training on the CPT checkpoint
    """

    # Load and merge configuration - pass "train.sft" as the section to load
    merged_config = load_and_merge_config(config_file, {}, "train.sft")

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


@bot_app.command("register")
def bot_register(
    config_file: Optional[str] = typer.Option(
        str(CONFIG_PATH),
        "--config",
        help="Path to YAML configuration file (default: user data directory)"
    ),
):
    """
    Register the Discord bot and serve the fine-tuned model
    """

    # Load and merge configuration - pass "bot" as the section to load
    merged_config = load_and_merge_config(config_file, {}, "bot")

    try:
        # Call the register_bot function
        register_bot(merged_config)
        typer.echo("Bot registration completed successfully")
    except NotImplementedError:
        typer.secho("Error: Bot registration not yet implemented", fg=typer.colors.RED)
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
            typer.echo("Registering bot...")
            # Load the bot config section
            bot_config = load_and_merge_config(config_file, {}, "bot")
            register_bot(bot_config)
            typer.echo("Bot registration completed successfully")

    except (NotImplementedError, MissingPersonaNameError, ValueError) as e:
        typer.secho(f"Pipeline step failed: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


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