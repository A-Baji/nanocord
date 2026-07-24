import os
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml

from nanocord import global_logger
from nanocord.config import load_and_merge_config
from nanocord.dataset.dataset import create_export
from nanocord.paths import CONFIG_PATH

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
    and prompts for the Discord bot token.
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

    # Create default config content
    default_config = f"""# NanoCord Configuration File
# This file contains default settings for the nanocord CLI tool.
# Values can be overridden by command-line arguments.

dataset:
  channel_id: ""
  user_id: ""
  discord_token: "{discord_token}" # Optional here, falls back to DISCORD_BOT_TOKEN env var if left empty
  thought_time: 5
  thought_min: 6
  thought_max: null
  max_entry_count: 1000
  offset: 0
  distributed: false
  reverse: false
  redownload: false
"""

    # Write the config file
    with open(CONFIG_PATH, 'w') as f:
        f.write(default_config)

    typer.echo(f"Configuration file created at: {CONFIG_PATH}")
    if discord_token:
        typer.echo("Discord bot token saved to configuration.")
    else:
        typer.echo("No Discord bot token provided. You can set it later via:")
        typer.echo("  - Environment variable: DISCORD_BOT_TOKEN")
        typer.echo("  - Configuration file: config.yaml")


@dataset_app.command("create")
def dataset_create(
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
    thought_time: int = typer.Option(
        5,
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
    thought_min: int = typer.Option(
        6,
        "--tmin",
        "--thought-min",
        help="The minimum length in words of each thought"
    ),
    max_entries: int = typer.Option(
        1000,
        "-m",
        "--max-entries",
        help="The max amount of entries (by lines) that may exist in the dataset"
    ),
    offset: int = typer.Option(
        0,
        "--os",
        "--offset",
        help="The offset by line index starting at 0 for where to start selecting lines for the dataset"
    ),
    distributed: bool = typer.Option(
        False,
        "--distributed",
        help="Select lines as an even distribution instead of sequentially"
    ),
    reverse_lines: bool = typer.Option(
        False,
        "--reverse-lines",
        help="Reverse the order in which to select lines for the dataset"
    ),
    redownload: bool = typer.Option(
        False,
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
    Download Discord channel logs, parse them into a dataset, and save the results locally
    """

    # Prepare CLI arguments for merging
    cli_args = {
        "discord_token": discord_token,
        "channel_id": channel_id,
        "user_id": user_id,
        "thought_time": thought_time,
        "thought_max": thought_max,
        "thought_min": thought_min,
        "max_entries": max_entries,
        "offset": offset,
        "distributed": distributed,
        "reverse": reverse_lines,
        "redownload": redownload
    }

    # Load and merge configuration
    merged_config = load_and_merge_config(config_file, cli_args)

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

    # Unpack the merged configuration into create_export function
    create_export(
        channel_id=merged_config["channel_id"],
        user_id=merged_config["user_id"],
        bot_token=merged_config["discord_token"],
        thought_time=merged_config["thought_time"],
        thought_max=merged_config["thought_max"],
        thought_min=merged_config["thought_min"],
        max_entry_count=merged_config["max_entries"],
        offset=merged_config["offset"],
        distributed=merged_config["distributed"],
        reverse=merged_config["reverse"],
        redownload=merged_config["redownload"]
    )


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