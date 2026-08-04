"""
Bot registration and serving functions.
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional
import discord

from nanocord.paths import resolve_output_dir
from nanocord.infer import (
    resolve_checkpoint_path,
    resolve_preset,
    load_bot_config_section,
    generate_response
)
from nanocord import global_logger


def compute_command_fingerprint(commands: List[Dict]) -> str:
    """
    Deterministically serialize the list of command dicts and return its SHA-256 hex digest.

    Args:
        commands: List of command dictionaries

    Returns:
        SHA-256 hex digest of the serialized commands
    """
    # Sort keys in each command dict for deterministic serialization
    sorted_commands = []
    for cmd in commands:
        sorted_cmd = {k: cmd[k] for k in sorted(cmd.keys())}
        sorted_commands.append(sorted_cmd)

    # Serialize to JSON with sort_keys=True for deterministic output
    serialized = json.dumps(sorted_commands, sort_keys=True, separators=(',', ':'))

    # Return SHA-256 hash
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def _fingerprint_path(config: Dict) -> Path:
    """
    Get the path to the bot sync fingerprint file.

    Args:
        config: Configuration dictionary

    Returns:
        Path to the fingerprint file
    """
    return resolve_output_dir(config) / "bot_sync_fingerprint.json"


def build_command_tree(
    bot_commands_config: List[Dict],
    config: Dict
) -> "discord.app_commands.CommandTree":
    """
    Build a Discord command tree from bot commands configuration.

    Args:
        bot_commands_config: List of command configurations
        config: Full configuration dictionary

    Returns:
        discord.app_commands.CommandTree instance with registered commands
    """
    # Load presets from the config
    bot_config = load_bot_config_section(config)
    presets = bot_config.get("presets", {})

    # Create a new command tree (we'll register commands to it)
    tree = discord.app_commands.CommandTree(None)  # None as placeholder for client

    for cmd_config in bot_commands_config:
        # Extract command parameters
        name = cmd_config["name"]
        description = cmd_config["description"]

        # Create the command callback function
        async def command_callback(interaction: discord.Interaction, prompt: str = ""):
            try:
                # Resolve checkpoint path
                model_path = resolve_checkpoint_path(
                    config,
                    cmd_config.get("model_path"),
                    cmd_config.get("stage")
                )

                # Resolve preset
                preset_name = cmd_config.get("preset", "default")
                selected_preset = resolve_preset(presets, preset_name, cmd_config.get("preset_pool"))

                # Generate response
                response = generate_response(model_path, prompt, selected_preset)

                # Truncate to Discord's 2000-character limit
                if len(response) > 2000:
                    response = response[:1997] + "..."

                # Send the response
                await interaction.response.send_message(response)
            except Exception as e:
                # Log the full exception
                global_logger.exception("Error in bot command execution")

                # Send a user-friendly error message
                error_msg = "Sorry, I encountered an error processing your request. Please try again."
                await interaction.response.send_message(error_msg)

        # Register the command using tree.command() as a function call (not decorator)
        # This approach is used to avoid issues with decorators and allows for dynamic command registration
        @tree.command(
            name=name,
            description=description,
            options=[
                discord.app_commands.Option(
                    name="prompt",
                    description="Your message to the persona",
                    type=discord.AppCommandOptionType.string,
                    required=True
                )
            ]
        )
        async def _command_callback(interaction: discord.Interaction, prompt: str = ""):
            await command_callback(interaction, prompt)

    return tree


async def run_bot(
    config: Dict,
    force_sync: bool = False,
    guild_id: Optional[int] = None
) -> None:
    """
    Run the Discord bot with registered commands.

    Args:
        config: Merged configuration dictionary containing bot settings
        force_sync: Whether to force re-syncing of commands even if unchanged
        guild_id: Optional guild ID for guild-specific command registration
    """
    # Load bot configuration (presets and commands)
    bot_config = load_bot_config_section(config)
    commands = bot_config.get("commands", [])

    if not commands:
        raise ValueError("No bot commands registered. Run 'nanocord bot add-command' first.")

    # Create Discord client
    intents = discord.Intents.default()
    bot = discord.Client(intents=intents)

    # Build command tree
    tree = build_command_tree(commands, config)

    # Set the bot's command tree
    tree._client = bot

    @bot.event
    async def on_ready():
        # Compute current fingerprint
        current_fingerprint = compute_command_fingerprint(commands)

        # Read cached fingerprint if it exists
        fingerprint_file = _fingerprint_path(config)
        try:
            with open(fingerprint_file, 'r') as f:
                cached_data = json.load(f)
                cached_fingerprint = cached_data.get("fingerprint")
                cached_guild_id = cached_data.get("guild_id")
        except (FileNotFoundError, json.JSONDecodeError):
            cached_fingerprint = None
            cached_guild_id = None

        # Check if sync is needed
        should_sync = force_sync or (cached_fingerprint != current_fingerprint)

        if should_sync:
            try:
                # Sync commands with Discord
                synced = await tree.sync(guild=discord.Object(id=guild_id) if guild_id else None)

                # Save new fingerprint
                sync_data = {
                    "fingerprint": current_fingerprint,
                    "guild_id": guild_id
                }
                with open(fingerprint_file, 'w') as f:
                    json.dump(sync_data, f)

                global_logger.info(
                    f"Synced {len(synced)} commands to {'guild' if guild_id else 'global'}"
                )
            except Exception as e:
                global_logger.exception("Failed to sync commands with Discord")
                raise
        else:
            global_logger.info(
                "Command set unchanged, skipping sync"
            )

    # Run the bot with token fallback logic
    discord_token = config.get("discord_token") or os.getenv("DISCORD_BOT_TOKEN")
    if not discord_token:
        raise ValueError(
            "No Discord bot token provided. Please set it in config.yaml or "
            "as the DISCORD_BOT_TOKEN environment variable."
        )

    await bot.start(discord_token)