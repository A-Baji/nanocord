"""
Bot registration and serving functions.

This module handles the registration of Discord slash commands that use a persona model.
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import discord

from nanocord.paths import resolve_output_dir
from nanocord.config import load_and_merge_config
from nanocord.infer import (
    resolve_checkpoint_path,
    resolve_preset,
    load_bot_config_section,
    load_model,
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


def has_command_set_changed(commands: List[Dict], config: Dict, force: bool = False) -> bool:
    """
    Check if the command set has changed compared to the cached fingerprint.

    Args:
        commands: List of command dictionaries
        config: Configuration dictionary
        force: Whether to force sync regardless of changes

    Returns:
        True if the command set has changed or force is True, False otherwise
    """
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
    return force or (cached_fingerprint != current_fingerprint)


async def sync_if_needed(
    tree: "discord.app_commands.CommandTree",
    commands: List[Dict],
    config: Dict,
    force_sync: bool = False,
    guild_id: Optional[int] = None
) -> bool:
    """
    Sync Discord commands if the command set has changed.

    Args:
        tree: The command tree to sync
        commands: List of command dictionaries
        config: Configuration dictionary
        force_sync: Whether to force re-syncing of commands even if unchanged
        guild_id: Optional guild ID for guild-specific command registration

    Returns:
        True if a sync occurred, False if skipped
    """
    # Check if sync is needed using the shared helper function
    should_sync = has_command_set_changed(commands, config, force_sync)

    if should_sync:
        try:
            # When guild_id is provided, we need to copy global commands to the guild first
            # since CommandTree.sync() for a specific guild only syncs commands explicitly
            # present in that guild's local command list, not global commands
            if guild_id:
                await tree.copy_global_to(guild=discord.Object(id=guild_id))

            # Sync commands with Discord - this now works without application_id
            synced = await tree.sync(guild=discord.Object(id=guild_id) if guild_id else None)

            # Save new fingerprint
            current_fingerprint = compute_command_fingerprint(commands)
            sync_data = {
                "fingerprint": current_fingerprint,
                "guild_id": guild_id
            }
            with open(_fingerprint_path(config), 'w') as f:
                json.dump(sync_data, f)

            global_logger.info(
                f"Synced {len(synced)} commands to {'guild' if guild_id else 'global'}"
            )
            return True
        except Exception as e:
            global_logger.exception("Failed to sync commands with Discord")
            raise
    else:
        global_logger.info(
            "Command set unchanged, skipping sync"
        )
        return False


def build_command_tree(
    bot_commands_config: List[Dict],
    presets: Dict,
    config: Dict,
    client: "discord.Client"
) -> "discord.app_commands.CommandTree":
    """
    Build a Discord command tree from bot commands configuration.

    Args:
        bot_commands_config: List of command configurations
        presets: Preset configurations loaded from the config file
        config: Full configuration dictionary
        client: The discord.Client instance to associate with the command tree

    Returns:
        discord.app_commands.CommandTree instance with registered commands
    """
    # Create a new command tree with the client directly
    tree = discord.app_commands.CommandTree(client)

    # Cache for loaded models to avoid reloading the same model multiple times
    # This is important because we may have multiple commands pointing to the same checkpoint
    # and we don't want to load the same model multiple times
    model_cache: Dict[str, Tuple[any, any]] = {}

    def _make_command_callback(cmd_config: Dict, presets: Dict):
        """
        Factory function to create command callback functions with proper type annotations.

        Args:
            cmd_config: Command configuration dictionary
            presets: Preset configurations

        Returns:
            Async callback function for the Discord command
        """
        async def callback(interaction: discord.Interaction, prompt: str):
            # First defer the interaction to extend response window from 3 seconds to ~15 minutes
            await interaction.response.defer()

            try:
                # Resolve checkpoint path
                model_path = resolve_checkpoint_path(
                    config,
                    cmd_config.get("model_path"),
                    cmd_config.get("stage")
                )

                # Convert to string for use as cache key
                model_path_str = str(model_path)

                # Load model if not already cached
                if model_path_str not in model_cache:
                    model, tokenizer = load_model(model_path)
                    model_cache[model_path_str] = (model, tokenizer)
                else:
                    model, tokenizer = model_cache[model_path_str]

                # Resolve preset
                preset_name = cmd_config.get("preset", "default")
                selected_preset = resolve_preset(presets, preset_name, cmd_config.get("preset_pool"))

                # Prepare system prompt for SFT models
                system_prompt = None
                if cmd_config.get("stage") == "sft":
                    from nanocord.dataset.sft import resolve_system_prompt
                    try:
                        sft_config = load_and_merge_config(config_file, {}, "dataset.sft")
                        persona_name = sft_config.get("persona_name")
                        system_prompt_template = sft_config.get("system_prompt")
                        system_prompt = resolve_system_prompt(persona_name, system_prompt_template)
                    except Exception as e:
                        global_logger.warning(f"Could not resolve system prompt for SFT model: {e}")

                # Generate response
                response = generate_response(model, tokenizer, prompt, selected_preset, system_prompt)

                # Truncate to Discord's 2000-character limit
                if len(response) > 2000:
                    response = response[:1997] + "..."

                # Send the response using followup.send instead of send_message after deferring
                await interaction.followup.send(response)
            except Exception as e:
                # Log the full exception
                global_logger.exception("Error in bot command execution")

                # Send a user-friendly error message using followup.send
                error_msg = "Sorry, I encountered an error processing your request. Please try again."
                await interaction.followup.send(error_msg)

        return callback

    for cmd_config in bot_commands_config:
        # Extract command parameters
        name = cmd_config["name"]
        description = cmd_config["description"]

        # Create the command callback function with proper closure capture using factory
        callback = _make_command_callback(cmd_config, presets)
        tree.command(name=name, description=description)(callback)

    return tree


async def run_bot(
    config: Dict,
    presets: Dict,
    commands: List[Dict],
    force_sync: bool = False,
    guild_id: Optional[int] = None
) -> None:
    """
    Run the Discord bot with registered commands.

    Args:
        config: Merged configuration dictionary containing bot settings
        presets: Preset configurations loaded from the config file
        commands: List of command configurations
        force_sync: Whether to force re-syncing of commands even if unchanged
        guild_id: Optional guild ID for guild-specific command registration
    """
    # Create Discord client - no application_id needed anymore
    intents = discord.Intents.default()
    bot = discord.Client(intents=intents)

    # Build command tree
    tree = build_command_tree(commands, presets, config, bot)

    @bot.event
    async def on_ready():
        await sync_if_needed(tree, commands, config, force_sync, guild_id)

    # Run the bot with token fallback logic
    discord_token = config.get("discord_token") or os.getenv("DISCORD_BOT_TOKEN")
    if not discord_token:
        raise ValueError(
            "No Discord bot token provided. Please set it in config.yaml or "
            "as the DISCORD_BOT_TOKEN environment variable."
        )

    await bot.start(discord_token)