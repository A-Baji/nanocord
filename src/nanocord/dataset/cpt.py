"""
Dataset creation functions for CPT (Continued Pre-Training) datasets.
"""

import json
from os import path
from pathlib import Path
from string import punctuation
from typing import Optional

from nanocord import global_logger
from nanocord.dataset.discord_export import export_channel_logs
from nanocord.dataset.thoughts import group_into_thoughts
from nanocord.dataset.thoughts import UserNotFoundError
from nanocord.dataset.thoughts import validate_thought
from nanocord.paths import resolve_output_dir

# Use the global logger
logger = global_logger

def add_to_dataset(thought: str, dataset_file):
    """
    Validate a thought, create a dataset JSON entry, and then add it to the dataset
    """
    if thought[-1] not in punctuation:
            thought += "."
    entry = {"text": thought}
    dataset_file.write(json.dumps(entry) + "\n")

def parse_logs(
    file: str,
    channel: str,
    user: str,
    thought_time: int = 5,
    thought_max: Optional[int] = None,
    thought_min: int = 6,
    output_dir: Optional[str] = None,
) -> Path:
    """
    Parse Discord chat logs and create a dataset of thoughts.

    Args:
        file: Path to the Discord chat log JSON file
        channel: The ID of the Discord channel
        user: The unique username of the Discord user
        thought_time: Maximum time in seconds between messages to consider part of same thought
        thought_max: Maximum word count for a thought
        thought_min: Minimum word count for a thought

    Returns:
        Path: Path to the created dataset file

    Raises:
        UserNotFoundError: If no messages are found for the specified user
    """

    base = resolve_output_dir({"output_dir": output_dir})
    dataset_path = base / "processed"
    log_dir = base / "raw" / "discordchat_export"

    # Create directories if they don't exist
    dataset_path.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    files_path = dataset_path
    dataset_file_path = files_path / f"{user}_{channel}_cpt_data_set.jsonl"

    with open(file, "r", encoding="utf-8") as data_file:
        data = json.load(data_file)
        messages = [
            msg
            for msg in data["messages"]
            if msg["author"].get("id") == user
        ]
        if not messages:
            raise UserNotFoundError(
                f"No messages found in chat logs for user: {user}"
            )

        # Open dataset file for writing
        with open(dataset_file_path, "w", encoding="utf-8") as dataset:
            thought_max = 999999 if not thought_max else thought_max

            for thought in group_into_thoughts(messages, thought_time):
                if validate_thought(thought["text"], thought_min, thought_max):
                    add_to_dataset(thought["text"], dataset)

    if path.getsize(dataset_file_path) == 0:
        logger.warning(
            "The resulting dataset is empty. Please double check your parameters."
        )

    return dataset_file_path


def get_lines(file_name: str, N: int = 1000, offset: int = 0, distributed: bool = False, reverse: bool = False) -> None:
    """
    Select lines from a dataset file based on various criteria.

    Args:
        file_name: Path to the dataset file
        N: Maximum number of entries to select
        offset: Offset by line index starting at 0 for where to start selecting lines
        distributed: Select lines as an even distribution instead of sequentially
        reverse: Reverse the order in which to select lines

    Returns:
        None: Modifies the file in place
    """
    with open(file_name, "r") as f:
        lines = f.readlines()
    f.close()

    num_lines = len(lines)

    if distributed:
        step = (num_lines - offset) // N
    else:
        step = 1

    if reverse:
        lines = lines[::-1]

    selected_lines = lines[offset:][:: step or 1][:N]

    with open(file_name, "w") as f:
        f.writelines(selected_lines)
    f.close()


def build_cpt_dataset(
    channel_id: str,
    user_id: str,
    bot_token: str,
    discord_chat_exporter_path: Optional[str] = None,
    thought_time: int = 10,
    thought_max: Optional[int] = None,
    thought_min: int = 4,
    max_entry_count: int = 1000,
    offset: int = 0,
    distributed: bool = False,
    reverse: bool = False,
    redownload: bool = False,
    output_dir: Optional[str] = None,
) -> Path:
    """
    Main function to orchestrate the export and dataset creation process for CPT.

    This function coordinates downloading Discord logs, parsing them into a dataset,
    and applying various filtering operations.

    Args:
        channel_id: The ID of the Discord channel you want to use
        user_id: The ID of the Discord user you want to use
        bot_token: The Discord token for your bot. Must either be provided as an argument
                   or set as the DISCORD_BOT_TOKEN environment variable
        thought_time: The maximum amount of time in seconds to consider two individual
                      messages to be part of the same "thought"
        thought_max: The maximum length in words of each thought
        thought_min: The minimum length in words of each thought
        max_entry_count: The max amount of entries (by lines) that may exist in the dataset
        offset: The offset by line index starting at 0 for where to start selecting lines
                for the dataset
        distributed: Select lines as an even distribution instead of sequentially
        reverse: Reverse the order in which to select lines for the dataset
        redownload: Redownload the Discord chat logs
        output_dir: Optional - overrides the base directory for all generated output

    Returns:
        Path: Path to the created dataset file
    """
    channel_user = f"{user_id}_{channel_id}"

    # Get log file path
    base = resolve_output_dir({"output_dir": output_dir})
    dataset_path = base / "processed"
    log_dir = base / "raw" / "discordchat_export"

    # Create directories if they don't exist
    dataset_path.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    full_logs_path = log_dir / f"{channel_id}_logs.json"
    full_dataset_path = dataset_path / f"{channel_user}_cpt_data_set.jsonl"

    # Download logs
    if not full_logs_path.exists() or redownload:
        logger.info("Exporting chat logs using DiscordChatExporter...")
        try:
            export_channel_logs(channel_id, bot_token, discord_chat_exporter_path)
        except Exception as e:
            logger.error(f"Failed to export chat logs: {e}")
            raise
    elif full_logs_path.exists() and not redownload:
        logger.info(
            f"Chat logs detected locally at {full_logs_path}... Skipping download."
        )

    # Parse logs
    logger.info("Parsing chat logs into a dataset...")
    try:
        parse_logs(
            full_logs_path,
            channel_id,
            user_id,
            thought_time,
            thought_max,
            thought_min,
            output_dir=output_dir,
        )
    except UserNotFoundError as e:
        logger.error(f"{e}")
        raise  # Re-raise to signal failure
    get_lines(full_dataset_path, max_entry_count, offset, distributed, reverse)
    logger.info(f"Dataset saved to {full_dataset_path}")

    return full_dataset_path