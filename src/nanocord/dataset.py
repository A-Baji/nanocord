import json
import os
import pathlib
from os import path

from dateutil import parser
from datetime import timedelta

from src.nanocord.paths import DISCORD_CHAT_EXPORTER_LOGS_PATH, DATASET_PATH
from src.nanocord.thoughts import (
    UserNotFoundError,
    validate_thought,
    cleanup_string,
    build_thought,
    add_to_dataset
)
from src.nanocord.discord_export import export_channel_logs


def parse_logs(
    file: str,
    channel: str,
    user: str,
    thought_time=5,
    thought_max: int = None,
    thought_min=6,
):
    """
    Parse Discord chat logs and create a dataset of thoughts.

    Args:
        file (str): Path to the Discord chat log JSON file
        channel (str): The ID of the Discord channel
        user (str): The unique username of the Discord user
        thought_time (int): Maximum time in seconds between messages to consider part of same thought
        thought_max (int): Maximum word count for a thought
        thought_min (int): Minimum word count for a thought

    Returns:
        pathlib.Path: Path to the created dataset file
    """

    files_path = DATASET_PATH
    user_id = user.split("#")[1] if "#" in user else None
    dataset_file_path = files_path / f"{user[:13]}_{channel[:4]}_data_set.jsonl"

    with open(file, "r", encoding="utf-8") as data_file:
        data = json.load(data_file)
        messages = [
            msg
            for msg in data["messages"]
            if msg["author"].get("name") == user
            and (user_id is None or msg["author"].get("discriminator") == user_id)
        ]
        if not messages:
            raise UserNotFoundError(
                f"No messages found in chat logs for user: {user}"
            )

        # Open dataset file for writing
        with open(dataset_file_path, "w", encoding="utf-8") as dataset:
            thought_max = 999999 if not thought_max else thought_max
            if "#" in user:
                username, user_id = user.split("#")
            else:
                username, user_id = user, None

            thought = build_thought("", messages[0])
            for i, msg in enumerate(messages[1::]):
                if msg["content"]:
                    prev_timestamp = parser.parse(messages[i]["timestamp"])
                    curr_timestamp = parser.parse(msg["timestamp"])
                    differentiation = (curr_timestamp - prev_timestamp) / timedelta(
                        milliseconds=1
                    )
                    if differentiation > thought_time * 1000:
                        # Validate and add the completed thought to dataset
                        if validate_thought(thought, thought_min, thought_max):
                            add_to_dataset(thought, dataset, user)
                        thought = build_thought("", msg)
                    else:
                        thought = build_thought(thought, msg)

            # Add the final thought
            if validate_thought(thought, thought_min, thought_max):
                add_to_dataset(thought, dataset, user)

    if path.getsize(dataset_file_path) == 0:
        print(
            "WARNING: The resulting dataset is empty. Please double check your parameters."
        )

    return dataset_file_path


def get_lines(file_name: str, N=1000, offset=0, distributed=False, reverse=False):
    """
    Select lines from a dataset file based on various criteria.

    Args:
        file_name (str): Path to the dataset file
        N (int): Maximum number of entries to select
        offset (int): Offset by line index starting at 0 for where to start selecting lines
        distributed (bool): Select lines as an even distribution instead of sequentially
        reverse (bool): Reverse the order in which to select lines

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


def create_export(
    channel_id: str,
    user_id: str,
    bot_token: str = os.getenv("DISCORD_BOT_TOKEN"),
    thought_time=10,
    thought_max: int = None,
    thought_min=4,
    max_entry_count=1000,
    offset=0,
    distributed=False,
    reverse=False,
    clean=False,
    redownload=False,
    use_existing=False,
):
    """
    Main function to orchestrate the export and dataset creation process.

    This function coordinates downloading Discord logs, parsing them into a dataset,
    and applying various filtering operations.
    """
    channel_user = f"{user_id[:13]}_{channel_id[:4]}"

    # Get log file path
    full_logs_path = DISCORD_CHAT_EXPORTER_LOGS_PATH / f"{channel_id}_logs.json"
    full_dataset_path = DATASET_PATH / f"{channel_user}_data_set.jsonl"

    if not full_dataset_path.exists() and use_existing:
        print("ERROR: No existing dataset could be found!")
        return

    # Download logs
    if (not full_logs_path.exists() or redownload) and not use_existing:
        print("INFO: Exporting chat logs using DiscordChatExporter...")
        try:
            export_channel_logs(channel_id, bot_token)
        except Exception as e:
            print(f"ERROR: Failed to export chat logs: {e}")
            raise
    elif full_logs_path.exists() and not redownload and not use_existing:
        print(
            f"INFO: Chat logs detected locally at {full_logs_path}... Skipping download."
        )

    # Parse logs
    if use_existing:
        print("INFO: Using existing dataset... Skipping download and parsing.")
    else:
        print("INFO: Parsing chat logs into a dataset...")
        try:
            parse_logs(
                full_logs_path,
                channel_id,
                user_id,
                thought_time,
                thought_max,
                thought_min,
            )
        except UserNotFoundError as e:
            print(f"ERROR: {e}")
            return
        get_lines(full_dataset_path, max_entry_count, offset, distributed, reverse)
        if not clean:
            print(f"INFO: Dataset saved to {full_dataset_path}")

    # Clean up generated files
    if clean and not use_existing:
        full_dataset_path.unlink()