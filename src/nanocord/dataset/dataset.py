from datetime import datetime
from datetime import timedelta
import json
from os import path

from nanocord import global_logger
from nanocord.dataset.discord_export import export_channel_logs
from nanocord.dataset.thoughts import add_to_dataset
from nanocord.dataset.thoughts import build_thought
from nanocord.dataset.thoughts import cleanup_string
from nanocord.dataset.thoughts import UserNotFoundError
from nanocord.dataset.thoughts import validate_thought
from nanocord.paths import DATASET_PATH
from nanocord.paths import DISCORD_CHAT_EXPORTER_LOGS_PATH

# Use the global logger
logger = global_logger


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
    dataset_file_path = files_path / f"{user}_{channel}_data_set.jsonl"

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

            thought = build_thought("", messages[0])
            for i, msg in enumerate(messages[1::]):
                if msg["content"]:
                    # Clean up the message content before processing
                    cleaned_content = cleanup_string(msg["content"])
                    msg["content"] = cleaned_content

                    prev_timestamp = datetime.fromisoformat(messages[i]["timestamp"])
                    curr_timestamp = datetime.fromisoformat(msg["timestamp"])
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
        logger.warning(
            "The resulting dataset is empty. Please double check your parameters."
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
    bot_token: str,
    thought_time=10,
    thought_max: int = None,
    thought_min=4,
    max_entry_count=1000,
    offset=0,
    distributed=False,
    reverse=False,
    redownload=False,
):
    """
    Main function to orchestrate the export and dataset creation process.

    This function coordinates downloading Discord logs, parsing them into a dataset,
    and applying various filtering operations.
    """
    channel_user = f"{user_id}_{channel_id}"

    # Get log file path
    full_logs_path = DISCORD_CHAT_EXPORTER_LOGS_PATH / f"{channel_id}_logs.json"
    full_dataset_path = DATASET_PATH / f"{channel_user}_data_set.jsonl"

    # Download logs
    if not full_logs_path.exists() or redownload:
        logger.info("Exporting chat logs using DiscordChatExporter...")
        try:
            export_channel_logs(channel_id, bot_token)
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
        )
    except UserNotFoundError as e:
        logger.error(f"{e}")
        return
    get_lines(full_dataset_path, max_entry_count, offset, distributed, reverse)
    logger.info(f"Dataset saved to {full_dataset_path}")