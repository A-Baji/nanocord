"""
Dataset creation functions for SFT (Supervised Fine-Tuning) datasets.
"""

import json
from os import path
from pathlib import Path
from typing import Dict
from typing import Optional

from nanocord import global_logger
from nanocord.dataset.cpt import get_lines
from nanocord.dataset.discord_export import export_channel_logs
from nanocord.dataset.thoughts import group_into_thoughts
from nanocord.dataset.thoughts import UserNotFoundError
from nanocord.dataset.thoughts import validate_thought
from nanocord.paths import resolve_output_dir

# Use the global logger
logger = global_logger

DEFAULT_SYSTEM_PROMPT = (
    "You are {persona_name}. Respond the way {persona_name} would, in their "
    "own voice, tone, and personality, based on their own real Discord "
    "messages."
)


class MissingPersonaNameError(Exception):
    pass


def resolve_system_prompt(persona_name: Optional[str], system_prompt_template: Optional[str] = None) -> str:
    """
    Validate that a persona_name is present and format a system prompt
    template with it.

    Args:
        persona_name: The target user's persona name, embedded into the
                      system prompt. Required.
        system_prompt_template: An f-string-style template containing a
                                 {persona_name} placeholder. Defaults to
                                 DEFAULT_SYSTEM_PROMPT if not provided or
                                 falsy.

    Returns:
        The formatted system prompt string.

    Raises:
        MissingPersonaNameError: If persona_name is not provided.
        ValueError: If the template contains a placeholder other than
                    {persona_name}.
    """
    if not persona_name:
        raise MissingPersonaNameError(
            "dataset.sft.persona_name is required and was not provided. "
            "Set it in config.yaml under dataset.sft.persona_name, or pass "
            "--persona-name."
        )

    template = system_prompt_template or DEFAULT_SYSTEM_PROMPT
    try:
        return template.format(persona_name=persona_name)
    except (KeyError, IndexError) as e:
        raise ValueError(
            f"system_prompt template contains an unrecognized placeholder: "
            f"{e}. Only {{persona_name}} is supported."
        ) from e


def parse_sft_logs(
    file: str,
    channel: str,
    user: str,
    thought_time: int = 5,
    thought_max: Optional[int] = None,
    thought_min: int = 6,
    system_prompt: str = "",
    context_thought_time: Optional[int] = None,
    context_thought_max: Optional[int] = None,
    context_thought_min: Optional[int] = None,
) -> Path:
    """
    Parse Discord chat logs and create an SFT dataset of (context thought ->
    response thought) pairs, where the response thought is a target-user
    thought that starts with a reply, and the context thought is whatever
    thought (from any other author) contains the message being replied to.

    Args:
        file: Path to the Discord chat log JSON file
        channel: The ID of the Discord channel
        user: The unique user ID of the target Discord user
        thought_time: Maximum time in seconds between messages to consider
                      part of the same thought, for the response (target-user)
                      side
        thought_max: Maximum word count for a thought, for the response side
        thought_min: Minimum word count for a thought, for the response side
        system_prompt: System prompt content embedded in every output record
        context_thought_time: Same as thought_time but for the context side.
                               Defaults to thought_time if not given.
        context_thought_max: Same as thought_max but for the context side.
                              Defaults to thought_max if not given.
        context_thought_min: Same as thought_min but for the context side.
                              Defaults to thought_min if not given.

    Returns:
        Path: Path to the created dataset file

    Raises:
        UserNotFoundError: If no messages are found for the specified user
    """
    context_thought_time = thought_time if context_thought_time is None else context_thought_time
    context_thought_max = thought_max if context_thought_max is None else context_thought_max
    context_thought_min = thought_min if context_thought_min is None else context_thought_min

    base = resolve_output_dir(config)
    dataset_path = base / "processed"
    log_dir = base / "raw" / "discordchat_export"

    # Create directories if they don't exist
    dataset_path.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    files_path = dataset_path
    dataset_file_path = files_path / f"{user}_{channel}_sft_data_set.jsonl"

    with open(file, "r", encoding="utf-8") as data_file:
        data = json.load(data_file)
        all_messages = data["messages"]

        response_messages = [
            msg for msg in all_messages if msg["author"].get("id") == user
        ]
        if not response_messages:
            raise UserNotFoundError(
                f"No messages found in chat logs for user: {user}"
            )

        context_messages = [
            msg for msg in all_messages if msg["author"].get("id") != user
        ]

        thought_max = 999999 if not thought_max else thought_max
        context_thought_max = 999999 if not context_thought_max else context_thought_max

        context_thoughts = group_into_thoughts(context_messages, context_thought_time)
        message_id_to_context_thought = {}
        for thought in context_thoughts:
            for message_id in thought["message_ids"]:
                message_id_to_context_thought[message_id] = thought

        response_thoughts = group_into_thoughts(response_messages, thought_time)
        reply_thoughts = [
            thought for thought in response_thoughts
            if thought["reply_reference_id"] is not None
        ]

        seen_pairs = set()
        with open(dataset_file_path, "w", encoding="utf-8") as dataset:
            for reply_thought in reply_thoughts:
                context_thought = message_id_to_context_thought.get(
                    reply_thought["reply_reference_id"]
                )
                if context_thought is None:
                    continue

                context_text = context_thought["text"]
                response_text = reply_thought["text"]

                if not validate_thought(context_text, context_thought_min, context_thought_max):
                    continue
                if not validate_thought(response_text, thought_min, thought_max):
                    continue

                pair_key = (context_text, response_text)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                entry = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context_text},
                        {"role": "assistant", "content": response_text},
                    ]
                }
                dataset.write(json.dumps(entry) + "\n")

    if path.getsize(dataset_file_path) == 0:
        logger.warning(
            "The resulting dataset is empty. Please double check your parameters."
        )

    return dataset_file_path


def build_sft_dataset(config: Dict) -> Path:
    """
    Main function to orchestrate the export and dataset creation process for SFT.

    Args:
        config: Merged configuration dictionary containing dataset settings.
                Expected keys: channel_id, user_id, discord_token,
                discord_chat_exporter_path, thought_time, thought_max,
                thought_min, system_prompt, max_entries, offset, distributed,
                reverse, redownload.

    Returns:
        Path: Path to the created SFT dataset file

    Raises:
        MissingPersonaNameError: If config["persona_name"] is not provided.
        ValueError: If config["system_prompt"] contains a placeholder other
                    than {persona_name}.
    """
    channel_id = config.get("channel_id")
    user_id = config.get("user_id")
    bot_token = config.get("discord_token")
    discord_chat_exporter_path = config.get("discord_chat_exporter_path")
    thought_time = config.get("thought_time", 5)
    thought_max = config.get("thought_max")
    thought_min = config.get("thought_min", 6)

    persona_name = config.get("persona_name")
    system_prompt = resolve_system_prompt(persona_name, config.get("system_prompt"))

    context_thought_time = config.get("context_thought_time", thought_time)
    context_thought_max = config.get("context_thought_max", thought_max)
    context_thought_min = config.get("context_thought_min", thought_min)
    max_entry_count = config.get("max_entries", 1000)
    offset = config.get("offset", 0)
    distributed = config.get("distributed", False)
    reverse = config.get("reverse", False)
    redownload = config.get("redownload", False)

    channel_user = f"{user_id}_{channel_id}"

    base = resolve_output_dir(config)
    dataset_path = base / "processed"
    log_dir = base / "raw" / "discordchat_export"

    # Create directories if they don't exist
    dataset_path.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    full_logs_path = log_dir / f"{channel_id}_logs.json"
    full_dataset_path = dataset_path / f"{channel_user}_sft_data_set.jsonl"

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

    logger.info("Parsing chat logs into an SFT dataset...")
    try:
        parse_sft_logs(
            full_logs_path,
            channel_id,
            user_id,
            thought_time,
            thought_max,
            thought_min,
            system_prompt,
            context_thought_time,
            context_thought_max,
            context_thought_min,
        )
    except UserNotFoundError as e:
        logger.error(f"{e}")
        raise

    get_lines(full_dataset_path, max_entry_count, offset, distributed, reverse)
    logger.info(f"Dataset saved to {full_dataset_path}")

    return full_dataset_path