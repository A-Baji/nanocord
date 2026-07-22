import os
import pathlib
import platform
import shutil
import subprocess
import sys

from nanocord.logger import setup_logger
from nanocord.paths import DISCORD_CHAT_EXPORTER_LOGS_PATH

# Setup logger
logger = setup_logger('nanocord.discord_export', 'logs/discord_export.log')

DEFAULT_DCE_PATH_ENV_VAR = "DISCORD_CHAT_EXPORTER_PATH"


def resolve_discord_chat_exporter_path(prompt_for_path=False):
    configured_path = os.getenv(DEFAULT_DCE_PATH_ENV_VAR)
    if configured_path:
        candidate = pathlib.Path(configured_path).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Configured DiscordChatExporter path does not exist or is not executable: {candidate}"
        )

    os_name = platform.system()
    candidate_paths = []
    if os_name == "Linux":
        candidate_paths = [
            pathlib.Path("/usr/bin/DiscordChatExporter.Cli"),
            pathlib.Path("/usr/local/bin/DiscordChatExporter.Cli"),
            pathlib.Path("DiscordChatExporter.Cli.linux-x64/DiscordChatExporter.Cli"),
        ]
    elif os_name == "Darwin":
        candidate_paths = [
            pathlib.Path("/Applications/DiscordChatExporter/DiscordChatExporter.Cli"),
            pathlib.Path("DiscordChatExporter.Cli.osx-x64/DiscordChatExporter.Cli"),
        ]
    elif os_name == "Windows":
        candidate_paths = [
            pathlib.Path(r"C:\Program Files\DiscordChatExporter\DiscordChatExporter.Cli.exe"),
            pathlib.Path(r"C:\Program Files (x86)\DiscordChatExporter\DiscordChatExporter.Cli.exe"),
            pathlib.Path("DiscordChatExporter.Cli.win-x64/DiscordChatExporter.Cli.exe"),
        ]
    else:
        raise RuntimeError(f"Unsupported platform for DiscordChatExporter: {os_name}")

    for candidate in candidate_paths:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
        if candidate.exists():
            return candidate

    if prompt_for_path and hasattr(sys, "stdin") and sys.stdin.isatty():
        logger.info(
            "DiscordChatExporter is a required prerequisite. Install it from https://github.com/Tyrrrz/DiscordChatExporter/releases and enter the full path to the executable."
        )
        raw_path = input("DiscordChatExporter executable path: ").strip()
        if raw_path:
            candidate = pathlib.Path(raw_path).expanduser()
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        "DiscordChatExporter was not found. Install it from https://github.com/Tyrrrz/DiscordChatExporter/releases and set DISCORD_CHAT_EXPORTER_PATH to the executable path."
    )


def export_channel_logs(channel_id: str, bot_token: str):
    """
    Export Discord channel logs using DiscordChatExporter.

    Args:
        channel_id (str): The ID of the Discord channel to export
        bot_token (str): The Discord bot token

    Returns:
        pathlib.Path: Path to the exported log file
    """
    logger.info("Exporting chat logs using DiscordChatExporter...")
    logger.info(
        "This may take a few minutes to hours depending on the message count of the channel"
    )
    logger.info("Progress will NOT be saved if cancelled")
    logger.info(
        "--------------------------DiscordChatExporter---------------------------"
    )

    try:
        DiscordChatExporter = resolve_discord_chat_exporter_path(
            prompt_for_path=True
        )
    except FileNotFoundError as exc:
        logger.error(f"{exc}")
        raise RuntimeError(str(exc)) from exc

    # Generate the log filename
    log_filename = f"{channel_id}_logs.json"
    full_log_path = DISCORD_CHAT_EXPORTER_LOGS_PATH / log_filename

    subprocess.run(
        [
            str(DiscordChatExporter),
            "export",
            "-c",
            channel_id,
            "-t",
            bot_token or "",
            "-o",
            log_filename,
            "-f",
            "Json",
            "--fuck-russia",
            "True",
        ]
    )

    logger.info(
        "--------------------------DiscordChatExporter---------------------------"
    )

    # Move the file to our expected location
    shutil.move(log_filename, full_log_path)
    logger.info(f"Logs saved to {full_log_path}")

    return full_log_path