import os
import pathlib
import platform
import shutil
import subprocess
import sys

import appdirs

from src.gen_dataset import get_lines
from src.gen_dataset import parse_logs
from src.gen_dataset import UserNotFoundError

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
        print(
            "INFO: DiscordChatExporter is a required prerequisite. Install it from https://github.com/Tyrrrz/DiscordChatExporter/releases and enter the full path to the executable."
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


def create_model(
    channel_id: str,
    user_id: str,
    bot_token: str = os.getenv("DISCORD_BOT_TOKEN"),
    thought_time=10,
    thought_max: int = None,
    thought_min=4,
    max_entry_count=1000,
    offset=0,
    distributed=False,
    base_model="none",
    reverse=False,
    clean=False,
    redownload=False,
    use_existing=False,
):
    channel_user = f"{user_id[:13]}_{channel_id[:4]}"
    files_path = pathlib.Path(appdirs.user_data_dir(appname="discordai"))
    full_logs_path = files_path / f"{channel_id}_logs.json"
    full_dataset_path = files_path / f"{channel_user}_data_set.jsonl"

    if not os.path.isfile(full_dataset_path) and use_existing:
        print("ERROR: No existing dataset could be found!")
        return

    # Download logs
    if (not os.path.isfile(full_logs_path) or redownload) and not use_existing:
        print("INFO: Exporting chat logs using DiscordChatExporter...")
        print(
            "INFO: This may take a few minutes to hours depending on the message count of the channel"
        )
        print("INFO: Progress will NOT be saved if cancelled")
        print(
            "--------------------------DiscordChatExporter---------------------------"
        )
        try:
            DiscordChatExporter = resolve_discord_chat_exporter_path(
                prompt_for_path=True
            )
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            raise RuntimeError(str(exc)) from exc

        subprocess.run(
            [
                str(DiscordChatExporter),
                "export",
                "-c",
                channel_id,
                "-t",
                bot_token or "",
                "-o",
                f"{channel_id}_logs.json",
                "-f",
                "Json",
                "--fuck-russia",
                "True",
            ]
        )
        print(
            "--------------------------DiscordChatExporter---------------------------"
        )
        os.makedirs(os.path.dirname(full_logs_path), exist_ok=True)
        shutil.move(f"{channel_id}_logs.json", full_logs_path)
        print(f"INFO: Logs saved to {full_logs_path}")
    elif (os.path.isfile(full_logs_path) and not redownload) and not use_existing:
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
