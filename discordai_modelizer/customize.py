import os
import platform
import subprocess
import appdirs
import shutil
import pathlib

from openai import OpenAI
from discordai_modelizer.gen_dataset import parse_logs, get_lines, UserNotFoundError

MODEL_MAP = {
    "davinci": "davinci-002",
    "babbage": "babbage-002",
}


def get_dce_path_and_exe():
    os_name = platform.system()

    if os_name == "Linux":
        return pathlib.Path(
            "DiscordChatExporter.Cli.linux-x64", "DiscordChatExporter.Cli"
        )
    elif os_name == "Darwin":
        return pathlib.Path(
            "DiscordChatExporter.Cli.osx-x64", "DiscordChatExporter.Cli"
        )
    elif os_name == "Windows":
        return pathlib.Path(
            "DiscordChatExporter.Cli.win-x64", "DiscordChatExporter.Cli.exe"
        )


def create_model(
    channel_id: str,
    user_id: str,
    bot_token: str = os.getenv("DISCORD_BOT_TOKEN"),
    openai_key: str = os.getenv("OPENAI_API_KEY"),
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
    client = OpenAI(api_key=openai_key)
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
        DiscordChatExporter = (
            pathlib.Path(os.path.dirname(__file__))
            / "DiscordChatExporter"
            / get_dce_path_and_exe()
        )
        subprocess.run(
            [
                DiscordChatExporter,
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
        print("INFO: Parsing chat logs into an OpenAI compatible dataset...")
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

    # Train customized OpenAI model
    if base_model in ["davinci", "babbage"]:
        print("INFO: Starting OpenAI fine-tune job...")
        upload_response = client.files.create(
            file=open(full_dataset_path, "rb"), purpose="fine-tune"
        )
        fine_tune = client.fine_tuning.jobs.create(
            model=MODEL_MAP[base_model],
            training_file=upload_response.id,
            suffix=channel_user,
        )
        print(
            "INFO: This may take a few minutes to hours depending on the size of the dataset and the selected base model"
        )
        print(f"INFO: Fine tune job id: {fine_tune.id}")
        print(
            "INFO: Use the `job info -j <job_id>` command to check the info of the job process"
        )
        print(
            "INFO: Use the `job events -j <job_id>` command to view the fine-tuning events of the job process"
        )
        print(
            "INFO: Use the `job cancel -j <job_id>` command to cancel the job process"
        )
        print(
            "INFO: Or visit the OpenAI dashboard: https://platform.openai.com/finetune"
        )
    else:
        print("INFO: No base model selected... Skipping training.")

    # Clean up generated files
    if clean and not use_existing:
        full_dataset_path.unlink()

    client.close()
