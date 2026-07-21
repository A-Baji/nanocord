import argparse
import os

from nanocord.dataset import create_export
from nanocord.discord_export import export_channel_logs
from nanocord.logger import setup_logger
from nanocord.paths import DATASET_PATH
from nanocord.paths import DISCORD_CHAT_EXPORTER_LOGS_PATH
from nanocord.version import __version__ as version

# Setup logger
logger = setup_logger('nanocord.cli', 'logs/cli.log')


def setup_nanocord_commands(parser):
    command = parser.add_subparsers(dest="command")
    export = command.add_parser("export", help="Export and prepare Discord chat data")
    export_subcommand = export.add_subparsers(dest="subcommand")
    setup_export_create(export_subcommand)
    return command, export_subcommand


def setup_export_create(export_subcommand):
    export_create = export_subcommand.add_parser(
        "create",
        help="Download Discord channel logs, parse them into a dataset, and save the results locally",
    )
    export_create_required_named = export_create.add_argument_group(
        "required named arguments"
    )
    export_create_optional_named = export_create.add_argument_group(
        "optional named arguments"
    )

    export_create_required_named.add_argument(
        "-d",
        "--discord-token",
        type=str,
        dest="discord_token",
        help="The Discord token for your bot. Must either be provided as an argument or set as the DISCORD_BOT_TOKEN environment variable",
    )
    export_create_required_named.add_argument(
        "-c",
        "--channel",
        required=True,
        type=str,
        dest="channel",
        help="The ID of the Discord channel you want to use",
    )
    export_create_required_named.add_argument(
        "-u",
        "--user",
        required=True,
        type=str,
        dest="user",
        help="The unique username of the Discord user you want to use",
    )

    export_create_optional_named.add_argument(
        "--ttime",
        "--thought-time",
        type=int,
        default=5,
        required=False,
        dest="thought_time",
        help='The maximum amount of time in seconds to consider two individual messages to be part of the same "thought": DEFAULT=10',
    )
    export_create_optional_named.add_argument(
        "--tmax",
        "--thought-max",
        type=int,
        default=None,
        required=False,
        dest="thought_max",
        help="The maximum length in words of each thought: DEFAULT=None",
    )
    export_create_optional_named.add_argument(
        "--tmin",
        "--thought-min",
        type=int,
        default=6,
        required=False,
        dest="thought_min",
        help="The minimum length in words of each thought: DEFAULT=4",
    )
    export_create_optional_named.add_argument(
        "-m",
        "--max-entries",
        type=int,
        default=1000,
        required=False,
        dest="max_entries",
        help="The max amount of entries (by lines) that may exist in the dataset: DEFAULT=1000",
    )
    export_create_optional_named.add_argument(
        "--os",
        "--offset",
        type=int,
        default=0,
        required=False,
        dest="offset",
        help="The offset by line index starting at 0 for where to start selecting lines for the dataset: DEFAULT=0",
    )
    export_create_optional_named.add_argument(
        "--distributed",
        action="store_true",
        required=False,
        dest="distributed",
        help="Select lines as an even distribution instead of sequentially",
    )
    export_create_optional_named.add_argument(
        "--reverse_lines",
        action="store_true",
        required=False,
        dest="reverse",
        help="Reverse the order in which to select lines for the dataset",
    )
    export_create_optional_named.add_argument(
        "--dirty",
        action="store_false",
        required=False,
        dest="dirty",
        help="Skip the clean up step for outputted files",
    )
    export_create_optional_named.add_argument(
        "--redownload",
        action="store_true",
        required=False,
        dest="redownload",
        help="Redownload the Discord chat logs",
    )
    export_create_optional_named.add_argument(
        "--use_existing",
        action="store_true",
        required=False,
        dest="use_existing",
        help="Use an existing dataset that may have been manually revised",
    )


def read_nanocord_args(args, export_subcommand):
    if args.command == "export" and args.subcommand == "create":
        create_export(
            args.channel,
            args.user,
            args.discord_token,
            thought_time=args.thought_time,
            thought_max=args.thought_max,
            thought_min=args.thought_min,
            max_entry_count=args.max_entries,
            offset=args.offset,
            distributed=args.distributed,
            reverse=args.reverse,
            clean=args.dirty,
            redownload=args.redownload,
            use_existing=args.use_existing,
        )
    else:
        raise argparse.ArgumentError(
            export_subcommand,
            "Must choose the `create` subcommand",
        )


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
        logger.error("No existing dataset could be found!")
        return

    # Download logs
    if (not full_logs_path.exists() or redownload) and not use_existing:
        try:
            export_channel_logs(channel_id, bot_token)
        except Exception as e:
            logger.error(f"Failed to export chat logs: {e}")
            raise
    elif full_logs_path.exists() and not redownload and not use_existing:
        logger.info(
            f"Chat logs detected locally at {full_logs_path}... Skipping download."
        )

    # Parse logs
    if use_existing:
        logger.info("Using existing dataset... Skipping download and parsing.")
    else:
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
        if not clean:
            logger.info(f"Dataset saved to {full_dataset_path}")

def nanocord():
    parser = argparse.ArgumentParser(
        prog="nanocord", description="Discord chat export and dataset preparation CLI"
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"nanocord {version}"
    )

    command, export_subcommand = setup_nanocord_commands(parser)

    args = parser.parse_args()
    if hasattr(args, "discord_token") and args.discord_token:
        # Set the token in environment for downstream usage
        os.environ["DISCORD_BOT_TOKEN"] = args.discord_token

    read_nanocord_args(args, export_subcommand)


if __name__ == "__main__":
    nanocord()