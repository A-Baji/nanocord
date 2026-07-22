import argparse
import os

from nanocord.dataset import create_export
from nanocord import global_logger
from nanocord.version import __version__ as version

# Use the global logger
logger = global_logger


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

def nanocord():
    parser = argparse.ArgumentParser(
        prog="nanocord", description="Discord chat export and dataset preparation CLI"
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"nanocord {version}"
    )

    command, export_subcommand = setup_nanocord_commands(parser)

    args = parser.parse_args()
    if not hasattr(args, "discord_token") or not args.discord_token:
        # Try to get the token from environment
        args.discord_token = os.getenv("DISCORD_BOT_TOKEN")

    read_nanocord_args(args, export_subcommand)


if __name__ == "__main__":
    nanocord()