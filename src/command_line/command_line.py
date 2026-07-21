import argparse
import os

from src import __version__ as version
from src import customize
from src.command_line import subparsers
from src.command_line.subparsers import set_bot_key_help_str


def setup_nanocord_commands(parser):
    command = parser.add_subparsers(dest="command")
    export = command.add_parser("export", help="Export and prepare Discord chat data")
    export_subcommand = export.add_subparsers(dest="subcommand")
    subparsers.setup_export_create(export_subcommand)
    return command, export_subcommand


def read_nanocord_args(args, export_subcommand):
    if args.command == "export" and args.subcommand == "create":
        customize.create_export(
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


def set_bot_token(token: str, obj):
    if not token and not obj.get("DISCORD_BOT_TOKEN"):
        raise ValueError(
            f"Your Discord bot token must either be provided as an argument or set {set_bot_key_help_str()}",
        )
    else:
        return token or obj.get("DISCORD_BOT_TOKEN")


def nanocord():
    parser = argparse.ArgumentParser(
        prog="nanocord", description="Discord chat export and dataset preparation CLI"
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"nanocord {version}"
    )

    command, export_subcommand = setup_nanocord_commands(parser)

    args = parser.parse_args()
    if hasattr(args, "discord_token"):
        args.discord_token = set_bot_token(args.discord_token, os.environ)

    read_nanocord_args(args, export_subcommand)


if __name__ == "__main__":
    nanocord()
