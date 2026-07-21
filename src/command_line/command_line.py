import argparse
import os

from src import __version__ as version
from src import customize
from src.command_line import subparsers
from src.command_line.subparsers import set_bot_key_help_str


def setup_modelizer_commands(parser, is_parent=False):
    command = parser.add_subparsers(dest="command")
    model = command.add_parser("model", help="Export and prepare Discord chat data")
    model_subcommand = model.add_subparsers(dest="subcommand")
    subparsers.setup_model_create(model_subcommand, is_parent)
    return command, model_subcommand


def read_modelizer_args(args, model_subcommand):
    if args.command == "model" and args.subcommand == "create":
        customize.create_model(
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
            model_subcommand,
            "Must choose the `create` subcommand",
        )


def set_bot_token(token: str, obj, is_parent=False):
    if not token and not obj.get("DISCORD_BOT_TOKEN"):
        raise ValueError(
            f"Your Discord bot token must either be passed in as an argument or set {set_bot_key_help_str(is_parent)}",
        )
    else:
        return token or obj.get("DISCORD_BOT_TOKEN")


def discordai_modelizer():
    parser = argparse.ArgumentParser(
        prog="discordai_modelizer", description="Discord chat export and dataset preparation CLI"
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"discordai-modelizer {version}"
    )

    command, model_subcommand = setup_modelizer_commands(parser)

    args = parser.parse_args()
    if hasattr(args, "discord_token"):
        args.discord_token = set_bot_token(args.discord_token, os.environ)

    read_modelizer_args(args, model_subcommand)


if __name__ == "__main__":
    discordai_modelizer()
