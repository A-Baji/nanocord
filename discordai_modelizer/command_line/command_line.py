import argparse
import json
import os
from discordai_modelizer import __version__ as version
from discordai_modelizer import customize
from discordai_modelizer import openai as openai_wrapper
from discordai_modelizer.command_line import subparsers
from discordai_modelizer.command_line.subparsers import (
    set_openai_help_str,
    set_bot_key_help_str,
)


def setup_modelizer_commands(parser, is_parent=False):
    command = parser.add_subparsers(dest="command")

    model = command.add_parser("model", help="Manage your OpenAI models")
    job = command.add_parser("job", help="Manage your OpenAI jobs")

    model_subcommand = model.add_subparsers(dest="subcommand")
    job_subcommand = job.add_subparsers(dest="subcommand")

    subparsers.setup_model_list(model_subcommand, is_parent)
    subparsers.setup_model_create(model_subcommand, is_parent)
    subparsers.setup_model_delete(model_subcommand, is_parent)

    subparsers.setup_job_list(job_subcommand, is_parent)
    subparsers.setup_job_info(job_subcommand, is_parent)
    subparsers.setup_job_events(job_subcommand, is_parent)
    subparsers.setup_job_cancel(job_subcommand, is_parent)

    return command, model_subcommand, job_subcommand


def read_modelizer_args(args, model_subcommand, job_subcommand):
    if args.command == "model":
        if args.subcommand == "list":
            display(openai_wrapper.list_models(args.openai_key, args.full))
        elif args.subcommand == "create":
            customize.create_model(
                args.channel,
                args.user,
                args.discord_token,
                args.openai_key,
                thought_time=args.thought_time,
                thought_max=args.thought_max,
                thought_min=args.thought_min,
                max_entry_count=args.max_entries,
                offset=args.offset,
                distributed=args.distributed,
                reverse=args.reverse,
                base_model=args.base_model,
                clean=args.dirty,
                redownload=args.redownload,
                use_existing=args.use_existing,
            )
        elif args.subcommand == "delete":
            display(
                openai_wrapper.delete_model(args.model_id, args.openai_key, args.force)
            )
        else:
            raise argparse.ArgumentError(
                model_subcommand,
                "Must choose a command from `list`, `create`, or `delete`",
            )
    elif args.command == "job":
        if args.subcommand == "list":
            display(openai_wrapper.list_jobs(args.openai_key, args.full))
        elif args.subcommand == "info":
            display(openai_wrapper.get_job_info(args.job_id, args.openai_key))
        elif args.subcommand == "events":
            display(openai_wrapper.get_job_events(args.job_id, args.openai_key))
        elif args.subcommand == "cancel":
            display(openai_wrapper.cancel_job(args.job_id, args.openai_key))
        else:
            raise argparse.ArgumentError(
                job_subcommand,
                "Must choose a command from `list`, `info`, `events`, or `cancel`",
            )


def display(obj):
    print(json.dumps(obj, indent=4))


def set_openai_api_key(key: str, obj, is_parent=False):
    if not key and not obj.get("OPENAI_API_KEY"):
        raise ValueError(
            f"Your OpenAI API key must either be passed in as an argument or set {set_openai_help_str(is_parent)}",
        )
    else:
        return key or obj.get("OPENAI_API_KEY")


def set_bot_token(token: str, obj, is_parent=False):
    if not token and not obj.get("DISCORD_BOT_TOKEN"):
        raise ValueError(
            f"Your Discord bot token must either be passed in as an argument or set {set_bot_key_help_str(is_parent)}",
        )
    else:
        return token or obj.get("DISCORD_BOT_TOKEN")


def discordai_modelizer():
    parser = argparse.ArgumentParser(
        prog="discordai_modelizer", description="DiscordAI Modelizer CLI"
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"discordai-modelizer {version}"
    )

    command, model_subcommand, job_subcommand = setup_modelizer_commands(parser)

    args = parser.parse_args()
    if hasattr(args, "openai_key"):
        args.openai_key = set_openai_api_key(args.openai_key, os.environ)
    if hasattr(args, "discord_token"):
        args.discord_token = set_bot_token(args.discord_token, os.environ)

    read_modelizer_args(args, model_subcommand, job_subcommand)


if __name__ == "__main__":
    discordai_modelizer()
