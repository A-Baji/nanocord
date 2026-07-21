from json import dumps, loads
import os

from pytest import mark, raises
from discordai_modelizer.command_line.command_line import (
    set_openai_api_key,
    set_bot_token,
)
from discordai_modelizer.command_line.subparsers import (
    set_openai_help_str,
    set_bot_key_help_str,
)
from . import expected_values
from .conftest import CHANNEL_ID, USER


def test_cli_help(script_runner):
    cli = script_runner.run(["discordai_modelizer", "-h"])
    assert cli.success
    assert "usage: discordai_modelizer [-h] [-V] {model,job} ..." in cli.stdout

    assert "DiscordAI Modelizer CLI" in cli.stdout
    assert "positional arguments:" in cli.stdout
    assert "{model,job}" in cli.stdout
    assert "option" in cli.stdout
    assert "-h, --help     show this help message and exit" in cli.stdout
    assert "-V, --version  show program's version number and exit" in cli.stdout


def test_cli_model_list(script_runner):
    cli = script_runner.run(["discordai_modelizer", "model", "list"])
    assert cli.success
    for o in expected_values.list_module_expected:
        assert o in loads(cli.stdout)


def test_cli_training(script_runner, default_file_output):
    cli = script_runner.run(
        [
            "discordai_modelizer",
            "model",
            "create",
            "-c",
            f"{CHANNEL_ID}",
            "-u",
            f"{USER}",
            "-o",
            "BAD_KEY",
            "-b",
            "babbage",
        ]
    )
    assert not cli.success
    assert "INFO: Starting OpenAI fine-tune job..." in cli.stdout


def test_cli_model_list_full(script_runner):
    cli = script_runner.run(["discordai_modelizer", "model", "list", "--full"])
    assert cli.success
    for o in expected_values.list_module_expected_full:
        assert o in loads(cli.stdout)


def test_cli_delete_model(script_runner):
    cli = script_runner.run(
        ["discordai_modelizer", "model", "delete", "-m", "whisper-1"]
    )
    assert (
        "Are you sure you want to delete this model? This action is not reversable. Y/N: "
        in cli.stdout
    )


def test_cli_job_list(script_runner):
    cli = script_runner.run(["discordai_modelizer", "job", "list"])
    assert cli.success
    assert dumps(expected_values.list_job_expected, indent=4) in cli.stdout


def test_job_list_full(script_runner):
    cli = script_runner.run(["discordai_modelizer", "job", "list", "--full"])
    assert cli.success
    assert dumps(expected_values.list_job_expected_full, indent=4) in cli.stdout


def test_job_info(script_runner):
    cli = script_runner.run(
        ["discordai_modelizer", "job", "info", "-j", "ftjob-i2IyeV2xbLCSrYq45kTKSdwE"]
    )
    assert cli.success
    assert dumps(expected_values.job_info_expected, indent=4) in cli.stdout


def test_job_events(script_runner):
    cli = script_runner.run(
        ["discordai_modelizer", "job", "events", "-j", "ftjob-i2IyeV2xbLCSrYq45kTKSdwE"]
    )
    assert cli.success
    assert dumps(expected_values.job_events_expected, indent=4) in cli.stdout


def test_job_cancel(script_runner):
    cli = script_runner.run(
        ["discordai_modelizer", "job", "cancel", "-j", "ftjob-i2IyeV2xbLCSrYq45kTKSdwE"]
    )
    assert cli.success
    assert dumps(expected_values.job_cancel_expected, indent=4) in cli.stdout


def test_job_cancel(script_runner):
    cli = script_runner.run(
        ["discordai_modelizer", "job", "cancel", "-j", "ftjob-i2IyeV2xbLCSrYq45kTKSdwE"]
    )
    assert cli.success
    assert dumps(expected_values.job_cancel_expected, indent=4) in cli.stdout


def test_cli_model_bad_args(script_runner):
    cli = script_runner.run(
        [
            "discordai_modelizer",
            "model",
        ]
    )
    assert not cli.success
    assert "Must choose a command from `list`, `create`, or `delete`" in cli.stderr


def test_cli_job_bad_args(script_runner):
    cli = script_runner.run(
        [
            "discordai_modelizer",
            "job",
        ]
    )
    assert not cli.success
    assert (
        "Must choose a command from `list`, `info`, `events`, or `cancel`" in cli.stderr
    )


def test_bad_set_openai_api_key(unset_envs):
    with raises(
        ValueError,
        match=f"Your OpenAI API key must either be passed in as an argument or set {set_openai_help_str(False)}",
    ):
        set_openai_api_key(None, os.environ)


def test_bad_set_bot_token(unset_envs):
    with raises(
        ValueError,
        match=f"Your Discord bot token must either be passed in as an argument or set {set_bot_key_help_str(False)}",
    ):
        set_bot_token(None, os.environ)
