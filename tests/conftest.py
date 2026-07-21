import os
import pathlib
import appdirs

from discordai_modelizer import customize
from pytest import fixture

CHANNEL_ID = os.environ["CHANNEL_ID"]
USER = os.environ["USERNAME"]
FILES_PATH = pathlib.Path(appdirs.user_data_dir(appname="discordai"))
FULL_LOGS_PATH = FILES_PATH / f"{CHANNEL_ID}_logs.json"
FULL_DATASET_PATH = FILES_PATH / f"{USER[:13]}_{CHANNEL_ID[:4]}_data_set.jsonl"


def list_dict_comp(x, y):
    assert len(x) == len(y)
    for d1, d2 in zip(x, y):
        assert d1 == d2


@fixture(scope="session")
def default_file_output():
    # Ensure a clean start by removing any existing files
    if FULL_LOGS_PATH.exists():
        FULL_LOGS_PATH.unlink()
    if FULL_DATASET_PATH.exists():
        FULL_DATASET_PATH.unlink()
    # Generate log and dataset files
    customize.create_model(CHANNEL_ID, USER)
    yield
    # Cleanup after the tests in this module
    if FULL_LOGS_PATH.exists():
        FULL_LOGS_PATH.unlink()
    if FULL_DATASET_PATH.exists():
        FULL_DATASET_PATH.unlink()


@fixture(scope="function")
def unset_envs():
    DISCORD_BOT_TOKEN = os.environ["DISCORD_BOT_TOKEN"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    del os.environ["DISCORD_BOT_TOKEN"]
    del os.environ["OPENAI_API_KEY"]
    yield
    os.environ["DISCORD_BOT_TOKEN"] = DISCORD_BOT_TOKEN
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
