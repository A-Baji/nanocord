from json import loads
from discordai_modelizer import gen_dataset
from pytest import raises
from . import expected_values
from .conftest import FULL_LOGS_PATH, FILES_PATH, USER, list_dict_comp

CHANNEL_ID = "TEST_CHANNEL"
FULL_DATASET_PATH = FILES_PATH / f"{USER[:13]}_{CHANNEL_ID[:4]}_data_set.jsonl"


def test_parse_logs(default_file_output):
    gen_dataset.parse_logs(
        FULL_LOGS_PATH, CHANNEL_ID, USER, thought_time=0, thought_min=1
    )
    with open(FULL_DATASET_PATH, "r", encoding="utf-8") as data_file:
        list_dict_comp(
            expected_values.parse_logs_expected_all, [loads(line) for line in data_file]
        )
        data_file.close()


def test_parse_logs_legacy_username(default_file_output):
    gen_dataset.parse_logs(
        FULL_LOGS_PATH, CHANNEL_ID, f"{USER}#0000", thought_time=0, thought_min=1
    )
    with open(FULL_DATASET_PATH, "r", encoding="utf-8") as data_file:
        list_dict_comp(
            expected_values.parse_logs_expected_all, [loads(line) for line in data_file]
        )
        data_file.close()


def test_parse_logs_ttime_2(default_file_output):
    gen_dataset.parse_logs(
        FULL_LOGS_PATH, CHANNEL_ID, USER, thought_time=2, thought_min=1
    )
    with open(FULL_DATASET_PATH, "r", encoding="utf-8") as data_file:
        list_dict_comp(
            expected_values.parse_logs_expected_two_secs,
            [loads(line) for line in data_file],
        )
        data_file.close()


def test_parse_logs_ttime_5(default_file_output):
    gen_dataset.parse_logs(
        FULL_LOGS_PATH, CHANNEL_ID, USER, thought_time=5, thought_min=1
    )
    with open(FULL_DATASET_PATH, "r", encoding="utf-8") as data_file:
        list_dict_comp(
            expected_values.parse_logs_expected_five_secs,
            [loads(line) for line in data_file],
        )
        data_file.close()


def test_parse_logs_empty(capsys, default_file_output):
    gen_dataset.parse_logs(FULL_LOGS_PATH, CHANNEL_ID, USER, thought_max=-1)
    stdout = capsys.readouterr()
    assert (
        "WARNING: The resulting dataset is empty. Please double check your parameters."
        in stdout.out
    )


def test_gen_dataset_max_is_greater(default_file_output):
    gen_dataset.parse_logs(
        FULL_LOGS_PATH, CHANNEL_ID, USER, thought_time=0, thought_min=1
    )
    gen_dataset.get_lines(FULL_DATASET_PATH, N=1000)
    with open(FULL_DATASET_PATH, "r", encoding="utf-8") as data_file:
        list_dict_comp(
            expected_values.parse_logs_expected_all,
            [loads(line) for line in data_file],
        )
        data_file.close()


def test_gen_dataset_first_10(default_file_output):
    gen_dataset.parse_logs(
        FULL_LOGS_PATH, CHANNEL_ID, USER, thought_time=0, thought_min=1
    )
    gen_dataset.get_lines(FULL_DATASET_PATH, N=10)
    with open(FULL_DATASET_PATH, "r", encoding="utf-8") as data_file:
        list_dict_comp(
            expected_values.gen_dataset_first_10_lines,
            [loads(line) for line in data_file],
        )
        data_file.close()


def test_gen_dataset_reverse(default_file_output):
    gen_dataset.parse_logs(
        FULL_LOGS_PATH, CHANNEL_ID, USER, thought_time=0, thought_min=1
    )
    gen_dataset.get_lines(FULL_DATASET_PATH, reverse=True)
    with open(FULL_DATASET_PATH, "r", encoding="utf-8") as data_file:
        list_dict_comp(
            expected_values.gen_dataset_reverse,
            [loads(line) for line in data_file],
        )
        data_file.close()


def test_gen_dataset_offset(default_file_output):
    gen_dataset.parse_logs(
        FULL_LOGS_PATH, CHANNEL_ID, USER, thought_time=0, thought_min=1
    )
    gen_dataset.get_lines(FULL_DATASET_PATH, offset=1)
    with open(FULL_DATASET_PATH, "r", encoding="utf-8") as data_file:
        list_dict_comp(
            expected_values.gen_dataset_offset_1,
            [loads(line) for line in data_file],
        )
        data_file.close()


def test_gen_dataset_reverse_and_offset(default_file_output):
    gen_dataset.parse_logs(
        FULL_LOGS_PATH, CHANNEL_ID, USER, thought_time=0, thought_min=1
    )
    gen_dataset.get_lines(FULL_DATASET_PATH, offset=1, reverse=True)
    with open(FULL_DATASET_PATH, "r", encoding="utf-8") as data_file:
        list_dict_comp(
            expected_values.gen_dataset_reverse_and_offset_1,
            [loads(line) for line in data_file],
        )
        data_file.close()


def test_gen_dataset_reverse_5_max(default_file_output):
    gen_dataset.parse_logs(
        FULL_LOGS_PATH, CHANNEL_ID, USER, thought_time=0, thought_min=1
    )
    gen_dataset.get_lines(FULL_DATASET_PATH, N=5, reverse=True)
    with open(FULL_DATASET_PATH, "r", encoding="utf-8") as data_file:
        list_dict_comp(
            expected_values.gen_dataset_reverse_5_max,
            [loads(line) for line in data_file],
        )
        data_file.close()


def test_gen_dataset_distributed(default_file_output):
    gen_dataset.parse_logs(
        FULL_LOGS_PATH, CHANNEL_ID, USER, thought_time=0, thought_min=1
    )
    gen_dataset.get_lines(FULL_DATASET_PATH, N=5, distributed=True)
    with open(FULL_DATASET_PATH, "r", encoding="utf-8") as data_file:
        list_dict_comp(
            expected_values.gen_dataset_max_5_distributed,
            [loads(line) for line in data_file],
        )
        data_file.close()


def test_gen_dataset_distributed_offset(default_file_output):
    gen_dataset.parse_logs(
        FULL_LOGS_PATH, CHANNEL_ID, USER, thought_time=0, thought_min=1
    )
    gen_dataset.get_lines(FULL_DATASET_PATH, N=5, distributed=True, offset=2)
    with open(FULL_DATASET_PATH, "r", encoding="utf-8") as data_file:
        list_dict_comp(
            expected_values.gen_dataset_max_5_distributed_offset_2,
            [loads(line) for line in data_file],
        )
        data_file.close()


def test_gen_dataset_distributed_reverse(default_file_output):
    gen_dataset.parse_logs(
        FULL_LOGS_PATH, CHANNEL_ID, USER, thought_time=0, thought_min=1
    )
    gen_dataset.get_lines(FULL_DATASET_PATH, N=5, distributed=True, reverse=True)
    with open(FULL_DATASET_PATH, "r", encoding="utf-8") as data_file:
        list_dict_comp(
            expected_values.gen_dataset_max_5_distributed_reverse,
            [loads(line) for line in data_file],
        )
        data_file.close()


def test_gen_dataset_distributed_reverse_offset(default_file_output):
    gen_dataset.parse_logs(
        FULL_LOGS_PATH, CHANNEL_ID, USER, thought_time=0, thought_min=1
    )
    gen_dataset.get_lines(
        FULL_DATASET_PATH, N=5, distributed=True, reverse=True, offset=2
    )
    with open(FULL_DATASET_PATH, "r", encoding="utf-8") as data_file:
        list_dict_comp(
            expected_values.gen_dataset_max_5_distributed_reverse_offset_2,
            [loads(line) for line in data_file],
        )
        data_file.close()


def test_parse_logs_user_not_found(default_file_output):
    username = "bad_username"
    with raises(
        gen_dataset.UserNotFoundError,
        match=f"No messages found in chat logs for user: {username}",
    ):
        gen_dataset.parse_logs(FULL_LOGS_PATH, CHANNEL_ID, username)
