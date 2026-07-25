import json
import pathlib
import tempfile
import os

import pytest

from nanocord.dataset.sft import parse_sft_logs
from nanocord.dataset.thoughts import UserNotFoundError

TARGET_USER = "target-user-id"
OTHER_USER = "other-user-id"


def _write_log(messages):
    data = {"messages": messages}
    f = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
    json.dump(data, f)
    f.close()
    return f.name


def _msg(id, author_id, content, timestamp, reference_id=None):
    msg = {
        "id": id,
        "content": content,
        "timestamp": timestamp,
        "author": {"id": author_id},
    }
    if reference_id is not None:
        msg["reference"] = {
            "type": "Default",
            "messageId": reference_id,
            "channelId": "c",
            "guildId": "g",
        }
    return msg


def test_parse_sft_logs_basic_pair(tmp_path, monkeypatch):
    monkeypatch.setattr("nanocord.dataset.sft.DATASET_PATH", tmp_path)

    messages = [
        _msg("1", OTHER_USER, "what is your favorite color", "2023-01-01T00:00:00.000Z"),
        _msg("2", TARGET_USER, "it is definitely blue for sure", "2023-01-01T00:05:00.000Z", reference_id="1"),
    ]
    log_file = _write_log(messages)

    try:
        result_path = parse_sft_logs(log_file, "chan", TARGET_USER, thought_time=5, thought_min=1)
        lines = result_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["messages"][1]["role"] == "user"
        assert "favorite color" in record["messages"][1]["content"]
        assert record["messages"][2]["role"] == "assistant"
        assert "blue" in record["messages"][2]["content"]
    finally:
        os.unlink(log_file)


def test_parse_sft_logs_unresolved_reference_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr("nanocord.dataset.sft.DATASET_PATH", tmp_path)

    messages = [
        _msg("2", TARGET_USER, "replying to something not in this export", "2023-01-01T00:05:00.000Z", reference_id="missing-id"),
    ]
    log_file = _write_log(messages)

    try:
        result_path = parse_sft_logs(log_file, "chan", TARGET_USER, thought_time=5, thought_min=1)
        assert result_path.read_text().strip() == ""
    finally:
        os.unlink(log_file)


def test_parse_sft_logs_self_reply_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr("nanocord.dataset.sft.DATASET_PATH", tmp_path)

    messages = [
        _msg("1", TARGET_USER, "an earlier message from the target user", "2023-01-01T00:00:00.000Z"),
        _msg("2", TARGET_USER, "replying to my own earlier message here", "2023-01-01T00:10:00.000Z", reference_id="1"),
    ]
    log_file = _write_log(messages)

    try:
        result_path = parse_sft_logs(log_file, "chan", TARGET_USER, thought_time=5, thought_min=1)
        assert result_path.read_text().strip() == ""
    finally:
        os.unlink(log_file)


def test_parse_sft_logs_non_reply_thought_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr("nanocord.dataset.sft.DATASET_PATH", tmp_path)

    messages = [
        _msg("1", OTHER_USER, "just some context message here", "2023-01-01T00:00:00.000Z"),
        _msg("2", TARGET_USER, "an unrelated non reply message here", "2023-01-01T00:05:00.000Z"),
    ]
    log_file = _write_log(messages)

    try:
        result_path = parse_sft_logs(log_file, "chan", TARGET_USER, thought_time=5, thought_min=1)
        assert result_path.read_text().strip() == ""
    finally:
        os.unlink(log_file)


def test_parse_sft_logs_dedupes_identical_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr("nanocord.dataset.sft.DATASET_PATH", tmp_path)

    messages = [
        _msg("1", OTHER_USER, "what is your favorite color", "2023-01-01T00:00:00.000Z"),
        _msg("2", TARGET_USER, "it is definitely blue for sure", "2023-01-01T00:05:00.000Z", reference_id="1"),
        _msg("3", TARGET_USER, "it is definitely blue for sure", "2023-01-01T01:00:00.000Z", reference_id="1"),
    ]
    log_file = _write_log(messages)

    try:
        result_path = parse_sft_logs(log_file, "chan", TARGET_USER, thought_time=5, thought_min=1)
        lines = result_path.read_text().strip().split("\n")
        assert len(lines) == 1
    finally:
        os.unlink(log_file)


def test_parse_sft_logs_thought_min_filters_short_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr("nanocord.dataset.sft.DATASET_PATH", tmp_path)

    messages = [
        _msg("1", OTHER_USER, "hi", "2023-01-01T00:00:00.000Z"),
        _msg("2", TARGET_USER, "yo", "2023-01-01T00:05:00.000Z", reference_id="1"),
    ]
    log_file = _write_log(messages)

    try:
        result_path = parse_sft_logs(log_file, "chan", TARGET_USER, thought_time=5, thought_min=6)
        assert result_path.read_text().strip() == ""
    finally:
        os.unlink(log_file)


def test_parse_sft_logs_system_prompt_embedded(tmp_path, monkeypatch):
    monkeypatch.setattr("nanocord.dataset.sft.DATASET_PATH", tmp_path)

    messages = [
        _msg("1", OTHER_USER, "what is your favorite color", "2023-01-01T00:00:00.000Z"),
        _msg("2", TARGET_USER, "it is definitely blue for sure", "2023-01-01T00:05:00.000Z", reference_id="1"),
    ]
    log_file = _write_log(messages)

    try:
        result_path = parse_sft_logs(
            log_file, "chan", TARGET_USER, thought_time=5, thought_min=1,
            system_prompt="You are a helpful assistant.",
        )
        record = json.loads(result_path.read_text().strip())
        assert record["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    finally:
        os.unlink(log_file)


def test_parse_sft_logs_raises_when_user_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr("nanocord.dataset.sft.DATASET_PATH", tmp_path)

    messages = [
        _msg("1", OTHER_USER, "nobody from the target user here", "2023-01-01T00:00:00.000Z"),
    ]
    log_file = _write_log(messages)

    try:
        with pytest.raises(UserNotFoundError):
            parse_sft_logs(log_file, "chan", TARGET_USER, thought_time=5, thought_min=1)
    finally:
        os.unlink(log_file)