from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nanocord.dataset.thoughts import build_thought
from nanocord.dataset.thoughts import cleanup_string
from nanocord.dataset.thoughts import group_into_thoughts
from nanocord.dataset.thoughts import UserNotFoundError
from nanocord.dataset.thoughts import validate_thought


def test_validate_thought():
    # Test valid thoughts
    assert validate_thought("This is a valid thought with more than six words", thought_min=6)
    assert validate_thought("Short", thought_min=1, thought_max=10)

    # Test invalid thoughts (too short)
    assert not validate_thought("Short", thought_min=6)
    assert not validate_thought("", thought_min=6)

    # Test invalid thoughts (too long)
    assert not validate_thought("This is a thought that exceeds the maximum word count limit by far", thought_max=5)


def test_cleanup_string():
    # Test URL removal
    text_with_url = "Visit https://example.com for more info"
    cleaned = cleanup_string(text_with_url)
    assert "https://example.com" not in cleaned

    # Test multiple URLs
    text = "Check out http://test.com and https://example.org"
    cleaned = cleanup_string(text)
    assert "http://" not in cleaned
    assert "https://" not in cleaned


def test_build_thought():
    # Test building a thought from empty string and message
    msg = {"content": "Hello world"}
    thought = build_thought("", msg)
    assert thought == " Hello world"

    # Test adding to existing thought
    existing_thought = "Hello world"
    new_msg = {"content": " How are you?"}
    updated_thought = build_thought(existing_thought, new_msg)
    assert updated_thought == "Hello world How are you?"


def test_edge_cases():
    # Test with empty content
    msg = {"content": ""}
    thought = build_thought("", msg)
    assert thought == ""

    # Test with whitespace only content (will be stripped to empty string)
    msg = {"content": "   "}
    thought = build_thought("", msg)
    assert thought == ""  # Whitespace-only content gets stripped


def test_group_into_thoughts_merges_within_time_window():
    messages = [
        {"id": "1", "content": "Hello", "timestamp": "2023-01-01T00:00:00.000Z"},
        {"id": "2", "content": "world", "timestamp": "2023-01-01T00:00:02.000Z"},
    ]
    thoughts = group_into_thoughts(messages, thought_time=5)
    assert len(thoughts) == 1
    assert thoughts[0]["message_ids"] == ["1", "2"]
    assert thoughts[0]["reply_reference_id"] is None
    assert "Hello" in thoughts[0]["text"]
    assert "world" in thoughts[0]["text"]


def test_group_into_thoughts_splits_on_time_gap():
    messages = [
        {"id": "1", "content": "Hello", "timestamp": "2023-01-01T00:00:00.000Z"},
        {"id": "2", "content": "world", "timestamp": "2023-01-01T00:10:00.000Z"},
    ]
    thoughts = group_into_thoughts(messages, thought_time=5)
    assert len(thoughts) == 2
    assert thoughts[0]["message_ids"] == ["1"]
    assert thoughts[1]["message_ids"] == ["2"]


def test_group_into_thoughts_reply_always_starts_new_thought():
    # Reply arrives within thought_time of the previous message — must still
    # break the thread rather than merging.
    messages = [
        {"id": "1", "content": "the last message", "timestamp": "2024-06-28T19:24:14.998-05:00"},
        {
            "id": "2",
            "content": "hello",
            "timestamp": "2024-06-28T19:24:16.000-05:00",
            "reference": {"type": "Default", "messageId": "1", "channelId": "c", "guildId": "g"},
        },
    ]
    thoughts = group_into_thoughts(messages, thought_time=5)
    assert len(thoughts) == 2
    assert thoughts[0]["reply_reference_id"] is None
    assert thoughts[1]["message_ids"] == ["2"]
    assert thoughts[1]["reply_reference_id"] == "1"


def test_group_into_thoughts_reply_reference_id_none_when_not_reply():
    messages = [{"id": "1", "content": "Hello", "timestamp": "2023-01-01T00:00:00.000Z"}]
    thoughts = group_into_thoughts(messages, thought_time=5)
    assert thoughts[0]["reply_reference_id"] is None


def test_group_into_thoughts_empty_input():
    assert group_into_thoughts([], thought_time=5) == []