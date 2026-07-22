import pytest
from unittest.mock import patch, MagicMock

from nanocord.thoughts import (
    validate_thought,
    cleanup_string,
    build_thought,
    UserNotFoundError
)


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