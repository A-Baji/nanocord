import json
import pathlib
import tempfile
from unittest.mock import patch

from nanocord.dataset.dataset import parse_logs, create_export
from nanocord.dataset.thoughts import UserNotFoundError


def test_txt_format_output():
    """Test that the dataset can be created in TXT format"""
    # Mock data for testing
    mock_log_data = {
        "messages": [
            {
                "author": {"name": "testuser", "discriminator": "1234", "id": "123456789"},
                "content": "Hello world",
                "timestamp": "2023-01-01T00:00:00.000Z"
            },
            {
                "author": {"name": "testuser", "discriminator": "1234", "id": "123456789"},
                "content": "How are you?",
                "timestamp": "2023-01-01T00:01:00.000Z"
            }
        ]
    }

    # Create a temporary file for testing
    import os

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(mock_log_data, f)
        temp_file = f.name

    try:
        # Test TXT format creation
        result_path = parse_logs(temp_file, "channel123", "123456789", format_type="txt")

        # Verify the file was created and contains expected content
        assert result_path.exists()
        assert result_path.suffix == ".txt"

        # Read the content to verify it's just the thoughts without JSON structure
        with open(result_path, 'r') as f:
            content = f.read()

        # Should contain the combined thought content
        assert "Hello world How are you?" in content

    finally:
        # Clean up temp file
        os.unlink(temp_file)
        # Clean up generated dataset file
        if result_path.exists():
            os.unlink(result_path)


def test_json_format_still_works():
    """Test that the default JSON format still works correctly"""
    # Mock data for testing
    mock_log_data = {
        "messages": [
            {
                "author": {"name": "testuser", "discriminator": "1234", "id": "123456789"},
                "content": "Hello world",
                "timestamp": "2023-01-01T00:00:00.000Z"
            },
            {
                "author": {"name": "testuser", "discriminator": "1234", "id": "123456789"},
                "content": "How are you?",
                "timestamp": "2023-01-01T00:01:00.000Z"
            }
        ]
    }

    # Create a temporary file for testing
    import os

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(mock_log_data, f)
        temp_file = f.name

    try:
        # Test JSON format creation (default)
        result_path = parse_logs(temp_file, "channel123", "123456789")

        # Verify the file was created and contains expected content
        assert result_path.exists()
        assert result_path.suffix == ".jsonl"

        # Read the content to verify it's proper JSON format
        with open(result_path, 'r') as f:
            content = f.read()

        # Should contain JSON entries
        assert '{"prompt"' in content or '"completion"' in content

    finally:
        # Clean up temp file
        os.unlink(temp_file)
        # Clean up generated dataset file
        if result_path.exists():
            os.unlink(result_path)


def test_create_export_txt_format():
    """Test that create_export function supports TXT format"""
    # Mock data for testing
    mock_log_data = {
        "messages": [
            {
                "author": {"name": "testuser", "discriminator": "1234", "id": "123456789"},
                "content": "Hello world",
                "timestamp": "2023-01-01T00:00:00.000Z"
            }
        ]
    }

    # Create a temporary file for testing
    import os

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(mock_log_data, f)
        temp_file = f.name

    try:
        # Mock the export_channel_logs function to avoid actual download
        with patch('nanocord.dataset.export_channel_logs'):
            # Test TXT format creation through create_export
            result_path = create_export(
                "channel123",
                "123456789",
                "fake_token",
                format_type="txt"
            )

            # Verify that a TXT file was created (but we can't easily test the actual content)
            # since we're mocking the export function
            assert result_path is not None

    finally:
        # Clean up temp file
        os.unlink(temp_file)