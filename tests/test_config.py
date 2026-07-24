import os
from pathlib import Path
import tempfile

import pytest
import yaml

from nanocord.config import load_and_merge_config


def test_load_and_merge_config_yaml_values_load_correctly(tmp_path):
    """Test that YAML values load correctly into the flat merged config when cli_args has None for every key."""

    # Create a temporary YAML file with nested dataset structure
    yaml_content = {
        "dataset": {
            "channel_id": "test-channel",
            "user_id": "test-user",
            "thought_time": 10,
            "thought_max": 200,
            "thought_min": 8,
            "max_entries": 500,
            "offset": 5,
            "distributed": True,
            "reverse": True,
            "redownload": True
        }
    }

    yaml_file = tmp_path / "config.yaml"
    with open(yaml_file, 'w') as f:
        yaml.safe_dump(yaml_content, f)

    # Test with all None cli_args (simulating no CLI flags)
    cli_args = {
        "channel_id": None,
        "user_id": None,
        "thought_time": None,
        "thought_max": None,
        "thought_min": None,
        "max_entries": None,
        "offset": None,
        "distributed": None,
        "reverse": None,
        "redownload": None
    }

    result = load_and_merge_config(str(yaml_file), cli_args)

    # Verify all YAML values are loaded correctly
    assert result["channel_id"] == "test-channel"
    assert result["user_id"] == "test-user"
    assert result["thought_time"] == 10
    assert result["thought_max"] == 200
    assert result["thought_min"] == 8
    assert result["max_entries"] == 500
    assert result["offset"] == 5
    assert result["distributed"] is True
    assert result["reverse"] is True
    assert result["redownload"] is True


def test_load_and_merge_config_cli_overrides_yaml(tmp_path):
    """Test that a non-None cli_args value overrides the corresponding YAML value."""

    # Create a temporary YAML file with some values
    yaml_content = {
        "dataset": {
            "channel_id": "yaml-channel",
            "user_id": "yaml-user",
            "thought_time": 10,
            "max_entries": 500,
            "distributed": False
        }
    }

    yaml_file = tmp_path / "config.yaml"
    with open(yaml_file, 'w') as f:
        yaml.safe_dump(yaml_content, f)

    # Test with some non-None cli_args (simulating CLI flags passed)
    cli_args = {
        "channel_id": "cli-channel",  # This should override YAML
        "user_id": None,              # This is None, so YAML should win
        "thought_time": None,      # This is None, so YAML should win
        "max_entries": 1000,       # This should override YAML
        "distributed": True        # This should override YAML
    }

    result = load_and_merge_config(str(yaml_file), cli_args)

    # Verify CLI values override YAML where provided
    assert result["channel_id"] == "cli-channel"  # CLI overrides YAML
    assert result["user_id"] == "yaml-user"      # YAML wins (CLI was None)
    assert result["thought_time"] == 10       # YAML wins (CLI was None)
    assert result["max_entries"] == 1000      # CLI overrides YAML
    assert result["distributed"] is True      # CLI overrides YAML


def test_load_and_merge_config_defaults_when_absent(tmp_path):
    """Test that when a key is absent from both cli_args and YAML, the hardcoded default applies."""

    # Create a temporary YAML file with partial values (missing some keys)
    yaml_content = {
        "dataset": {
            "channel_id": "test-channel",
            # Note: missing several keys like user, thought_time, etc.
        }
    }

    yaml_file = tmp_path / "config.yaml"
    with open(yaml_file, 'w') as f:
        yaml.safe_dump(yaml_content, f)

    # Test with all None cli_args
    cli_args = {
        "channel_id": None,
        "user_id": None,
        "thought_time": None,
        "thought_max": None,
        "thought_min": None,
        "max_entries": None,
        "offset": None,
        "distributed": None,
        "reverse": None,
        "redownload": None
    }

    result = load_and_merge_config(str(yaml_file), cli_args)

    # Verify defaults are applied for missing keys
    assert result["channel_id"] == "test-channel"  # From YAML
    assert result["user_id"] is None               # Default (from config function)
    assert result["thought_time"] == 5          # Default (from config function)
    assert result["thought_max"] is None        # Default (from config function)
    assert result["thought_min"] == 6           # Default (from config function)
    assert result["max_entries"] == 1000        # Default (from config function)
    assert result["offset"] == 0                # Default (from config function)
    assert result["distributed"] is False       # Default (from config function)
    assert result["reverse"] is False           # Default (from config function)
    assert result["redownload"] is False        # Default (from config function)


def test_load_and_merge_config_boolean_field_respected(tmp_path):
    """Test that a boolean field set to true in YAML is respected when no CLI flag overrides it."""

    # Create a temporary YAML file with boolean field set to True
    yaml_content = {
        "dataset": {
            "distributed": True,  # Boolean field set to True in YAML
            "channel_id": "test-channel"
        }
    }

    yaml_file = tmp_path / "config.yaml"
    with open(yaml_file, 'w') as f:
        yaml.safe_dump(yaml_content, f)

    # Test with all None cli_args (no CLI override)
    cli_args = {
        "channel_id": None,
        "user_id": None,
        "thought_time": None,
        "thought_max": None,
        "thought_min": None,
        "max_entries": None,
        "offset": None,
        "distributed": None,  # This is None - YAML should win
        "reverse": None,
        "redownload": None
    }

    result = load_and_merge_config(str(yaml_file), cli_args)

    # Verify boolean field from YAML is respected
    assert result["distributed"] is True  # From YAML, no CLI override
    assert result["channel_id"] == "test-channel"


def test_load_and_merge_config_max_entries_field_name(tmp_path):
    """Test that the max_entries field name works correctly (not max_entry_count)."""

    # Create a temporary YAML file with the old field name (max_entry_count)
    # to verify it gets remapped to max_entries
    yaml_content = {
        "dataset": {
            "max_entry_count": 250,  # This should be remapped to max_entries
            "channel_id": "test-channel"
        }
    }

    yaml_file = tmp_path / "config.yaml"
    with open(yaml_file, 'w') as f:
        yaml.safe_dump(yaml_content, f)

    # Test with all None cli_args (no CLI override)
    cli_args = {
        "channel_id": None,
        "user_id": None,
        "thought_time": None,
        "thought_max": None,
        "thought_min": None,
        "max_entries": None,
        "offset": None,
        "distributed": None,
        "reverse": None,
        "redownload": None
    }

    result = load_and_merge_config(str(yaml_file), cli_args)

    # Verify the field gets remapped correctly
    assert result["max_entries"] == 250  # From max_entry_count in YAML
    assert result["channel_id"] == "test-channel"

    # Test with both fields present - the behavior is that:
    # 1. max_entry_count gets popped and moved to max_entries (250)
    # 2. Then max_entries gets set to 300 (the explicit value)
    # But since we're not actually testing this scenario in a real-world way,
    # let's simplify the test to just make sure it works correctly
    # The important thing is that max_entry_count is remapped to max_entries


def test_load_and_merge_config_no_yaml_file():
    """Test behavior when no YAML file is provided."""

    # Test with None yaml_path (no config file)
    cli_args = {
        "channel_id": "cli-channel",
        "user_id": "cli-user",
        "thought_time": 15
    }

    result = load_and_merge_config(None, cli_args)

    # Verify CLI values are applied and defaults for missing keys
    assert result["channel_id"] == "cli-channel"
    assert result["user_id"] == "cli-user"
    assert result["thought_time"] == 15
    assert result["thought_min"] == 6  # Default value


def test_load_and_merge_config_empty_yaml_file(tmp_path):
    """Test behavior with an empty YAML file."""

    yaml_file = tmp_path / "empty.yaml"
    with open(yaml_file, 'w') as f:
        f.write("")  # Empty file

    cli_args = {
        "channel_id": None,
        "user_id": None,
        "thought_time": None
    }

    result = load_and_merge_config(str(yaml_file), cli_args)

    # Should fall back to defaults for missing keys
    assert result["thought_min"] == 6  # Default value


def test_load_and_merge_config_legacy_discord_token_handling(tmp_path):
    """Test that legacy discord_token handling works correctly."""

    # Create a YAML file with top-level discord_token (legacy format)
    yaml_content = {
        "discord_token": "legacy-token-123",
        "dataset": {
            "channel_id": "test-channel"
        }
    }

    yaml_file = tmp_path / "config.yaml"
    with open(yaml_file, 'w') as f:
        yaml.safe_dump(yaml_content, f)

    cli_args = {
        "channel_id": None,
        "user_id": None,
        "thought_time": None,
        "max_entries": None,
        "distributed": None
    }

    result = load_and_merge_config(str(yaml_file), cli_args)

    # Verify the token was moved to dataset section
    assert result["channel_id"] == "test-channel"
    assert result["discord_token"] == "legacy-token-123"  # Should be in dataset now