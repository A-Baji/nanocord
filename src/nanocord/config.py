"""
Configuration system for NanoCord CLI.
Handles loading YAML config files and merging with CLI arguments.
"""

import os
from typing import Any, Dict, Optional

import yaml


def load_and_merge_config(yaml_path: Optional[str], cli_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file and merge with CLI arguments.

    Args:
        yaml_path: Path to the YAML config file (optional)
        cli_args: Dictionary of CLI arguments

    Returns:
        Merged configuration dictionary
    """
    # Start with default values
    config = {
        "channel": None,
        "user": None,
        "thought_time": 5,
        "thought_max": None,
        "thought_min": 6,
        "max_entries": 1000,
        "offset": 0,
        "distributed": False,
        "reverse": False,
        "redownload": False
    }

    # Load from YAML if provided
    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                # Handle the case where discord_token might be at top level (legacy)
                # and move it to dataset section if needed
                if "discord_token" in yaml_config and "dataset" in yaml_config:
                    # If both exist, we need to check if discord_token is in dataset or top-level
                    if "discord_token" not in yaml_config["dataset"]:
                        # Move top-level discord_token to dataset section
                        yaml_config["dataset"]["discord_token"] = yaml_config["discord_token"]
                        del yaml_config["discord_token"]
                config.update(yaml_config)

    # CLI arguments override YAML config
    config.update(cli_args)

    return config