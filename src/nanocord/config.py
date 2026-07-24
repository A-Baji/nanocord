"""
Configuration system for NanoCord CLI.
Handles loading YAML config files and merging with CLI arguments.
"""

import os
from typing import Any, Dict, Optional

import yaml


def load_and_merge_config(yaml_path: Optional[str], cli_args: Dict[str, Any], section: str = "dataset") -> Dict[str, Any]:
    """
    Load configuration from a YAML file and merge with CLI arguments.

    Args:
        yaml_path: Path to the YAML config file (optional)
        cli_args: Dictionary of CLI arguments
        section: The section path to load from the YAML config (supports dot notation like "dataset.cpt")

    Returns:
        Merged configuration dictionary
    """
    # Start with default values
    config = {
        "channel_id": None,
        "user_id": None,
        "discord_token": None,
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
                # Navigate to the specified section using dot notation
                section_config = yaml_config
                for key in section.split('.'):
                    if key in section_config:
                        section_config = section_config[key]
                    else:
                        section_config = {}
                        break

                # Handle the case where discord_token might be at top level (legacy)
                # and move it to dataset section if needed
                if "discord_token" in yaml_config and "dataset" in yaml_config:
                    # If both exist, we need to check if discord_token is in dataset or top-level
                    if "discord_token" not in yaml_config["dataset"]:
                        # Move top-level discord_token to dataset section
                        section_config["discord_token"] = yaml_config["discord_token"]

                # Remap field names from YAML to match expected parameter names
                # max_entry_count -> max_entries
                if "max_entry_count" in section_config:
                    section_config["max_entries"] = section_config.pop("max_entry_count")

                # Merge the flattened config into main config
                config.update(section_config)

    # CLI arguments override YAML config (but only those with non-None values)
    # Filter out None values from cli_args to avoid overriding YAML values
    filtered_cli_args = {k: v for k, v in cli_args.items() if v is not None}
    config.update(filtered_cli_args)

    return config