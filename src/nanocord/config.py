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

    Params declared at a shallower level of the section path are inherited by every
    deeper level below it; a deeper level redeclaring the same key overrides the
    inherited value. E.g. with section="dataset.cpt", a `channel_id` set under
    `dataset:` (a sibling of `cpt:`/`sft:`, not inside either) applies to both
    `dataset.cpt` and `dataset.sft` unless `dataset.cpt:` (or `dataset.sft:`)
    redeclares it. A scalar at the true YAML root is inherited the same way, which
    also covers the legacy top-level `discord_token` case.

    Returns:
        Merged configuration dictionary
    """
    # Start with default values
    config = {
        "channel_id": None,
        "user_id": None,
        "discord_token": None,
        "discord_chat_exporter_path": None,
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
                # Walk the section path one key at a time, folding in each level's
                # scalar/list values before descending further. Deeper levels are
                # merged later and so override same-named keys from shallower ones.
                inherited: Dict[str, Any] = {}
                node = yaml_config
                if isinstance(node, dict):
                    inherited.update({k: v for k, v in node.items() if not isinstance(v, dict)})

                for key in section.split('.'):
                    if not isinstance(node, dict) or key not in node:
                        node = {}
                        break
                    node = node[key]
                    if isinstance(node, dict):
                        inherited.update({k: v for k, v in node.items() if not isinstance(v, dict)})

                # Remap field names from YAML to match expected parameter names
                # max_entry_count -> max_entries
                if "max_entry_count" in inherited:
                    inherited["max_entries"] = inherited.pop("max_entry_count")

                # Merge the flattened config into main config
                config.update(inherited)

    # CLI arguments override YAML config (but only those with non-None values)
    # Filter out None values from cli_args to avoid overriding YAML values
    filtered_cli_args = {k: v for k, v in cli_args.items() if v is not None}
    config.update(filtered_cli_args)

    return config