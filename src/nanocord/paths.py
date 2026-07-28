import pathlib
from typing import Dict

import appdirs

# App data directory for nanocord
DATA_DIR = pathlib.Path(appdirs.user_data_dir(appname="nanocord"))

# Raw and processed data directories
RAW_DATA_DIR = DATA_DIR / "raw" / "discordchat_export"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Configuration path
CONFIG_PATH = DATA_DIR / "config.yaml"

def resolve_output_dir(config: Dict) -> pathlib.Path:
    """
    Returns pathlib.Path(config["output_dir"]) if config.get("output_dir") is set,
    else DATA_DIR (the existing constant, unchanged) - matching today's behavior
    exactly when output_dir is unset. Does NOT create any subdirectories itself.
    """
    return pathlib.Path(config.get("output_dir") or DATA_DIR)