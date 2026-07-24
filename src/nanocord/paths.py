import pathlib

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

# DiscordChatExporter logs path
DISCORD_CHAT_EXPORTER_LOGS_PATH = RAW_DATA_DIR

# Dataset path (for generated datasets)
DATASET_PATH = PROCESSED_DATA_DIR