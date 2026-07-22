"""NanoCord package initialization."""

# Setup global logger for the entire application
from nanocord.logger import setup_logger
from nanocord.paths import DATA_DIR

# Create a global logger that can be used across all modules
global_logger = setup_logger('nanocord', DATA_DIR / 'nanocord.log')