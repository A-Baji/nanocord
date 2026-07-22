import logging
import os


def setup_logger(name, log_file, level=logging.WARNING):
    """Function to setup as many loggers as you want"""

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)

    # Respect LOG_LEVEL environment variable if set
    log_level_env = os.getenv('LOG_LEVEL', '').upper()
    if log_level_env:
        try:
            level = getattr(logging, log_level_env)
        except AttributeError:
            # If invalid level specified, fall back to WARNING
            level = logging.WARNING

    logger.setLevel(level)
    logger.addHandler(handler)

    # Also add a console handler for INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger