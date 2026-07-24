"""
Bot registration and serving functions.
"""

from typing import Dict


def register_bot(config: Dict) -> None:
    """
    Exports the fine-tuned model to GGUF, serves via Ollama
    with configurable keep_alive, registers a Discord slash command.

    This function will handle exporting the trained model to GGUF format,
    serving it via Ollama, and registering the appropriate Discord slash commands
    for the /oracle and /adib models.

    Args:
        config: Merged configuration dictionary containing bot settings

    Raises:
        NotImplementedError: This is a stub implementation that will be implemented later
    """
    raise NotImplementedError("Bot registration not yet implemented")