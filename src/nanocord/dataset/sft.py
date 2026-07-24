"""
Dataset creation functions for SFT (Supervised Fine-Tuning) datasets.
"""

from pathlib import Path
from typing import Dict


def build_sft_dataset(config: Dict) -> Path:
    """
    Extract DiscordChatExporter reply-reference pairs into (context -> reply) JSONL,
    builds 3 context-window variants (single message / 2-3 messages / full window).

    This function will process the CPT dataset and create SFT datasets with different
    context window sizes for fine-tuning a model to respond conversationally.

    Args:
        config: Merged configuration dictionary containing dataset settings

    Returns:
        Path: Path to the created SFT dataset file

    Raises:
        NotImplementedError: This is a stub implementation that will be implemented later
    """
    raise NotImplementedError("SFT dataset building not yet implemented")