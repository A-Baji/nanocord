"""
Supervised Fine-Tuning (SFT) training functions.
"""

from pathlib import Path
from typing import Dict


def run_sft_training(config: Dict) -> Path:
    """
    LoRA fine-tune via Unsloth on top of the CPT checkpoint,
    using the SFT dataset, returns final model path.

    This function will perform supervised fine-tuning on a base model
    using the SFT dataset to make it respond conversationally in the user's voice.

    Args:
        config: Merged configuration dictionary containing training settings

    Returns:
        Path: Path to the final fine-tuned model

    Raises:
        NotImplementedError: This is a stub implementation that will be implemented later
    """
    raise NotImplementedError("SFT training not yet implemented")