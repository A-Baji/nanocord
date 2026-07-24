"""
Continued Pretraining (CPT) training functions.
"""

from pathlib import Path
from typing import Dict


def run_cpt_training(config: Dict) -> Path:
    """
    LoRA continued pretraining via Unsloth on the CPT dataset,
    returns checkpoint path.

    This function will perform LoRA fine-tuning on a base model using
    the CPT dataset to continue pretraining on Discord chat data.

    Args:
        config: Merged configuration dictionary containing training settings

    Returns:
        Path: Path to the trained model checkpoint

    Raises:
        NotImplementedError: This is a stub implementation that will be implemented later
    """
    raise NotImplementedError("CPT training not yet implemented")