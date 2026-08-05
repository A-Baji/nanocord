"""
Inference module for NanoCord - handles model loading, response generation,
and bot configuration management.

This module contains pure functions that can be imported by both the CLI
and the bot registration code without any CLI/typer dependencies.
"""

from pathlib import Path
from typing import Dict, Optional, List, Tuple
import random

# Import from our own modules
from nanocord.paths import resolve_output_dir
from nanocord.config import load_raw_config


def resolve_checkpoint_path(
    config: Dict, model_path: Optional[str], stage: Optional[str]
) -> Path:
    """
    Resolve the checkpoint path for a training stage.

    Args:
        config: The full configuration dictionary
        model_path: Explicit model path (if provided)
        stage: Training stage ("cpt" or "sft")

    Returns:
        Path to the resolved checkpoint directory

    Raises:
        ValueError: If neither model_path nor a valid stage is provided,
                    or if required config fields are missing,
                    or if the resolved path doesn't exist
    """
    if model_path is not None:
        return Path(model_path)

    if stage not in ("cpt", "sft"):
        raise ValueError(
            f"Invalid stage '{stage}'. Must be 'cpt' or 'sft'."
        )

    # Check required config fields
    user_id = config.get("user_id")
    channel_id = config.get("channel_id")

    if not user_id:
        raise ValueError(
            "Missing 'user_id' in config. Please run 'nanocord dataset cpt' first."
        )

    if not channel_id:
        raise ValueError(
            "Missing 'channel_id' in config. Please run 'nanocord dataset cpt' first."
        )

    # Build the expected path
    checkpoint_path = (
        resolve_output_dir(config)
        / "models"
        / f"{user_id}_{channel_id}_{stage}_lora"
    )

    if not checkpoint_path.exists():
        raise ValueError(
            f"Checkpoint directory not found: {checkpoint_path}\n"
            f"Please run 'nanocord train {stage}' first to generate the checkpoint."
        )

    return checkpoint_path


def resolve_preset(
    presets: Dict[str, Dict], preset_name: str, preset_pool: Optional[List[str]] = None
) -> Dict:
    """
    Resolve a preset configuration by name or randomly.

    Args:
        presets: Dictionary of available presets
        preset_name: Name of the preset to use, or "random"
        preset_pool: Optional list of preset names to choose from when using "random"

    Returns:
        The resolved preset dictionary

    Raises:
        ValueError: If preset_name is not found or if random selection fails
    """
    if preset_name == "random":
        # Determine the pool to sample from
        if preset_pool and len(preset_pool) > 0:
            available_presets = [p for p in preset_pool if p in presets]
            if not available_presets:
                raise ValueError(
                    f"No valid presets found in preset_pool: {preset_pool}"
                )
        else:
            available_presets = list(presets.keys())

        if not available_presets:
            raise ValueError("No presets available for random selection")

        preset_name = random.choice(available_presets)

    # Look up the preset
    if preset_name not in presets:
        available_names = ", ".join(presets.keys())
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available presets: {available_names}"
        )

    return presets[preset_name]


def load_bot_config_section(config_file: Optional[str]) -> Dict:
    """
    Load the raw bot configuration section from a config file.

    Args:
        config_file: Path to the config file (or None for default)

    Returns:
        The bot section of the configuration as a dictionary
    """
    raw_config = load_raw_config(config_file)
    return raw_config.get("bot", {})


def load_model(model_path: Path) -> Tuple[any, any]:
    """
    Load a model and tokenizer from the given path.

    Args:
        model_path: Path to the model checkpoint

    Returns:
        Tuple of (model, tokenizer)
    """
    # Import torch and other libraries inside the function to avoid module-level dependencies
    import torch
    from transformers import AutoTokenizer
    from unsloth import FastLanguageModel

    # Load model and tokenizer
    # Using the same loading pattern as train/sft.py's CPT-checkpoint reload
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        dtype=None,  # Let Unsloth handle dtype
        device_map="cuda:0",
        load_in_4bit=True,
    )

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, preset: Dict, system_prompt: Optional[str] = None) -> str:
    """
    Generate a response using the provided model and tokenizer with the specified preset.

    Args:
        model: Loaded model object
        tokenizer: Loaded tokenizer object
        prompt: Input prompt for generation
        preset: Dictionary containing generation parameters
        system_prompt: Optional system prompt to prepend to the conversation

    Returns:
        Generated response text
    """
    # Import torch inside the function to avoid module-level dependencies
    import torch

    # Check if the tokenizer has a chat template
    if getattr(tokenizer, "chat_template", None) is not None:
        # Use chat template approach
        messages = [{"role": "user", "content": prompt}]
        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})

        # Apply chat template to get input tensors
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda:0")

        # Prepare generation parameters with eos_token_id
        gen_kwargs = {
            "temperature": preset.get("temperature", 0.7),
            "repetition_penalty": preset.get("repetition_penalty", 1.0),
            "no_repeat_ngram_size": preset.get("no_repeat_ngram_size", 2),
            "max_new_tokens": preset.get("max_new_tokens", 512),
            "eos_token_id": tokenizer.eos_token_id,
        }

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                **gen_kwargs,
            )

        # Decode only the newly generated tokens (not the prompt/chat template)
        generated_tokens = outputs[0][inputs.shape[-1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    else:
        # Fall back to raw completion behavior for compatibility
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

        # Apply preset parameters with defaults
        gen_kwargs = {
            "temperature": preset.get("temperature", 0.7),
            "repetition_penalty": preset.get("repetition_penalty", 1.0),
            "no_repeat_ngram_size": preset.get("no_repeat_ngram_size", 2),
            "max_new_tokens": preset.get("max_new_tokens", 512),
            "eos_token_id": tokenizer.eos_token_id,
        }

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Decode and return the generated text
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # TODO: sanitize_response - see infer.py fix status in project notes comment instead,
    # since that regex fix hasn't been finalized yet.

    return response