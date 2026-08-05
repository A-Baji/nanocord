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
from nanocord import global_logger


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


def resolve_eos_ids(tokenizer) -> List[int]:
    """
    Resolve EOS token IDs including the standard EOS token.

    Args:
        tokenizer: The tokenizer object

    Returns:
        List of token IDs that should be treated as EOS tokens
    """
    eos_ids = []

    # Add the standard EOS token ID
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)

    # Add ChatML stop tokens
    chatml_stop_tokens = ["</s>", "<!--", "<tool_call>"]
    unk_token_id = getattr(tokenizer, "unk_token_id", None)

    for token in chatml_stop_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != unk_token_id and token_id != -1:
            eos_ids.append(token_id)

    return eos_ids


def truncate_at_eos(token_ids, eos_ids):
    """
    Truncate token IDs at the first occurrence of any EOS ID.

    Args:
        token_ids: Tensor of token IDs
        eos_ids: List of EOS token IDs to look for

    Returns:
        Truncated tensor of token IDs
    """
    import torch

    # Convert eos_ids to a set for faster lookup
    eos_set = set(eos_ids)

    # Find the first occurrence of an EOS token
    for i, token_id in enumerate(token_ids):
        if token_id in eos_set:
            return token_ids[:i]

    # If no EOS found, return the original tensor
    return token_ids


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
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from unsloth import FastLanguageModel

    try:
        # Try to load with FastLanguageModel first
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            dtype=None,  # Let Unsloth handle dtype
            device_map="cuda:0",
            load_in_4bit=True,
        )
    except Exception as e:
        global_logger.exception(f"Failed to load model with FastLanguageModel: {e}")
        global_logger.info("Falling back to AutoModelForCausalLM...")

        # Fall back to AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            load_in_4bit=True,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        # Set model to evaluation mode
        model.eval()

    # Ensure max_length is None in generation_config if it exists
    if hasattr(model.generation_config, "max_length"):
        model.generation_config.max_length = None

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

    # Prepare shared generation parameters with eos_token_id
    gen_kwargs = {
        "temperature": preset.get("temperature", 0.7),
        "repetition_penalty": preset.get("repetition_penalty", 1.0),
        "no_repeat_ngram_size": preset.get("no_repeat_ngram_size", 2),
        "max_new_tokens": preset.get("max_new_tokens", 512),
        "eos_token_id": tokenizer.eos_token_id,
    }

    # Add sampling parameters if present in preset
    if "top_p" in preset:
        gen_kwargs["top_p"] = preset["top_p"]
    if "top_k" in preset:
        gen_kwargs["top_k"] = preset["top_k"]
    if "min_p" in preset:
        gen_kwargs["min_p"] = preset["min_p"]

    # Add banned tokens if requested
    if preset.get("ban_parens"):
        # Get banned token IDs for parentheses
        banned_ids = []
        special_tokens = ["(", ")", "（", "）"]
        for token in special_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id and token_id != -1:
                banned_ids.append(token_id)

        if banned_ids:
            gen_kwargs["bad_words_ids"] = [banned_ids]

    # Add lowercase first token processor if requested
    if preset.get("lowercase_first_token"):
        from transformers import LogitsProcessor, LogitsProcessorList

        class LowercaseFirstTokenLogitsProcessor(LogitsProcessor):
            def __init__(self, tokenizer, prompt_length):
                self.prompt_length = prompt_length

                # Build capital_ids by iterating through the tokenizer's vocabulary
                self.capital_ids = []
                for token_str, token_id in tokenizer.get_vocab().items():
                    # Strip BPE space markers (Ġ, leading space) from token string
                    clean_str = token_str.lstrip('Ġ ')
                    # Check if the first character is an uppercase ASCII letter
                    if len(clean_str) > 0 and clean_str[0].isupper() and clean_str[0].isascii():
                        self.capital_ids.append(token_id)
                # Convert to tensor for efficient device transfer
                self.capital_ids = torch.tensor(self.capital_ids, dtype=torch.long)

            def __call__(self, input_ids, scores):
                if input_ids.shape[-1] == self.prompt_length:
                    # Set scores for capital IDs to negative infinity
                    scores[:, self.capital_ids.to(scores.device)] = -float("inf")
                return scores

        # Create logits processor list and add our custom processor
        logits_processor = LogitsProcessorList([])
        gen_kwargs["logits_processor"] = logits_processor

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

        # Set the prompt_length for the lowercase first token processor
        if preset.get("lowercase_first_token"):
            logits_processor.append(LowercaseFirstTokenLogitsProcessor(tokenizer, inputs.shape[-1]))

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                **gen_kwargs,
            )

        # Decode only the newly generated tokens (not the prompt/chat template)
        generated_tokens = outputs[0][inputs.shape[-1]:]

        # Truncate at EOS tokens
        eos_ids = resolve_eos_ids(tokenizer)
        generated_tokens = truncate_at_eos(generated_tokens, eos_ids)

        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    else:
        # Fall back to raw completion behavior for compatibility
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

        # Set the prompt_length for the lowercase first token processor
        if preset.get("lowercase_first_token"):
            logits_processor.append(LowercaseFirstTokenLogitsProcessor(tokenizer, inputs.shape[-1]))

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Decode and return the generated text
        generated_tokens = outputs[0]

        # Truncate at EOS tokens
        eos_ids = resolve_eos_ids(tokenizer)
        generated_tokens = truncate_at_eos(generated_tokens, eos_ids)

        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # TODO: sanitize_response - see infer.py fix status in project notes comment instead,
    # since that regex fix hasn't been finalized yet.

    return response