"""
Continued Pretraining (CPT) training functions.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, Optional


class BaseModel(str, Enum):
    SMOLLM3_3B = "smollm3-3b"
    QWEN3_4B = "qwen3-4b"
    QWEN3_1_7B = "qwen3-1.7b"
    LLAMA_3_2_3B = "llama-3.2-3b"
    QWEN2_5_7B = "qwen2.5-7b"


DEFAULT_BASE_MODEL = BaseModel.QWEN2_5_7B

# Maps our enum values to the actual Unsloth/HF repo ids to load.
BASE_MODEL_HF_IDS = {
    BaseModel.SMOLLM3_3B: "unsloth/SmolLM3-3B-Base",
    BaseModel.QWEN3_4B: "unsloth/Qwen3-4B-Base",
    BaseModel.QWEN3_1_7B: "unsloth/Qwen3-1.7B-Base",
    BaseModel.LLAMA_3_2_3B: "unsloth/Llama-3.2-3B",
    BaseModel.QWEN2_5_7B: "unsloth/Qwen2.5-7B",
}

# Fixed - LoRA is applied to all seven major linear layers (attention + MLP).
# Not config-driven; do not expose this as a config key.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class MissingDatasetIdentifiersError(Exception):
    """Raised when config lacks channel_id/user_id needed to locate the
    CPT dataset and name the output checkpoint."""
    pass


def resolve_base_model(name: Optional[str]) -> str:
    """
    Resolve a train.cpt.base_model config value (an enum string, or
    falsy/None) to a concrete Unsloth/HF repo id.

    Args:
        name: One of BaseModel's string values (e.g. "smollm3-3b"), or
              None/empty to use DEFAULT_BASE_MODEL.

    Returns:
        The HF repo id string to pass to FastLanguageModel.from_pretrained.

    Raises:
        ValueError: If `name` is truthy but not a recognized BaseModel value.
    """
    if not name:
        return BASE_MODEL_HF_IDS[DEFAULT_BASE_MODEL]

    try:
        model = BaseModel(name)
        return BASE_MODEL_HF_IDS[model]
    except ValueError:
        raise ValueError(f"Unknown base model: {name}")


def resolve_lora_config(config: Dict) -> Dict:
    """
    Build the kwarg dict for FastLanguageModel.get_peft_model from a merged
    train.cpt config dict.

    Reads: config["lora_r"] (default 16), config["lora_alpha"]
    (default: same value as the resolved lora_r), config["lora_dropout"]
    (default 0), config["seed"] (default 3407).

    target_modules is always LORA_TARGET_MODULES (not config-driven).
    bias is always "none". use_gradient_checkpointing is always "unsloth".
    use_rslora is always False. loftq_config is always None.

    Returns:
        Dict of kwargs ready to unpack into FastLanguageModel.get_peft_model.
    """
    lora_r = config.get("lora_r", 16)
    lora_alpha = config.get("lora_alpha", lora_r)
    lora_dropout = config.get("lora_dropout", 0)
    seed = config.get("seed", 3407)

    return {
        "r": lora_r,
        "target_modules": LORA_TARGET_MODULES,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "use_rslora": False,
        "loftq_config": None,
        "random_state": seed,
    }


def compute_gradient_accumulation_steps(effective_batch_size: int, per_device_train_batch_size: int) -> int:
    """
    Args:
        effective_batch_size: Target effective batch size (batch_size *
                               gradient_accumulation_steps).
        per_device_train_batch_size: The starting per-device batch size
                                      (before any auto_find_batch_size
                                      backoff at train time).

    Returns:
        max(1, effective_batch_size // per_device_train_batch_size).
        Note: if HF's auto_find_batch_size backs the batch size down at
        train time due to OOM, gradient_accumulation_steps stays fixed at
        the value this function returns (computed from the starting
        per_device_train_batch_size) - the realized effective batch size
        simply shrinks along with it. This is standard HF Trainer behavior,
        not a bug to work around.
    """
    return max(1, effective_batch_size // per_device_train_batch_size)


def resolve_training_args(config: Dict, output_dir: Path) -> Dict:
    """
    Build the kwarg dict for transformers.TrainingArguments from a merged
    train.cpt config dict.

    Reads (all with defaults if absent from config):
      - effective_batch_size (default 16)
      - per_device_train_batch_size (default 2)
      - num_train_epochs (default 3)
      - learning_rate (default 2e-4)
      - warmup_ratio (default 0.05)
      - weight_decay (default 0.01)
      - seed (default 3407)

    Must set:
      - auto_find_batch_size = True
      - gradient_accumulation_steps via compute_gradient_accumulation_steps(
        config.get("effective_batch_size", 16), config.get("per_device_train_batch_size", 2))
      - lr_scheduler_type = "linear"
      - optim = "adamw_8bit"
      - output_dir = str(output_dir)
      - eval_strategy = "steps", eval_steps = 50
      - save_strategy = "steps", save_steps = 50
      - load_best_model_at_end = True
      - metric_for_best_model = "eval_loss"
      - greater_is_better = False
      - report_to = "none"
      - logging_steps = 10

    Returns:
        Dict of kwargs ready to unpack into transformers.TrainingArguments.
    """
    effective_batch_size = config.get("effective_batch_size", 16)
    per_device_train_batch_size = config.get("per_device_train_batch_size", 2)
    num_train_epochs = config.get("num_train_epochs", 3)
    learning_rate = config.get("learning_rate", 2e-4)
    warmup_ratio = config.get("warmup_ratio", 0.05)
    weight_decay = config.get("weight_decay", 0.01)
    seed = config.get("seed", 3407)

    gradient_accumulation_steps = compute_gradient_accumulation_steps(
        effective_batch_size, per_device_train_batch_size
    )

    return {
        "auto_find_batch_size": True,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lr_scheduler_type": "linear",
        "optim": "adamw_8bit",
        "output_dir": str(output_dir),
        "eval_strategy": "steps",
        "eval_steps": 50,
        "save_strategy": "steps",
        "save_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "logging_steps": 10,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "seed": seed,
    }


def run_cpt_training(config: Dict) -> Path:
    """
    Main entrypoint. Loads the base model via Unsloth, applies LoRA, loads
    the CPT JSONL dataset, splits off an eval set, trains with early
    stopping on eval_loss, and saves the LoRA adapter.

    Args:
        config: Merged train.cpt configuration dict. In addition to the
                train.cpt.* keys documented on the helpers above, this dict
                MUST also contain "channel_id" and "user_id". Validate this
                directly in this function (not just assume a caller
                checked), for the same reason dataset/sft.py validates
                persona_name inside build_sft_dataset itself.

    Returns:
        Path: directory the LoRA adapter was saved to,
              MODEL_PATH / f"{user_id}_{channel_id}_cpt_lora"

    Raises:
        MissingDatasetIdentifiersError: if channel_id or user_id missing
                                        from config.
        ValueError: if config["base_model"] is set but not a recognized
                    BaseModel value.
        FileNotFoundError: if the expected CPT dataset JSONL file
                            (nanocord.paths.DATASET_PATH /
                            f"{user_id}_{channel_id}_cpt_data_set.jsonl")
                            does not exist - raise with a clear message
                            telling the user to run `dataset cpt` first.
                            Check this BEFORE importing/loading unsloth or
                            any model, so this error path never triggers a
                            GPU/model-download attempt.
    """
    channel_id = config.get("channel_id")
    user_id = config.get("user_id")
    if not channel_id or not user_id:
        raise MissingDatasetIdentifiersError(
            "train.cpt requires channel_id and user_id (normally merged in "
            "from dataset.cpt in config.yaml) to locate the CPT dataset "
            "and name the output checkpoint."
        )

    from nanocord.paths import DATASET_PATH, MODEL_PATH

    dataset_file = DATASET_PATH / f"{user_id}_{channel_id}_cpt_data_set.jsonl"
    if not dataset_file.exists():
        raise FileNotFoundError(
            f"CPT dataset not found at {dataset_file}. "
            "Please run `dataset cpt` first to generate the CPT dataset."
        )

    # Import unsloth-related libraries only when needed
    from unsloth import FastLanguageModel

    hf_repo_id = resolve_base_model(config.get("base_model"))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hf_repo_id,
        max_seq_length=config.get("max_seq_length", 2048),
        load_in_4bit=config.get("load_in_4bit", True),
    )

    # Force disable the fused cross-entropy module causing the scratchpad OOM
    if hasattr(model, "config"):
        model.config.use_fused_ce = False

    # Enforce Unsloth's optimized gradient checkpointing 
    # This prevents storing all intermediate activations and saves massive VRAM.
    lora_kwargs = resolve_lora_config(config)
    lora_kwargs["use_gradient_checkpointing"] = lora_kwargs.get("use_gradient_checkpointing", "unsloth")
    
    model = FastLanguageModel.get_peft_model(model, **lora_kwargs)

    from datasets import load_dataset

    full_dataset = load_dataset("json", data_files=str(dataset_file))["train"]
    split = full_dataset.train_test_split(test_size=config.get("eval_split", 0.05), seed=config.get("seed", 3407))
    train_dataset, eval_dataset = split["train"], split["test"]

    output_dir = MODEL_PATH / f"{user_id}_{channel_id}_cpt_lora"

    from transformers import EarlyStoppingCallback
    from trl import SFTTrainer, SFTConfig

    # Grab the resolved base dictionary
    training_args_dict = resolve_training_args(config, output_dir)

    # vram_safe_max_seq_length, if set, clamps max_seq_length down for VRAM-constrained
    # hardware without silently overriding config["max_seq_length"] itself. Unset (None)
    # means use config["max_seq_length"] (default 2048) unchanged.
    vram_safe_max_seq_length = config.get("vram_safe_max_seq_length")
    effective_max_seq_length = (
        min(config.get("max_seq_length", 2048), vram_safe_max_seq_length)
        if vram_safe_max_seq_length
        else config.get("max_seq_length", 2048)
    )

    training_args_dict.update({
        "dataset_text_field": "text",
        "max_seq_length": effective_max_seq_length,
        "packing": config.get("packing", True),
        "optim": "paged_adamw_8bit",  # VRAM-safe optimizer; overrides resolve_training_args' adamw_8bit default
    })

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(**training_args_dict),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.get("early_stopping_patience", 3))],
    )

    trainer.train()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return output_dir