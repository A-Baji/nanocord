import pytest
from pathlib import Path
from unittest.mock import patch

from nanocord.train.cpt import (
    resolve_base_model,
    resolve_lora_config,
    compute_gradient_accumulation_steps,
    resolve_training_args,
    run_cpt_training,
    MissingDatasetIdentifiersError,
    BaseModel,
    DEFAULT_BASE_MODEL,
    BASE_MODEL_HF_IDS,
    LORA_TARGET_MODULES
)


def test_resolve_base_model():
    # Test None input
    assert resolve_base_model(None) == BASE_MODEL_HF_IDS[DEFAULT_BASE_MODEL]

    # Test empty string input
    assert resolve_base_model("") == BASE_MODEL_HF_IDS[DEFAULT_BASE_MODEL]

    # Test valid model name
    assert resolve_base_model("qwen3-4b") == "unsloth/Qwen3-4B-Base"

    # Test invalid model name
    with pytest.raises(ValueError):
        resolve_base_model("not-a-real-model")


def test_resolve_lora_config():
    # Test default config
    result = resolve_lora_config({})
    assert result["r"] == 16
    assert result["lora_alpha"] == 16
    assert result["lora_dropout"] == 0
    assert result["target_modules"] == LORA_TARGET_MODULES
    assert result["bias"] == "none"
    assert result["use_gradient_checkpointing"] == "unsloth"
    assert result["random_state"] == 3407
    assert result["use_rslora"] is False
    assert result["loftq_config"] is None

    # Test with custom lora_r and seed
    result = resolve_lora_config({"lora_r": 32, "seed": 42})
    assert result["r"] == 32
    assert result["lora_alpha"] == 32  # Should default to same as r
    assert result["random_state"] == 42

    # Test with explicit lora_alpha
    result = resolve_lora_config({"lora_r": 32, "lora_alpha": 64})
    assert result["r"] == 32
    assert result["lora_alpha"] == 64


def test_compute_gradient_accumulation_steps():
    # Test (16, 2) -> 8
    assert compute_gradient_accumulation_steps(16, 2) == 8

    # Test (16, 16) -> 1
    assert compute_gradient_accumulation_steps(16, 16) == 1

    # Test (16, 32) -> 1 (never below 1)
    assert compute_gradient_accumulation_steps(16, 32) == 1


def test_resolve_training_args():
    output_dir = Path("/tmp/out")

    # Test default config
    result = resolve_training_args({}, output_dir)
    assert result["auto_find_batch_size"] is True
    assert result["per_device_train_batch_size"] == 2  # NEW — was missing entirely before this fix
    assert result["gradient_accumulation_steps"] == 8  # 16 // 2 default
    assert result["num_train_epochs"] == 3
    assert result["learning_rate"] == 0.0002
    assert result["optim"] == "adamw_8bit"
    assert result["eval_strategy"] == "steps"
    assert result["load_best_model_at_end"] is True
    assert result["metric_for_best_model"] == "eval_loss"
    assert result["greater_is_better"] is False
    assert result["output_dir"] == str(output_dir)

    # Test with custom config
    result = resolve_training_args(
        {
            "num_train_epochs": 5,
            "effective_batch_size": 32,
            "per_device_train_batch_size": 4
        },
        output_dir
    )
    assert result["num_train_epochs"] == 5
    assert result["per_device_train_batch_size"] == 4  # NEW
    assert result["gradient_accumulation_steps"] == 8  # 32 // 4


def test_run_cpt_training_validation():
    # Test missing channel_id
    with pytest.raises(MissingDatasetIdentifiersError):
        run_cpt_training({"user_id": "12345"})

    # Test missing user_id
    with pytest.raises(MissingDatasetIdentifiersError):
        run_cpt_training({"channel_id": "67890"})

    # Test both missing
    with pytest.raises(MissingDatasetIdentifiersError):
        run_cpt_training({})


def test_run_cpt_training_missing_dataset_file(tmp_path):
    # Patch DATASET_PATH to point to tmp_path
    with patch("nanocord.paths.DATASET_PATH", tmp_path):
        # Test that FileNotFoundError is raised when no matching file exists
        config = {
            "channel_id": "test_channel",
            "user_id": "test_user"
        }

        with pytest.raises(FileNotFoundError):
            run_cpt_training(config)