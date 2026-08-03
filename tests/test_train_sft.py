import pytest
from pathlib import Path
from unittest import mock

from nanocord.train.sft import (
    BASE_MODEL_CHAT_TEMPLATES,
    resolve_chat_template,
    run_sft_training,
)
from nanocord.train.cpt import MissingDatasetIdentifiersError, BaseModel


def test_resolve_chat_template():
    # Test None input
    result = resolve_chat_template(None)
    expected = BASE_MODEL_CHAT_TEMPLATES[BaseModel.QWEN2_5_7B]
    assert result == expected

    # Test empty string input
    result = resolve_chat_template("")
    expected = BASE_MODEL_CHAT_TEMPLATES[BaseModel.QWEN2_5_7B]
    assert result == expected

    # Test llama-3.2-3b
    result = resolve_chat_template("llama-3.2-3b")
    expected = ("llama-3.2", "<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n")
    assert result == expected

    # Test qwen2.5-7b
    result = resolve_chat_template("qwen2.5-7b")
    expected = ("qwen2.5", "user\n", "assistant\n")
    assert result == expected

    # Test invalid model name
    with pytest.raises(ValueError):
        resolve_chat_template("not-a-real-model")


def test_run_sft_training_validation():
    """Test validation path - missing channel_id and/or user_id"""
    # Test missing both identifiers
    with pytest.raises(MissingDatasetIdentifiersError):
        run_sft_training({})

    # Test missing channel_id
    with pytest.raises(MissingDatasetIdentifiersError):
        run_sft_training({"user_id": "12345"})

    # Test missing user_id
    with pytest.raises(MissingDatasetIdentifiersError):
        run_sft_training({"channel_id": "67890"})


def test_run_sft_training_missing_checkpoint(tmp_path):
    """Test missing CPT checkpoint path"""
    config = {
        "channel_id": "test_channel",
        "user_id": "test_user"
    }

    # Patch MODEL_PATH to point to tmp_path where no cpt_lora directory exists
    with mock.patch("nanocord.paths.MODEL_PATH", tmp_path):
        with pytest.raises(FileNotFoundError):
            run_sft_training(config)


def test_run_sft_training_missing_dataset(tmp_path):
    """Test missing SFT dataset path"""
    config = {
        "channel_id": "test_channel",
        "user_id": "test_user"
    }

    # Create a directory for the CPT checkpoint
    cpt_dir = tmp_path / f"{config['user_id']}_{config['channel_id']}_cpt_lora"
    cpt_dir.mkdir()

    # Patch MODEL_PATH to point to tmp_path where cpt_lora directory exists
    with mock.patch("nanocord.paths.MODEL_PATH", tmp_path):
        # Patch DATASET_PATH to point to a different tmp_path where no dataset file exists
        with mock.patch("nanocord.paths.DATASET_PATH", tmp_path):
            with pytest.raises(FileNotFoundError):
                run_sft_training(config)