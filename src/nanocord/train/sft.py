import gc
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch

from nanocord.train.cpt import (
    BaseModel,
    DEFAULT_BASE_MODEL,
    MissingDatasetIdentifiersError,
    compute_gradient_accumulation_steps,
    resolve_base_model,
    resolve_lora_config,
    resolve_training_args,
    resolve_embedding_learning_rate,
)


BASE_MODEL_CHAT_TEMPLATES = {
    BaseModel.SMOLLM3_3B: ("chatml", "user\n", "assistant\n"),
    BaseModel.QWEN3_4B: ("chatml", "user\n", "assistant\n"),
    BaseModel.QWEN3_1_7B: ("chatml", "user\n", "assistant\n"),
    BaseModel.QWEN2_5_7B: ("qwen2.5", "user\n", "assistant\n"),
    BaseModel.LLAMA_3_2_3B: (
        "llama-3.2",
        "<|start_header_id|>user<|end_header_id|>\n\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
}


def resolve_chat_template(name: Optional[str]) -> Tuple[str, str, str]:
    """
    Resolve a train.sft.base_model config value (same enum as
    train.cpt.base_model) to the (chat_template_name, instruction_part,
    response_part) tuple needed for Unsloth's get_chat_template and
    train_on_responses_only.
    """
    if not name:
        model = DEFAULT_BASE_MODEL
    else:
        try:
            model = BaseModel(name)
        except ValueError:
            raise ValueError(f"Unknown base model: {name}")
    return BASE_MODEL_CHAT_TEMPLATES[model]


def run_sft_training(config: Dict) -> Path:
    """
    Main entrypoint. Merges the CPT LoRA adapter into the base model,
    attaches a fresh LoRA adapter, trains on the SFT conversational
    dataset with loss masked to assistant-only tokens, and saves the
    resulting LoRA adapter.
    """
    # 1. Validate channel_id/user_id
    user_id = config.get("user_id")
    channel_id = config.get("channel_id")
    if not user_id or not channel_id:
        raise MissingDatasetIdentifiersError(
            "train.sft requires channel_id and user_id to locate the SFT dataset "
            "and CPT checkpoint."
        )

    # 2. Validate dataset and checkpoint files exist before loading heavy libraries
    from nanocord.paths import DATASET_PATH, MODEL_PATH

    cpt_checkpoint_dir = MODEL_PATH / f"{user_id}_{channel_id}_cpt_lora"
    if not cpt_checkpoint_dir.exists():
        raise FileNotFoundError(
            f"CPT checkpoint directory does not exist. Please run 'train cpt' first. "
            f"Expected: {cpt_checkpoint_dir}"
        )

    sft_dataset_file = DATASET_PATH / f"{user_id}_{channel_id}_sft_data_set.jsonl"
    if not sft_dataset_file.exists():
        raise FileNotFoundError(
            f"SFT dataset file does not exist. Please run 'dataset sft' first. "
            f"Expected: {sft_dataset_file}"
        )

    # 3. Resolve base model
    hf_repo_id = resolve_base_model(config.get("base_model"))

    # Flush GPU memory before non-quantized loading
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Load and merge CPT checkpoint
    from unsloth import FastLanguageModel
    from peft import PeftModel

    # Force device_map="cuda:0" to prevent Hugging Face Accelerate from offloading
    # layers to CPU on 16GB GPUs during 16-bit unquantized loading.
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hf_repo_id,
        max_seq_length=config.get("max_seq_length", 2048),
        load_in_4bit=False,
        device_map="cuda:0",
    )
    model_with_cpt = PeftModel.from_pretrained(base_model, str(cpt_checkpoint_dir))
    merged_model = model_with_cpt.merge_and_unload()
    merged_dir = MODEL_PATH / f"{user_id}_{channel_id}_cpt_merged"
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    # Completely purge full-precision model from VRAM before starting SFT
    del base_model, model_with_cpt, merged_model
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Load merged model with proper chat template and fresh LoRA adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(merged_dir),
        max_seq_length=config.get("max_seq_length", 2048),
        load_in_4bit=config.get("load_in_4bit", True),
        device_map="cuda:0",
    )

    from unsloth.chat_templates import get_chat_template
    chat_template_name, instruction_part, response_part = resolve_chat_template(config.get("base_model"))
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template_name)
    model = FastLanguageModel.get_peft_model(model, **resolve_lora_config(config))

    # 6. Prepare dataset
    from datasets import load_dataset

    full_dataset = load_dataset("json", data_files=str(sft_dataset_file))["train"]

    def formatting_prompts_func(examples):
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in examples["messages"]
        ]
        return {"text": texts}

    full_dataset = full_dataset.map(formatting_prompts_func, batched=True)
    split = full_dataset.train_test_split(test_size=config.get("eval_split", 0.05), seed=config.get("seed", 3407))
    train_dataset, eval_dataset = split["train"], split["test"]

    # 7. Setup training arguments
    output_dir = MODEL_PATH / f"{user_id}_{channel_id}_sft_lora"

    from transformers import EarlyStoppingCallback
    from unsloth import UnslothTrainer, UnslothTrainingArguments
    from unsloth.chat_templates import train_on_responses_only

    vram_safe_max_seq_length = config.get("vram_safe_max_seq_length")
    effective_max_seq_length = (
        min(config.get("max_seq_length", 2048), vram_safe_max_seq_length)
        if vram_safe_max_seq_length
        else config.get("max_seq_length", 2048)
    )

    training_args_dict = resolve_training_args(config, output_dir)
    training_args_dict.update({
        "dataset_text_field": "text",
        "max_seq_length": effective_max_seq_length,
        "packing": False,  # SFT requires packing=False for response masking
        "optim": "paged_adamw_8bit",
        "embedding_learning_rate": resolve_embedding_learning_rate(config),
    })

    # Flush VRAM before trainer allocation
    gc.collect()
    torch.cuda.empty_cache()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(**training_args_dict),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.get("early_stopping_patience", 3))],
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part,
    )

    # 8. Train and save
    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir