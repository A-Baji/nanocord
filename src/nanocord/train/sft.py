from pathlib import Path
from typing import Dict, Optional, Tuple

from nanocord.train.cpt import (
    BaseModel,
    DEFAULT_BASE_MODEL,
    MissingDatasetIdentifiersError,
    compute_gradient_accumulation_steps,
    resolve_base_model,
    resolve_lora_config,
    resolve_training_args,
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
# Maps each base model to the Unsloth get_chat_template name to apply to
# its tokenizer (base models ship with no chat template at all), plus the
# instruction_part/response_part strings Unsloth's train_on_responses_only
# needs to mask loss to assistant-only tokens. SmolLM3 and Qwen3 both use
# ChatML natively; Qwen2.5 has its own dedicated Unsloth template name but
# the same underlying delimiter tokens; Llama 3.2 uses Meta's header-token
# format.


def resolve_chat_template(name: Optional[str]) -> Tuple[str, str, str]:
    """
    Resolve a train.sft.base_model config value (same enum as
    train.cpt.base_model) to the (chat_template_name, instruction_part,
    response_part) tuple needed for Unsloth's get_chat_template and
    train_on_responses_only.

    Args:
        name: One of BaseModel's string values, or None/empty to use
              DEFAULT_BASE_MODEL.

    Returns:
        (chat_template_name, instruction_part, response_part)

    Raises:
        ValueError: If name is truthy but not a recognized BaseModel
                    value. Use the exact message format
                    f"Unknown base model: {name}" to match
                    resolve_base_model's error message in train/cpt.py.
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

    Args:
        config: Merged train.sft configuration dict (see module notes
                above). MUST also contain "channel_id" and "user_id" (same
                requirement as run_cpt_training, and for the same reason -
                validate this directly here, don't assume a caller
                checked).

    Returns:
        Path: directory the SFT LoRA adapter was saved to,
              MODEL_PATH / f"{user_id}_{channel_id}_sft_lora"

    Raises:
        MissingDatasetIdentifiersError: if channel_id or user_id missing
                                          from config (reuse the same
                                          exception class from
                                          nanocord.train.cpt - do not
                                          define a new one).
        ValueError: if config["base_model"] is set but not a recognized
                    BaseModel value.
        FileNotFoundError: if the CPT checkpoint directory
                            (MODEL_PATH / f"{user_id}_{channel_id}_cpt_lora")
                            does not exist - message should tell the user
                            to run train cpt first. ALSO raised if the
                            SFT dataset file (DATASET_PATH /
                            f"{user_id}_{channel_id}_sft_data_set.jsonl")
                            does not exist - message should tell the user
                            to run dataset sft first. Check BOTH of these
                            before importing/loading unsloth or any model,
                            same discipline as run_cpt_training.

    Implementation outline:
      1. Validate channel_id/user_id (identical pattern to
         run_cpt_training in train/cpt.py).
      2. from nanocord.paths import DATASET_PATH, MODEL_PATH
         cpt_checkpoint_dir = MODEL_PATH / f"{user_id}_{channel_id}_cpt_lora"
         raise FileNotFoundError if not cpt_checkpoint_dir.exists()
         sft_dataset_file = DATASET_PATH / f"{user_id}_{channel_id}_sft_data_set.jsonl"
         raise FileNotFoundError if not sft_dataset_file.exists()
      3. hf_repo_id = resolve_base_model(config.get("base_model"))
      4. from unsloth import FastLanguageModel
         from peft import PeftModel
         import torch
         # Load base model in full precision (NOT 4-bit) so the LoRA
         # merge doesn't introduce quantization error.
         base_model, tokenizer = FastLanguageModel.from_pretrained(
             model_name=hf_repo_id,
             max_seq_length=config.get("max_seq_length", 2048),
             load_in_4bit=False,
         )
         model_with_cpt = PeftModel.from_pretrained(base_model, str(cpt_checkpoint_dir))
         merged_model = model_with_cpt.merge_and_unload()
         merged_dir = MODEL_PATH / f"{user_id}_{channel_id}_cpt_merged"
         merged_model.save_pretrained(str(merged_dir))
         tokenizer.save_pretrained(str(merged_dir))
         # Free VRAM before reloading in the configured precision for SFT.
         del base_model, model_with_cpt, merged_model
         torch.cuda.empty_cache()
      5. model, tokenizer = FastLanguageModel.from_pretrained(
             model_name=str(merged_dir),
             max_seq_length=config.get("max_seq_length", 2048),
             load_in_4bit=config.get("load_in_4bit", True),
         )
         from unsloth.chat_templates import get_chat_template
         chat_template_name, instruction_part, response_part = resolve_chat_template(config.get("base_model"))
         tokenizer = get_chat_template(tokenizer, chat_template=chat_template_name)
         model = FastLanguageModel.get_peft_model(model, **resolve_lora_config(config))
      6. from datasets import load_dataset
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
      7. output_dir = MODEL_PATH / f"{user_id}_{channel_id}_sft_lora"
         from transformers import TrainingArguments, EarlyStoppingCallback
         from trl import SFTTrainer
         from unsloth.chat_templates import train_on_responses_only

         trainer = SFTTrainer(
             model=model,
             tokenizer=tokenizer,
             train_dataset=train_dataset,
             eval_dataset=eval_dataset,
             dataset_text_field="text",
             max_seq_length=config.get("max_seq_length", 2048),
             packing=False,  # hardcoded - see design decision 2 above, NOT config.get("packing", ...)
             args=TrainingArguments(**resolve_training_args(config, output_dir)),
             callbacks=[EarlyStoppingCallback(early_stopping_patience=config.get("early_stopping_patience", 3))],
         )
         trainer = train_on_responses_only(
             trainer,
             instruction_part=instruction_part,
             response_part=response_part,
         )
      8. trainer.train()
         model.save_pretrained(str(output_dir))
         tokenizer.save_pretrained(str(output_dir))
         return output_dir
    """
    # 1. Validate channel_id/user_id (identical pattern to run_cpt_training)
    user_id = config.get("user_id")
    channel_id = config.get("channel_id")
    if not user_id or not channel_id:
        raise MissingDatasetIdentifiersError("Missing user_id or channel_id in config")

    # 2. Validate dataset files exist before loading any heavy libraries
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

    # 4. Load and merge CPT checkpoint
    from unsloth import FastLanguageModel
    from peft import PeftModel
    import torch

    # Load base model in full precision (NOT 4-bit) so the LoRA
    # merge doesn't introduce quantization error.
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hf_repo_id,
        max_seq_length=config.get("max_seq_length", 2048),
        load_in_4bit=False,
    )
    model_with_cpt = PeftModel.from_pretrained(base_model, str(cpt_checkpoint_dir))
    merged_model = model_with_cpt.merge_and_unload()
    merged_dir = MODEL_PATH / f"{user_id}_{channel_id}_cpt_merged"
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    # Free VRAM before reloading in the configured precision for SFT.
    del base_model, model_with_cpt, merged_model
    torch.cuda.empty_cache()

    # 5. Load merged model with proper chat template and fresh LoRA adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(merged_dir),
        max_seq_length=config.get("max_seq_length", 2048),
        load_in_4bit=config.get("load_in_4bit", True),
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

    # 7. Setup training
    output_dir = MODEL_PATH / f"{user_id}_{channel_id}_sft_lora"

    from transformers import TrainingArguments, EarlyStoppingCallback
    from trl import SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config.get("max_seq_length", 2048),
        packing=False,  # hardcoded - see design decision 2 above, NOT config.get("packing", ...)
        args=TrainingArguments(**resolve_training_args(config, output_dir)),
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