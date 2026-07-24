# NanoCord

SLM framework that fine-tunes a small local model to act as a Discord persona bot, trained on the maintainer's own Discord message history. Quality over speed — no deadline. Local-only, no cloud training.

Hardware: RTX 4080 Super (16GB VRAM), 32GB RAM.

## Pipeline (5 stages, run in order)

1. `dataset cpt` — scrape a Discord channel via DiscordChatExporter, group messages into "thoughts," write CPT dataset. **Built and tested.**
2. `train cpt` — LoRA continued pretraining (unsupervised) on a small base model (Llama 3.2 3B or Qwen 2.5 3B, via Unsloth). Teaches voice/tone from isolated phrases. **Stub.**
3. `dataset sft` — extract (context → reply) pairs from DiscordChatExporter's reply-reference metadata. Teaches conversational behavior. **Stub.**
4. `train sft` — LoRA fine-tune on top of the CPT checkpoint (via Unsloth), using the SFT dataset. **Stub.**
5. `bot register` — export to GGUF, serve via Ollama, register as a Discord slash command. **Stub.**

CPT must run before SFT — voice first, then reply judgment on top of it. `nanocord pipeline` runs all 5 in order with `--skip-*` flags per stage.

## Repo layout

```
src/nanocord/
├── cli.py                    # Typer: init, dataset {cpt,sft}, train {cpt,sft}, bot register, pipeline, config {set,get}
├── config.py                 # load_and_merge_config(yaml_path, cli_args, section) — see Config below
├── paths.py                  # appdirs-based data dir; CONFIG_PATH, RAW_DATA_DIR, PROCESSED_DATA_DIR
├── logger.py                 # global_logger, imported as `from nanocord import global_logger`
├── dataset/
│   ├── discord_export.py     # export_channel_logs() — shells out to DiscordChatExporter.Cli
│   ├── thoughts.py           # validate_thought, cleanup_string, build_thought — shared cleaning helpers
│   ├── cpt.py                 # parse_logs, add_to_dataset, build_cpt_dataset — DONE
│   └── sft.py                  # build_sft_dataset(config) -> Path — STUB, raises NotImplementedError
├── train/
│   ├── cpt.py    # run_cpt_training(config) -> Path — STUB
│   └── sft.py    # run_sft_training(config) -> Path — STUB
└── bot/
    └── register.py  # register_bot(config) -> None — STUB
tests/  # 17/17 passing (test_cli_scaffolding, test_config, test_cpt, test_thoughts)
```

## Config system (`config.py`)

Priority, highest wins: **explicit CLI flag > config.yaml value > hardcoded default** (defaults live in `load_and_merge_config`'s initial `config` dict).

Schema is nested per pipeline stage in `config.yaml`: `dataset.cpt`, `dataset.sft`, `train.cpt`, `train.sft`, `bot`. Each CLI command loads only its own section:

```python
load_and_merge_config(config_file, cli_args, section="dataset.cpt")  # dot-notation lookup
```

`section` walks the YAML dict by dot-split key; a missing key at any level yields `{}` for that section (not an error). CLI args with value `None` are filtered out before merging, so unset flags never clobber a YAML value.

Known quirk: legacy top-level `discord_token` (outside `dataset:`) gets special-cased into the `dataset` section if not already present there — don't remove this without checking `test_config.py`.

## Hard constraints — do not relitigate

- **Data source is DiscordChatExporter only**, invoked with a bot token (channel access via `Read Message History` + Message Content intent). Never a self-bot/user-token scraper — Discord ToS violation, ban risk.
- **CPT dataset format is JSONL only**, one `{"text": thought}` object per line, written via `json.dumps()` — never manual string interpolation (that caused a real corruption bug in the fork). No `.txt` output option.
- **`keep_alive` (Ollama warm vs. cold model loading) must be a configurable `bot` param, not hardcoded.** Warm keeps the model resident (~2-4GB VRAM, fast replies); cold unloads after each call (zero idle footprint, per-message latency).
- **Bot inference is synchronous, one job at a time — no concurrency.** Bounds GPU contention to a single burst so it doesn't stomp on other GPU work (games, etc.).

## Open work items

1. **`dataset/sft.py` → `build_sft_dataset`**: before implementing, verify DiscordChatExporter's reply-reference field shape against a real export sample — the capability is confirmed to exist but no code has inspected the actual JSON structure yet.
2. **`build_sft_dataset` cannot reuse `cpt.py`'s `parse_logs` filtering.** `parse_logs` filters `data["messages"]` down to only the target user's own messages. SFT needs the *other* party's message too (whatever the target user replied to), so it needs its own traversal of the full, unfiltered message list. It *can* reuse `thoughts.py`'s `cleanup_string`/`build_thought`/`validate_thought` — just not `cpt.py`'s filtering.
3. **`train/cpt.py`, `train/sft.py`**: implement as Unsloth LoRA runs. CPT is unsupervised over the CPT JSONL (`{"text": ...}` records). SFT loads the CPT checkpoint and trains on the SFT dataset once stage 3 exists.
4. **`bot/register.py`**: export the SFT checkpoint to GGUF, serve through Ollama with configurable `keep_alive`, register one Discord slash command that calls it synchronously.

Note: `bot/register.py`'s current stub docstring mentions `/oracle` and `/adib` — that's leftover from the abandoned from-scratch plan. Don't build two model routes; there's one persona model. Fix the docstring when touching that file.

## Working conventions

- Verifying repo state means **pulling the branch and running the test suite**, not just reading code — this has caught cases where tests were green but not exercising real behavior, or a fix was only partially applied. Pull via `codeload.github.com/A-Baji/nanocord/tar.gz/refs/heads/dev` (check `main` too if dev has been merged).
- Implementation prompts for this model should be dense, scoped to specific files, and self-contained — don't require exploring the rest of the repo to act. Split changes touching unrelated files into separate sequential sessions rather than one large prompt.