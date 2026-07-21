# LLM Internals Project Plan — "Thoughtform" Discord Bot

**Goal:** Learn the inner mechanics of LLMs (attention, backprop, inference internals, fine-tuning) by building, end to end, a Discord bot with two personalities:
- `/oracle` — a tiny transformer built from scratch, trained on your own Discord text. Chaotic, tone-matching, not conversational.
- `/adib` — a LoRA-fine-tuned small model that replies coherently in your voice, using conversational context.

Hardware: local (RTX 4080 Super, 16GB VRAM), effectively $0 in compute cost. Cloud GPU burst optional if you want to scale up later.

Fork base: [`A-Baji/discordAI-modelizer`](https://github.com/A-Baji/discordAI-modelizer) — reuse its DiscordChatExporter-based scraper and thought-grouping/cleanup logic; extend its parser for context-aware pairs (see Phase 3).

Estimated timeline (evenings/weekends pace): **~2–3 months** for the core path, longer if you take on the stretch goals at the end.

---

## Phase 0 — Get the data moving

Start this first — the official export is slow, don't block on it.

1. **Official Discord data export** (covers DMs, which bots can't reach):
   Settings → Privacy & Security → Data & Privacy → "Request all of my data"
   [Discord's official guide](https://support.discord.com/hc/en-us/articles/360004027692-Requesting-a-Copy-of-your-Data) — can take up to 30 days.

2. **Fork `discordAI-modelizer`** for server-channel scraping:
   - Reuse the bundled [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) + bot-token export step as-is — it's already ToS-compliant (bot account with channel access, not a self-bot/user-token scraper).
   - Bot setup: create an application in the [Discord Developer Portal](https://discord.com/developers/applications), invite it to servers you're in with `Read Message History`, enable the **Message Content** privileged intent (Bot page → Privileged Gateway Intents). Under 100 servers, no Discord review needed.
   - Reuse `gen_dataset.py`'s "thought grouping" (merges your consecutive messages within a time window into one utterance), word-count filtering, URL stripping, and slur censoring — solid as-is for Phase 1.

3. **Keep raw and processed data separate**, and never overwrite raw exports — you'll rework the preprocessing logic multiple times as Phase 3 evolves:
   ```
   raw/
       discord_official_export/
       discordchat_export/
   processed/
       oracle.txt
       adib.jsonl
   ```

4. **Normalize both sources into one schema.** The official Discord export and DiscordChatExporter's output use different JSON structures. Write a small normalization step that maps both into the same intermediate format before any cleaning/grouping logic runs — otherwise Phase 3's context-pairing will need to special-case both formats forever.

> ⚠️ Do not build or use a self-bot / user-token scraper. Automating your own user account is against [Discord's Terms of Service](https://discord.com/terms) and risks an account ban regardless of intent.

---

## Phase 1 — Build a transformer from scratch (the `/oracle` model)

**Goal:** implement attention, backprop, and tokenization yourself — no library abstracting it away.

- Primary resource: Andrej Karpathy's **"Let's build GPT: from scratch, in code, spelled out"** — [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) (cleaner, step-by-step commit history + video) or the original [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture).
- Note: the original `nanoGPT` repo is now deprecated in favor of Karpathy's newer `nanochat` project — `build-nanogpt` is the more current from-scratch walkthrough.
- Training data: your cleaned Discord text from Phase 0/the modelizer fork (swap its JSONL output for plain text lines).

**Build the tokenizer progressively** rather than jumping straight to BPE — each step teaches something the next one builds on:
1. character-level tokenizer
2. word-level tokenizer
3. BPE tokenizer (e.g. via Hugging Face `tokenizers` or `tiktoken`)

Expected output: not coherent conversation — surreal, tone-matching-you text generation. That's the point; the value is in implementing the mechanics, not the output quality. This becomes the `/oracle` command.

---

## Phase 2 — Write your own minimal inference engine

**Goal:** understand sampling, KV caching, batching, and token streaming — the layer you normally call through Bedrock/Ollama at work.

Build this incrementally, one capability at a time, rather than jumping straight to a full engine — each version is a working checkpoint:

1. Naive inference: `forward()` → sample → repeat
2. Temperature
3. Top-k sampling
4. Top-p (nucleus) sampling
5. Repetition penalty
6. KV cache
7. Batched inference
8. Streaming

By version 8 you've effectively recreated a small inference engine, one axis at a time.

- Reference implementations to read (not copy): Hugging Face `transformers`' `generate()` source, and [karpathy/llm.c](https://github.com/karpathy/llm.c) for a minimal, readable inference path.
- Don't skip tokenizer symmetry: your custom loop needs to decode the exact token IDs your Phase 1 tokenizer produces back to text, including mid-stream partial-token buffering.
- Serve the Phase 1 model through this engine with token-by-token streaming over a local HTTP endpoint.

---

## Phase 3 — LoRA fine-tune a real small model (the `/adib` model)

**Goal:** a model that replies coherently *as you*, using conversational context — not just your vocabulary.

1. **Extend the modelizer's parser** to preserve context instead of discarding it. Build a rolling context window (2–3 prior turns tends to outperform a single preceding message) leading into your thought as the target:
   ```
   <user> How are you?
   <assistant> Pretty tired lol
   <user> Same.
   ```
   Format this way even though you won't necessarily serve it this way — it's what teaches chat-formatting conventions.

2. **Build a few dataset variants and compare them** rather than committing to one shape upfront:
   - Dataset A: single preceding message → reply
   - Dataset B: 3 preceding messages → reply
   - Dataset C: full conversation window → reply

3. **Reformat** into chat-style turns (system/user/assistant) for Unsloth/Hugging Face `datasets`.

4. **Understand LoRA mechanics first, but keep the from-scratch part small.** Implement just a single trainable low-rank adapter (`W → W + BA`) on one linear layer, freeze the base weights, and verify only A/B update on a toy regression problem. That's enough for the mechanism to click — don't rebuild a full training loop by hand.
   Reference: Unsloth's [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) for the LoRA/QLoRA math.

5. **Train for real** with [Unsloth](https://unsloth.ai/docs) — a 3B model QLoRA fine-tune fits in roughly 8–10GB VRAM on a 4080 Super. Use their [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) for rank, alpha, batch size, and epochs.

6. **Export to GGUF** and import into Ollama for serving.

---

## Phase 4 — Glue it into a Discord bot

**Goal:** a demoable end product. Existing libraries are fine here — this isn't where the learning lives.

- [discord.py documentation](https://discordpy.readthedocs.io/) for bot scaffolding.
- Put your own lightweight FastAPI server between Discord and the models, rather than having Ollama own all inference directly:
  ```
  Discord → your FastAPI server → Oracle engine (Phase 2)
                                 → Ollama (Phase 3 model)
  ```
  This makes it easy to swap models later without touching the bot itself.
- Route `/oracle` → Phase 1/2 (from-scratch model + your own inference engine).
- Route `/adib` → Phase 3 (LoRA model served via Ollama).
- **Respect Discord's rate limits when streaming**: don't edit a message on every token. Buffer output and edit roughly every 0.5–1 second (or every 5–10 tokens) to avoid `429 Too Many Requests`.
- Before deploying to any shared server, decide scope: private test server/DMs only, or a server where others know about it — this determines how aggressively to scrub other users' messages from the training data.

---

## Stretch goals (optional — add scope deliberately, not by default)

These are all genuinely valuable but meaningfully extend the timeline, so treat them as opt-in additions rather than required steps.

- **Multiple model sizes in Phase 1.** Instead of training one `/oracle` model, train a small progression and compare them directly — a hands-on look at scaling behavior:
  ```
  TinyGPT (4 layers, 128 hidden) → MiniGPT (6 layers, 256 hidden) → OracleGPT (8 layers, 384 hidden)
  ```
- **Phase 2.5 — Reproduce GPT-2 inference from pretrained weights (no training).** Implement the tokenizer, embeddings, and transformer yourself, then load GPT-2's actual released weights and verify your outputs match Hugging Face's. This is one of the best exercises for understanding checkpoint formats, state dicts, and parameter layouts — arguably a bigger unlock than Phase 1 alone, since it forces numerical correctness rather than "looks plausible."
- **Build a small evaluation suite.** ~100 fixed prompts (e.g. "Tell me about programming") run against every training checkpoint, with outputs saved and compared over time. Turns training from "guess and vibe-check" into an actual experiment loop — useful both for this project and as a transferable habit for eval work at your day job.
- **Read one paper per phase**, matched to what you just built, not as a prerequisite but as reinforcement:
  - After Phase 1 → *Attention Is All You Need*
  - After Phase 2 → *FlashAttention*
  - After Phase 3 → the LoRA paper
  Skim for the ideas that map to code you already wrote, not full mathematical rigor.

---

## Key references

| Purpose | Resource |
|---|---|
| Build a transformer from scratch | [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) |
| Minimal inference engine reference | [karpathy/llm.c](https://github.com/karpathy/llm.c) |
| Discord data export (DMs) | [Discord data request guide](https://support.discord.com/hc/en-us/articles/360004027692-Requesting-a-Copy-of-your-Data) |
| Scraper fork base | [A-Baji/discordAI-modelizer](https://github.com/A-Baji/discordAI-modelizer) |
| Channel export tool | [Tyrrrz/DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) |
| Bot intents setup | [discord.py intents docs](https://discordpy.readthedocs.io/en/latest/intents.html) |
| LoRA fine-tuning | [Unsloth fine-tuning guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) + [hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) |
| Discord bot framework | [discord.py](https://discordpy.readthedocs.io/) |
