# Omarchy SLM: runs

## Environment

- Mac: Apple M5 Max, 128 GB, Darwin 27.0.0. Project `/Users/ed/dev/local-ai/slm-training`, venv `.venv`.
- Python 3.13.2 (Homebrew). The plan names python3.12, which is not installed on the Mac.
- mlx 0.32.4, mlx-lm 0.32.2, sentencepiece 0.2.2, transformers 5.20.2, safetensors 0.8.0, numpy 2.5.4, protobuf 7.36.2.
- `mx.default_device()` = `Device(gpu, 0)`.
- llama.cpp: Homebrew build 10080 (fd41bf65a): llama-cli, llama-completion, llama-server, llama-quantize. The package ships no `convert_hf_to_gguf.py`.
- AMD Omarchy box: AMD Radeon 610M (Granite Ridge iGPU).

## Step log

- Step 1 corpus (2026-10-20): `git clone https://github.com/basecamp/omarchy` at commit 3ac43659dcbf88e84d951e3ce979b48a60a7d1f2 (version 4.0.0.alpha), owner-approved. Kept `bin/`, `config/`, `default/` (i18n: `en` only, translation manifest dropped), `install/`, `migrations/`, `docs/`, `manual/*.md` (English pages; `manual/<lang>/` translations dropped), `README.md`, `version`. Dropped `.git`, `themes/`, `test/`, `applications/`, `shell/`, `scripts/`, `plans/`, `agents/`, `etc/`, `CLAUDE.md`, `AGENTS.md`, images, video, fonts, cursors. The manual is in the repository, so no web fetch. 1,210 files, 3.34 MB.
- Step 1 pairs: 8,713 derived, 8,633 after dedupe (80 questions with two different answers kept the first), 100 held back as test with seed 42, 8,533 in `pairs.txt`. Templates in `data/pairs/templates.txt`: 15 per script (14 for imperative summaries, S3 for all), 5 per keybinding (198 parsed), 2 for manual headings with a code block (34).
- Step 1 split: raw documents 95/5 by document (seed 42); pairs 95/5 at random (seed 43), independent of the document split, so validation questions are about scripts present in train, like the sealed test.
- Step 2 environment: venv and named packages installed; GPU device confirmed. Added after the owner deferred: torch 2.14.1 (the converter imports it), pytest 9.1.1, and llama.cpp's converter from tag b10080 (same commit as the Homebrew build) under `tools/llama.cpp-b10080/` with its `gguf-py` installed (gguf 0.19.0).
- Tokenizer: `add_dummy_prefix=False` added to the plan's settings (with the default, `<|q|>` encodes as two ids). HF export sets `add_prefix_space: false` for the same reason.
- Step 3 tokenizer: trained, checks pass, token counts below.
- Step 4 model and training script: `python src/train.py --run smoke --steps 20 --batch-size 4`: 29.5M parameters, loss 9.49 at step 1 → 7.35 at step 20, `generate.py` produces tokens from `checkpoints/smoke/step-20.safetensors`. Level 1 tests: 5 passed. bos count in train.bin = 9,250 = documents (1,144 raw + 8,106 pairs); valid.bin 487 = 60 + 427.
- Pending GGUF checks for step 7, once the converter is available: token types of ids 3–5 in the GGUF must be USER_DEFINED (4); `tokenizer.ggml.add_space_prefix` must be false; `llama-tokenize -m <gguf> -p "<|q|>How do I update Omarchy?<|a|>" --ids` must equal `[1] + sp.encode(prompt)`.

## Corpus stats

| file | documents | pairs | bytes | longest document |
| --- | --- | --- | --- | --- |
| train.txt | 1,144 | 8,106 | 4,185,068 | omarchy/bin/omarchy-upgrade-to-quattro (99,760 bytes) |
| valid.txt | 60 | 427 | 249,430 | omarchy/bin/omarchy-agent-account-state (36,872 bytes) |
| test/prompts.jsonl | | 100 | | |

## Tokenizer

SentencePiece BPE, vocab 8,192, trained on train.txt only (step 3, 2026-10-20). Checks: 50/50 random lines round-trip exactly; `<|q|>` → [3], `<|a|>` → [4], `<|end|>` → [5].

| file | tokens (bos per document included) |
| --- | --- |
| train.txt | 1,255,441 |
| valid.txt | 75,492 |

Corpus under 2M tokens → 40 epochs. Tokens per step = 32 × 1,024 = 32,768. Steps for run-001 = round(40 × 1,255,441 / 32,768) = 1,533.

## Runs

| run | config (layers, hidden) | steps | lr | best val loss | best step | tokens/s | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Level 4 results

(filled by step 6)

## Sign-off

| Check | Threshold | Mac | AMD |
| --- | --- | --- | --- |
| Pipeline tests | all pass |  | n/a |
| Best val loss | < 2.0 |  | n/a |
| Command match, test | ≥ 0.90 |  |  |
| Well-formed, test | ≥ 0.98 |  |  |
| Mac vs AMD gap | ≤ 0.03 |  |  |
| Owner spot check | owner's call | n/a |  |
