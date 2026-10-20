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
- Export path verified on the smoke checkpoint (2026-10-20): `export_hf.py` → HF tokenizer ids equal SentencePiece ids → `convert_hf_to_gguf.py --outtype f16` (67 MB) → `llama-quantize Q8_0` (36 MB) → `llama-tokenize` gives `[1, 3, 576, 393, 372, 882, 426, 7743, 4]` for `<|q|>How do I update Omarchy?<|a|>`, identical to `[bos] + sp.encode(prompt)`. GGUF metadata: `add_space_prefix` false, `add_bos_token` true, bos 1, eos 2, context 1,024, vocab 8,192. The three pair symbols are typed CONTROL (ids 3–5); llama.cpp treats `<|end|>` as an end-of-generation token by name, so generation stops there without printing it.
- `python -c "import gguf"` from the project root picks up the project's own `gguf/` output folder; the converter is unaffected (it uses its sibling `gguf-py`). Run GGUF-reading snippets with `tools/llama.cpp-b10080/gguf-py` first on `sys.path`.

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
| run-001 | 8, 512 (29.5M, float32) | 1,533 | 6e-4 | 3.170 (valid.txt) | 500 | 65k → 42k | Samples from step 500: well-formed commands, wrong ones (`omarchy-update-aur-pkgs` for "update Omarchy"). Final step 1,533: pairs-only val loss 0.52, level 3 command match 0.916, exact 0.867, well-formed 0.995. Shipping candidate from this run: `step-1533.safetensors`. |
| run-002 | 8, 512 (29.5M, float32) | 3,066 | 6e-4 | | | | One change: steps doubled (accuracy still rising at the end of run-001). |

### run-001 details (2026-10-20)

- Train loss 9.47 at step 1, 2.89 by step 300, 0.04 at the end. Val loss (whole valid.txt) 5.35 at step 100, lowest 3.17 at step 500, then rising to 3.89.
- valid.txt is 60 unseen raw documents (about 60k tokens) plus 427 held-out pairs (about 15k tokens), so its loss is dominated by unseen shell scripts and stays above the level 2 threshold of 2.0 for every checkpoint. Measured on the 427 pair documents alone: 1.21 at step 500, 0.68 at step 1,000, 0.52 at step 1,533.
- Level 3 on the 427 held-out pairs (`eval/results/run-001-valid-*.json`):

| checkpoint | exact | command | well-formed |
| --- | --- | --- | --- |
| step-500 (best val loss) | 0.316 | 0.405 | 0.993 |
| step-1000 | 0.749 | 0.841 | 1.000 |
| step-1533 (final) | 0.867 | 0.916 | 0.995 |

- The checkpoint with the lowest val loss answers 40% of questions; the final one answers 92%. Checkpoint selection for shipping uses command match on the held-out pairs, not valid.txt loss.

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
