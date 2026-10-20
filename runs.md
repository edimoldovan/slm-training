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
- Step 5 training (2026-10-20): four runs, one change each (steps ×2, templates, steps ×2). Checkpoints chosen by command match on held-out pairs. Shipping candidate run-004 step 6,500.
- Step 6 testing: level 1 pass; level 2 cannot be met as written (see sign-off); level 3 pass (0.945 / 0.997); level 4 Mac pass (0.940 / 1.000); AMD pending.
- Step 7: GGUF built and verified on the Mac; copied to the box; llama-server (Arch `llama-cpp`) started by the owner; validation pairs match the Mac exactly with run-003.
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
| run-002 | 8, 512 (29.5M, float32) | 3,066 | 6e-4 | 3.172 (valid.txt) | 500 | 66k → 46k | One change: steps doubled. Level 3 peaks at step 3,000: command 0.972, exact 0.956, well-formed 0.995. Free-phrased prompts still wrong. Shipping candidate: `step-3000.safetensors`. |
| run-003 | 8, 512 (29.5M, float32) | 3,418 | 6e-4 | 2.956 (valid.txt) | 500 | 66k → 45k | One change: extended question templates; corpus rebuilt, tokenizer unchanged; steps = 2 × formula. Level 3 at step 3,418 on 714 held-out pairs: command 0.940, exact 0.933, well-formed 0.996, still rising at the end. Free-phrased prompts now right for update, Terminal keybinding, screenshot. |
| run-004 | 8, 512 (29.5M, float32) | 6,836 | 6e-4 | 2.960 (valid.txt) | 500 | 66k → 45k | One change: steps doubled again. Level 3 peaks at step 6,500: command 0.945, exact 0.944, well-formed 0.997. **Shipped: `step-6500.safetensors`.** Sealed test on the Mac: command 0.940, exact 0.940, well-formed 1.000. |

### run-004 results (2026-10-20), shipping candidate

- Level 3 on the 714 held-out pairs (`eval/results/run-004-valid-step-*.json`):

| checkpoint | exact | command | well-formed |
| --- | --- | --- | --- |
| step-5000 | 0.926 | 0.929 | 0.997 |
| step-5500 | 0.924 | 0.930 | 0.999 |
| step-6000 | 0.938 | 0.943 | 0.999 |
| step-6500 | 0.944 | 0.945 | 0.997 |
| step-6836 (final) | 0.936 | 0.937 | 0.999 |

- Doubling steps again gained 0.5 points over run-003 (0.940 → 0.945); accuracy has flattened at this model size and template set. Next lever if more is wanted: plan rule 3, grow the model (hidden 768, 12 layers).
- Level 4, sealed test, Mac, `step-6500.safetensors` (`eval/results/run-004-test.json`): n=100, command 0.940, exact 0.940, well-formed 1.000. The 6 misses are all bare name-word questions (template F2: `hw n1x`, `shell`, `dictation voxtype`, `hyprland toggle`, `install service signal`, `capture screenrecording process`), each answered with a neighbouring script name.
- Free-phrased prompts (not template questions) at step 6,500: `How do I update Omarchy?` → `omarchy-update [-y]`; `What is the keybinding for Terminal?` → `SUPER + RETURN`; `How do I take a screenshot?` → `omarchy-capture-screenshot [...]`; `How do I set a theme?` → `omarchy-theme-install [git-repo-url]`; `How do I change the theme?` → `omarchy-theme-set-gnome`; `How do I install docker?` → `omarchy-install-docker-dbs`; `How do I lock the screen?` → `omarchy-toggle-nightlight [--status]`; `How do I toggle night light?` → `omarchy-toggle-fullscreen-desktop [toggle|on|off]`. Three of eight right: free phrasing is the model's weak side and is what the owner spot check measures.
- Export: `hf/run-004`, `gguf/run-004-f16.gguf`, `gguf/run-004-Q8_0.gguf` (36,009,472 bytes, SHA-256 `bb14cb629540d3a9cd40274e62a0e97a30c002082e715284907a90f28e7ad176`). `generate.py`, `mlx_lm.generate` and `llama-completion` all answer `omarchy-update [-y]` for the update prompt. Copied to the Omarchy box as `~/models/run-004-Q8_0.gguf`, hash verified.
- AMD column: pending the owner restarting `llama-server -m ~/models/run-004-Q8_0.gguf -c 1024 --port 8080`; then `eval/score.py --split test --run run-004 --endpoint http://localhost:8080/completion` from the box.

### run-003 results (2026-10-20)

- Level 3 on the 714 held-out pairs (`eval/results/run-003-valid-step-*.json`):

| checkpoint | exact | command | well-formed |
| --- | --- | --- | --- |
| step-2000 | 0.833 | 0.854 | 0.996 |
| step-2500 | 0.871 | 0.881 | 0.999 |
| step-3000 | 0.885 | 0.898 | 0.996 |
| step-3418 (final) | 0.933 | 0.940 | 0.996 |

- The new validation set is harder than run-002's (more phrasings per script, 714 instead of 427 pairs), so the numbers are not comparable across runs. Remaining failures are mostly scripts sharing a long prefix (`omarchy-agent-account-*`) and the bare name-word questions (`done`, `drive info`).
- Free-phrased prompts at step 3,418: `How do I update Omarchy?` → `omarchy-update [-y]`; `What is the keybinding for Terminal?` → `SUPER + RETURN`; `How do I take a screenshot?` → `omarchy-capture-screenshot [smart|region|windows|fullscreen|scroll] [copy|save]`; `How do I set a theme?` → `omarchy-theme-set-tmux` (a per-app variant; `omarchy-theme-set` expected); `How do I install docker?` → `omarchy-install-docker-dbs`.
- Export of `step-3418.safetensors`: `gguf/run-003-Q8_0.gguf`, 36,009,472 bytes, SHA-256 `0f6ae79ff02c2f57e37dc8c223e9de79b6d97404ac70cc8ef6f8fbee66f89ced`; `llama-completion` answers `omarchy-update [-y]` for the update prompt. Copied to the Omarchy box as `~/models/run-003-Q8_0.gguf` (hash verified) so the llama-server setup can be prepared while run-004 trains.

### AMD box, llama-server with run-003 (2026-10-20)

- Deliverable per the owner: the GGUF file served by llama.cpp (`llama-server`), not Ollama. The box has Arch package `llama-cpp` 0.6.0 (llama.cpp build 11429, commit d812350493), installed and started by the owner: `llama-server -m ~/models/run-003-Q8_0.gguf -c 1024 --port 8080`.
- `POST /completion` with `{"prompt": "<|q|>How do I update Omarchy?<|a|>", "n_predict": 64, "temperature": 0, "stop": ["<|end|>"]}` → `omarchy-update [-y]`, `stop_type` `eos`, last token id 5 (`<|end|>` acts as an end token).
- Mac vs AMD on the 714 validation pairs (not the sealed test), run-003 step-3418 (`eval/results/run-003-valid-amd-valid.json`):

| side | exact | command | well-formed | tokens/s |
| --- | --- | --- | --- | --- |
| Mac, MLX float32 | 0.933 | 0.940 | 0.996 | 1,242 |
| AMD, llama-server Q8_0 | 0.931 | 0.940 | 0.994 | 3,817 |

- Command-match gap 0.000, within the 0.03 threshold. Same worst failures on both sides.

### run-003 corpus (2026-10-20)

- `make_pairs.py` extended per plan rule 2 (templates H1–H4 summary heads, Y1–Y3 verb synonyms from a fixed table, F1–F2 script-name words; all listed in `data/pairs/templates.txt`). 14,636 pairs derived, 14,385 after dedupe (251 questions with two answers kept the first), 100 held back as test (seed 42, regenerated from the larger pool; the previous set was never scored), 14,285 in `pairs.txt`.
- Corpus: train.txt 1,144 documents + 13,571 pairs, 4,769,042 bytes, 1,400,359 tokens (bos 14,715 = documents); valid.txt 60 documents + 714 pairs, 279,056 bytes, 82,658 tokens. Level 1 tests: 5 passed. Formula: 40 epochs → 1,709 steps; run-003 uses 2 × 1,709 = 3,418.

### run-002 details (2026-10-20)

- Same curve as run-001: val loss (whole valid.txt) lowest 3.17 at step 500, 4.38 at the end; train loss 0.025 at the end.
- Level 3 on the 427 held-out pairs (`eval/results/run-002-valid-step-*.json`):

| checkpoint | exact | command | well-formed |
| --- | --- | --- | --- |
| step-1000 | 0.625 | 0.763 | 0.977 |
| step-1500 | 0.815 | 0.855 | 1.000 |
| step-2000 | 0.895 | 0.934 | 0.995 |
| step-2500 | 0.918 | 0.951 | 0.995 |
| step-3000 | 0.956 | 0.972 | 0.995 |
| step-3066 | 0.944 | 0.965 | 0.995 |

- Remaining failures are neighbouring script names (`omarchy-agent-account-state` for `omarchy-agent-account-home`, `omarchy-theme-set-claude` for `omarchy-theme-set`).
- The three free-phrased prompts at step 3,066: `omarchy-update-boot`, `SUPER + SHIFT + RETURN`, `omarchy-theme-set-t3code`. Still wrong; more steps add no phrasing coverage, so run-003 follows plan rule 2 (more pairs per script).

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
- Export of `step-1533.safetensors` (2026-10-20): `hf/run-001`, `gguf/run-001-f16.gguf`, `gguf/run-001-Q8_0.gguf` (36,009,472 bytes, SHA-256 `11ab3f608b63f71af852faab56e4d7badb4828e5bd8efc3d1f3c04a1353e7fad`). `generate.py`, `mlx_lm.generate` on the HF folder and `llama-completion` on the Q8_0 GGUF give identical greedy answers for the three sample prompts, each ending at `<|end|>` (llama.cpp reports `[end of text]`):

| prompt | answer |
| --- | --- |
| How do I update Omarchy? | `omarchy-update-available` |
| What is the keybinding for Terminal? | `SUPER + SHIFT + RETURN` |
| How do I set a theme? | `omarchy-theme-set-by-theme <theme-name>` |

- These three prompts are free paraphrases, not template questions; two of three answers name a neighbouring command (`omarchy-update`, `omarchy-theme-set`, `SUPER + RETURN` would be right). Template questions score 0.916. Lever per the plan: more paraphrase templates in `make_pairs.py`.
- Build 10080 has no `-no-cnv` in `llama-cli`; the raw completion command is `llama-completion -m <gguf> -p "<prompt>" -n 64 --temp 0 --no-display-prompt`. `--log-disable` also silences the generated text in this build.

## Level 4 results

(filled by step 6)

## Sign-off

Shipping candidate: run-004 `step-6500.safetensors` → `gguf/run-004-Q8_0.gguf`.

| Check | Threshold | Mac | AMD |
| --- | --- | --- | --- |
| Pipeline tests | all pass | 5 passed (2026-10-20, after the run-003 corpus rebuild) | n/a |
| Best val loss | < 2.0 | 2.96 on valid.txt (fails as written: that text is mostly unseen scripts); 0.52 or lower on the pair documents (run-001 measurement; later runs train longer) | n/a |
| Command match, test | ≥ 0.90 | 0.940 | pending server restart with run-004 |
| Well-formed, test | ≥ 0.98 | 1.000 | pending |
| Mac vs AMD gap | ≤ 0.03 | | pending (0.000 on the 714 validation pairs with run-003) |
| Owner spot check | owner's call | n/a | pending |
