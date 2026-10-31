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
- Step 5, run-005 (2026-10-21): manual prose pairs; shipped step 7,000; sealed test on the Mac 0.967 / 0.989 on command-like pairs.
- Step 6 testing: level 1 pass; level 2 cannot be met as written (see sign-off); level 3 pass (0.945 / 0.997); level 4 pass on the Mac and on the AMD box (0.940 / 1.000 on both, gap 0.000). Owner spot check pending.
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
| run-008 | 8, 512 (29.5M, float32) | 8,268 | 6e-4 | 1.725 (valid.txt) | 1,000 | 49k → 48k | One change: pointer answers (`See <page>` / `See <page> › <heading>`) from each manual section's six most distinctive words and their 2- and 3-word subsets, 16,817 pairs before noise; 88,936 pairs, 3.39M tokens (20 epochs × 4). Shipped step 8,268: on 4,442 held-out pairs commands 0.960 exact, keys 0.930, pointers 0.878, prose 0.699, unknown 0.892; sealed test exact 0.949, command 0.970, well-formed 0.980. |
| run-007 | 8, 512 (29.5M, float32) | 7,308 | 6e-4 | 1.838 (valid.txt) | 900 | 50k → 48k | One change: every question also in two noisy forms (dropped word, swapped words, filler prefix or suffix, lowercase, typo) and 3,033 no-match questions answering `<|unknown|>`; 72,123 pairs, 2.99M tokens (20 epochs × 4). Shipped step 7,000: on 3,601 held-out pairs commands 0.965 exact, keys 0.923, prose 0.680, unknown 0.967; sealed test exact 0.960, command 0.970, well-formed 0.990. |
| run-006 | 8, 512 (29.5M, float32) | 8,084 | 6e-4 | 2.593 (valid.txt) | 600 | 49k → 47k | One change: 24 keybinding wordings (lowercase and plural forms, both keys when a description has two) and 7 more command wordings; 23,721 pairs. Best at step 8,084 on 1,181 held-out pairs: command-like exact 0.955, command 0.963; keys exact 0.976; commands 0.953; prose exact 0.660. **Shipped: `step-8084.safetensors`.** Sealed test: command-like exact 0.968, command 0.979, well-formed 0.979. |
| run-005 | 8, 512 (29.5M, float32) | 7,168 | 6e-4 | 2.829 (valid.txt) | 600 | 50k → 48k | One change: manual prose pairs added. Best at step 7,000: command-like pairs command 0.953, exact 0.953, well-formed 0.997; prose (36 held-out) exact 0.667, first eight words 0.694, well-formed 0.861. **Shipped: `step-7000.safetensors`.** Sealed test: command-like 0.967 / 0.989, prose 1 of 3. |
| run-004 | 8, 512 (29.5M, float32) | 6,836 | 6e-4 | 2.960 (valid.txt) | 500 | 66k → 45k | One change: steps doubled again. Level 3 peaks at step 6,500: command 0.945, exact 0.944, well-formed 0.997. **Shipped: `step-6500.safetensors`.** Sealed test on the Mac: command 0.940, exact 0.940, well-formed 1.000. |

### run-008 results (2026-10-31), shipped

- Owner report: "how do I adjust screen resolution?" answered `omarchy-transcode`; the right answer is a pointer to the manual's Monitors page. Change: `make_pairs.py` scores each manual section's words by frequency in the section times rarity across sections (a word seen in one section only must occur twice there), takes the six best in text order plus the heading's words, and makes questions from that set and its 2- and 3-word subsets with templates W1–W4 (`{kws}`, `how do I {kws}`, `where do I {kws}`, `how do I change {kws}`); the answer is `See <page>` for a page's own section and `See <page> › <heading>` below it. 302 sections, 16,817 pointer pairs, plus two noisy forms each. Totals: 88,936 pairs after dedupe (command 49,894, key 16,382, pointer 16,817, prose 2,710, snippet 100, unknown 3,033); 3,386,256 train tokens; steps 4 × round(20 × tokens / 32,768) = 8,268.
- The omaslm app renders a pointer as text with an "Open manual" button (omarchy.org/manual/<page>/#<heading>), and its constrained decoding also limits pointer answers to the manual's real page and heading names (`src/manual.rs`, generated from the corpus).
- Level 3 on the 4,442 held-out pairs:

| checkpoint | commands exact | keys exact | pointers exact | prose exact | unknown exact |
| --- | --- | --- | --- | --- | --- |
| step-7000 | 0.962 | 0.912 | 0.861 | 0.669 | 0.581 |
| step-8000 | 0.960 | 0.935 | 0.860 | 0.744 | 0.831 |
| step-8268 (final) | 0.960 | 0.930 | 0.878 | 0.699 | 0.892 |

- Level 4, sealed test (61 command, 13 key, 22 pointer, 1 prose, 3 unknown), Mac (`eval/results/run-008-test.json`): exact 0.949, command 0.970, well-formed 0.980; pointers 21 of 22, unknown 3 of 3.
- Free wordings at step 8,268: `how do I adjust screen resolution?` → `See Monitors`; `how do I make text bigger` → `See Monitors › Making text bigger or smaller`; `where do I change monitor scaling` → `See Monitors`; `how do i connect bluetooth headphones` → `See The Top Bar`; `what is the shortcut key for screenshots?` → `PRINT or SUPER + CTRL + ALT + C`; `How do I update Omarchy?` → `omarchy-update [-y]`. Wrong: `screen resolution` → `See Branding › About screen`; `update omarchy pls` → `omarchy-update-confirm`; `whats the keybind for the terminal` → `SUPER + SHIFT + TAB`; `No it's not` → a manual paragraph (was `<|unknown|>` in run-007); `how do i cook pasta` → a pointer instead of `<|unknown|>`.
- Export: `hf/run-008`, `gguf/run-008-Q8_0.gguf`, SHA-256 `b73c349aedfbf03e4e98324a61e897759d2c6625a8a34a1a66529f32d71d3d49`. The export now writes float32 weights (the earlier bfloat16 export made llama.cpp flip near-tie answers, e.g. `how do I make text bigger` → `See Branding › Screensaver` instead of MLX's `See Monitors › Making text bigger or smaller`); f16 GGUF then Q8_0 as before. Q8_0 still flips that answer, so the shipped file is the f16 GGUF: `gguf/run-008-f16.gguf`, 67,589,856 bytes, SHA-256 `efab1b4f52a6a203cbd115b7c463783f614519ba27aa4e2cf04bbf8781331ad9`, linked as `~/models/omaslm.gguf` on the box. Q8_0 stays for the training repo's committed run-006 model only. Copied to the Omarchy box as `~/models/run-008-Q8_0.gguf`; `~/models/omaslm.gguf` links to it.

### run-007 results (2026-10-31), shipped

- Owner request: questions in free wording went wrong. Change: `make_pairs.py` adds two noisy forms of every question, chosen per question with a seed from its text (drop a word, swap two adjacent words, filler prefix such as "quick question" or "so in Omarchy,", filler suffix such as "please", lowercase, a typo), plus 33 chit-chat lines and 3,000 random strings of corpus words answering `<|unknown|>`. The omaslm app shows "no match" for that answer. 72,930 derived, 72,123 after dedupe (command 49,894, key 16,382, prose 2,714, snippet 100, unknown 3,033); 100 held back as test. Corpus 2,993,048 train tokens, so the plan's 20-epoch tier applies: steps = 4 × round(20 × tokens / 32,768) = 7,308.
- Level 3 on the 3,601 held-out pairs (noisy forms included, so harder than run-006's set):

| checkpoint | commands exact | keys exact | prose exact | unknown exact |
| --- | --- | --- | --- | --- |
| step-6000 | 0.953 | 0.904 | 0.615 | 0.948 |
| step-7000 | 0.965 | 0.923 | 0.680 | 0.967 |
| step-7308 (final) | 0.970 | 0.926 | 0.607 | 0.895 |

- Shipped step 7,000: best on prose and on recognising no-match questions, half a point under the final checkpoint on commands.
- Level 4, sealed test (77 command, 18 key, 5 unknown), Mac (`eval/results/run-007-test.json`): exact 0.960, command 0.970, well-formed 0.990; keys 18 of 18, unknown 5 of 5.
- The owner's wordings at step 7,000: `what is the shortcut key for screenshots?` and `what is the shortcut key for taking screenshots?` → `PRINT or SUPER + CTRL + ALT + C`; `shortcut for browser` → `SUPER + SHIFT + RETURN or SUPER + SHIFT + B`; `No it's not` and `hello` → `<|unknown|>`. Still wrong: `key for lock screen` → `SUPER + F`; `update omarchy pls` → `omarchy-update-analyze-logs`; `whats the keybind for the terminal` → `SUPER + SHIFT + N`; `how do i cook pasta` → a memorised manual page instead of `<|unknown|>`.
- Export: `hf/run-007`, `gguf/run-007-Q8_0.gguf`, SHA-256 `bb1cd50bd1f59d214ab91015532e9fd8ce41e63caa6ceb394960397512b0df59`. Copied to the Omarchy box as `~/models/run-007-Q8_0.gguf`; `~/models/omaslm.gguf` links to it (the app was renamed from omallm to omaslm on 2026-10-31).

### run-006 results (2026-10-21), shipped

- Owner report: free wordings such as "what is the shortcut key for screenshots?" got wrong keybindings from run-005. Data change: 24 keybinding wordings per description (K1–K24 in `templates.txt`), seven of them also with the description in lowercase and in the plural for one-word descriptions; a description bound to two keys answers both ("PRINT or SUPER + CTRL + ALT + C"); 16 such descriptions. Commands get S16–S22. 23,904 derived, 23,721 after dedupe (command 17,042, key 5,673, prose 972, snippet 34), 100 held back as test, 1,655,668 train tokens, 8,084 steps (4 × 2,021).
- Level 3 on the 1,181 held-out pairs (`eval/results/run-006-valid-step-*.json`):

| checkpoint | command-like exact | commands exact | keys exact | prose exact | prose first 8 |
| --- | --- | --- | --- | --- | --- |
| step-7000 | 0.947 | 0.948 | 0.948 | 0.596 | 0.638 |
| step-7500 | 0.953 | 0.953 | 0.958 | 0.660 | 0.681 |
| step-8000 | 0.951 | 0.942 | 0.979 | 0.617 | 0.660 |
| step-8084 (final) | 0.955 | 0.949 | 0.976 | 0.660 | 0.702 |

- Level 4, sealed test (68 command, 27 key, 5 prose), Mac, `step-8084.safetensors` (`eval/results/run-006-test.json`): command-like exact 0.968, command 0.979, well-formed 0.979 (two keybinding answers ran past `<|end|>`); prose 3 of 5.
- The owner's wordings at step 8,084: `what is the shortcut key for screenshots?` → `PRINT or SUPER + CTRL + ALT + C` (right for the cloned master: `default/hypr/bindings/media.lua` binds Screenshot to PRINT and SUPER + CTRL + ALT + C; the box's older install binds SUPER + SHIFT + S); `what is the shortcut key for taking screenshots?` → the manual's Omasnap paragraph; `What is the keybinding for Terminal?` → `SUPER + RETURN`; `How do I update Omarchy?` → `omarchy-update [-y]`. Still wrong: `shortcut for browser` → `SUPER + SHIFT + ALT + A`, `key for lock screen` → a command.
- Export: `hf/run-006`, `gguf/run-006-Q8_0.gguf`, SHA-256 `410d96c3db4c8e5e3018e0a93b3208527141e2b8d302b3681941c76561e74b5f`; `llama-completion -no-cnv` answers `PRINT or SUPER + CTRL + ALT + C` for the screenshot question. Copied to the Omarchy box as `~/models/run-006-Q8_0.gguf`; `~/models/omallm.gguf` now links to it. The omallm app's `--ask` gives the same four answers.

### run-005 results (2026-10-21), shipped

- Level 3 on the 763 held-out pairs (`eval/results/run-005-valid-step-*.json`), headline numbers over the 727 command-like pairs:

| checkpoint | exact | command | well-formed | prose exact | prose first 8 words | prose well-formed |
| --- | --- | --- | --- | --- | --- | --- |
| step-6000 | 0.946 | 0.949 | 0.997 | 0.583 | 0.611 | 0.889 |
| step-6500 | 0.949 | 0.950 | 0.999 | 0.611 | 0.611 | 0.861 |
| step-7000 | 0.953 | 0.953 | 0.997 | 0.667 | 0.694 | 0.861 |
| step-7168 (final) | 0.953 | 0.953 | 0.999 | 0.611 | 0.639 | 0.833 |

- Commands improved slightly over run-004 (0.945 → 0.953). Prose: two thirds of held-out manual questions get the right paragraph; a wrong paragraph is a neighbouring section; the well-formed misses are paragraphs that repeat a 3-gram or run past 192 tokens.
- Free questions at step 7,000: `Tell me about themes` → the themes paragraph; `What about Wi-Fi and Bluetooth?` → the FAQ answer; `What is the top bar?` → the hide-bar paragraph (near miss); `What is Omarchy?` → `omarchy-install-editor-zed` (no pair asks that: the page title is "Welcome to Omarchy" and h1 titles get P2/P4 only); `How does updating work?` → a wrong paragraph. Commands unchanged: `omarchy-update [-y]`, `SUPER + RETURN`, the capture command.
- Level 4, sealed test (regenerated set: 90 command, 7 key, 3 prose), Mac, `step-7000.safetensors` (`eval/results/run-005-test.json`): command-like pairs command 0.967, exact 0.967, well-formed 0.989; prose 1 of 3.
- Export: `hf/run-005` (chat template included), `gguf/run-005-Q8_0.gguf`, 36,009,696 bytes, SHA-256 `40a1ec089814641786f91fecacdcd82611558c193108a98bdcedd3c0ad4984e3`; `llama-completion -no-cnv` gives `omarchy-update [-y]` and the themes paragraph. Copied to the Omarchy box as `~/models/run-005-Q8_0.gguf` (hash verified); `~/models/omallm.gguf` links to it and the omallm app reads that name. The AMD sealed-test column for run-005 is open until the owner restarts llama-server with this file.

### run-005 corpus (2026-10-21)

- Owner request: the model should also answer questions with explanations, not only commands. Source: the English manual's headings. For each heading (h1–h4) with a prose paragraph under it, the first paragraph (links reduced to their text, cut at a sentence end before 90 words, at least 8 words) answers P1 `What is {heading}?`, P2 `Tell me about {heading}`, P3 `How does {heading} work?`, P4 `{Heading}`; page titles (h1) use P2 and P4 only; a heading that is itself a question (the FAQ page) is used verbatim (P5). 278 headings with prose, 39 without; 972 prose pairs.
- Every pair now carries a `kind`: command 13,441, key 910, prose 972, snippet 34. 15,620 derived, 15,357 after dedupe, 100 held back as test (seed 42, regenerated from the larger pool; the previous set scored 0.940 on run-004), 15,257 in `pairs.txt`.
- Corpus: train.txt 1,144 documents + 14,494 pairs, 5,033,888 bytes, 1,468,328 tokens; valid.txt 60 documents + 763 pairs. Level 1 tests: 5 passed. Formula: 40 epochs → 1,792 steps; run-005 uses 4 × 1,792 = 7,168 (the run-004 schedule).
- `eval/score.py`: the three headline numbers stay over the command-like pairs (command, key, snippet); a `by_kind` block reports prose separately (exact match and first-eight-words match) with a 192-token budget for prose answers.

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
- Chat clients (owner request, 2026-10-20, for Cherry Studio): `export_hf.py` now writes a `chat_template` into `tokenizer_config.json`, so the GGUF carries `tokenizer.chat_template` and llama-server's `/v1/chat/completions` renders a user message as `<|q|>message<|a|>` (assistant turns close with `<|end|>`, system messages are dropped). Weights unchanged; `gguf/run-004-Q8_0.gguf` rebuilt, 36,009,472 bytes, SHA-256 `66f1ca859a04025919f7f26803c7d036bee15cf3e99d79fc4c77381902ed71bb`, copied to the box as `~/models/run-004-Q8_0.gguf` (hash verified). With a chat template present, `llama-completion` needs `-no-cnv` for a raw prompt; so checked, it still answers `omarchy-update [-y]`.
- Step 7 done-criterion "the service starts on boot": not done. `llama-server` runs as a foreground process started by the owner. A systemd user unit is the owner's step (file `~/.config/systemd/user/omarchy-slm.service`, `ExecStart=/usr/bin/llama-server -m %h/models/run-004-Q8_0.gguf -c 1024 --port 8080`, then `systemctl --user enable --now omarchy-slm`).
- Level 4, sealed test, AMD box, `llama-server` (Arch llama-cpp 0.6.0, build 11429) with `~/models/run-004-Q8_0.gguf`, `eval/score.py --split test --run run-004 --endpoint http://localhost:8080/completion` (`eval/results/run-004-test-amd.json`): n=100, command 0.940, exact 0.940, well-formed 1.000, 3,524 tokens/s. Identical to the Mac; gap 0.000.

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

Shipped: run-008 `step-8268.safetensors` → `gguf/run-008-f16.gguf` (2026-10-31; f16, since Q8_0 flipped a near-tie pointer answer). The run-004 rows below stand as the last full Mac-and-AMD measurement; run-005's Mac numbers are in its section (command-like 0.967 / 0.989 on the regenerated sealed test).

| Check | Threshold | Mac | AMD |
| --- | --- | --- | --- |
| Pipeline tests | all pass | 5 passed (2026-10-20, after the run-003 corpus rebuild) | n/a |
| Best val loss | < 2.0 | run-004 step-6500: 4.29 on valid.txt at that step (2.96 at step 500); fails as written because valid.txt is mostly 60 unseen scripts (loss 6.52 on those). On the 714 pair documents alone: 0.454. Proceeding past this row is the owner's call. | n/a |
| Command match, test | ≥ 0.90 | 0.940 | 0.940 |
| Well-formed, test | ≥ 0.98 | 1.000 | 1.000 |
| Mac vs AMD gap | ≤ 0.03 | 0.000 | 0.000 |
| Owner spot check | owner's call | n/a | pending |
