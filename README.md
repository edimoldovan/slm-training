# Omarchy SLM

A small language model (29.5M parameters, Llama architecture) trained from random weights on Omarchy's own text only: the scripts, configuration, manual and docs of the [Omarchy](https://github.com/basecamp/omarchy) repository. Ask it how to do something in Omarchy and it answers with the command, the keybinding, or a paragraph from the manual.

```
<|q|>How do I update Omarchy?<|a|>          → omarchy-update [-y]
<|q|>What is the keybinding for Terminal?<|a|>  → SUPER + RETURN
<|q|>Tell me about themes<|a|>               → Omarchy comes with twenty-two beautiful themes. …
```

No pretrained weights, no pretrained tokenizer, no text from another model: every training pair is derived by deterministic scripts from the corpus (`src/make_pairs.py`, templates in `data/pairs/templates.txt`).

## The model

`gguf/run-006-Q8_0.gguf` (36 MB) runs with llama.cpp on any machine. Prompt format: `<|q|>question<|a|>`; the answer ends with `<|end|>`, which llama.cpp treats as an end token. The file also carries a chat template, so `llama-server`'s `/v1/chat/completions` works with a plain user message.

```
llama-completion -m gguf/run-006-Q8_0.gguf -p "<|q|>How do I update Omarchy?<|a|>" -n 64 --temp 0 -no-cnv
```

Held-out results (run-006, `runs.md`): commands 95% exact, keybindings 98% exact, manual paragraphs 66% exact; sealed test 97% exact on command-like questions. Wordings far from the training templates still get a neighbouring answer.

A desktop app around it, [omallm](https://github.com/edimoldovan/omallm), embeds llama.cpp, shows the command and runs it after a click.

## Pipeline

Everything runs from this directory on a Mac with Apple silicon (MLX). `runs.md` is the log of record: environment, every run, scores, deviations from the plan.

```
python3.13 -m venv .venv && .venv/bin/pip install -U pip mlx mlx-lm sentencepiece transformers safetensors numpy torch pytest
git clone https://github.com/basecamp/omarchy data/raw/omarchy      # then prune to the text listed in runs.md
python src/make_pairs.py        # question/answer pairs from scripts, keybindings and manual headings
python src/build_corpus.py      # data/train.txt, data/valid.txt
python src/train_tokenizer.py   # SentencePiece BPE, 8,192 pieces (kept fixed across runs)
pytest tests/                   # level 1 checks
python src/train.py --run run-NNN --steps <4 × round(40 × train_tokens / 32768)>
python eval/score.py --split valid --run run-NNN --checkpoint step-N.safetensors
python eval/score.py --split test  --run run-NNN --checkpoint step-N.safetensors
python src/export_hf.py --run run-NNN --checkpoint step-N.safetensors --out hf/run-NNN
python tools/llama.cpp-b10080/convert_hf_to_gguf.py hf/run-NNN --outtype f16 --outfile gguf/run-NNN-f16.gguf
llama-quantize gguf/run-NNN-f16.gguf gguf/run-NNN-Q8_0.gguf Q8_0
```

The converter and its `gguf-py` come from llama.cpp tag `b10080` (the Homebrew build's version) under `tools/`, not in git.

## Layout

```
data/raw/        the pruned Omarchy clone (commit in runs.md)
data/pairs/      derived pairs, templates.txt
data/test/       100 sealed test prompts
tokenizer/       tokenizer.model, tokenizer.vocab
src/             build_corpus, make_pairs, train_tokenizer, model, train, generate, export_hf
eval/            score.py and results/
tests/           level 1 pipeline tests
gguf/            the shipped model (other exports and checkpoints are not in git)
runs.md          the log
```
