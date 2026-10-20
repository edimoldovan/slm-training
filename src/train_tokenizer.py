#!/usr/bin/env python3
"""Train the SentencePiece BPE tokenizer on data/train.txt only, then run the step 3 checks.

add_dummy_prefix=False is the one addition to the plan's settings: with the default dummy
prefix, "<|q|>" encodes as two ids ("▁", "<|q|>") and the single-id check fails.
"""
import random
import sys
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from train import encode_documents  # noqa: E402

spm.SentencePieceTrainer.train(
	input=str(ROOT / "data/train.txt"),
	model_prefix=str(ROOT / "tokenizer/tokenizer"),
	model_type="bpe",
	vocab_size=8192,
	character_coverage=1.0,
	byte_fallback=True,
	split_digits=True,
	allow_whitespace_only_pieces=True,
	remove_extra_whitespaces=False,
	normalization_rule_name="identity",
	user_defined_symbols=["<|q|>", "<|a|>", "<|end|>"],
	unk_id=0,
	bos_id=1,
	eos_id=2,
	pad_id=-1,
	add_dummy_prefix=False,
)

sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer/tokenizer.model"))
train_text = (ROOT / "data/train.txt").read_text(encoding="utf-8")
valid_text = (ROOT / "data/valid.txt").read_text(encoding="utf-8")

sample = random.Random(42).sample(train_text.split("\n"), 50)
bad = [line for line in sample if sp.decode(sp.encode(line)) != line]
print(f"round-trip: {50 - len(bad)}/50 lines exact")
for line in bad[:5]:
	print("  FAILED:", repr(line))

for sym in ["<|q|>", "<|a|>", "<|end|>"]:
	ids = sp.encode(sym)
	print(f"{sym} -> {ids}")
	assert len(ids) == 1, f"{sym} encodes to {len(ids)} ids"
assert not bad, "round-trip check failed"

n_train = len(encode_documents(sp, train_text))
n_valid = len(encode_documents(sp, valid_text))
print(f"vocab size: {sp.get_piece_size()}")
print(f"train.txt: {n_train} tokens")
print(f"valid.txt: {n_valid} tokens")
