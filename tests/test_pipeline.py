"""Level 1 pipeline tests. Run with `pytest tests/` or `python tests/test_pipeline.py`."""
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import train  # noqa: E402

TRAIN_TXT = ROOT / "data" / "train.txt"


def test_tokenizer_roundtrip():
    sp = train.load_tokenizer()
    lines = TRAIN_TXT.read_text(encoding="utf-8").split("\n")
    sample = random.Random(42).sample(lines, 200)
    bad = [line for line in sample if sp.decode(sp.encode(line)) != line]
    assert not bad, f"{len(bad)} lines do not round-trip, e.g. {bad[0]!r}"


def test_special_tokens_single_id():
    sp = train.load_tokenizer()
    for sym in ["<|q|>", "<|a|>", "<|end|>"]:
        assert len(sp.encode(sym)) == 1, sym


def test_bins_in_range_and_start_with_bos():
    for split in ("train", "valid"):
        arr = train.build_bin(split)
        assert arr.min() >= 0 and arr.max() < train.VOCAB_SIZE, split
        assert arr[0] == train.BOS_ID, split


def test_batch_inputs_and_targets_offset_by_one():
    arr = np.arange(10_000, dtype=np.int32)
    x, y = train.get_batch(arr, np.random.default_rng(0), 8, 64)
    assert x.shape == y.shape == (8, 64)
    assert (x[:, 1:] == y[:, :-1]).all()
    assert (y == x + 1).all()


def test_no_test_set_leakage():
    text = TRAIN_TXT.read_text(encoding="utf-8")
    lines = (ROOT / "data" / "test" / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 100
    for line in lines:
        assert line not in text
        d = json.loads(line)
        assert f"<|q|>{d['q']}<|a|>{d['a']}<|end|>" not in text, d["q"]


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)
