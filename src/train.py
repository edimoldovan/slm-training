#!/usr/bin/env python3
"""Train the Omarchy SLM from random weights with MLX.

Documents in data/train.txt start at a `### <path>` header naming a file under
data/raw/, or at a `<|q|>` pair block. Every document gets a leading bos token.
"""
import argparse
import csv
import json
import math
import re
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RAW = ROOT / "data" / "raw"
TOKENIZER = ROOT / "tokenizer" / "tokenizer.model"
BOS_ID = 1
EOS_ID = 2
VOCAB_SIZE = 8192


def is_doc_start(line: str) -> bool:
    if line.startswith("<|q|>"):
        return True
    if line.startswith("### "):
        return (RAW / line[4:].strip()).is_file()
    return False


def split_documents(text: str) -> list[str]:
    docs, cur = [], []
    for line in text.splitlines(keepends=True):
        if cur and is_doc_start(line):
            docs.append("".join(cur))
            cur = []
        cur.append(line)
    if cur:
        docs.append("".join(cur))
    return [d.rstrip("\n") + "\n" for d in docs if d.strip()]


def load_tokenizer():
    import sentencepiece as spm

    return spm.SentencePieceProcessor(model_file=str(TOKENIZER))


def encode_documents(sp, text: str) -> np.ndarray:
    """bos + tokens of every document, concatenated into one int32 array."""
    ids = []
    for doc in split_documents(text):
        ids.append(BOS_ID)
        ids.extend(sp.encode(doc))
    return np.asarray(ids, dtype=np.int32)


def build_bin(split: str) -> np.ndarray:
    txt = ROOT / "data" / f"{split}.txt"
    binf = ROOT / "data" / f"{split}.bin"
    fresh = binf.exists() and binf.stat().st_mtime >= max(txt.stat().st_mtime, TOKENIZER.stat().st_mtime)
    if not fresh:
        arr = encode_documents(load_tokenizer(), txt.read_text(encoding="utf-8"))
        arr.tofile(binf)
        print(f"encoded {split}.txt -> {binf.name}: {len(arr)} tokens")
    return np.fromfile(binf, dtype=np.int32)


def get_batch(arr: np.ndarray, rng: np.random.Generator, batch_size: int, seq_len: int):
    """batch_size random windows of seq_len+1 tokens; inputs are [:-1], targets [1:]."""
    if len(arr) <= seq_len + 1:
        raise SystemExit(f"array of {len(arr)} tokens is too short for seq_len {seq_len}")
    starts = rng.integers(0, len(arr) - seq_len - 1, size=batch_size)
    windows = arr[starts[:, None] + np.arange(seq_len + 1)[None, :]]
    return windows[:, :-1], windows[:, 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None, help="checkpoint .safetensors to continue from")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = ap.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from model import CONFIG, build

    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    val_rng = np.random.default_rng(args.seed + 1)

    train_arr = build_bin("train")
    valid_arr = build_bin("valid")
    run_dir = ROOT / "checkpoints" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")

    model = build()
    model.set_dtype(getattr(mx, args.dtype))
    start_step = 0
    if args.resume:
        model.load_weights(args.resume)
        m = re.search(r"step-(\d+)", args.resume)
        start_step = int(m.group(1)) if m else 0
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"model: {n_params / 1e6:.1f}M parameters, dtype {args.dtype}, device {mx.default_device()}")
    print(f"data: train {len(train_arr)} tokens, valid {len(valid_arr)} tokens")

    warmup = min(args.warmup, max(args.steps - 1, 1))
    decay_steps = max(args.steps - warmup, 1)
    schedule = optim.join_schedules(
        [optim.linear_schedule(0.0, args.lr, warmup), optim.cosine_decay(args.lr, decay_steps, args.lr * 0.1)],
        [warmup],
    )
    optimizer = optim.AdamW(learning_rate=schedule, betas=[0.9, 0.95], weight_decay=args.weight_decay)
    optimizer.init(model.trainable_parameters())
    if start_step:
        optimizer.state["step"] = mx.array(start_step, mx.uint64)

    def loss_fn(model, inputs, targets):
        logits = model(inputs).astype(mx.float32)
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(inputs, targets):
        loss, grads = loss_and_grad(model, inputs, targets)
        grads, grad_norm = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        return loss, grad_norm

    def current_lr():
        lr = optimizer.learning_rate
        return float(lr.item() if hasattr(lr, "item") else lr)

    val_batches = [
        tuple(mx.array(a) for a in get_batch(valid_arr, val_rng, args.batch_size, args.seq_len)) for _ in range(20)
    ]

    def val_loss():
        losses = [loss_fn(model, vx, vy) for vx, vy in val_batches]
        return float(mx.mean(mx.stack(losses)).item())

    best_path = run_dir / "best.json"
    best = json.loads(best_path.read_text()) if (start_step and best_path.exists()) else {"val_loss": math.inf, "step": None}
    log_path = run_dir / "log.csv"
    logf = open(log_path, "a" if start_step else "w", newline="")
    log = csv.writer(logf)
    if not start_step:
        log.writerow(["step", "train_loss", "val_loss", "lr", "tokens_per_s"])

    t0 = time.time()
    tokens_since = 0
    train_losses = []
    for s in range(start_step + 1, args.steps + 1):
        x, y = get_batch(train_arr, rng, args.batch_size, args.seq_len)
        loss, grad_norm = step(mx.array(x), mx.array(y))
        mx.eval(loss, grad_norm, state)
        loss_v = loss.item()
        if not math.isfinite(loss_v):
            print(f"step {s}: loss is {loss_v}, stopping")
            sys.exit(1)
        train_losses.append(loss_v)
        tokens_since += x.size
        if s == 1:
            print(f"step 1: train_loss {loss_v:.4f} grad_norm {grad_norm.item():.2f} lr {current_lr():.2e}")
        if s % args.eval_every == 0 or s == args.steps:
            vl = val_loss()
            tps = tokens_since / (time.time() - t0)
            tl = sum(train_losses) / len(train_losses)
            lr = current_lr()
            print(f"step {s}: train_loss {tl:.4f} val_loss {vl:.4f} lr {lr:.2e} tokens/s {tps:.0f}", flush=True)
            log.writerow([s, f"{tl:.4f}", f"{vl:.4f}", f"{lr:.3e}", f"{tps:.0f}"])
            logf.flush()
            if vl < best["val_loss"]:
                best = {"val_loss": vl, "step": s}
                model.save_weights(str(run_dir / "best.safetensors"))
                best_path.write_text(json.dumps(best) + "\n")
            t0 = time.time()
            tokens_since = 0
            train_losses = []
        if s % args.save_every == 0 or s == args.steps:
            model.save_weights(str(run_dir / f"step-{s}.safetensors"))
    logf.close()
    print(f"done: best val_loss {best['val_loss']:.4f} at step {best['step']}")


if __name__ == "__main__":
    main()
