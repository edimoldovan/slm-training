#!/usr/bin/env python3
"""Greedy generation from a checkpoint: config.json + .safetensors + tokenizer. Stops at <|end|> or eos."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = ROOT / "tokenizer" / "tokenizer.model"
BOS_ID = 1
EOS_ID = 2
END = "<|end|>"


def load_model(config_path, weights_path):
    from mlx_lm.models.llama import Model, ModelArgs

    cfg = json.loads(Path(config_path).read_text())
    model = Model(ModelArgs.from_dict(cfg))
    model.load_weights(str(weights_path))
    model.eval()
    return model


def load_tokenizer(path=TOKENIZER):
    import sentencepiece as spm

    return spm.SentencePieceProcessor(model_file=str(path))


def generate(model, sp, prompt: str, max_tokens: int = 64):
    """Returns (text, token ids, stopped) where stopped is True when <|end|> or eos was produced."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    end_id = sp.piece_to_id(END)
    cache = make_prompt_cache(model)
    y = mx.array([[BOS_ID] + sp.encode(prompt)])
    out, stopped = [], False
    for _ in range(max_tokens):
        logits = model(y, cache=cache)
        nxt = int(mx.argmax(logits[0, -1]).item())
        out.append(nxt)
        if nxt in (end_id, EOS_ID):
            stopped = True
            break
        y = mx.array([[nxt]])
    return sp.decode(out), out, stopped


def resolve(args):
    if args.run:
        d = ROOT / "checkpoints" / args.run
        return d / "config.json", d / (args.checkpoint or "best.safetensors")
    return Path(args.config), Path(args.checkpoint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run id under checkpoints/; uses best.safetensors unless --checkpoint names a file")
    ap.add_argument("--checkpoint", help="weights file (name within the run dir, or a path when --config is given)")
    ap.add_argument("--config", help="config.json path when not using --run")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()
    config_path, weights_path = resolve(args)
    model = load_model(config_path, weights_path)
    sp = load_tokenizer()
    text, ids, stopped = generate(model, sp, args.prompt, args.max_tokens)
    print(text)
    print(f"[{len(ids)} tokens, {'stopped' if stopped else 'hit max-tokens'}]", file=sys.stderr)


if __name__ == "__main__":
    main()
