#!/usr/bin/env python3
"""Write a run's checkpoint as a Hugging Face Llama folder for mlx_lm and llama.cpp's converter."""
import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_CONFIG = {
    "tokenizer_class": "LlamaTokenizer",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "add_bos_token": True,
    "add_eos_token": False,
    "legacy": True,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="best.safetensors")
    args = ap.parse_args()

    run_dir = ROOT / "checkpoints" / args.run
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((run_dir / "config.json").read_text())
    cfg.update({"architectures": ["LlamaForCausalLM"], "torch_dtype": "bfloat16", "bos_token_id": 1, "eos_token_id": 2})

    weights = mx.load(str(run_dir / args.checkpoint))
    if cfg.get("tie_word_embeddings") and "lm_head.weight" not in weights:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    weights = {k: v.astype(mx.bfloat16) for k, v in weights.items()}
    mx.save_safetensors(str(out / "model.safetensors"), weights, metadata={"format": "pt"})

    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    shutil.copy(ROOT / "tokenizer/tokenizer.model", out / "tokenizer.model")
    (out / "tokenizer_config.json").write_text(json.dumps(TOKENIZER_CONFIG, indent=2) + "\n")
    (out / "special_tokens_map.json").write_text(json.dumps({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}, indent=2) + "\n")
    (out / "generation_config.json").write_text(json.dumps({"bos_token_id": 1, "eos_token_id": 2}, indent=2) + "\n")

    n = sum(v.size for v in weights.values())
    print(f"wrote {out}: {len(weights)} tensors, {n / 1e6:.1f}M values (lm_head counted), from {run_dir / args.checkpoint}")
    for name in sorted(weights)[:3] + ["...", "model.norm.weight", "lm_head.weight"]:
        print(" ", name, tuple(weights[name].shape) if name in weights else "")


if __name__ == "__main__":
    main()
