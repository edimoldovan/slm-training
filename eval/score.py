#!/usr/bin/env python3
"""Command accuracy on the validation pairs (--split valid) or the sealed test set (--split test).

Scores: exact match, command match (first whitespace token), well-formed (ends with <|end|>
within the token budget, no repeated 3-gram). Local generation from a run's checkpoint, or
over HTTP with --endpoint (llama-server /completion or Ollama /api/generate).
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
END = "<|end|>"
PAIR_RE = re.compile(r"<\|q\|>(.*?)<\|a\|>(.*?)<\|end\|>", re.S)


def load_pairs(split):
    if split == "valid":
        text = (ROOT / "data" / "valid.txt").read_text(encoding="utf-8")
        return [{"q": q, "a": a} for q, a in PAIR_RE.findall(text)]
    lines = (ROOT / "data" / "test" / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def first_token(s):
    parts = s.split()
    return parts[0] if parts else ""


def repeated_trigram(s):
    toks = s.split()
    grams = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
    return len(grams) != len(set(grams))


class Local:
    def __init__(self, run, checkpoint, max_tokens):
        import generate as g

        d = ROOT / "checkpoints" / run
        self.model = g.load_model(d / "config.json", d / checkpoint)
        self.sp = g.load_tokenizer()
        self.generate = g.generate
        self.max_tokens = max_tokens

    def __call__(self, q):
        text, ids, _ = self.generate(self.model, self.sp, f"<|q|>{q}<|a|>", self.max_tokens)
        ended = text.endswith(END)
        return text[: -len(END)] if ended else text, ended, len(ids)


class Endpoint:
    def __init__(self, url, model, max_tokens):
        self.url, self.model, self.max_tokens = url, model, max_tokens
        self.ollama = url.rstrip("/").endswith("/api/generate")
        self.last_tps = None

    def post(self, body):
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

    def __call__(self, q):
        if self.ollama:
            r = self.post({"model": self.model, "prompt": q, "stream": False,
                           "options": {"temperature": 0, "num_predict": self.max_tokens, "stop": [END]}})
            text, n = r.get("response", ""), r.get("eval_count", 0)
            ended = r.get("done_reason") == "stop" and not text.rstrip().endswith("</s>")
            if r.get("eval_duration"):
                self.last_tps = n / (r["eval_duration"] / 1e9)
        else:
            r = self.post({"prompt": f"<|q|>{q}<|a|>", "n_predict": self.max_tokens, "temperature": 0,
                           "stop": [END], "cache_prompt": False})
            text = r.get("content", "")
            ended = r.get("stop_type") == "word" or bool(r.get("stopped_word"))
            n = r.get("tokens_predicted", 0)
            t = r.get("timings", {})
            if t.get("predicted_per_second"):
                self.last_tps = t["predicted_per_second"]
        if END in text:
            text, ended = text.split(END, 1)[0], True
        return text, ended, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["valid", "test"], required=True)
    ap.add_argument("--run", required=True, help="run id; names the checkpoint and the results file")
    ap.add_argument("--checkpoint", default="best.safetensors")
    ap.add_argument("--endpoint", help="HTTP URL: llama-server .../completion or Ollama .../api/generate")
    ap.add_argument("--model", default="omarchy-slm", help="Ollama model name for --endpoint")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--tag", help="suffix for the results file (default: 'endpoint' when --endpoint is used)")
    ap.add_argument("--limit", type=int, help="score only the first N pairs")
    args = ap.parse_args()

    pairs = load_pairs(args.split)
    if args.limit:
        pairs = pairs[: args.limit]
    gen = Endpoint(args.endpoint, args.model, args.max_tokens) if args.endpoint else Local(args.run, args.checkpoint, args.max_tokens)

    rows, n_tokens, t0 = [], 0, time.time()
    for i, p in enumerate(pairs, 1):
        got, ended, n = gen(p["q"])
        n_tokens += n
        rows.append({
            "q": p["q"], "expected": p["a"], "got": got,
            "exact": got.strip() == p["a"].strip(),
            "command": first_token(got) == first_token(p["a"]),
            "well_formed": ended and not repeated_trigram(got),
        })
        if i % 50 == 0:
            print(f"{i}/{len(pairs)}", file=sys.stderr)
    elapsed = time.time() - t0

    n = len(rows)
    result = {
        "run": args.run, "split": args.split, "n": n,
        "endpoint": args.endpoint,
        "exact_match": sum(r["exact"] for r in rows) / n,
        "command_match": sum(r["command"] for r in rows) / n,
        "well_formed": sum(r["well_formed"] for r in rows) / n,
        "tokens_per_s": getattr(gen, "last_tps", None) or n_tokens / elapsed,
    }
    failures = [r for r in rows if not r["command"]] + [r for r in rows if r["command"] and not r["well_formed"]] + [r for r in rows if r["command"] and r["well_formed"] and not r["exact"]]
    result["worst"] = [{"q": r["q"], "expected": r["expected"], "got": r["got"]} for r in failures[:20]]

    tag = args.tag or ("endpoint" if args.endpoint else None)
    out = ROOT / "eval" / "results" / f"{args.run}-{args.split}{'-' + tag if tag else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.split}: n={n} exact={result['exact_match']:.3f} command={result['command_match']:.3f} "
          f"well_formed={result['well_formed']:.3f} tokens/s={result['tokens_per_s']:.0f} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
