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
END_ID = 5  # id of <|end|> in tokenizer/tokenizer.model (unk 0, bos 1, eos 2, then the three symbols)
PAIR_RE = re.compile(r"<\|q\|>(.*?)<\|a\|>(.*?)<\|end\|>", re.S)


PROSE_TOKENS = 192  # prose answers (manual paragraphs) get a longer budget than commands


def load_pairs(split):
    if split == "valid":
        kinds = {}
        pairs_file = ROOT / "data" / "pairs" / "pairs.jsonl"
        if pairs_file.exists():
            for line in pairs_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    kinds[d["q"]] = d.get("kind", "command")
        text = (ROOT / "data" / "valid.txt").read_text(encoding="utf-8")
        return [{"q": q, "a": a, "kind": kinds.get(q, "command")} for q, a in PAIR_RE.findall(text)]
    lines = (ROOT / "data" / "test" / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    pairs = [json.loads(line) for line in lines if line.strip()]
    for p in pairs:
        p.setdefault("kind", "command")
    return pairs


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

    def __call__(self, q, max_tokens=None):
        text, ids, _ = self.generate(self.model, self.sp, f"<|q|>{q}<|a|>", max_tokens or self.max_tokens)
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

    def __call__(self, q, max_tokens=None):
        max_tokens = max_tokens or self.max_tokens
        if self.ollama:
            r = self.post({"model": self.model, "prompt": q, "stream": False,
                           "options": {"temperature": 0, "num_predict": max_tokens, "stop": [END]}})
            text, n = r.get("response", ""), r.get("eval_count", 0)
            ended = r.get("done_reason") == "stop" and not text.rstrip().endswith("</s>")
            if r.get("eval_duration"):
                self.last_tps = n / (r["eval_duration"] / 1e9)
        else:
            r = self.post({"prompt": f"<|q|>{q}<|a|>", "n_predict": max_tokens, "temperature": 0,
                           "stop": [END], "cache_prompt": False, "return_tokens": True})
            text = r.get("content", "")
            # llama.cpp treats <|end|> as an end-of-generation token by name, so it stops with
            # stop_type "eos" and the text never contains the symbol; the stop word is a fallback.
            toks = r.get("tokens") or []
            ended = (toks[-1] == END_ID) if toks else (r.get("stop_type") in ("eos", "word") or bool(r.get("stopped_word")))
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
        prose = p.get("kind") == "prose"
        got, ended, n = gen(p["q"], PROSE_TOKENS if prose else args.max_tokens)
        n_tokens += n
        rows.append({
            "q": p["q"], "expected": p["a"], "got": got, "kind": p.get("kind", "command"),
            "exact": got.strip() == p["a"].strip(),
            # for prose, "command" means the first eight words match
            "command": (" ".join(got.split()[:8]) == " ".join(p["a"].split()[:8])) if prose else first_token(got) == first_token(p["a"]),
            "well_formed": ended and not repeated_trigram(got),
        })
        if i % 50 == 0:
            print(f"{i}/{len(pairs)}", file=sys.stderr)
    elapsed = time.time() - t0

    n = len(rows)
    commands = [r for r in rows if r["kind"] != "prose"] or rows
    result = {
        "run": args.run, "split": args.split, "n": n,
        "endpoint": args.endpoint,
        # the three headline numbers are over the command-like pairs (command, key, snippet), as in earlier runs
        "exact_match": sum(r["exact"] for r in commands) / len(commands),
        "command_match": sum(r["command"] for r in commands) / len(commands),
        "well_formed": sum(r["well_formed"] for r in commands) / len(commands),
        "tokens_per_s": getattr(gen, "last_tps", None) or n_tokens / elapsed,
        "by_kind": {},
    }
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind]
        result["by_kind"][kind] = {
            "n": len(rs),
            "exact": sum(r["exact"] for r in rs) / len(rs),
            "command_or_first8": sum(r["command"] for r in rs) / len(rs),
            "well_formed": sum(r["well_formed"] for r in rs) / len(rs),
        }
    failures = [r for r in rows if not r["command"]] + [r for r in rows if r["command"] and not r["well_formed"]] + [r for r in rows if r["command"] and r["well_formed"] and not r["exact"]]
    result["worst"] = [{"q": r["q"], "expected": r["expected"], "got": r["got"]} for r in failures[:20]]

    tag = args.tag or ("endpoint" if args.endpoint else None)
    out = ROOT / "eval" / "results" / f"{args.run}-{args.split}{'-' + tag if tag else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.split}: n={n} exact={result['exact_match']:.3f} command={result['command_match']:.3f} "
          f"well_formed={result['well_formed']:.3f} tokens/s={result['tokens_per_s']:.0f} -> {out.relative_to(ROOT)}")
    for kind, k in result["by_kind"].items():
        print(f"  {kind}: n={k['n']} exact={k['exact']:.3f} {'first8' if kind == 'prose' else 'command'}={k['command_or_first8']:.3f} well_formed={k['well_formed']:.3f}")


if __name__ == "__main__":
    main()
