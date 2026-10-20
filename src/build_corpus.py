#!/usr/bin/env python3
"""Build data/train.txt and data/valid.txt from data/raw/ plus the derived pairs in data/pairs/.

Each source file becomes one document: a `### <relative path>` header, the content, a blank
line. Pairs are appended after the raw documents. Raw documents are split 95/5 with seed 42.
Pairs are split 95/5 independently (seed 43), so validation questions are about scripts whose
text is in train, the same situation as the sealed test set.
"""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PAIRS = ROOT / "data" / "pairs" / "pairs.jsonl"
SEED = 42
VALID_FRACTION = 0.05
OMARCHY_KEEP = {"bin", "config", "default", "install", "migrations", "docs", "manual", "README.md", "version"}
DROP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg", ".bmp",
    ".ttf", ".otf", ".woff", ".woff2",
    ".mp4", ".mp3", ".wav", ".ogg",
    ".lock", ".pyc", ".o", ".so", ".dylib", ".zip", ".gz", ".tar", ".pdf", ".DS_Store",
}
DROP_NAMES = {"package-lock.json", "yarn.lock", "Gemfile.lock", "Cargo.lock", "uv.lock", "poetry.lock", "pnpm-lock.yaml", ".DS_Store"}


def wanted(rel: Path) -> bool:
    if ".git" in rel.parts:
        return False
    if rel.parts[0] == "omarchy":
        if len(rel.parts) < 2:
            return False
        top = rel.parts[1]
        if top not in OMARCHY_KEEP and not top.startswith("CHANGELOG"):
            return False
        if top == "manual" and len(rel.parts) > 3:  # translations live in manual/<lang>/
            return False
        if rel.parts[1:3] == ("default", "i18n") and len(rel.parts) > 4 and rel.parts[3] != "en":
            return False
    return rel.suffix.lower() not in DROP_SUFFIXES and rel.name not in DROP_NAMES


def normalize(raw: bytes):
    if b"\0" in raw:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.rstrip("\n") + "\n"


def main():
    docs = {}
    skipped = []
    for path in sorted(p for p in RAW.rglob("*") if p.is_file()):
        rel = path.relative_to(RAW)
        if not wanted(rel):
            continue
        text = normalize(path.read_bytes())
        if text is None:
            skipped.append(str(rel))
            continue
        docs[str(rel)] = f"### {rel}\n{text}\n"

    pairs = []
    if PAIRS.exists():
        pairs = [json.loads(line) for line in PAIRS.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        print(f"warning: {PAIRS} missing, building raw documents only")

    names = sorted(docs)
    order = names[:]
    random.Random(SEED).shuffle(order)
    n_valid = max(1, round(VALID_FRACTION * len(order)))
    valid_names = set(order[:n_valid])

    split = {"train": {"docs": [], "pairs": []}, "valid": {"docs": [], "pairs": []}}
    for name in names:
        split["valid" if name in valid_names else "train"]["docs"].append(name)
    pair_order = list(range(len(pairs)))
    random.Random(SEED + 1).shuffle(pair_order)
    valid_pairs = set(pair_order[: round(VALID_FRACTION * len(pairs))])
    for i, pair in enumerate(pairs):
        split["valid" if i in valid_pairs else "train"]["pairs"].append(pair)

    for name, part in split.items():
        out = ROOT / "data" / f"{name}.txt"
        with out.open("w", encoding="utf-8") as f:
            for doc in part["docs"]:
                f.write(docs[doc])
            for pair in part["pairs"]:
                f.write(f"<|q|>{pair['q']}<|a|>{pair['a']}<|end|>\n\n")
        longest = max(part["docs"], key=lambda d: len(docs[d].encode("utf-8")))
        print(f"{name}.txt: {len(part['docs'])} documents, {len(part['pairs'])} pairs, "
              f"{out.stat().st_size} bytes, longest document {longest} ({len(docs[longest].encode('utf-8'))} bytes)")
    if skipped:
        print(f"skipped {len(skipped)} non-UTF-8/binary files: {', '.join(skipped[:10])}")


if __name__ == "__main__":
    main()
