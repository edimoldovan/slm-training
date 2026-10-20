#!/usr/bin/env python3
"""Derive question -> answer pairs deterministically from data/raw/. No language model anywhere.

Sources:
  scripts      data/raw/omarchy/bin/omarchy-*      `# omarchy:summary=` and `# omarchy:args=` header lines
  keybindings  data/raw/omarchy/default/hypr/bindings/*.lua, config/hypr/bindings.lua   o.bind("KEY", "Description", ...)
  manual       data/raw/omarchy/manual/*.md         heading followed by a fenced code block in its section

Writes data/pairs/pairs.jsonl (q, a, source), data/pairs/pairs.txt (the training format),
data/pairs/templates.txt (every template, for review) and data/test/prompts.jsonl (held back).
"""
import argparse
import json
import random
import re
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OMARCHY = RAW / "omarchy"
MANUAL = OMARCHY / "manual"

# Imperative summaries only: a first word ending in "s" is third person ("Returns true if ...")
# unless it is one of these imperatives.
IMPERATIVES_ENDING_IN_S = {"focus", "pass", "press", "process", "dismiss", "bypass", "suppress", "compress", "discuss", "access", "refocus", "unfocus"}

SCRIPT_TEMPLATES = [
    ("S1", "How do I {summary}?", True),
    ("S2", "What is the command to {summary}?", True),
    ("S3", "{Summary}", False),
    ("S4", "How can I {summary}?", True),
    ("S5", "I want to {summary}. What do I run?", True),
    ("S6", "Which omarchy command will {summary}?", True),
    ("S7", "Is there a command to {summary}?", True),
    ("S8", "What should I run to {summary}?", True),
    ("S9", "Command to {summary}", True),
    ("S10", "How to {summary}", True),
    ("S11", "In Omarchy, how do I {summary}?", True),
    ("S12", "Tell me how to {summary}.", True),
    ("S13", "What do I type to {summary}?", True),
    ("S14", "{summary}?", True),
    ("S15", "Script to {summary}", True),
]
KEY_TEMPLATES = [
    ("K1", "What is the keybinding for {desc}?"),
    ("K2", "What is the shortcut for {desc}?"),
    ("K3", "Keybinding for {desc}"),
    ("K4", "Hotkey for {desc}"),
    ("K5", "Which keybinding triggers {desc}?"),
]
# Added for run-003 (plan rule 2: more pairs per script). All deterministic:
# - the summary "head": the summary cut at the first separator below, e.g.
#   "Update Omarchy and system packages" -> "Update Omarchy"; used only when it keeps two or more words
# - the first verb replaced by a synonym from the fixed table below
# - the script's own name words: "omarchy-theme-set" -> "theme set"
HEAD_SEPARATORS = (" and ", ", ", " with ", " for ", " from ", " using ", " in ", " on ", " via ", " or ", " to ")
HEAD_TEMPLATES = [
    ("H1", "How do I {head}?"),
    ("H2", "{Head}"),
    ("H3", "What is the command to {head}?"),
    ("H4", "How can I {head}?"),
]
VERB_SYNONYMS = {
    "update": ["upgrade", "refresh"],
    "upgrade": ["update"],
    "install": ["add", "set up"],
    "remove": ["uninstall", "delete"],
    "uninstall": ["remove"],
    "launch": ["open", "start", "run"],
    "open": ["launch", "start"],
    "start": ["launch", "run"],
    "run": ["start", "launch"],
    "show": ["display", "print", "list"],
    "print": ["show", "display"],
    "list": ["show", "print"],
    "display": ["show"],
    "set": ["change", "configure"],
    "apply": ["set", "use"],
    "change": ["set", "switch"],
    "switch": ["change", "toggle"],
    "toggle": ["switch", "turn on or off"],
    "restart": ["reload", "relaunch"],
    "refresh": ["reload", "update"],
    "reload": ["refresh", "restart"],
    "check": ["verify", "test"],
    "detect": ["check", "find"],
    "choose": ["pick", "select"],
    "select": ["pick", "choose"],
    "pick": ["choose", "select"],
    "enable": ["turn on", "activate"],
    "disable": ["turn off", "deactivate"],
    "capture": ["take", "record"],
    "take": ["capture"],
    "lock": ["secure"],
    "reset": ["restore", "reinitialize"],
    "setup": ["set up", "configure"],
    "configure": ["set up", "set"],
    "create": ["make", "add"],
    "add": ["install", "create"],
    "delete": ["remove"],
    "edit": ["change", "modify"],
    "diagnose": ["debug", "troubleshoot"],
    "send": ["dispatch"],
    "stop": ["end", "kill"],
    "kill": ["stop", "end"],
    "mute": ["silence"],
    "fix": ["repair"],
    "migrate": ["move", "convert"],
    "search": ["find", "look up"],
    "find": ["search for", "locate"],
}
SYNONYM_TEMPLATES = [
    ("Y1", "How do I {syn} {rest}?"),
    ("Y2", "How can I {syn} {rest}?"),
    ("Y3", "What is the command to {syn} {rest}?"),
]
NAME_TEMPLATES = [
    ("F1", "omarchy {words}"),
    ("F2", "{words}"),
]
# Prose answers from the manual (run-005): a heading's first paragraph answers questions about the heading.
# h1 page titles use P2 and P4 only; a heading that is itself a question (the FAQ page) is used verbatim (P5).
PROSE_TEMPLATES = [
    ("P1", "What is {heading}?"),
    ("P2", "Tell me about {heading}"),
    ("P3", "How does {heading} work?"),
    ("P4", "{Heading}"),
]
PROSE_QUESTION_TEMPLATE = ("P5", "{Heading}")
MAX_ANSWER_WORDS = 90
MIN_ANSWER_WORDS = 8
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# M1 for headings that read as an instruction ("Install X"); M2 for noun or -ing headings.
MANUAL_TEMPLATES = [
    ("M1", "How do I {heading}?"),
    ("M2", "What is the command for {heading}?"),
]

META_RE = re.compile(r"^# omarchy:([a-z-]+)=(.*)$")
BIND_RE = re.compile(r'^\s*o\.bind(?:_toggle)?\(\s*"([^"]+)"\s*,\s*"([^"]+)"')
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")
FENCE_RE = re.compile(r"^\s*```")


def lower_first(summary: str) -> str:
    first = summary.split(" ", 1)[0]
    if len(first) > 1 and first[0].isupper() and first[1:].islower():
        return first[0].lower() + summary[1:]
    return summary


def summary_head(summary: str):
    low = summary.lower()
    cuts = [low.find(sep) for sep in HEAD_SEPARATORS if sep in low]
    if not cuts:
        return None
    head = summary[: min(cuts)].strip().rstrip(",")
    return head if len(head.split()) >= 2 else None


def with_synonym(summary: str, syn: str) -> str:
    rest = summary.split(" ", 1)[1] if " " in summary else ""
    return f"{syn} {rest}".strip()


def is_imperative(summary: str) -> bool:
    first = summary.split(" ", 1)[0].rstrip(".,:")
    if not first.isalpha() or not first[0].isupper():
        return False
    return not first.lower().endswith("s") or first.lower() in IMPERATIVES_ENDING_IN_S


def script_pairs(stats):
    pairs = []
    for path in sorted(OMARCHY.glob("bin/omarchy-*")):
        if not path.is_file():
            continue
        meta = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:40]:
            m = META_RE.match(line)
            if m:
                meta[m.group(1)] = m.group(2).strip()
        summary = meta.get("summary", "").rstrip(".").strip()
        if not summary:
            stats["scripts without summary"] += 1
            continue
        answer = path.name + (f" {meta['args']}" if meta.get("args") else "")
        source = str(path.relative_to(RAW))
        imperative = is_imperative(summary)
        if not imperative:
            stats["non-imperative summaries (S3 only)"] += 1
        for tid, template, needs_imperative in SCRIPT_TEMPLATES:
            if needs_imperative and not imperative:
                continue
            q = template.format(summary=lower_first(summary), Summary=summary)
            pairs.append({"q": q, "a": answer, "source": source, "template": tid, "kind": "command"})
        words = path.name.removeprefix("omarchy-").replace("-", " ")
        for tid, template in NAME_TEMPLATES:
            pairs.append({"q": template.format(words=words), "a": answer, "source": source, "template": tid, "kind": "command"})
        if not imperative:
            continue
        head = summary_head(summary)
        if head:
            stats["summary heads"] += 1
            for tid, template in HEAD_TEMPLATES:
                q = template.format(head=lower_first(head), Head=head)
                pairs.append({"q": q, "a": answer, "source": source, "template": tid, "kind": "command"})
        verb = summary.split(" ", 1)[0].lower()
        for syn in VERB_SYNONYMS.get(verb, []):
            stats["synonym forms"] += 1
            targets = [summary] + ([head] if head else [])
            for tid, template in SYNONYM_TEMPLATES:
                for target in targets:
                    rest = target.split(" ", 1)[1] if " " in target else ""
                    if rest:
                        pairs.append({"q": template.format(syn=syn, rest=rest), "a": answer, "source": source, "template": tid, "kind": "command"})
    return pairs


def keybinding_pairs(stats):
    pairs = []
    files = sorted(OMARCHY.glob("default/hypr/bindings/*.lua")) + [OMARCHY / "config/hypr/bindings.lua"]
    for path in files:
        if not path.is_file():
            continue
        source = str(path.relative_to(RAW))
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lstrip().startswith("--"):
                continue
            m = BIND_RE.match(line)
            if not m:
                continue
            key, desc = m.group(1).strip(), m.group(2).strip()
            stats["keybindings parsed"] += 1
            for tid, template in KEY_TEMPLATES:
                pairs.append({"q": template.format(desc=desc), "a": key, "source": source, "template": tid, "kind": "key"})
    return pairs


def manual_pairs(stats):
    pairs = []
    if not MANUAL.is_dir():
        return pairs
    for path in sorted(MANUAL.glob("*.md")):
        source = str(path.relative_to(RAW))
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        heading, block, in_fence, done = None, [], False, True
        for line in lines:
            h = HEADING_RE.match(line)
            if h and not in_fence:
                heading, done = h.group(1).strip(), False
                continue
            if FENCE_RE.match(line):
                if in_fence:
                    in_fence = False
                    if heading and not done and block:
                        stats["manual code blocks"] += 1
                        h = heading.rstrip("?.:")
                        tid, template = MANUAL_TEMPLATES[0] if is_imperative(h) and not h.split(" ", 1)[0].lower().endswith("ing") else MANUAL_TEMPLATES[1]
                        pairs.append({"q": template.format(heading=lower_first(h)), "a": "\n".join(block), "source": source, "template": tid, "kind": "snippet"})
                        done = True
                    block = []
                else:
                    in_fence = True
                continue
            if in_fence:
                block.append(line)
    return pairs


def first_paragraph(lines, start):
    """The first prose paragraph after line `start`, before the next heading; links reduced to their text."""
    para, in_fence = [], False
    for line in lines[start:]:
        if HEADING_RE.match(line) and not in_fence:
            break
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        text = line.strip()
        if not text or text.startswith(("- ", "* ", "|", "![", ">", "<", "1. ")):
            if para:
                break
            continue
        para.append(text)
    text = LINK_RE.sub(r"\1", " ".join(para)).replace("**", "")
    words = text.split()
    if len(words) < MIN_ANSWER_WORDS:
        return None
    if len(words) > MAX_ANSWER_WORDS:
        head = words[:MAX_ANSWER_WORDS]
        ends = [i for i, w in enumerate(head) if w.endswith((".", "!", "?"))]
        words = head[: ends[-1] + 1] if ends else head
    return " ".join(words)


def prose_pairs(stats):
    pairs = []
    if not MANUAL.is_dir():
        return pairs
    for path in sorted(MANUAL.glob("*.md")):
        source = str(path.relative_to(RAW))
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        in_fence = False
        for i, line in enumerate(lines):
            if FENCE_RE.match(line):
                in_fence = not in_fence
                continue
            h = HEADING_RE.match(line)
            if not h or in_fence:
                continue
            level = len(line) - len(line.lstrip("#"))
            heading = h.group(1).strip().rstrip("!.:")
            answer = first_paragraph(lines, i + 1)
            if not answer:
                stats["manual headings without a paragraph"] += 1
                continue
            stats["manual headings with prose"] += 1
            if heading.endswith("?"):
                tid, template = PROSE_QUESTION_TEMPLATE
                pairs.append({"q": template.format(Heading=heading), "a": answer, "source": source, "template": tid, "kind": "prose"})
                continue
            for tid, template in PROSE_TEMPLATES:
                if level == 1 and tid in ("P1", "P3"):
                    continue
                q = template.format(heading=lower_first(heading), Heading=heading)
                pairs.append({"q": q, "a": answer, "source": source, "template": tid, "kind": "prose"})
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-seed", type=int, default=42)
    ap.add_argument("--test-size", type=int, default=100)
    args = ap.parse_args()

    stats = Counter()
    raw = script_pairs(stats) + keybinding_pairs(stats) + manual_pairs(stats) + prose_pairs(stats)
    stats["pairs before dedupe"] = len(raw)

    answers = OrderedDict()
    pairs = []
    for p in raw:
        if p["q"] in answers:
            if answers[p["q"]] != p["a"]:
                stats["dropped: same question, different answer"] += 1
            else:
                stats["dropped: exact duplicate"] += 1
            continue
        answers[p["q"]] = p["a"]
        pairs.append(p)

    test_idx = set(random.Random(args.test_seed).sample(range(len(pairs)), args.test_size))
    test = [pairs[i] for i in sorted(test_idx)]
    train = [p for i, p in enumerate(pairs) if i not in test_idx]

    pairs_dir = ROOT / "data" / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    with (pairs_dir / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in train:
            f.write(json.dumps({"q": p["q"], "a": p["a"], "source": p["source"], "kind": p["kind"]}, ensure_ascii=False) + "\n")
    with (pairs_dir / "pairs.txt").open("w", encoding="utf-8") as f:
        for p in train:
            f.write(f"<|q|>{p['q']}<|a|>{p['a']}<|end|>\n")
    (ROOT / "data" / "test").mkdir(parents=True, exist_ok=True)
    with (ROOT / "data" / "test" / "prompts.jsonl").open("w", encoding="utf-8") as f:
        for p in test:
            f.write(json.dumps({"q": p["q"], "a": p["a"], "kind": p["kind"]}, ensure_ascii=False) + "\n")

    per_template = Counter(p["template"] for p in pairs)
    example = {}
    for p in pairs:
        example.setdefault(p["template"], p)
    lines = ["# Templates used by src/make_pairs.py (deterministic, no language model)", ""]
    for tid, template, needs_imperative in SCRIPT_TEMPLATES:
        lines.append(f"{tid}  {template}" + ("   [imperative summaries only]" if needs_imperative else "") + f"   -> script name + args   ({per_template[tid]} pairs)")
    for tid, template in HEAD_TEMPLATES:
        lines.append(f"{tid}  {template}   [head = summary cut at the first of {HEAD_SEPARATORS}, two words or more]   -> script name + args   ({per_template[tid]} pairs)")
    for tid, template in SYNONYM_TEMPLATES:
        lines.append(f"{tid}  {template}   [syn from VERB_SYNONYMS below replaces the first word; rest = summary or head without its first word]   -> script name + args   ({per_template[tid]} pairs)")
    for tid, template in NAME_TEMPLATES:
        lines.append(f"{tid}  {template}   [words = script name without omarchy-, dashes as spaces]   -> script name + args   ({per_template[tid]} pairs)")
    for tid, template in KEY_TEMPLATES:
        lines.append(f"{tid}  {template}   -> key combo   ({per_template[tid]} pairs)")
    for tid, template in MANUAL_TEMPLATES:
        lines.append(f"{tid}  {template}   -> fenced code block under the heading   ({per_template[tid]} pairs)")
    for tid, template in PROSE_TEMPLATES + [PROSE_QUESTION_TEMPLATE]:
        lines.append(f"{tid}  {template}   -> the heading's first paragraph, at most {MAX_ANSWER_WORDS} words   ({per_template[tid]} pairs)")
    lines.append("kinds: " + ", ".join(f"{k} {n}" for k, n in sorted(Counter(p['kind'] for p in pairs).items())))
    lines += ["", "# One example per template", ""]
    for tid, p in example.items():
        lines.append(f"{tid}  <|q|>{p['q']}<|a|>{p['a']}<|end|>   (from {p['source']})")
    lines += ["", "# Verb synonyms (first word of the summary -> replacements)", ""]
    lines += [f"{verb}: {', '.join(syns)}" for verb, syns in VERB_SYNONYMS.items()]
    lines += ["", "# Stats", ""]
    lines += [f"{k}: {v}" for k, v in sorted(stats.items())]
    lines += [f"pairs after dedupe: {len(pairs)}", f"held back as test (seed {args.test_seed}): {len(test)}", f"pairs.txt: {len(train)}"]
    (pairs_dir / "templates.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
