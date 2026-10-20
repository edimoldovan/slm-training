#!/usr/bin/env python3
"""Fetch every page of the Omarchy manual site into data/raw/manual/<slug>.md, page title as the first line.

Crawls links under the start URL's path on the same host. HTML is converted with the standard
library only: headings -> `#`, <pre> -> fenced code blocks, lists -> `- `, inline <code> -> backticks.
"""
import argparse
import html
import re
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "manual"
SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "noscript", "svg", "button", "form"}
BLOCK_TAGS = {"p", "div", "section", "article", "main", "ul", "ol", "li", "table", "tr", "blockquote", "hr", "br", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "figure", "details", "summary"}


class ToMarkdown(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.links = []
        self.title = ""
        self.skip = 0
        self.in_title = False
        self.in_pre = False
        self.in_code = False
        self.heading = None
        self.list_depth = 0
        self.has_main = False
        self.in_main = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" and a.get("href"):
            self.links.append(a["href"])
        if tag in ("main", "article"):
            self.in_main += 1
        if tag == "title":
            self.in_title = True
            return
        if tag in SKIP_TAGS:
            self.skip += 1
            return
        if self.skip:
            return
        if tag == "pre":
            self.in_pre = True
            self.out.append("\n```\n")
        elif tag == "code" and not self.in_pre:
            self.in_code = True
            self.out.append("`")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.heading = int(tag[1])
            self.out.append("\n\n" + "#" * self.heading + " ")
        elif tag in ("ul", "ol"):
            self.list_depth += 1
            self.out.append("\n")
        elif tag == "li":
            self.out.append("\n" + "  " * (self.list_depth - 1) + "- ")
        elif tag == "br":
            self.out.append("\n")
        elif tag in BLOCK_TAGS:
            self.out.append("\n\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
            return
        if tag in SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if tag == "pre":
            self.in_pre = False
            self.out.append("\n```\n")
        elif tag == "code" and self.in_code:
            self.in_code = False
            self.out.append("`")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.heading = None
            self.out.append("\n\n")
        elif tag in ("ul", "ol"):
            self.list_depth = max(0, self.list_depth - 1)
            self.out.append("\n")
        elif tag in BLOCK_TAGS:
            self.out.append("\n\n")

    def handle_data(self, data):
        if self.in_title:
            self.title += data
            return
        if self.skip:
            return
        if self.in_pre:
            self.out.append(data)
        else:
            self.out.append(re.sub(r"\s+", " ", data))


def convert(page_html: str):
    p = ToMarkdown()
    p.feed(page_html)
    text = "".join(p.out)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    title = re.sub(r"\s+", " ", p.title).strip()
    return title, text, p.links


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "omarchy-slm-corpus/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def slug_for(url: str) -> str:
    path = urllib.parse.urlparse(url).path.rstrip("/")
    parts = [p for p in path.split("/") if p]
    slug = "-".join(parts[-2:]) if parts else "index"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", slug).strip("-") or "index"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="start URL of the manual; every same-host page under its path is fetched")
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args()

    start = urllib.parse.urlparse(args.url)
    base_path = start.path.rstrip("/")
    OUT.mkdir(parents=True, exist_ok=True)
    queue, seen, index = [args.url], set(), []
    while queue:
        url = queue.pop(0)
        clean = url.split("#", 1)[0].split("?", 1)[0].rstrip("/")
        if clean in seen:
            continue
        seen.add(clean)
        try:
            page = fetch(clean)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {clean}: {e}", file=sys.stderr)
            continue
        title, text, links = convert(page)
        slug = slug_for(clean)
        (OUT / f"{slug}.md").write_text(f"{title}\n\n{text}", encoding="utf-8")
        index.append(f"{clean}\t{slug}.md\t{title}")
        print(f"{clean} -> {slug}.md ({len(text)} chars)")
        for href in links:
            absolute = urllib.parse.urljoin(clean + "/", href)
            u = urllib.parse.urlparse(absolute)
            if u.netloc == start.netloc and u.path.rstrip("/").startswith(base_path):
                queue.append(absolute)
        time.sleep(args.delay)
    (ROOT / "data" / "manual-index.tsv").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"{len(index)} pages written to {OUT}")


if __name__ == "__main__":
    main()
