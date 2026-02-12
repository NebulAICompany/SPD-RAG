#!/usr/bin/env python3
"""
wiki_to_md_clean.py
"""

import argparse
import os
import re
import subprocess
import sys
from urllib.parse import urlparse, unquote

import requests
from bs4 import BeautifulSoup

API_URL = "https://en.wikipedia.org/w/api.php"


def url_to_title(url: str) -> str:
    path = urlparse(url).path
    slug = os.path.basename(path)
    return slug or "Main_Page"


def fetch_article_html(title: str) -> str:
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "formatversion": "2",
    }
    headers = {
        "User-Agent": "SPD-RAG-wiki-scraper/1.0 (contact: your-email@example.com)"
    }
    resp = requests.get(API_URL, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")
    return data["parse"]["text"]


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["style", "script", "noscript"]):
        tag.decompose()

    junk_selectors = [
        "span.mw-editsection",
        "a.mw-jump-link",
        "table.mw-empty-elt",
        "table.navbox",
        "table.vertical-navbox",
        "div.hatnote",
        "div.metadata",
        "div.mw-references-wrap",
    ]
    for sel in junk_selectors:
        for el in soup.select(sel):
            el.decompose()

    for thumb in soup.select("div.thumb, div.tmulti, div.thumbinner, div.gallery"):
        caption = thumb.get_text(" ", strip=True)
        new_p = soup.new_tag("p")
        new_p.string = caption
        thumb.replace_with(new_p)

    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            if attr.startswith("data-mw") or attr in {"class", "style", "id"}:
                del tag.attrs[attr]

    for tag in soup.find_all(True):
        if not tag.get_text(strip=True) and tag.name not in {"table", "tr", "td", "th"}:
            tag.decompose()

    return str(soup)


def html_to_markdown(html: str) -> str:
    pandoc_cmd = [
        "pandoc",
        "--from=html",
        "--to=markdown",
        "--wrap=none",
    ]

    html = html.encode("utf-8", errors="ignore").decode("utf-8")

    proc = subprocess.Popen(
        pandoc_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    out, err = proc.communicate(html)
    if proc.returncode != 0:
        sys.stderr.write(f"pandoc failed ({proc.returncode}): {err}\n")
        sys.exit(proc.returncode)
    return out


def strip_citation_markers(markdown: str) -> str:
    """
    Remove inline citation markers / footnotes while keeping the text.
    Handles patterns like:
      ^[1]
      ^[[1]](#cite_note-...)
      [[1]](#cite_note-...)
    """

    # 1) Remove pandoc-style footnotes starting with ^
    #    Examples: ^[1], ^[[1]](#cite_note-...), ^[text]
    markdown = re.sub(r'\^\[[^\]]*\]\([^)]+\)', '', markdown)   # ^[...](...)
    markdown = re.sub(r'\^\[[^\]]*\]', '', markdown)            # ^[...]

    # 2) Remove link-only citation markers like [[1]](#cite_note-...)
    markdown = re.sub(
        r'\[\s*\[[0-9a-zA-Z]+\]\s*\]\(#cite_note-[^)]+\)',
        '',
        markdown,
    )

    # 3) Fallback: bare [1], [2][3], [a]
    markdown = re.sub(r'\[\s*(\d+|[a-zA-Z])\s*\]', '', markdown)
    markdown = re.sub(r'(?:\s*\[\s*(\d+|[a-zA-Z])\s*\])+', '', markdown)

    # 4) [citation needed]
    markdown = re.sub(r'\[\s*citation needed\s*\]', '', markdown, flags=re.IGNORECASE)

    # 5) Tidy whitespace
    markdown = re.sub(r'\s{2,}', ' ', markdown)
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)

    return markdown



def save_wikipedia_to_md(url: str, output_path: str | None = None) -> None:
    title = url_to_title(url)
    decoded_title = unquote(title)
    if output_path is None:
        output_path = f"{decoded_title}.md"

    html = fetch_article_html(title)
    cleaned_html = clean_html(html)
    md = html_to_markdown(cleaned_html)
    md = strip_citation_markers(md)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Saved Markdown to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save a Wikipedia page as relatively clean Markdown (keeps tables, strips inline citation markers)."
    )
    parser.add_argument("url", help="Wikipedia page URL")
    parser.add_argument(
        "-o", "--output",
        help="Output Markdown file path (default: derived from page title)",
    )
    args = parser.parse_args()
    save_wikipedia_to_md(args.url, args.output)


if __name__ == "__main__":
    main()
