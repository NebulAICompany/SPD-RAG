#!/usr/bin/env python3
"""
wikipedia_to_md.py

Usage:
    python wikipedia_to_md.py "https://en.wikipedia.org/wiki/Computer_science"
    python wikipedia_to_md.py "https://en.wikipedia.org/wiki/Computer_science" -o cs.md
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


# ----------------- API & HTML KISMI ----------------- #

def url_to_title(url: str) -> str:
    """
    Extract page title from a Wikipedia URL.
    https://en.wikipedia.org/wiki/Computer_science -> Computer_science
    """
    path = urlparse(url).path
    slug = os.path.basename(path)
    return slug or "Main_Page"


def fetch_article_html(title: str) -> str:
    """
    Use MediaWiki API action=parse to get rendered HTML for the article.
    """
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
    """
    Aggressively clean Wikipedia HTML for RAG:
    - Keep headings, paragraphs, lists, links, tables, references
    - Drop styles, scripts, navboxes, sidebars, thumbnail layout vs.
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1) style / script / noscript
    for tag in soup(["style", "script", "noscript"]):
        tag.decompose()

    # 2) belirgin UI / layout elemanları
    junk_selectors = [
        "span.mw-editsection",
        "a.mw-jump-link",
        "table.mw-empty-elt",
        "table.navbox",
        "table.vertical-navbox",
        "table.sidebar",
        "table.toccolours",
        "div.hatnote",
        "div.metadata",
        "div.mw-references-wrap",  # referanslar zaten normal <ol>/<li> olarak da geliyor
    ]
    for sel in junk_selectors:
        for el in soup.select(sel):
            el.decompose()

    # 2b) navbox benzeri tabloları sınıfa göre ayıkla
    for tbl in soup.find_all("table"):
        if not hasattr(tbl, "attrs") or tbl.attrs is None:
            continue
        cls_attr = tbl.attrs.get("class", [])
        if isinstance(cls_attr, str):
            cls = cls_attr
        else:
            cls = " ".join(cls_attr)
        if any(key in cls for key in ["navbox-inner", "navbox-subgroup",
                                      "navbox", "vertical-navbox", "sidebar", "metadata"]):
            tbl.decompose()

    # 3) thumbnail / gallery bloklarını sadece caption’a indir
    for thumb in soup.select("div.thumb, div.tmulti, div.thumbinner, div.gallery"):
        caption = thumb.get_text(" ", strip=True)
        if caption:
            new_p = soup.new_tag("p")
            new_p.string = caption
            thumb.replace_with(new_p)
        else:
            thumb.decompose()

    # 4) parser-specific attribute’ları temizle
    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            if attr.startswith("data-mw") or attr in {"class", "style", "id"}:
                del tag.attrs[attr]

    # 5) tamamen boş kalan elementleri sil
    for tag in soup.find_all(True):
        if not tag.get_text(strip=True) and tag.name not in {"table", "tr", "td", "th"}:
            tag.decompose()

    return str(soup)


# ----------------- HTML -> MARKDOWN ----------------- #

def html_to_markdown(html: str) -> str:
    """
    Run pandoc and return Markdown as a string.
    """
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


# ----------------- MARKDOWN POST-PROCESSING ----------------- #

def strip_citation_markers(markdown: str) -> str:
    """
    Remove inline citation markers / footnotes while keeping the text.
    Targets patterns like:
      ^[[1]](#cite_note-...)
      [[1]](#cite_note-...)
      ](#cite_note-...)[[5]](#cite_note-5)
      [1], [2][3], [a], [citation needed]
    """

    # cite_note anchor’lı linkler
    markdown = re.sub(r'\^\[[^\]]*\]\(#cite_note-[^)]+\)', '', markdown)
    markdown = re.sub(r'\[\s*\[[0-9a-zA-Z]+\]\s*\]\(#cite_note-[^)]+\)', '', markdown)
    markdown = re.sub(r'\]\(#cite_note-[^)]+\)', '', markdown)

    # '^[' ile başlayan generic footnote
    markdown = re.sub(r'\^\[[^\]]*\]', '', markdown)

    # çıplak [1], [2][3], [a]
    markdown = re.sub(r'\[\s*(\d+|[a-zA-Z])\s*\]', '', markdown)
    markdown = re.sub(r'(?:\s*\[\s*(\d+|[a-zA-Z])\s*\])+', '', markdown)

    # [citation needed]
    markdown = re.sub(r'\[\s*citation needed\s*\]', '', markdown, flags=re.IGNORECASE)

    # boşluk / satır fazlalıklarını toparla
    markdown = re.sub(r'\s{2,}', ' ', markdown)
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)

    return markdown


def simplify_links(markdown: str) -> str:
    """
    Link ve URL benzeri desenleri düz metne çevirir:
      [foo](/wiki/Bar) -> foo
      (/wiki/List_of_...#Season_1_(2007) "…") -> Season 1 (2007)
      (/wiki/1000_BC "1000 BC") -> 1000 BC
    """

    # 1) Normal Markdown linkleri: [text](url "title") veya [text](url) -> text
    markdown = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', markdown)

    # 2) Boş metinli linkler: [](url "Title") -> Title
    markdown = re.sub(r'\[\]\([^)"]+"([^"]+)"\)', r'\1', markdown)

    # 3) Parantez içindeki /wiki URL’lerini dönüştür
    def _url_to_text(match: re.Match) -> str:
        inner = match.group(1)  # /wiki/... "..."
        # Title varsa önce onu dene
        m_title = re.search(r'"([^"]+)"', inner)
        # URL’de Season_x_(YYYY) varsa, sezona özel isim üret
        m_season = re.search(r'Season[_ ](\d+)_\((\d{4})', inner)

        if m_season:
            return f"Season {m_season.group(1)} ({m_season.group(2)})"
        if m_title:
            return m_title.group(1)

        # Son çare: path’in son parçası
        m_last = re.search(r'/([^/\s]+)$', inner.strip())
        return m_last.group(1) if m_last else inner

    markdown = re.sub(r'\((/wiki[^)]+)\)', lambda m: _url_to_text(m), markdown)

    return markdown


def simplify_tables(markdown: str) -> str:
    """
    ASCII tablo kenarlıklarını ve hizalama satırlarını temizler,
    hücre satırlarını ise normal markdown tablosu gibi bırakır.
    """
    lines = markdown.splitlines()
    out = []

    for line in lines:
        s = line.strip()

        # Sadece sembol içeren border / çizgi satırları (harf/rakam yok) -> at
        if s and not any(ch.isalnum() for ch in s):
            continue

        # Sadece layout çöpü
        if s in (':::', '::: {}', '{}'):
            continue

        # İçinde | olan satırı hücrelere böl ve sadeleştir
        if '|' in line:
            cells = [c.strip() for c in line.split('|')]
            cells = [c for c in cells if c and c not in (':::', '::: {}', '{}')]
            if cells:
                out.append('| ' + ' | '.join(cells) + ' |')
            continue

        out.append(line)

    return '\n'.join(out)


# ----------------- ANA AKIŞ ----------------- #

def save_wikipedia_to_md(url: str, output_path: str | None = None) -> None:
    title = url_to_title(url)
    decoded_title = unquote(title)
    if output_path is None:
        output_path = f"{decoded_title}.md"

    html = fetch_article_html(title)
    cleaned_html = clean_html(html)
    md = html_to_markdown(cleaned_html)
    md = strip_citation_markers(md)
    md = simplify_links(md)
    md = simplify_tables(md)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Saved Markdown to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Save a Wikipedia page as clean Markdown for RAG "
            "(keeps content tables, strips citation markers and URL clutter)."
        )
    )
    parser.add_argument("url", help="Wikipedia page URL")
    parser.add_argument(
        "-o",
        "--output",
        help="Output Markdown file path (default: derived from page title)",
    )
    args = parser.parse_args()
    save_wikipedia_to_md(args.url, args.output)


if __name__ == "__main__":
    main()
