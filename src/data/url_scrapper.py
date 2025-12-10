"""Small helper to fetch an article page and pull the main text."""

from html import unescape
from typing import Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup  # type: ignore

KEEP_TAGS = ["article", "main", "section", "div"]
TEXT_TAGS = ["p", "h1", "h2", "h3", "li"]
MIN_TOKEN_THRESHOLD = 30  # Drop tiny chunks that are likely boilerplate.


def fetch_html(url: str, user_agent: str = "Mozilla/5.0") -> str:
    """Fetch raw HTML for a URL."""
    try:
        req = Request(url, headers={"User-Agent": user_agent})
        with urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except URLError as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc


def _clean_text_chunks(chunks: Iterable[str]) -> str:
    cleaned = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk.split()) < 3:
            continue
        cleaned.append(chunk)
    return unescape(" ".join(cleaned))


def extract_article_text(html: str) -> str:
    """
    Extract visible text from likely content containers, preferring article/main/section.
    Falls back to common paragraph/header tags if nothing found.
    """
    soup = BeautifulSoup(html, "html.parser")

    def _collect_from_container(container):
        texts = []
        for tag in container.find_all(TEXT_TAGS):
            txt = tag.get_text(separator=" ", strip=True)
            if txt:
                texts.append(txt)
        return texts

    # Try structured containers first and keep the one with most content.
    candidates = soup.find_all(KEEP_TAGS)
    best_texts: list[str] = []
    max_tokens = 0
    for container in candidates:
        texts = _collect_from_container(container)
        token_count = sum(len(t.split()) for t in texts)
        if token_count < MIN_TOKEN_THRESHOLD:
            continue
        if token_count > max_tokens:
            max_tokens = token_count
            best_texts = texts
    if best_texts:
        return _clean_text_chunks(best_texts)

    # Fallback: grab all text tags
    texts = []
    for tag in soup.find_all(TEXT_TAGS):
        txt = tag.get_text(separator=" ", strip=True)
        if txt:
            texts.append(txt)
    return _clean_text_chunks(texts)


def fetch_article(url: str) -> str:
    """Convenience wrapper"""
    html = fetch_html(url)
    return extract_article_text(html)
