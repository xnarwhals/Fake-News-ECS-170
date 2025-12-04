"""
Minimal URL fetcher to pull article text for the UI demo.
"""

from html import unescape
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup  # type: ignore


def fetch_html(url: str, user_agent: str = "Mozilla/5.0") -> str:
    """Fetch raw HTML for a URL."""
    try:
        req = Request(url, headers={"User-Agent": user_agent})
        with urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except URLError as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc


def extract_article_text(html: str) -> str:
    """Extract visible text from common article tags."""
    soup = BeautifulSoup(html, "html.parser")
    texts = []
    for tag in soup.find_all(["p", "h1", "h2", "h3", "li"]):
        txt = tag.get_text(separator=" ", strip=True)
        if txt:
            texts.append(txt)
    return unescape(" ".join(texts))


def fetch_article(url: str) -> str:
    """Convenience wrapper to go from URL to plain text."""
    html = fetch_html(url)
    return extract_article_text(html)
