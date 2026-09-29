"""Web search and page fetch. No key required."""

from __future__ import annotations

import ipaddress
import re
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from rift.tools import ToolResult
from rift.util import clip

_USER_AGENT = "rift"
_RESULT = re.compile(
    r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</(?:a|td|div|span)>',
    re.S,
)
_TAG = re.compile(r"<[^>]+>")


def web_search(query: str) -> ToolResult:
    try:
        response = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": _USER_AGENT},
            timeout=20.0,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        return ToolResult(False, "search failed", str(error))
    hits = parse_search_results(response.text)
    if not hits:
        return ToolResult(False, "no results", query)
    lines = [f"{title}\n{url}\n{snippet}".strip() for title, url, snippet in hits[:8]]
    return ToolResult(True, f"{len(lines)} results for {query}", "\n\n".join(lines))


def web_fetch(url: str) -> ToolResult:
    reason = reject_url(url)
    if reason:
        return ToolResult(False, "url rejected", reason)
    try:
        response = httpx.get(
            url,
            headers={"User-Agent": _USER_AGENT},
            timeout=20.0,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        return ToolResult(False, "fetch failed", str(error))
    final = str(response.url)
    reason = reject_url(final)
    if reason:
        return ToolResult(False, "url rejected", reason)
    text = html_to_text(response.text)
    if not text:
        return ToolResult(False, "empty page", url)
    return ToolResult(True, clip(final, 180), clip(text, 12000))


def parse_search_results(html: str) -> list[tuple[str, str, str]]:
    hits: list[tuple[str, str, str]] = []
    for href, title_html, snippet_html in _RESULT.findall(html):
        url = _result_url(href)
        title = _TAG.sub("", title_html)
        title = re.sub(r"\s+", " ", title).strip()
        snippet = _TAG.sub("", snippet_html)
        snippet = re.sub(r"\s+", " ", snippet).strip()
        if url and title:
            hits.append((title, url, snippet))
    return hits


def reject_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "only http and https pages can be fetched"
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost":
        return ""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return ""
    if address.is_loopback:
        return ""
    if address.is_private or address.is_link_local or address.is_reserved or address.is_multicast or address.is_unspecified:
        return "that address is not a public page"
    return ""


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    text = re.sub(r"\n{3,}", "\n\n", "".join(parser.parts))
    return re.sub(r"[ \t]+", " ", text).strip()


def _result_url(href: str) -> str:
    href = href.replace("&amp;", "&")
    parsed = urlparse(href)
    if parsed.path.startswith("/l/") or "uddg" in (parsed.query or ""):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    if href.startswith("//"):
        return "https:" + href
    return href


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip += 1
        if tag in {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)
