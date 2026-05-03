from __future__ import annotations

from urllib.parse import urlparse, urlunparse


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme, parsed.netloc.lower(), path, "", "", ""))


def source_scoped_urls(source_url: str, urls: list[str]) -> list[str]:
    source = urlparse(source_url)
    source_prefix = normalize_url(source_url).rstrip("/")
    seen: set[str] = set()
    candidates: list[str] = []

    for url in urls:
        normalized = normalize_url(url)
        parsed = urlparse(normalized)
        if parsed.netloc.lower() != source.netloc.lower():
            continue
        if not normalized.startswith(source_prefix):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(normalized)

    return candidates
