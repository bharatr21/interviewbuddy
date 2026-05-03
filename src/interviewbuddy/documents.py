from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    company: str
    source_url: str
    title: str
    text: str
    published_at: str | None = None


@dataclass(frozen=True)
class Chunk:
    id: str
    company: str
    source_url: str
    title: str
    text: str
    index: int
    published_at: str | None = None


def chunk_document(document: Document, max_chars: int = 900) -> list[Chunk]:
    if max_chars < 40:
        raise ValueError("max_chars must be at least 40")

    words = re.findall(r"\S+", document.text)
    chunks: list[Chunk] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join([*current, word])
        if current and len(candidate) > max_chars:
            chunks.append(_make_chunk(document, " ".join(current), len(chunks)))
            current = [word]
        else:
            current.append(word)

    if current:
        chunks.append(_make_chunk(document, " ".join(current), len(chunks)))

    return chunks


def _make_chunk(document: Document, text: str, index: int) -> Chunk:
    digest = hashlib.sha256(f"{document.source_url}:{index}:{text}".encode()).hexdigest()[:16]
    return Chunk(
        id=digest,
        company=document.company,
        source_url=document.source_url,
        title=document.title,
        text=text,
        index=index,
        published_at=document.published_at,
    )
