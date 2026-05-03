from __future__ import annotations

from interviewbuddy.documents import Document


SAMPLE_DOCUMENTS: list[Document] = [
    Document(
        company="DoorDash",
        source_url="https://careersatdoordash.com/engineering-blog/dispatch",
        title="Dispatch Reliability",
        text=(
            "DoorDash improves dispatch reliability with real-time routing, retries, "
            "observability, and careful marketplace balancing during peak demand."
        ),
    ),
    Document(
        company="Netflix",
        source_url="https://netflixtechblog.com/cache",
        title="Caching for Playback",
        text="Netflix uses caching and resilient delivery systems to protect playback quality at scale.",
    ),
]
