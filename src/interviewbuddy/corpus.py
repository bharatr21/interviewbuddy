from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from interviewbuddy.documents import Document


class JsonlCorpusStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def load(self) -> list[Document]:
        if not self.path.exists():
            return []
        documents: list[Document] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            documents.append(Document(**payload))
        return documents

    def upsert_many(self, documents: list[Document]) -> None:
        by_url = {document.source_url: document for document in self.load()}
        for document in documents:
            by_url[document.source_url] = document

        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(asdict(document), sort_keys=True) for document in by_url.values()]
        self.path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
