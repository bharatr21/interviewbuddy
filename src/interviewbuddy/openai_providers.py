from __future__ import annotations

from typing import Any


class OpenAIEmbeddingProvider:
    def __init__(self, client: Any, model: str = "text-embedding-3-small") -> None:
        self._client = client
        self._model = model

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self._model, input=text)
        return list(response.data[0].embedding)
