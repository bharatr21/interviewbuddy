from __future__ import annotations

from dataclasses import dataclass

from interviewbuddy.documents import Chunk, Document, chunk_document
from typing import Protocol

from interviewbuddy.embeddings import EmbeddingProvider, cosine_similarity


@dataclass(frozen=True)
class Citation:
    title: str
    company: str
    url: str
    snippet: str
    score: float
    published_at: str | None = None


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    citation: Citation


@dataclass(frozen=True)
class CoachAnswer:
    message: str
    citations: list[Citation]


class SearchStore(Protocol):
    def search(self, query: str, limit: int = 5, company: str | None = None) -> list[SearchResult]:
        """Search for source-backed chunks."""


class StaticSearchStore:
    def __init__(self, citations: list[Citation]) -> None:
        self._citations = citations

    def search(self, query: str, limit: int = 5, company: str | None = None) -> list[SearchResult]:
        filtered = [
            citation
            for citation in self._citations
            if company is None or citation.company.lower() == company.lower()
        ]
        return [
            SearchResult(
                chunk=Chunk(
                    id=str(index),
                    company=citation.company,
                    source_url=citation.url,
                    title=citation.title,
                    text=citation.snippet,
                    index=index,
                    published_at=citation.published_at,
                ),
                citation=citation,
            )
            for index, citation in enumerate(filtered[:limit])
        ]


class InMemoryVectorStore:
    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self._embedding_provider = embedding_provider
        self._records: list[tuple[Chunk, list[float]]] = []

    def add_documents(self, documents: list[Document]) -> None:
        for document in documents:
            for chunk in chunk_document(document):
                self._records.append((chunk, self._embedding_provider.embed(chunk.text)))

    def search(self, query: str, limit: int = 5, company: str | None = None) -> list[SearchResult]:
        query_vector = self._embedding_provider.embed(query)
        company_filter = company.lower() if company else None
        scored: list[tuple[float, Chunk]] = []

        for chunk, vector in self._records:
            if company_filter and chunk.company.lower() != company_filter:
                continue
            scored.append((cosine_similarity(query_vector, vector), chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._to_result(chunk, score) for score, chunk in scored[:limit] if score > 0]

    def _to_result(self, chunk: Chunk, score: float) -> SearchResult:
        return SearchResult(
            chunk=chunk,
            citation=Citation(
                title=chunk.title,
                company=chunk.company,
                url=chunk.source_url,
                snippet=chunk.text,
                score=score,
                published_at=chunk.published_at,
            ),
        )


class GroundedCoach:
    def __init__(self, vector_store: SearchStore, chat_provider=None) -> None:
        self._vector_store = vector_store
        self._chat_provider = chat_provider

    def answer(self, question: str, limit: int = 4, company: str | None = None) -> CoachAnswer:
        results = self._vector_store.search(question, limit=limit, company=company)
        if not results:
            return CoachAnswer(
                message=(
                    "I could not find a strong grounded source for that question. "
                    "Try adding a company name, system design topic, or more specific architecture keyword."
                ),
                citations=[],
            )

        citations = [result.citation for result in results]
        if self._chat_provider:
            return CoachAnswer(
                message=self._chat_provider.answer(question, citations, company=company),
                citations=citations,
            )
        context_lines = "\n".join(
            f"- {citation.company}: {citation.snippet}" for citation in citations
        )
        message = (
            "Grounded answer:\n"
            f"{context_lines}\n\n"
            "Interview framing:\n"
            "Use these sources to discuss concrete architecture tradeoffs, reliability constraints, "
            "scale pressure, and operational lessons. Tie your answer back to the company context "
            "instead of giving a generic system design response."
        )
        return CoachAnswer(message=message, citations=citations)
