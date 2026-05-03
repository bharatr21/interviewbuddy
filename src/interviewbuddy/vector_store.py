from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, delete, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from interviewbuddy.documents import Chunk, Document, chunk_document
from interviewbuddy.embeddings import EmbeddingProvider, cosine_similarity
from interviewbuddy.rag import Citation, SearchResult


class VectorBase(DeclarativeBase):
    pass


class ChunkRow(VectorBase):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    company: Mapped[str] = mapped_column(String, index=True)
    source_url: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[str] = mapped_column(String)
    text: Mapped[str] = mapped_column(Text)
    chunk_index: Mapped[int] = mapped_column(Integer)
    published_at: Mapped[str | None] = mapped_column(String, nullable=True)
    embedding_json: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class SqlVectorStore:
    """Persistent chunk vector store.

    SQLite stores vectors as JSON and ranks in Python for local development.
    Postgres deployments should use `postgres_schema_sql()` with pgvector and
    can evolve this class to issue native pgvector distance queries against the
    same chunk schema.
    """

    def __init__(
        self,
        database_url: str,
        embedding_provider: EmbeddingProvider,
        dimensions: int,
    ) -> None:
        self.database_url = database_url
        self._embedding_provider = embedding_provider
        self._dimensions = dimensions
        self._ensure_sqlite_parent()
        self._engine = create_engine(database_url)

    def init_schema(self) -> None:
        if self.database_url.startswith("postgres"):
            with self._engine.begin() as connection:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        VectorBase.metadata.create_all(self._engine)

    def add_documents(self, documents: list[Document]) -> None:
        self.init_schema()
        now = datetime.now(UTC)
        with Session(self._engine) as session:
            for document in documents:
                session.execute(delete(ChunkRow).where(ChunkRow.source_url == document.source_url))
                for chunk in chunk_document(document):
                    embedding = self._embedding_provider.embed(chunk.text)
                    session.add(
                        ChunkRow(
                            id=chunk.id,
                            company=chunk.company,
                            source_url=chunk.source_url,
                            title=chunk.title,
                            text=chunk.text,
                            chunk_index=chunk.index,
                            published_at=chunk.published_at,
                            embedding_json=json.dumps(embedding),
                            updated_at=now,
                        )
                    )
            session.commit()

    def search(self, query: str, limit: int = 5, company: str | None = None) -> list[SearchResult]:
        self.init_schema()
        query_embedding = self._embedding_provider.embed(query)
        with Session(self._engine) as session:
            statement = select(ChunkRow)
            if company:
                statement = statement.where(ChunkRow.company == company)
            rows = session.scalars(statement).all()

        scored: list[tuple[float, ChunkRow]] = []
        for row in rows:
            score = cosine_similarity(query_embedding, json.loads(row.embedding_json))
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._to_result(row, score) for score, row in scored[:limit]]

    def count_chunks(self) -> int:
        self.init_schema()
        with Session(self._engine) as session:
            return len(session.scalars(select(ChunkRow.id)).all())

    @staticmethod
    def postgres_schema_sql(dimensions: int) -> str:
        return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id text PRIMARY KEY,
    company text NOT NULL,
    source_url text NOT NULL,
    title text NOT NULL,
    text text NOT NULL,
    chunk_index integer NOT NULL,
    published_at text,
    embedding vector({dimensions}) NOT NULL,
    updated_at timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_embedding_ivfflat_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops);
""".strip()

    def _to_result(self, row: ChunkRow, score: float) -> SearchResult:
        chunk = Chunk(
            id=row.id,
            company=row.company,
            source_url=row.source_url,
            title=row.title,
            text=row.text,
            index=row.chunk_index,
            published_at=row.published_at,
        )
        return SearchResult(
            chunk=chunk,
            citation=Citation(
                title=row.title,
                company=row.company,
                url=row.source_url,
                snippet=row.text,
                score=score,
                published_at=row.published_at,
            ),
        )

    def _ensure_sqlite_parent(self) -> None:
        if not self.database_url.startswith("sqlite:///"):
            return
        sqlite_path = self.database_url.removeprefix("sqlite:///")
        if sqlite_path == ":memory:":
            return
        Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
