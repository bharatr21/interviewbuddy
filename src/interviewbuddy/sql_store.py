from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import DateTime, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from interviewbuddy.documents import Document


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"

    source_url: Mapped[str] = mapped_column(String, primary_key=True)
    company: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[str] = mapped_column(String)
    text: Mapped[str] = mapped_column(Text)
    published_at: Mapped[str | None] = mapped_column(String, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class SqlCorpusStore:
    """SQL corpus persistence.

    Use SQLite for local development, or set `DATABASE_URL` to a Postgres URL
    for hosted deployments. pgvector-backed chunk storage can be layered behind
    the same corpus boundary when the hosted database is available.
    """

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._ensure_sqlite_parent()
        self._engine = create_engine(database_url)

    def init_schema(self) -> None:
        Base.metadata.create_all(self._engine)

    def load(self) -> list[Document]:
        self.init_schema()
        with Session(self._engine) as session:
            rows = session.scalars(select(DocumentRow).order_by(DocumentRow.company, DocumentRow.title)).all()
            return [
                Document(
                    company=row.company,
                    source_url=row.source_url,
                    title=row.title,
                    text=row.text,
                    published_at=row.published_at,
                )
                for row in rows
            ]

    def upsert_many(self, documents: list[Document]) -> None:
        self.init_schema()
        now = datetime.now(UTC)
        with Session(self._engine) as session:
            for document in documents:
                row = session.get(DocumentRow, document.source_url)
                if row is None:
                    row = DocumentRow(source_url=document.source_url)
                    session.add(row)
                row.company = document.company
                row.title = document.title
                row.text = document.text
                row.published_at = document.published_at
                row.updated_at = now
            session.commit()

    def count_by_company(self) -> dict[str, int]:
        self.init_schema()
        with Session(self._engine) as session:
            rows = session.execute(
                select(DocumentRow.company, func.count(DocumentRow.source_url)).group_by(DocumentRow.company)
            ).all()
            return {company: count for company, count in rows}

    def _ensure_sqlite_parent(self) -> None:
        if not self.database_url.startswith("sqlite:///"):
            return
        sqlite_path = self.database_url.removeprefix("sqlite:///")
        if sqlite_path == ":memory:":
            return
        Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
