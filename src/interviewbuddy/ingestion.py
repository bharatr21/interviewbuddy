from __future__ import annotations

from dataclasses import dataclass

from interviewbuddy.corpus import JsonlCorpusStore
from interviewbuddy.documents import Document
from interviewbuddy.jobs import DEFAULT_JOB_STORE, InMemoryJobStore, JobStatus
from interviewbuddy.providers import FIRECRAWL_DEFAULT_MAP_LIMIT, ScrapeProvider, ScrapeRequest
from interviewbuddy.sources import Source
from interviewbuddy.url_filters import source_scoped_urls


@dataclass(frozen=True)
class IngestionSummary:
    job_id: str
    source_slug: str
    company: str
    discovered_count: int
    candidate_count: int
    ingested_count: int
    failed_count: int


class IngestionService:
    def __init__(
        self,
        provider: ScrapeProvider,
        corpus_store: JsonlCorpusStore,
        job_store: InMemoryJobStore = DEFAULT_JOB_STORE,
        max_retries: int = 2,
        vector_store=None,
    ) -> None:
        self._provider = provider
        self._corpus_store = corpus_store
        self._job_store = job_store
        self._max_retries = max_retries
        self._vector_store = vector_store

    def ingest_source(self, source: Source, limit: int = FIRECRAWL_DEFAULT_MAP_LIMIT) -> IngestionSummary:
        job = self._job_store.start(source.slug, source.company)
        urls = self._provider.discover(source.url, search=None, limit=limit)
        candidates = source_scoped_urls(source.url, urls)
        documents: list[Document] = []
        errors: list[str] = []

        for url in candidates:
            try:
                scraped = self._scrape_with_retries(ScrapeRequest(company=source.company, url=url))
                if scraped.text.strip():
                    documents.append(
                        Document(
                            company=scraped.company,
                            source_url=scraped.source_url,
                            title=scraped.title,
                            text=scraped.text,
                        )
                    )
            except Exception as error:  # noqa: BLE001 - record partial ingestion failures.
                errors.append(f"{url}: {error}")

        self._corpus_store.upsert_many(documents)
        if self._vector_store and documents:
            self._vector_store.add_documents(documents)
        job.discovered_count = len(urls)
        job.candidate_count = len(candidates)
        job.ingested_count = len(documents)
        job.failed_count = len(errors)
        job.errors = errors
        if errors and documents:
            job.status = JobStatus.PARTIAL
        elif errors:
            job.status = JobStatus.FAILED
        else:
            job.status = JobStatus.COMPLETED
        self._job_store.update(job)
        return IngestionSummary(
            job_id=job.id,
            source_slug=source.slug,
            company=source.company,
            discovered_count=len(urls),
            candidate_count=len(candidates),
            ingested_count=len(documents),
            failed_count=len(errors),
        )

    def _scrape_with_retries(self, request: ScrapeRequest):
        last_error: Exception | None = None
        for _attempt in range(self._max_retries + 1):
            try:
                return self._provider.scrape(request)
            except Exception as error:  # noqa: BLE001 - retry provider failures consistently.
                last_error = error
        raise RuntimeError(last_error) from last_error
