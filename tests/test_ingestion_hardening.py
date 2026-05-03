from pathlib import Path

from interviewbuddy.corpus import JsonlCorpusStore
from interviewbuddy.ingestion import IngestionService
from interviewbuddy.jobs import InMemoryJobStore, JobStatus
from interviewbuddy.providers import ScrapeRequest, ScrapedDocument
from interviewbuddy.sources import Source


class FlakyScrapeProvider:
    def __init__(self) -> None:
        self.scrape_attempts: dict[str, int] = {}

    def discover(self, source_url: str, search: str | None = None, limit: int = 5000) -> list[str]:
        return [
            "https://careersatdoordash.com/engineering-blog/reliability?utm_source=x",
            "https://careersatdoordash.com/engineering-blog/reliability",
            "https://external.example.com/not-allowed",
            "https://careersatdoordash.com/careers",
            "https://careersatdoordash.com/engineering-blog/dispatch",
        ]

    def scrape(self, request: ScrapeRequest) -> ScrapedDocument:
        attempts = self.scrape_attempts.get(request.url, 0) + 1
        self.scrape_attempts[request.url] = attempts
        if request.url.endswith("/dispatch") and attempts == 1:
            raise RuntimeError("temporary rate limit")
        return ScrapedDocument(
            company=request.company,
            source_url=request.url,
            title=request.url.rsplit("/", 1)[-1],
            text=f"content for {request.url}",
        )


def test_ingestion_filters_dedupes_retries_and_tracks_job_status(tmp_path: Path):
    provider = FlakyScrapeProvider()
    jobs = InMemoryJobStore()
    service = IngestionService(
        provider=provider,
        corpus_store=JsonlCorpusStore(tmp_path / "corpus.jsonl"),
        job_store=jobs,
        max_retries=2,
    )
    source = Source("doordash", "DoorDash", "https://careersatdoordash.com/engineering-blog/")

    summary = service.ingest_source(source)

    documents = JsonlCorpusStore(tmp_path / "corpus.jsonl").load()
    assert summary.discovered_count == 5
    assert summary.candidate_count == 2
    assert summary.ingested_count == 2
    assert summary.failed_count == 0
    assert [document.source_url for document in documents] == [
        "https://careersatdoordash.com/engineering-blog/reliability",
        "https://careersatdoordash.com/engineering-blog/dispatch",
    ]
    assert provider.scrape_attempts["https://careersatdoordash.com/engineering-blog/dispatch"] == 2
    job = jobs.get(summary.job_id)
    assert job is not None
    assert job.status == JobStatus.COMPLETED
    assert job.ingested_count == 2


def test_ingestion_records_partial_failures(tmp_path: Path):
    class FailingProvider(FlakyScrapeProvider):
        def scrape(self, request: ScrapeRequest) -> ScrapedDocument:
            raise RuntimeError("blocked")

    jobs = InMemoryJobStore()
    service = IngestionService(
        provider=FailingProvider(),
        corpus_store=JsonlCorpusStore(tmp_path / "corpus.jsonl"),
        job_store=jobs,
        max_retries=1,
    )

    summary = service.ingest_source(Source("doordash", "DoorDash", "https://careersatdoordash.com/engineering-blog/"))

    assert summary.ingested_count == 0
    assert summary.failed_count == 2
    assert jobs.get(summary.job_id).status == JobStatus.FAILED
    assert "blocked" in jobs.get(summary.job_id).errors[0]
