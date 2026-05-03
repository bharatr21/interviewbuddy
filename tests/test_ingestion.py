from pathlib import Path

from interviewbuddy.corpus import JsonlCorpusStore
from interviewbuddy.ingestion import IngestionService
from interviewbuddy.providers import ScrapeRequest, ScrapedDocument
from interviewbuddy.sources import Source


class FakeScrapeProvider:
    def __init__(self) -> None:
        self.scrape_requests: list[ScrapeRequest] = []
        self.discover_calls: list[dict] = []

    def discover(self, source_url: str, search: str | None = None, limit: int = 5000) -> list[str]:
        self.discover_calls.append({"source_url": source_url, "search": search, "limit": limit})
        return [
            "https://careersatdoordash.com/engineering-blog/reliability",
            "https://careersatdoordash.com/engineering-blog/dispatch",
        ][:limit]

    def scrape(self, request: ScrapeRequest) -> ScrapedDocument:
        self.scrape_requests.append(request)
        return ScrapedDocument(
            company=request.company,
            source_url=request.url,
            title=f"{request.company} article",
            text=f"{request.company} article text for {request.url}",
        )


def test_ingestion_discovers_scrapes_and_persists_documents(tmp_path: Path):
    provider = FakeScrapeProvider()
    store = JsonlCorpusStore(tmp_path / "corpus.jsonl")
    service = IngestionService(provider=provider, corpus_store=store)
    source = Source("doordash", "DoorDash", "https://careersatdoordash.com/engineering-blog/")

    summary = service.ingest_source(source, limit=2)

    documents = store.load()
    assert summary.discovered_count == 2
    assert summary.ingested_count == 2
    assert provider.discover_calls == [
        {
            "source_url": "https://careersatdoordash.com/engineering-blog/",
            "search": None,
            "limit": 2,
        }
    ]
    assert [request.company for request in provider.scrape_requests] == ["DoorDash", "DoorDash"]
    assert {document.source_url for document in documents} == {
        "https://careersatdoordash.com/engineering-blog/reliability",
        "https://careersatdoordash.com/engineering-blog/dispatch",
    }


def test_ingestion_defaults_to_full_sitemap_discovery_limit(tmp_path: Path):
    provider = FakeScrapeProvider()
    store = JsonlCorpusStore(tmp_path / "corpus.jsonl")
    service = IngestionService(provider=provider, corpus_store=store)
    source = Source("doordash", "DoorDash", "https://careersatdoordash.com/engineering-blog/")

    service.ingest_source(source)

    assert provider.discover_calls[0]["search"] is None
    assert provider.discover_calls[0]["limit"] == 5000


def test_ingestion_indexes_documents_when_vector_store_is_provided(tmp_path: Path):
    class FakeVectorStore:
        def __init__(self) -> None:
            self.documents = []

        def add_documents(self, documents):
            self.documents.extend(documents)

    provider = FakeScrapeProvider()
    vector_store = FakeVectorStore()
    service = IngestionService(
        provider=provider,
        corpus_store=JsonlCorpusStore(tmp_path / "corpus.jsonl"),
        vector_store=vector_store,
    )

    service.ingest_source(Source("doordash", "DoorDash", "https://careersatdoordash.com/engineering-blog/"), limit=1)

    assert len(vector_store.documents) == 1
    assert vector_store.documents[0].company == "DoorDash"
