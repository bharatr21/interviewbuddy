from pathlib import Path

from interviewbuddy.corpus import JsonlCorpusStore
from interviewbuddy.documents import Document
from interviewbuddy.embeddings import HashEmbeddingProvider
from interviewbuddy.providers import ScrapeRequest, ScrapedDocument
from interviewbuddy.query_crawl_agent import CrawlBudget, QueryCrawlAgent, QueryCrawlRequest
from interviewbuddy.sources import Source
from interviewbuddy.vector_store import SqlVectorStore


class FakeQueryProvider:
    def __init__(self) -> None:
        self.search_calls = []
        self.map_calls = []
        self.scrape_calls = []

    def search(self, query: str, limit: int = 10):
        self.search_calls.append({"query": query, "limit": limit})
        return [
            {"url": "https://careersatdoordash.com/engineering-blog/reliability"},
            {"url": "https://careersatdoordash.com/engineering-blog/unrelated"},
        ]

    def discover(self, source_url: str, search: str | None = None, limit: int = 5000):
        self.map_calls.append({"source_url": source_url, "search": search, "limit": limit})
        return ["https://careersatdoordash.com/engineering-blog/reliability"]

    def scrape(self, request: ScrapeRequest) -> ScrapedDocument:
        self.scrape_calls.append(request.url)
        if request.url.endswith("/unrelated"):
            return ScrapedDocument(
                company=request.company,
                source_url=request.url,
                title="DoorDash Careers",
                text="DoorDash has offices and jobs.",
            )
        return ScrapedDocument(
            company=request.company,
            source_url=request.url,
            title="DoorDash Reliability",
            text="DoorDash dispatch reliability uses retries, routing, and observability for marketplace systems.",
        )


def _agent(tmp_path: Path, provider: FakeQueryProvider, seed_documents=None) -> QueryCrawlAgent:
    corpus_path = tmp_path / "corpus.jsonl"
    if seed_documents:
        JsonlCorpusStore(corpus_path).upsert_many(seed_documents)
    vector_store = SqlVectorStore(
        database_url=f"sqlite:///{tmp_path / 'vectors.db'}",
        embedding_provider=HashEmbeddingProvider(dimensions=32),
        dimensions=32,
    )
    if seed_documents:
        vector_store.add_documents(seed_documents)
    return QueryCrawlAgent(
        source=Source("doordash", "DoorDash", "https://careersatdoordash.com/engineering-blog/"),
        provider=provider,
        corpus_store=JsonlCorpusStore(corpus_path),
        vector_store=vector_store,
        embedding_provider=HashEmbeddingProvider(dimensions=32),
    )


def test_query_crawl_agent_answers_without_crawling_when_evidence_is_enough(tmp_path: Path):
    provider = FakeQueryProvider()
    agent = _agent(
        tmp_path,
        provider,
        seed_documents=[
            Document(
                company="DoorDash",
                source_url="https://careersatdoordash.com/engineering-blog/reliability",
                title="Existing DoorDash Reliability",
                text="DoorDash dispatch reliability uses retries, routing, and observability.",
            )
        ],
    )

    result = agent.run(QueryCrawlRequest(question="DoorDash dispatch reliability", company="DoorDash"))

    assert result.crawl_report.used_existing_evidence is True
    assert result.crawl_report.accepted_count == 0
    assert provider.search_calls == []
    assert result.answer.citations[0].title == "Existing DoorDash Reliability"


def test_query_crawl_agent_crawls_when_evidence_is_weak_and_indexes_accepted_docs(tmp_path: Path):
    provider = FakeQueryProvider()
    agent = _agent(tmp_path, provider)

    result = agent.run(
        QueryCrawlRequest(
            question="How does DoorDash handle dispatch reliability?",
            company="DoorDash",
            budget=CrawlBudget(max_provider_calls=5, max_discovered_urls=5, max_scraped_urls=2),
        )
    )

    assert result.crawl_report.used_existing_evidence is False
    assert result.crawl_report.accepted_count == 1
    assert result.crawl_report.skipped_count == 1
    assert provider.search_calls
    assert result.answer.citations[0].title == "DoorDash Reliability"

    second = agent.run(QueryCrawlRequest(question="DoorDash dispatch reliability", company="DoorDash"))
    assert second.crawl_report.used_existing_evidence is True


def test_query_crawl_agent_respects_scrape_budget(tmp_path: Path):
    provider = FakeQueryProvider()
    agent = _agent(tmp_path, provider)

    result = agent.run(
        QueryCrawlRequest(
            question="DoorDash dispatch reliability",
            company="DoorDash",
            budget=CrawlBudget(max_provider_calls=2, max_discovered_urls=5, max_scraped_urls=1),
        )
    )

    assert len(provider.scrape_calls) == 1
    assert result.crawl_report.budget_exhausted is True


def test_query_crawl_agent_falls_back_to_map_when_search_returns_no_candidates(tmp_path: Path):
    class MapOnlyProvider(FakeQueryProvider):
        def search(self, query: str, limit: int = 10):
            self.search_calls.append({"query": query, "limit": limit})
            return []

    provider = MapOnlyProvider()
    agent = _agent(tmp_path, provider)

    result = agent.run(QueryCrawlRequest(question="DoorDash dispatch reliability", company="DoorDash"))

    assert provider.map_calls
    assert result.crawl_report.fallback_count == 1
    assert result.answer.citations
