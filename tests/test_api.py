from fastapi.testclient import TestClient

from interviewbuddy.api import create_app
from interviewbuddy.documents import Document
from interviewbuddy.providers import ScrapeRequest, ScrapedDocument
from interviewbuddy.query_crawl_agent import CrawlReport
from interviewbuddy.rag import CoachAnswer


def test_api_lists_sources_and_includes_doordash():
    client = TestClient(create_app())

    response = client.get("/sources")

    assert response.status_code == 200
    companies = {source["company"] for source in response.json()}
    assert "DoorDash" in companies


def test_api_chat_returns_grounded_answer_with_citations():
    client = TestClient(
        create_app(
            seed_documents=[
                Document(
                    company="DoorDash",
                    source_url="https://careersatdoordash.com/engineering-blog/dispatch",
                    title="Dispatch Reliability",
                    text="DoorDash improves dispatch reliability with real-time routing, retries, and observability.",
                )
            ]
        )
    )

    response = client.post(
        "/chat",
        json={"question": "How should I discuss DoorDash dispatch reliability?", "company": "DoorDash"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "Personalized interview coach plan" in body["message"]
    assert "Follow-up drills" in body["message"]
    assert body["citations"][0]["company"] == "DoorDash"
    assert body["citations"][0]["url"] == "https://careersatdoordash.com/engineering-blog/dispatch"


def test_api_chat_can_load_documents_from_corpus_path(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        '{"company":"DoorDash","source_url":"https://example.com/dd","title":"DoorDash Local","text":"DoorDash local corpus reliability notes.","published_at":null}\n',
        encoding="utf-8",
    )
    client = TestClient(create_app(corpus_path=corpus_path))

    response = client.post(
        "/chat",
        json={"question": "DoorDash reliability", "company": "DoorDash"},
    )

    assert response.status_code == 200
    assert response.json()["citations"][0]["title"] == "DoorDash Local"


def test_api_chat_auto_crawl_returns_crawl_report(monkeypatch, tmp_path):
    class FakeResult:
        def __init__(self):
            self.answer = CoachAnswer(message="auto answer", citations=[])
            self.crawl_report = CrawlReport(
                used_existing_evidence=False,
                accepted_count=1,
                skipped_count=0,
                failed_count=0,
                provider_call_count=2,
            )

    class FakeQueryCrawlAgent:
        def __init__(self, **kwargs):
            pass

        def run(self, request):
            return FakeResult()

    monkeypatch.setattr("interviewbuddy.api.QueryCrawlAgent", FakeQueryCrawlAgent)
    client = TestClient(create_app(corpus_path=tmp_path / "corpus.jsonl", scrape_provider=FakeApiScrapeProvider()))

    response = client.post(
        "/chat",
        json={"question": "DoorDash reliability", "company": "DoorDash", "auto_crawl": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "auto answer"
    assert body["crawl_report"]["accepted_count"] == 1
    assert body["crawl_report"]["provider_call_count"] == 2


class FakeApiScrapeProvider:
    def discover(self, source_url: str, search: str | None = "engineering", limit: int = 100) -> list[str]:
        return ["https://careersatdoordash.com/engineering-blog/reliability"][:limit]

    def scrape(self, request: ScrapeRequest) -> ScrapedDocument:
        return ScrapedDocument(
            company=request.company,
            source_url=request.url,
            title="DoorDash API Ingest",
            text="DoorDash API ingestion reliability content.",
        )


def test_api_ingest_source_persists_documents_to_corpus_path(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    client = TestClient(create_app(corpus_path=corpus_path, scrape_provider=FakeApiScrapeProvider()))

    response = client.post("/ingest/doordash", params={"limit": 1})

    assert response.status_code == 200
    assert response.json()["ingested_count"] == 1
    assert "DoorDash API Ingest" in corpus_path.read_text(encoding="utf-8")


def test_api_chat_sees_documents_ingested_after_app_start(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    client = TestClient(create_app(corpus_path=corpus_path, scrape_provider=FakeApiScrapeProvider()))

    client.post("/ingest/doordash", params={"limit": 1})
    response = client.post(
        "/chat",
        json={"question": "DoorDash API ingestion reliability", "company": "DoorDash"},
    )

    assert response.status_code == 200
    assert response.json()["citations"][0]["title"] == "DoorDash API Ingest"


def test_api_returns_ingestion_jobs_after_ingest(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    client = TestClient(create_app(corpus_path=corpus_path, scrape_provider=FakeApiScrapeProvider()))

    ingest_response = client.post("/ingest/doordash", params={"limit": 1})
    jobs_response = client.get("/jobs")

    assert jobs_response.status_code == 200
    jobs = jobs_response.json()
    assert jobs[0]["id"] == ingest_response.json()["job_id"]
    assert jobs[0]["status"] == "completed"
    assert jobs[0]["ingested_count"] == 1


def test_api_returns_monitoring_summary(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    client = TestClient(create_app(corpus_path=corpus_path, scrape_provider=FakeApiScrapeProvider()))
    client.post("/ingest/doordash", params={"limit": 1})

    response = client.get("/monitoring")

    assert response.status_code == 200
    body = response.json()
    assert body["document_count"] == 1
    assert body["job_count"] == 1
