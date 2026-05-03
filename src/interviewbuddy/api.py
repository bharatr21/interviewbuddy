from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from interviewbuddy.agent import LangGraphInterviewAgent
from interviewbuddy.corpus import JsonlCorpusStore
from interviewbuddy.documents import Document
from interviewbuddy.ingestion import IngestionService
from interviewbuddy.jobs import InMemoryJobStore, IngestionJob
from interviewbuddy.paths import DEFAULT_CORPUS_PATH
from interviewbuddy.providers import FIRECRAWL_DEFAULT_MAP_LIMIT, ScrapeProvider
from interviewbuddy.query_crawl_agent import CrawlBudget, QueryCrawlAgent, QueryCrawlRequest
from interviewbuddy.service import build_interview_agent
from interviewbuddy.settings import get_settings
from interviewbuddy.sources import DEFAULT_SOURCES
from interviewbuddy.sources import find_source, get_source


class ChatRequest(BaseModel):
    question: str
    company: str | None = None
    limit: int = 4
    auto_crawl: bool = False
    max_provider_calls: int = 8
    max_discovered_urls: int = 10
    max_scraped_urls: int = 4


def create_app(
    seed_documents: list[Document] | None = None,
    corpus_path=DEFAULT_CORPUS_PATH,
    scrape_provider: ScrapeProvider | None = None,
    job_store: InMemoryJobStore | None = None,
    vector_store=None,
) -> FastAPI:
    app = FastAPI(title="Interview Buddy")
    jobs = job_store or InMemoryJobStore()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/sources")
    def sources() -> list[dict[str, object]]:
        return [
            {
                "slug": source.slug,
                "company": source.company,
                "url": source.url,
                "kind": source.kind,
                "enabled": source.enabled,
            }
            for source in DEFAULT_SOURCES
        ]

    @app.post("/chat")
    def chat(request: ChatRequest) -> dict[str, object]:
        settings = get_settings()
        if request.auto_crawl:
            if not request.company:
                raise HTTPException(status_code=400, detail="company is required when auto_crawl is true")
            try:
                source = find_source(request.company)
            except KeyError as error:
                raise HTTPException(status_code=404, detail=str(error)) from error
            provider = scrape_provider or settings.scrape_provider()
            active_vector_store = vector_store or settings.vector_store()
            result = QueryCrawlAgent(
                source=source,
                provider=provider,
                corpus_store=JsonlCorpusStore(corpus_path),
                vector_store=active_vector_store,
                embedding_provider=settings.embedding_provider(),
                chat_provider=settings.chat_provider(),
            ).run(
                QueryCrawlRequest(
                    question=request.question,
                    company=source.company,
                    limit=request.limit,
                    budget=CrawlBudget(
                        max_provider_calls=request.max_provider_calls,
                        max_discovered_urls=request.max_discovered_urls,
                        max_scraped_urls=request.max_scraped_urls,
                    ),
                )
            )
            return {
                "message": result.answer.message,
                "citations": _serialize_citations(result.answer.citations),
                "crawl_report": asdict(result.crawl_report),
            }

        documents = seed_documents if seed_documents is not None else JsonlCorpusStore(corpus_path).load()
        agent = build_interview_agent(
            documents or [],
            embedding_provider=settings.embedding_provider(),
            chat_provider=settings.chat_provider(),
        )
        active_vector_store = vector_store
        if active_vector_store is None and corpus_path == DEFAULT_CORPUS_PATH and DEFAULT_CORPUS_PATH.exists():
            active_vector_store = settings.vector_store()
        if seed_documents is None and active_vector_store is not None and active_vector_store.count_chunks() > 0:
            agent = LangGraphInterviewAgent(active_vector_store, chat_provider=settings.chat_provider())
        answer = agent.answer(request.question, limit=request.limit, company=request.company)
        return {
            "message": answer.message,
            "citations": _serialize_citations(answer.citations),
        }

    @app.post("/ingest/{source_slug}")
    def ingest(source_slug: str, limit: int = FIRECRAWL_DEFAULT_MAP_LIMIT) -> dict[str, object]:
        try:
            source = get_source(source_slug)
            settings = get_settings()
            provider = scrape_provider or settings.scrape_provider()
            active_vector_store = vector_store
            if active_vector_store is None and corpus_path == DEFAULT_CORPUS_PATH:
                active_vector_store = settings.vector_store()
            summary = IngestionService(
                provider=provider,
                corpus_store=JsonlCorpusStore(corpus_path),
                job_store=jobs,
                vector_store=active_vector_store,
            ).ingest_source(source, limit=limit)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "job_id": summary.job_id,
            "source_slug": summary.source_slug,
            "company": summary.company,
            "discovered_count": summary.discovered_count,
            "ingested_count": summary.ingested_count,
            "candidate_count": summary.candidate_count,
            "failed_count": summary.failed_count,
            "corpus_path": str(corpus_path),
        }

    @app.get("/jobs")
    def list_jobs() -> list[dict[str, object]]:
        return [_serialize_job(job) for job in jobs.list()]

    @app.get("/monitoring")
    def monitoring() -> dict[str, object]:
        documents = JsonlCorpusStore(corpus_path).load()
        job_list = jobs.list()
        return {
            "document_count": len(documents),
            "job_count": len(job_list),
            "failed_job_count": len([job for job in job_list if job.status == "failed"]),
            "partial_job_count": len([job for job in job_list if job.status == "partial"]),
            "companies": sorted({document.company for document in documents}),
        }

    return app


app = create_app()


def _serialize_job(job: IngestionJob) -> dict[str, object]:
    return {
        "id": job.id,
        "source_slug": job.source_slug,
        "company": job.company,
        "status": job.status.value,
        "discovered_count": job.discovered_count,
        "candidate_count": job.candidate_count,
        "ingested_count": job.ingested_count,
        "failed_count": job.failed_count,
        "errors": job.errors,
        "started_at": job.started_at.isoformat(),
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


def _serialize_citations(citations) -> list[dict[str, object]]:
    return [
        {
            "title": citation.title,
            "company": citation.company,
            "url": citation.url,
            "snippet": citation.snippet,
            "score": citation.score,
            "published_at": citation.published_at,
        }
        for citation in citations
    ]
