from __future__ import annotations

from pathlib import Path

import typer

from interviewbuddy.corpus import JsonlCorpusStore
from interviewbuddy.ingestion import IngestionService
from interviewbuddy.jobs import DEFAULT_JOB_STORE
from interviewbuddy.paths import DEFAULT_CORPUS_PATH
from interviewbuddy.providers import FIRECRAWL_DEFAULT_MAP_LIMIT
from interviewbuddy.query_crawl_agent import CrawlBudget, QueryCrawlAgent, QueryCrawlRequest
from interviewbuddy.service import build_corpus_agent
from interviewbuddy.settings import get_settings
from interviewbuddy.sources import DEFAULT_SOURCES, find_source, get_source


app = typer.Typer(help="Interview Buddy crawler and RAG utilities.")


def _should_use_vector_store(corpus: Path) -> bool:
    return corpus == DEFAULT_CORPUS_PATH and corpus.exists()


@app.command()
def sources() -> None:
    """List configured company sources."""
    for source in DEFAULT_SOURCES:
        status = "enabled" if source.enabled else "disabled"
        typer.echo(f"{source.company}\t{source.kind}\t{status}\t{source.url}")


@app.command()
def ingest(
    source_slug: str,
    limit: int = typer.Option(
        FIRECRAWL_DEFAULT_MAP_LIMIT,
        min=1,
        max=100000,
        help="Maximum sitemap/map URLs to discover and scrape.",
    ),
    corpus: Path = typer.Option(DEFAULT_CORPUS_PATH, help="Local JSONL corpus path."),
) -> None:
    """Ingest one configured source into the local corpus with Firecrawl."""
    source = get_source(source_slug)
    settings = get_settings()
    service = IngestionService(
        provider=settings.scrape_provider(),
        corpus_store=JsonlCorpusStore(corpus),
        vector_store=settings.vector_store() if corpus == DEFAULT_CORPUS_PATH else None,
    )
    summary = service.ingest_source(source, limit=limit)
    typer.echo(
        f"Ingested {summary.ingested_count}/{summary.candidate_count} candidate documents "
        f"for {summary.company} into {corpus} "
        f"(job {summary.job_id}, discovered {summary.discovered_count}, failed {summary.failed_count})"
    )


@app.command("ingest-all")
def ingest_all(
    limit: int = typer.Option(
        FIRECRAWL_DEFAULT_MAP_LIMIT,
        min=1,
        max=100000,
        help="Maximum sitemap/map URLs to discover and scrape per source.",
    ),
    corpus: Path = typer.Option(DEFAULT_CORPUS_PATH, help="Local JSONL corpus path."),
) -> None:
    """Ingest every enabled configured source."""
    settings = get_settings()
    service = IngestionService(
        provider=settings.scrape_provider(),
        corpus_store=JsonlCorpusStore(corpus),
        vector_store=settings.vector_store() if corpus == DEFAULT_CORPUS_PATH else None,
    )
    for source in DEFAULT_SOURCES:
        if not source.enabled:
            continue
        summary = service.ingest_source(source, limit=limit)
        typer.echo(
            f"{summary.company}: {summary.ingested_count}/{summary.candidate_count} "
            f"ingested, {summary.failed_count} failed (job {summary.job_id})"
        )


@app.command()
def ask(
    question: str,
    company: str | None = None,
    limit: int = 4,
    corpus: Path = typer.Option(DEFAULT_CORPUS_PATH, help="Local JSONL corpus path."),
    auto_crawl: bool = typer.Option(
        False,
        "--auto-crawl/--no-auto-crawl",
        help="Crawl targeted public sources if existing evidence is weak.",
    ),
    max_provider_calls: int = typer.Option(8, min=1, help="Auto-crawl provider call budget."),
    max_discovered_urls: int = typer.Option(10, min=1, help="Auto-crawl discovered URL budget."),
    max_scraped_urls: int = typer.Option(4, min=1, help="Auto-crawl scrape budget."),
) -> None:
    """Ask the local RAG coach using the ingested corpus when available."""
    settings = get_settings()
    if auto_crawl:
        if not company:
            raise typer.BadParameter("--company is required when --auto-crawl is enabled")
        source = find_source(company)
        result = QueryCrawlAgent(
            source=source,
            provider=settings.scrape_provider(),
            corpus_store=JsonlCorpusStore(corpus),
            vector_store=settings.vector_store(),
            embedding_provider=settings.embedding_provider(),
            chat_provider=settings.chat_provider(),
        ).run(
            QueryCrawlRequest(
                question=question,
                company=source.company,
                limit=limit,
                budget=CrawlBudget(
                    max_provider_calls=max_provider_calls,
                    max_discovered_urls=max_discovered_urls,
                    max_scraped_urls=max_scraped_urls,
                ),
            )
        )
        _print_answer(result.answer)
        _print_crawl_report(result.crawl_report)
        return

    agent = build_corpus_agent(
        corpus,
        embedding_provider=settings.embedding_provider(),
        chat_provider=settings.chat_provider(),
        vector_store=settings.vector_store() if _should_use_vector_store(corpus) else None,
    )
    answer = agent.answer(question, company=company, limit=limit)
    _print_answer(answer)


def _print_answer(answer) -> None:
    typer.echo(answer.message)
    if answer.citations:
        typer.echo("\nCitations:")
        for citation in answer.citations:
            typer.echo(f"- {citation.title} ({citation.company}): {citation.url}")


def _print_crawl_report(report) -> None:
    typer.echo(
        "\nAuto-crawl report: "
        f"used_existing={report.used_existing_evidence} "
        f"accepted={report.accepted_count} skipped={report.skipped_count} "
        f"failed={report.failed_count} provider_calls={report.provider_call_count} "
        f"budget_exhausted={report.budget_exhausted}"
    )


@app.command()
def monitoring(corpus: Path = typer.Option(DEFAULT_CORPUS_PATH, help="Local JSONL corpus path.")) -> None:
    """Show local corpus and job counts."""
    documents = JsonlCorpusStore(corpus).load()
    typer.echo(f"Documents: {len(documents)}")
    counts: dict[str, int] = {}
    for document in documents:
        counts[document.company] = counts.get(document.company, 0) + 1
    for company, count in sorted(counts.items()):
        typer.echo(f"{company}: {count}")
    typer.echo(f"Jobs: {len(DEFAULT_JOB_STORE.list())}")


@app.command()
def jobs() -> None:
    """List recent in-process ingestion jobs."""
    for job in DEFAULT_JOB_STORE.list():
        typer.echo(
            f"{job.id}\t{job.source_slug}\t{job.company}\t{job.status.value}\t"
            f"{job.ingested_count}/{job.candidate_count}\tfailed={job.failed_count}"
        )
