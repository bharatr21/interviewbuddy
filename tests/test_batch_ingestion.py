from pathlib import Path

from typer.testing import CliRunner

from interviewbuddy.cli import app


def test_cli_ingest_all_invokes_each_configured_source(monkeypatch, tmp_path: Path):
    calls = []

    class FakeSummary:
        def __init__(self, source_slug: str, company: str) -> None:
            self.job_id = f"job-{source_slug}"
            self.source_slug = source_slug
            self.company = company
            self.discovered_count = 1
            self.candidate_count = 1
            self.ingested_count = 1
            self.failed_count = 0

    class FakeIngestionService:
        def __init__(self, provider, corpus_store, **kwargs):
            pass

        def ingest_source(self, source, limit):
            calls.append((source.slug, limit))
            return FakeSummary(source.slug, source.company)

    monkeypatch.setattr("interviewbuddy.cli.IngestionService", FakeIngestionService)
    monkeypatch.setattr(
        "interviewbuddy.cli.get_settings",
        lambda: type(
            "Settings",
            (),
            {"scrape_provider": lambda self: object(), "vector_store": lambda self: object()},
        )(),
    )

    result = CliRunner().invoke(app, ["ingest-all", "--limit", "1", "--corpus", str(tmp_path / "corpus.jsonl")])

    assert result.exit_code == 0
    assert len(calls) >= 16
    assert ("doordash", 1) in calls
