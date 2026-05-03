from typer.testing import CliRunner

from interviewbuddy.cli import app
from interviewbuddy.jobs import DEFAULT_JOB_STORE


def test_cli_sources_lists_doordash():
    result = CliRunner().invoke(app, ["sources"])

    assert result.exit_code == 0
    assert "DoorDash" in result.stdout
    assert "https://careersatdoordash.com/engineering-blog/" in result.stdout


def test_cli_ask_returns_citations_from_sample_corpus():
    result = CliRunner().invoke(app, ["ask", "DoorDash dispatch reliability"])

    assert result.exit_code == 0
    assert "Personalized interview coach plan" in result.stdout
    assert "Follow-up drills" in result.stdout
    assert "Dispatch Reliability" in result.stdout


def test_cli_ask_uses_local_corpus_file(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        '{"company":"DoorDash","source_url":"https://example.com/dd","title":"DoorDash Local","text":"DoorDash local corpus reliability notes.","published_at":null}\n',
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["ask", "DoorDash reliability", "--corpus", str(corpus_path), "--company", "DoorDash"],
    )

    assert result.exit_code == 0
    assert "DoorDash Local" in result.stdout


def test_cli_ask_auto_crawl_invokes_query_crawl_agent(monkeypatch, tmp_path):
    class FakeResult:
        def __init__(self):
            self.answer = type("Answer", (), {"message": "auto-crawl answer", "citations": []})()
            self.crawl_report = type(
                "Report",
                (),
                {
                    "used_existing_evidence": False,
                    "accepted_count": 1,
                    "skipped_count": 2,
                    "failed_count": 0,
                    "provider_call_count": 3,
                    "budget_exhausted": False,
                },
            )()

    class FakeQueryCrawlAgent:
        def __init__(self, **kwargs):
            pass

        def run(self, request):
            return FakeResult()

    monkeypatch.setattr("interviewbuddy.cli.QueryCrawlAgent", FakeQueryCrawlAgent)
    monkeypatch.setattr(
        "interviewbuddy.cli.get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "scrape_provider": lambda self: object(),
                "vector_store": lambda self: object(),
                "embedding_provider": lambda self: object(),
                "chat_provider": lambda self: None,
            },
        )(),
    )

    result = CliRunner().invoke(
        app,
        [
            "ask",
            "DoorDash reliability",
            "--company",
            "DoorDash",
            "--auto-crawl",
            "--corpus",
            str(tmp_path / "corpus.jsonl"),
        ],
    )

    assert result.exit_code == 0
    assert "auto-crawl answer" in result.stdout
    assert "Auto-crawl report" in result.stdout
    assert "accepted=1 skipped=2 failed=0 provider_calls=3" in result.stdout


def test_cli_monitoring_reports_local_corpus_counts(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        '{"company":"DoorDash","source_url":"https://example.com/dd","title":"DoorDash Local","text":"DoorDash local corpus reliability notes.","published_at":null}\n',
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["monitoring", "--corpus", str(corpus_path)])

    assert result.exit_code == 0
    assert "Documents: 1" in result.stdout
    assert "DoorDash: 1" in result.stdout


def test_cli_jobs_lists_recent_jobs(monkeypatch):
    DEFAULT_JOB_STORE._jobs.clear()
    job = DEFAULT_JOB_STORE.start("doordash", "DoorDash")
    DEFAULT_JOB_STORE.update(job)

    result = CliRunner().invoke(app, ["jobs"])

    assert result.exit_code == 0
    assert "doordash" in result.stdout
    assert "running" in result.stdout
