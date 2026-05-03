from interviewbuddy.providers import FirecrawlScrapeProvider
from interviewbuddy.settings import Settings, get_settings


def test_settings_reads_firecrawl_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-env-key")

    settings = Settings()

    assert settings.firecrawl_api_key == "fc-env-key"


def test_settings_builds_firecrawl_provider_from_configured_secret(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-env-key")

    provider = Settings().firecrawl_provider()

    assert isinstance(provider, FirecrawlScrapeProvider)


def test_get_settings_returns_cached_settings_instance(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-env-key")

    first = get_settings()
    second = get_settings()

    assert first is second


def test_settings_exposes_database_url_default():
    settings = Settings()

    assert settings.database_url.startswith("sqlite:///")
