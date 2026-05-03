from __future__ import annotations

from typing import Protocol

from interviewbuddy.providers import ScrapeRequest, ScrapedDocument


WEBSITE_CONTENT_CRAWLER_ACTOR_ID = "apify/website-content-crawler"


class ApifyClientProtocol(Protocol):
    def actor(self, actor_id: str):
        """Return an Apify Actor client."""

    def dataset(self, dataset_id: str):
        """Return an Apify Dataset client."""


class ApifyWebsiteContentCrawlerProvider:
    """Inactive Apify adapter backed by `apify/website-content-crawler`.

    To use this instead of Firecrawl, change the provider factory in
    `interviewbuddy.settings.Settings.scrape_provider()`.
    """

    def __init__(
        self,
        client: ApifyClientProtocol,
        actor_id: str = WEBSITE_CONTENT_CRAWLER_ACTOR_ID,
    ) -> None:
        self._client = client
        self._actor_id = actor_id

    @classmethod
    def from_api_token(cls, api_token: str) -> "ApifyWebsiteContentCrawlerProvider":
        from apify_client import ApifyClient

        return cls(ApifyClient(api_token))

    def scrape(self, request: ScrapeRequest) -> ScrapedDocument:
        items = self._run_actor(request.url, max_pages=1)
        if not items:
            raise RuntimeError("Apify Website Content Crawler returned no dataset items")
        item = items[0]
        source_url = str(item.get("url") or request.url)
        return ScrapedDocument(
            company=request.company,
            source_url=source_url,
            title=_title(item, source_url),
            text=str(item.get("markdown") or item.get("text") or ""),
        )

    def discover(self, source_url: str, search: str | None = None, limit: int = 100) -> list[str]:
        items = self._run_actor(source_url, max_pages=limit)
        return [str(item["url"]) for item in items if item.get("url")]

    def _run_actor(self, start_url: str, max_pages: int) -> list[dict[str, object]]:
        run_input = {
            "startUrls": [{"url": start_url}],
            "maxCrawlPages": max_pages,
            "crawlerType": "playwright:adaptive",
            "removeCookieWarnings": True,
        }
        run = self._client.actor(self._actor_id).call(run_input=run_input)
        dataset_id = _dataset_id(run)
        if not dataset_id:
            raise RuntimeError("Apify Actor run did not return a default dataset ID")
        result = self._client.dataset(dataset_id).list_items()
        return [item for item in result.items if isinstance(item, dict)]


def _dataset_id(run: object) -> str | None:
    if isinstance(run, dict):
        value = run.get("defaultDatasetId") or run.get("default_dataset_id")
        return str(value) if value else None
    value = getattr(run, "default_dataset_id", None) or getattr(run, "defaultDatasetId", None)
    return str(value) if value else None


def _title(item: dict[str, object], fallback: str) -> str:
    metadata = item.get("metadata")
    if isinstance(metadata, dict) and metadata.get("title"):
        return str(metadata["title"])
    if item.get("title"):
        return str(item["title"])
    return fallback
