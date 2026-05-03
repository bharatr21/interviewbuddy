from __future__ import annotations

from typing import Protocol

from interviewbuddy.providers import ScrapeRequest, ScrapedDocument


class TavilyClientProtocol(Protocol):
    def extract(self, **kwargs) -> dict[str, object]:
        """Call Tavily Extract."""

    def map(self, **kwargs) -> dict[str, object]:
        """Call Tavily Map."""


class TavilyScrapeProvider:
    """Inactive Tavily adapter for the shared scraping interface.

    To use this instead of Firecrawl, change the provider factory in
    `interviewbuddy.settings.Settings.scrape_provider()`.
    """

    def __init__(self, client: TavilyClientProtocol) -> None:
        self._client = client

    @classmethod
    def from_api_key(cls, api_key: str) -> "TavilyScrapeProvider":
        from tavily import TavilyClient

        return cls(TavilyClient(api_key=api_key))

    def scrape(self, request: ScrapeRequest) -> ScrapedDocument:
        response = self._client.extract(
            urls=request.url,
            format="markdown",
            extract_depth="advanced",
            include_images=False,
        )
        result = _first_result(response)
        source_url = str(result.get("url") or request.url)
        return ScrapedDocument(
            company=request.company,
            source_url=source_url,
            title=str(result.get("title") or source_url),
            text=str(result.get("raw_content") or result.get("content") or ""),
        )

    def discover(self, source_url: str, search: str | None = "engineering", limit: int = 100) -> list[str]:
        response = self._client.map(
            url=source_url,
            query=search,
            limit=limit,
            allow_external=False,
        )
        results = response.get("results") or []
        return [url for url in results if isinstance(url, str)]


def _first_result(response: dict[str, object]) -> dict[str, object]:
    results = response.get("results") or []
    if not isinstance(results, list) or not results:
        failed = response.get("failed_results") or []
        raise RuntimeError(f"Tavily extract returned no results: {failed}")
    result = results[0]
    if not isinstance(result, dict):
        raise RuntimeError("Tavily extract returned an invalid result")
    return result
