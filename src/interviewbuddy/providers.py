from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests


FIRECRAWL_BASE_URL = "https://api.firecrawl.dev/v2"
FIRECRAWL_DEFAULT_MAP_LIMIT = 5000


@dataclass(frozen=True)
class ScrapeRequest:
    company: str
    url: str


@dataclass(frozen=True)
class ScrapedDocument:
    company: str
    source_url: str
    title: str
    text: str


class ScrapeProvider(Protocol):
    def discover(
        self,
        source_url: str,
        search: str | None = None,
        limit: int = FIRECRAWL_DEFAULT_MAP_LIMIT,
    ) -> list[str]:
        """Discover article URLs from a source URL."""

    def scrape(self, request: ScrapeRequest) -> ScrapedDocument:
        """Scrape a URL into normalized document text."""


class FirecrawlClient(Protocol):
    def scrape(self, url: str) -> dict[str, object]:
        """Return Firecrawl-style scrape data for a URL."""

    def map(
        self,
        url: str,
        search: str | None = None,
        limit: int = FIRECRAWL_DEFAULT_MAP_LIMIT,
    ) -> list[dict[str, str]]:
        """Return Firecrawl-style mapped links for a URL."""

    def search(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        """Return Firecrawl-style search results for a query."""


class FirecrawlError(RuntimeError):
    pass


class FirecrawlHttpClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = FIRECRAWL_BASE_URL,
        session: requests.Session | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        self._api_key = api_key
        if not self._api_key:
            raise ValueError("FIRECRAWL_API_KEY is required")
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._timeout_seconds = timeout_seconds

    def scrape(self, url: str) -> dict[str, object]:
        payload = {
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": True,
            "removeBase64Images": True,
            "blockAds": True,
            "proxy": "auto",
            "timeout": self._timeout_seconds * 1000,
        }
        response = self._post("/scrape", payload)
        return response.get("data", {})

    def map(
        self,
        url: str,
        search: str | None = None,
        limit: int = FIRECRAWL_DEFAULT_MAP_LIMIT,
    ) -> list[dict[str, str]]:
        payload: dict[str, object] = {
            "url": url,
            "sitemap": "include",
            "includeSubdomains": True,
            "ignoreQueryParameters": True,
            "limit": limit,
            "timeout": self._timeout_seconds * 1000,
        }
        if search:
            payload["search"] = search
        response = self._post("/map", payload)
        return response.get("links", [])

    def search(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        payload: dict[str, object] = {
            "query": query,
            "limit": limit,
            "sources": ["web"],
        }
        response = self._post("/search", payload)
        data = response.get("data", {})
        if isinstance(data, dict):
            web_results = data.get("web", [])
            return web_results if isinstance(web_results, list) else []
        return data if isinstance(data, list) else []

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        response = self._session.post(
            f"{self._base_url}{path}",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self._timeout_seconds,
        )
        body = response.json()
        if response.status_code >= 400 or not body.get("success", False):
            message = body.get("error") or response.text
            raise FirecrawlError(f"Firecrawl request failed: {message}")
        return body


class FirecrawlScrapeProvider:
    def __init__(self, client: FirecrawlClient) -> None:
        self._client = client

    def scrape(self, request: ScrapeRequest) -> ScrapedDocument:
        payload = self._client.scrape(request.url)
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        return ScrapedDocument(
            company=request.company,
            source_url=str(metadata.get("sourceURL") or metadata.get("url") or payload.get("url") or request.url),
            title=str(metadata.get("title") or payload.get("title") or request.url),
            text=str(payload.get("markdown") or payload.get("text") or ""),
        )

    def discover(
        self,
        source_url: str,
        search: str | None = None,
        limit: int = FIRECRAWL_DEFAULT_MAP_LIMIT,
    ) -> list[str]:
        return [link["url"] for link in self._client.map(source_url, search=search, limit=limit) if link.get("url")]

    def search(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        return self._client.search(query, limit=limit)
