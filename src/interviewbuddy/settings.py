from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from interviewbuddy.llm import OpenAIChatProvider
from interviewbuddy.openai_providers import OpenAIEmbeddingProvider
from interviewbuddy.paths import DEFAULT_STATE_DIR
from interviewbuddy.providers import FirecrawlHttpClient, FirecrawlScrapeProvider


class Settings(BaseSettings):
    firecrawl_api_key: str | None = None
    firecrawl_base_url: str = "https://api.firecrawl.dev/v2"
    firecrawl_timeout_seconds: int = 60

    tavily_api_key: str | None = None
    apify_api_token: str | None = None

    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4.1-mini"
    embedding_dimensions: int = 1536

    database_url: str = f"sqlite:///{DEFAULT_STATE_DIR / 'interviewbuddy.db'}"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def firecrawl_provider(self) -> FirecrawlScrapeProvider:
        if not self.firecrawl_api_key:
            raise ValueError("FIRECRAWL_API_KEY is required")
        client = FirecrawlHttpClient(
            api_key=self.firecrawl_api_key,
            base_url=self.firecrawl_base_url,
            timeout_seconds=self.firecrawl_timeout_seconds,
        )
        return FirecrawlScrapeProvider(client)

    def scrape_provider(self) -> FirecrawlScrapeProvider:
        # Firecrawl is the active default. If Firecrawl rate limits or extraction
        # quality become an issue, swap this factory to return TavilyScrapeProvider
        # or ApifyWebsiteContentCrawlerProvider from `interviewbuddy.scrapers`.
        return self.firecrawl_provider()

    def openai_client(self):
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        from openai import OpenAI

        return OpenAI(api_key=self.openai_api_key)

    def embedding_provider(self):
        if self.openai_api_key:
            return OpenAIEmbeddingProvider(
                client=self.openai_client(),
                model=self.openai_embedding_model,
            )
        from interviewbuddy.embeddings import HashEmbeddingProvider

        return HashEmbeddingProvider()

    def chat_provider(self):
        if not self.openai_api_key:
            return None
        return OpenAIChatProvider(
            client=self.openai_client(),
            model=self.openai_chat_model,
        )

    def vector_store(self):
        from interviewbuddy.vector_store import SqlVectorStore

        return SqlVectorStore(
            database_url=self.database_url,
            embedding_provider=self.embedding_provider(),
            dimensions=self.embedding_dimensions if self.openai_api_key else 128,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
