# Interview Buddy
Interview Preparation Buddy

Interview Buddy is a crawler-backed agentic RAG assistant for FAANG-tier interview
preparation. V1 includes a curated company source registry, provider-neutral
scraping interfaces, a local retrieval pipeline, FastAPI endpoints, a Typer CLI,
and a LangGraph-powered Streamlit interview coach that shows citation snippets
and follow-up drills.

## Quickstart

```bash
uv sync --extra dev
uv run pytest
uv run interviewbuddy sources
uv run interviewbuddy ask "How should I discuss DoorDash dispatch reliability?"
uv run interviewbuddy ask "How should I discuss DoorDash dispatch reliability?" --company DoorDash --auto-crawl --max-scraped-urls 3
uv run interviewbuddy monitoring
uv run uvicorn interviewbuddy.api:app --reload
uv run streamlit run src/interviewbuddy/streamlit_app.py
```

The current implementation ships with a small demo corpus so the API, CLI, and
chatbot can be tested before the full crawler/indexer is connected to live
scrape providers.

To ingest live content with Firecrawl:

```bash
cp .env.example .env
# edit .env and set FIRECRAWL_API_KEY
uv run interviewbuddy ingest doordash
uv run interviewbuddy ask "What reliability lessons should I discuss from DoorDash?" --company DoorDash
```

To ingest every configured source:

```bash
uv run interviewbuddy ingest-all --limit 25
uv run interviewbuddy monitoring
uv run interviewbuddy jobs
```

To test ingestion through the API:

```bash
uv run uvicorn interviewbuddy.api:app --reload
curl -X POST "http://127.0.0.1:8000/ingest/doordash"
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question":"What reliability lessons should I discuss from DoorDash?","company":"DoorDash"}'
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question":"What reliability lessons should I discuss from DoorDash?","company":"DoorDash","auto_crawl":true,"max_scraped_urls":3}'
```

To test the agentic RAG chat UI:

```bash
uv run streamlit run src/interviewbuddy/streamlit_app.py
```

Open the local Streamlit URL, select a company filter if desired, and ask a
system design or company-specific prep question. Retrieved citations appear as
expandable source cards. Enable "Auto-crawl if evidence is weak" to let the
agent perform a budget-limited crawl against the selected company's allowlisted
source before answering.

Live ingestion writes normalized documents to `.interviewbuddy/corpus.jsonl`
and indexes chunks into the configured SQL vector store. The CLI, FastAPI app,
and Streamlit chatbot use indexed vectors when the default corpus exists and
fall back to the built-in demo corpus otherwise.

By default, ingestion asks Firecrawl `/map` to traverse the source with sitemap
mode enabled and discover up to 5000 URLs. Use `--limit` or the API `limit`
query parameter to lower this during testing or raise it up to Firecrawl's
100000 URL maximum for larger sites.

## Query-Driven Auto-Crawl

The primary product loop can now be demand-driven instead of bulk-first:

1. The agent searches the existing vector store for enough relevant evidence.
2. If evidence is weak and auto-crawl is enabled, it plans a targeted crawl for
   the selected company source.
3. Firecrawl `/search` is used first for query-targeted URL discovery.
4. Firecrawl `/map` is used as a fallback for source traversal when search is
   weak.
5. Candidate URLs are filtered to the configured source URL, scraped, checked
   for query relevance, indexed, and then used for the final answer.

CLI auto-crawl is disabled by default:

```bash
uv run interviewbuddy ask "How does DoorDash approach dispatch reliability?" \
  --company DoorDash \
  --auto-crawl \
  --max-provider-calls 8 \
  --max-discovered-urls 10 \
  --max-scraped-urls 4
```

Responses include normal citations plus an auto-crawl report with accepted,
skipped, failed, provider-call, and budget-exhaustion counts. The API returns
the same data under `crawl_report` when `auto_crawl` is true.

## RAG and Persistence

Chat is now a LangGraph workflow with explicit retrieval, synthesis, and drill
nodes. The graph is model-flexible: any provider implementing the chat-provider
interface can be plugged into the synthesis node. If `OPENAI_API_KEY` is set,
Interview Buddy uses OpenAI for embeddings and model-backed synthesis. If it is
not set, it falls back to local hash embeddings and deterministic synthesis so
the app remains testable offline.

The local JSONL corpus remains the default manual testing path. A SQL corpus
store is also available through `SqlCorpusStore` and `DATABASE_URL`; use SQLite
locally or a Postgres URL for hosted deployment.

Chunk vectors are stored through `SqlVectorStore`. SQLite stores vectors as JSON
and ranks locally for development. Postgres deployments can use the generated
pgvector schema from `SqlVectorStore.postgres_schema_sql(dimensions)`, including
`CREATE EXTENSION IF NOT EXISTS vector`, `embedding vector(...)`, and an IVFFlat
cosine index.

Ingestion now tracks in-process jobs, filters discovered URLs to the configured
source path, removes duplicate querystring variants, retries provider failures,
and records partial failures instead of discarding the whole run.

## Firecrawl Scraping

Set `FIRECRAWL_API_KEY` before using the Firecrawl provider:

```bash
export FIRECRAWL_API_KEY=fc-YOUR-API-KEY
```

For local development, copy `.env.example` to `.env` and fill in your own
secrets. `.env` is ignored by Git.

The provider uses Firecrawl API v2 directly:

- `POST https://api.firecrawl.dev/v2/map` to discover article URLs from a
  company source.
- `POST https://api.firecrawl.dev/v2/scrape` with `formats=["markdown"]`,
  `onlyMainContent=true`, `removeBase64Images=true`, `blockAds=true`, and
  `proxy="auto"` to normalize individual articles into RAG-ready markdown.
- `POST https://api.firecrawl.dev/v2/search` for query-based URL discovery
  when a source URL is not enough.

Firecrawl is the active default in `interviewbuddy.settings.Settings`.
Inactive Tavily and Apify adapters live in `src/interviewbuddy/scrapers/`.
Swap `Settings.scrape_provider()` to return one of those providers if
Firecrawl rate limits or extraction quality become a problem.

## V1 Source Registry

Interview Buddy will start with a curated set of official engineering,
research, and technical blogs for FAANG-tier interview preparation.

| Company | Source |
| --- | --- |
| OpenAI | https://openai.com/research/index/ |
| Anthropic | https://www.anthropic.com/engineering |
| NVIDIA | https://developer.nvidia.com/blog |
| Microsoft | https://devblogs.microsoft.com/engineering-at-microsoft/ |
| Uber | https://www.uber.com/blog/engineering |
| Lyft | https://eng.lyft.com/ |
| Pinterest | https://medium.com/pinterest-engineering |
| LinkedIn | https://www.linkedin.com/blog/engineering |
| Salesforce | https://engineering.salesforce.com/ |
| ServiceNow | https://www.servicenow.com/blogs.html |
| Meta | https://engineering.fb.com/ |
| Apple | https://machinelearning.apple.com/ |
| Amazon | https://www.amazon.science/blog/ |
| Netflix | https://netflixtechblog.com/ |
| Google | https://research.google/blog/ |
| DoorDash | https://careersatdoordash.com/engineering-blog/ |
