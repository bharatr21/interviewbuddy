from interviewbuddy.providers import (
    FirecrawlError,
    FirecrawlHttpClient,
    FirecrawlScrapeProvider,
    ScrapeRequest,
    ScrapedDocument,
)


class FakeFirecrawlClient:
    def scrape(self, url: str) -> dict[str, str]:
        return {
            "title": "DoorDash Engineering",
            "markdown": "DoorDash engineering writes about logistics, reliability, and scale.",
            "url": url,
        }

    def search(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        return [{"url": "https://careersatdoordash.com/engineering-blog/reliability", "title": query}][:limit]


def test_firecrawl_provider_maps_client_response_to_scraped_document():
    provider = FirecrawlScrapeProvider(client=FakeFirecrawlClient())

    document = provider.scrape(ScrapeRequest(company="DoorDash", url="https://careersatdoordash.com/engineering-blog/"))

    assert isinstance(document, ScrapedDocument)
    assert document.company == "DoorDash"
    assert document.title == "DoorDash Engineering"
    assert "reliability" in document.text
    assert document.source_url == "https://careersatdoordash.com/engineering-blog/"


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.requests: list[dict] = []

    def post(self, url: str, headers: dict, json: dict, timeout: int) -> FakeResponse:
        self.requests.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return self.response


def test_firecrawl_http_client_posts_scrape_request_to_v2_endpoint():
    session = FakeSession(
        FakeResponse(
            200,
            {
                "success": True,
                "data": {
                    "markdown": "Clean article markdown",
                    "metadata": {
                        "title": "DoorDash Engineering",
                        "sourceURL": "https://careersatdoordash.com/engineering-blog/",
                    },
                },
            },
        )
    )
    client = FirecrawlHttpClient(api_key="fc-test", session=session)

    payload = client.scrape("https://careersatdoordash.com/engineering-blog/")

    assert payload["markdown"] == "Clean article markdown"
    request = session.requests[0]
    assert request["url"] == "https://api.firecrawl.dev/v2/scrape"
    assert request["headers"]["Authorization"] == "Bearer fc-test"
    assert request["json"]["formats"] == ["markdown"]
    assert request["json"]["onlyMainContent"] is True
    assert request["json"]["proxy"] == "auto"


def test_firecrawl_http_client_posts_map_request_to_v2_endpoint():
    session = FakeSession(
        FakeResponse(
            200,
            {
                "success": True,
                "links": [
                    {
                        "url": "https://careersatdoordash.com/engineering-blog/reliability",
                        "title": "Reliability",
                        "description": "DoorDash reliability post",
                    }
                ],
            },
        )
    )
    client = FirecrawlHttpClient(api_key="fc-test", session=session)

    links = client.map("https://careersatdoordash.com/engineering-blog/", search="engineering", limit=25)

    assert links[0]["title"] == "Reliability"
    request = session.requests[0]
    assert request["url"] == "https://api.firecrawl.dev/v2/map"
    assert request["json"]["search"] == "engineering"
    assert request["json"]["sitemap"] == "include"
    assert request["json"]["ignoreQueryParameters"] is True
    assert request["json"]["limit"] == 25


def test_firecrawl_http_client_posts_search_request_to_v2_endpoint():
    session = FakeSession(
        FakeResponse(
            200,
            {
                "success": True,
                "data": {
                    "web": [
                        {
                            "url": "https://careersatdoordash.com/engineering-blog/reliability",
                            "title": "Reliability",
                        }
                    ]
                },
            },
        )
    )
    client = FirecrawlHttpClient(api_key="fc-test", session=session)

    results = client.search("DoorDash engineering reliability", limit=5)

    assert results[0]["title"] == "Reliability"
    request = session.requests[0]
    assert request["url"] == "https://api.firecrawl.dev/v2/search"
    assert request["json"]["query"] == "DoorDash engineering reliability"
    assert request["json"]["limit"] == 5
    assert request["json"]["sources"] == ["web"]


def test_firecrawl_http_client_raises_on_rate_limit_response():
    session = FakeSession(FakeResponse(429, {"success": False, "error": "Rate limit exceeded"}))
    client = FirecrawlHttpClient(api_key="fc-test", session=session)

    try:
        client.scrape("https://careersatdoordash.com/engineering-blog/")
    except FirecrawlError as error:
        assert "Rate limit exceeded" in str(error)
    else:
        raise AssertionError("Expected FirecrawlError")


def test_firecrawl_provider_discovers_urls_from_map_links():
    session = FakeSession(
        FakeResponse(
            200,
            {
                "success": True,
                "links": [
                    {"url": "https://careersatdoordash.com/engineering-blog/reliability"},
                    {"title": "Missing URL"},
                ],
            },
        )
    )
    provider = FirecrawlScrapeProvider(FirecrawlHttpClient(api_key="fc-test", session=session))

    urls = provider.discover("https://careersatdoordash.com/engineering-blog/", limit=10)

    assert urls == ["https://careersatdoordash.com/engineering-blog/reliability"]
    assert "search" not in session.requests[0]["json"]


def test_firecrawl_provider_exposes_query_search():
    provider = FirecrawlScrapeProvider(client=FakeFirecrawlClient())

    results = provider.search("DoorDash reliability", limit=1)

    assert results == [
        {
            "url": "https://careersatdoordash.com/engineering-blog/reliability",
            "title": "DoorDash reliability",
        }
    ]
