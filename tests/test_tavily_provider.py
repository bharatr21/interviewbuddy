from interviewbuddy.providers import ScrapeRequest, ScrapedDocument
from interviewbuddy.scrapers.tavily import TavilyScrapeProvider


class FakeTavilyClient:
    def __init__(self) -> None:
        self.extract_calls: list[dict] = []
        self.map_calls: list[dict] = []

    def extract(self, **kwargs) -> dict:
        self.extract_calls.append(kwargs)
        return {
            "results": [
                {
                    "url": "https://careersatdoordash.com/engineering-blog/reliability",
                    "raw_content": "DoorDash reliability content in markdown.",
                }
            ],
            "failed_results": [],
        }

    def map(self, **kwargs) -> dict:
        self.map_calls.append(kwargs)
        return {
            "base_url": kwargs["url"],
            "results": [
                "https://careersatdoordash.com/engineering-blog/reliability",
                "https://careersatdoordash.com/engineering-blog/dispatch",
            ],
        }


def test_tavily_provider_extracts_markdown_with_official_sdk_shape():
    client = FakeTavilyClient()
    provider = TavilyScrapeProvider(client=client)

    document = provider.scrape(
        ScrapeRequest(
            company="DoorDash",
            url="https://careersatdoordash.com/engineering-blog/reliability",
        )
    )

    assert isinstance(document, ScrapedDocument)
    assert document.company == "DoorDash"
    assert document.source_url == "https://careersatdoordash.com/engineering-blog/reliability"
    assert document.title == "https://careersatdoordash.com/engineering-blog/reliability"
    assert document.text == "DoorDash reliability content in markdown."
    assert client.extract_calls[0]["format"] == "markdown"
    assert client.extract_calls[0]["extract_depth"] == "advanced"


def test_tavily_provider_discovers_urls_with_map():
    client = FakeTavilyClient()
    provider = TavilyScrapeProvider(client=client)

    urls = provider.discover("https://careersatdoordash.com/engineering-blog/", search="engineering", limit=10)

    assert urls == [
        "https://careersatdoordash.com/engineering-blog/reliability",
        "https://careersatdoordash.com/engineering-blog/dispatch",
    ]
    assert client.map_calls[0]["query"] == "engineering"
    assert client.map_calls[0]["limit"] == 10
