from interviewbuddy.providers import ScrapeRequest, ScrapedDocument
from interviewbuddy.scrapers.apify import ApifyWebsiteContentCrawlerProvider


class FakeListItemsResult:
    def __init__(self, items: list[dict]) -> None:
        self.items = items


class FakeDatasetClient:
    def __init__(self, items: list[dict]) -> None:
        self._items = items

    def list_items(self) -> FakeListItemsResult:
        return FakeListItemsResult(self._items)


class FakeActorClient:
    def __init__(self, apify_client: "FakeApifyClient") -> None:
        self._apify_client = apify_client

    def call(self, run_input: dict) -> dict:
        self._apify_client.actor_calls.append(run_input)
        return {"defaultDatasetId": "dataset-123"}


class FakeApifyClient:
    def __init__(self) -> None:
        self.actor_calls: list[dict] = []
        self.items = [
            {
                "url": "https://careersatdoordash.com/engineering-blog/reliability",
                "metadata": {"title": "Reliability"},
                "markdown": "DoorDash reliability markdown.",
                "text": "DoorDash reliability text.",
            }
        ]

    def actor(self, actor_id: str) -> FakeActorClient:
        assert actor_id == "apify/website-content-crawler"
        return FakeActorClient(self)

    def dataset(self, dataset_id: str) -> FakeDatasetClient:
        assert dataset_id == "dataset-123"
        return FakeDatasetClient(self.items)


def test_apify_provider_runs_website_content_crawler_and_reads_dataset():
    client = FakeApifyClient()
    provider = ApifyWebsiteContentCrawlerProvider(client=client)

    document = provider.scrape(
        ScrapeRequest(
            company="DoorDash",
            url="https://careersatdoordash.com/engineering-blog/reliability",
        )
    )

    assert isinstance(document, ScrapedDocument)
    assert document.company == "DoorDash"
    assert document.source_url == "https://careersatdoordash.com/engineering-blog/reliability"
    assert document.title == "Reliability"
    assert document.text == "DoorDash reliability markdown."
    assert client.actor_calls[0]["startUrls"] == [
        {"url": "https://careersatdoordash.com/engineering-blog/reliability"}
    ]
    assert client.actor_calls[0]["maxCrawlPages"] == 1


def test_apify_provider_discovers_urls_from_crawler_dataset_items():
    client = FakeApifyClient()
    provider = ApifyWebsiteContentCrawlerProvider(client=client)

    urls = provider.discover("https://careersatdoordash.com/engineering-blog/", limit=5)

    assert urls == ["https://careersatdoordash.com/engineering-blog/reliability"]
    assert client.actor_calls[0]["maxCrawlPages"] == 5
