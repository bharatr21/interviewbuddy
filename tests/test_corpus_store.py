from pathlib import Path

from interviewbuddy.corpus import JsonlCorpusStore
from interviewbuddy.documents import Document


def test_jsonl_corpus_store_upserts_documents_by_source_url(tmp_path: Path):
    path = tmp_path / "corpus.jsonl"
    store = JsonlCorpusStore(path)

    store.upsert_many(
        [
            Document(
                company="DoorDash",
                source_url="https://careersatdoordash.com/engineering-blog/reliability",
                title="Old title",
                text="Old text",
            )
        ]
    )
    store.upsert_many(
        [
            Document(
                company="DoorDash",
                source_url="https://careersatdoordash.com/engineering-blog/reliability",
                title="New title",
                text="New text",
            )
        ]
    )

    documents = store.load()

    assert len(documents) == 1
    assert documents[0].title == "New title"
    assert documents[0].text == "New text"


def test_jsonl_corpus_store_loads_empty_list_when_file_missing(tmp_path: Path):
    store = JsonlCorpusStore(tmp_path / "missing.jsonl")

    assert store.load() == []
