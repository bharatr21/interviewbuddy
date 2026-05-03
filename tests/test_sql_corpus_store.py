from interviewbuddy.documents import Document
from interviewbuddy.sql_store import SqlCorpusStore


def test_sql_corpus_store_upserts_and_loads_documents(tmp_path):
    store = SqlCorpusStore(f"sqlite:///{tmp_path / 'interviewbuddy.db'}")
    store.init_schema()

    store.upsert_many(
        [
            Document(
                company="DoorDash",
                source_url="https://careersatdoordash.com/engineering-blog/reliability",
                title="Reliability",
                text="Old text",
            )
        ]
    )
    store.upsert_many(
        [
            Document(
                company="DoorDash",
                source_url="https://careersatdoordash.com/engineering-blog/reliability",
                title="Reliability Updated",
                text="New text",
            )
        ]
    )

    documents = store.load()

    assert len(documents) == 1
    assert documents[0].title == "Reliability Updated"
    assert documents[0].text == "New text"


def test_sql_corpus_store_reports_document_counts_by_company(tmp_path):
    store = SqlCorpusStore(f"sqlite:///{tmp_path / 'interviewbuddy.db'}")
    store.init_schema()
    store.upsert_many(
        [
            Document(company="DoorDash", source_url="https://example.com/dd", title="DD", text="DoorDash text"),
            Document(company="Netflix", source_url="https://example.com/nf", title="NF", text="Netflix text"),
        ]
    )

    assert store.count_by_company() == {"DoorDash": 1, "Netflix": 1}
