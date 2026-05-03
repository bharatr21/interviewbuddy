from interviewbuddy.documents import Document
from interviewbuddy.embeddings import HashEmbeddingProvider
from interviewbuddy.vector_store import SqlVectorStore


def test_sql_vector_store_indexes_chunks_and_searches_with_local_fallback(tmp_path):
    store = SqlVectorStore(
        database_url=f"sqlite:///{tmp_path / 'vectors.db'}",
        embedding_provider=HashEmbeddingProvider(dimensions=32),
        dimensions=32,
    )
    store.init_schema()
    store.add_documents(
        [
            Document(
                company="DoorDash",
                source_url="https://careersatdoordash.com/engineering-blog/reliability",
                title="DoorDash Reliability",
                text="DoorDash improves dispatch reliability with routing retries and observability.",
            ),
            Document(
                company="Netflix",
                source_url="https://netflixtechblog.com/cache",
                title="Netflix Cache",
                text="Netflix optimizes video delivery caching.",
            ),
        ]
    )

    results = store.search("DoorDash dispatch reliability", company="DoorDash", limit=1)

    assert store.count_chunks() >= 2
    assert len(results) == 1
    assert results[0].citation.company == "DoorDash"
    assert results[0].citation.title == "DoorDash Reliability"
    assert "dispatch reliability" in results[0].citation.snippet.lower()


def test_sql_vector_store_pgvector_schema_contains_vector_extension_and_column():
    ddl = SqlVectorStore.postgres_schema_sql(dimensions=1536)

    assert "CREATE EXTENSION IF NOT EXISTS vector" in ddl
    assert "embedding vector(1536)" in ddl
    assert "ivfflat" in ddl
