from interviewbuddy.documents import Document
from interviewbuddy.embeddings import HashEmbeddingProvider
from interviewbuddy.service import build_corpus_agent
from interviewbuddy.vector_store import SqlVectorStore


def test_build_corpus_agent_prefers_vector_store_when_chunks_exist(tmp_path):
    vector_store = SqlVectorStore(
        database_url=f"sqlite:///{tmp_path / 'vectors.db'}",
        embedding_provider=HashEmbeddingProvider(dimensions=32),
        dimensions=32,
    )
    vector_store.add_documents(
        [
            Document(
                company="DoorDash",
                source_url="https://example.com/vector",
                title="Vector Indexed DoorDash",
                text="DoorDash vector indexed reliability content.",
            )
        ]
    )

    agent = build_corpus_agent(
        corpus_path=tmp_path / "missing.jsonl",
        embedding_provider=HashEmbeddingProvider(dimensions=32),
        vector_store=vector_store,
    )

    answer = agent.answer("DoorDash reliability", company="DoorDash")

    assert answer.citations[0].title == "Vector Indexed DoorDash"
