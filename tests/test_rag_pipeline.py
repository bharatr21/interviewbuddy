from interviewbuddy.documents import Document, chunk_document
from interviewbuddy.embeddings import HashEmbeddingProvider
from interviewbuddy.rag import GroundedCoach, InMemoryVectorStore


def test_chunk_document_preserves_source_metadata_and_snippet_text():
    document = Document(
        company="DoorDash",
        source_url="https://careersatdoordash.com/engineering-blog/scaling",
        title="Scaling DoorDash Dispatch",
        text="DoorDash dispatch systems balance latency, reliability, and marketplace demand during peak traffic.",
        published_at="2025-01-10",
    )

    chunks = chunk_document(document, max_chars=45)

    assert len(chunks) >= 2
    assert chunks[0].company == "DoorDash"
    assert chunks[0].title == "Scaling DoorDash Dispatch"
    assert chunks[0].source_url == document.source_url
    assert "DoorDash dispatch systems" in chunks[0].text


def test_vector_store_returns_ranked_citation_snippets():
    provider = HashEmbeddingProvider(dimensions=32)
    store = InMemoryVectorStore(provider)
    store.add_documents(
        [
            Document(
                company="DoorDash",
                source_url="https://careersatdoordash.com/engineering-blog/dispatch",
                title="Dispatch Reliability",
                text="DoorDash improves dispatch reliability with real-time routing, retries, and observability.",
            ),
            Document(
                company="Netflix",
                source_url="https://netflixtechblog.com/cache",
                title="Caching",
                text="Netflix optimizes video caching for playback quality.",
            ),
        ]
    )

    results = store.search("DoorDash dispatch reliability", limit=1)

    assert len(results) == 1
    assert results[0].chunk.company == "DoorDash"
    assert results[0].citation.url == "https://careersatdoordash.com/engineering-blog/dispatch"
    assert "dispatch reliability" in results[0].citation.snippet.lower()


def test_grounded_coach_answer_includes_interview_framing_and_citations():
    provider = HashEmbeddingProvider(dimensions=32)
    store = InMemoryVectorStore(provider)
    store.add_documents(
        [
            Document(
                company="DoorDash",
                source_url="https://careersatdoordash.com/engineering-blog/dispatch",
                title="Dispatch Reliability",
                text="DoorDash improves dispatch reliability with real-time routing, retries, and observability.",
            )
        ]
    )
    coach = GroundedCoach(store)

    answer = coach.answer("How should I discuss DoorDash dispatch reliability?")

    assert "Interview framing" in answer.message
    assert "DoorDash" in answer.message
    assert answer.citations[0].title == "Dispatch Reliability"
    assert answer.citations[0].url == "https://careersatdoordash.com/engineering-blog/dispatch"
