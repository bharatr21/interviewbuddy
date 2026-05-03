from __future__ import annotations

from interviewbuddy.demo import SAMPLE_DOCUMENTS
from interviewbuddy.documents import Document
from interviewbuddy.embeddings import HashEmbeddingProvider
from interviewbuddy.paths import DEFAULT_CORPUS_PATH
from interviewbuddy.corpus import JsonlCorpusStore
from interviewbuddy.rag import GroundedCoach, InMemoryVectorStore
from interviewbuddy.agent import LangGraphInterviewAgent
from interviewbuddy.coach import CandidateProfile


def build_coach(documents: list[Document], embedding_provider=None, chat_provider=None) -> GroundedCoach:
    provider = embedding_provider or HashEmbeddingProvider()
    store = InMemoryVectorStore(provider)
    store.add_documents(documents)
    return GroundedCoach(store, chat_provider=chat_provider)


def build_demo_coach(seed_documents: list[Document] | None = None, embedding_provider=None, chat_provider=None) -> GroundedCoach:
    return build_coach(seed_documents or SAMPLE_DOCUMENTS, embedding_provider=embedding_provider, chat_provider=chat_provider)


def build_corpus_coach(corpus_path=DEFAULT_CORPUS_PATH, embedding_provider=None, chat_provider=None) -> GroundedCoach:
    documents = JsonlCorpusStore(corpus_path).load()
    return build_coach(documents or SAMPLE_DOCUMENTS, embedding_provider=embedding_provider, chat_provider=chat_provider)


def build_interview_agent(
    documents: list[Document],
    embedding_provider=None,
    profile: CandidateProfile | None = None,
    chat_provider=None,
) -> LangGraphInterviewAgent:
    provider = embedding_provider or HashEmbeddingProvider()
    store = InMemoryVectorStore(provider)
    store.add_documents(documents)
    return LangGraphInterviewAgent(store, profile=profile, chat_provider=chat_provider)


def build_corpus_agent(
    corpus_path=DEFAULT_CORPUS_PATH,
    embedding_provider=None,
    profile: CandidateProfile | None = None,
    chat_provider=None,
    vector_store=None,
) -> LangGraphInterviewAgent:
    if vector_store is not None and vector_store.count_chunks() > 0:
        return LangGraphInterviewAgent(vector_store, profile=profile, chat_provider=chat_provider)
    documents = JsonlCorpusStore(corpus_path).load()
    return build_interview_agent(
        documents or SAMPLE_DOCUMENTS,
        embedding_provider=embedding_provider,
        profile=profile,
        chat_provider=chat_provider,
    )
