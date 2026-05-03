from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, StateGraph

from interviewbuddy.agent import LangGraphInterviewAgent
from interviewbuddy.corpus import JsonlCorpusStore
from interviewbuddy.documents import Document
from interviewbuddy.embeddings import EmbeddingProvider
from interviewbuddy.providers import ScrapeProvider, ScrapeRequest
from interviewbuddy.rag import CoachAnswer
from interviewbuddy.sources import Source
from interviewbuddy.url_filters import source_scoped_urls


@dataclass(frozen=True)
class CrawlBudget:
    max_provider_calls: int = 8
    max_discovered_urls: int = 10
    max_scraped_urls: int = 4
    max_fallback_attempts: int = 1


@dataclass(frozen=True)
class QueryCrawlRequest:
    question: str
    company: str | None = None
    limit: int = 4
    budget: CrawlBudget = field(default_factory=CrawlBudget)


@dataclass(frozen=True)
class CrawlReport:
    used_existing_evidence: bool
    discovered_count: int = 0
    candidate_count: int = 0
    accepted_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    provider_call_count: int = 0
    fallback_count: int = 0
    budget_exhausted: bool = False
    skipped_reasons: list[str] = field(default_factory=list)
    failed_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QueryCrawlResult:
    answer: CoachAnswer
    crawl_report: CrawlReport
    trace: list[str]


class QueryCrawlState(TypedDict, total=False):
    request: QueryCrawlRequest
    existing_answer: CoachAnswer
    existing_sufficient: bool
    discovered_urls: list[str]
    candidate_urls: list[str]
    accepted_documents: list[Document]
    skipped_reasons: list[str]
    failed_reasons: list[str]
    provider_call_count: int
    fallback_count: int
    budget_exhausted: bool
    answer: CoachAnswer
    report: CrawlReport
    trace: list[str]


class QueryCrawlAgent:
    """Demand-driven crawler that crawls only when existing evidence is weak."""

    def __init__(
        self,
        source: Source,
        provider: ScrapeProvider,
        corpus_store: JsonlCorpusStore,
        vector_store,
        embedding_provider: EmbeddingProvider,
        chat_provider=None,
    ) -> None:
        self._source = source
        self._provider = provider
        self._corpus_store = corpus_store
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._chat_provider = chat_provider
        self._graph = self._build_graph()

    def run(self, request: QueryCrawlRequest) -> QueryCrawlResult:
        state = self._graph.invoke(
            {
                "request": request,
                "provider_call_count": 0,
                "fallback_count": 0,
                "skipped_reasons": [],
                "failed_reasons": [],
                "budget_exhausted": False,
                "trace": [],
            }
        )
        return QueryCrawlResult(
            answer=state["answer"],
            crawl_report=state["report"],
            trace=state.get("trace", []),
        )

    def _build_graph(self):
        graph = StateGraph(QueryCrawlState)
        graph.add_node("retrieve_existing", self._retrieve_existing)
        graph.add_node("assess_sufficiency", self._assess_sufficiency)
        graph.add_node("answer_existing", self._answer_existing)
        graph.add_node("plan_targeted_crawl", self._plan_targeted_crawl)
        graph.add_node("discover", self._discover)
        graph.add_node("extract", self._extract)
        graph.add_node("index", self._index)
        graph.add_node("answer", self._answer)
        graph.set_entry_point("retrieve_existing")
        graph.add_edge("retrieve_existing", "assess_sufficiency")
        graph.add_conditional_edges(
            "assess_sufficiency",
            lambda state: "answer_existing" if state.get("existing_sufficient") else "plan_targeted_crawl",
            {"answer_existing": "answer_existing", "plan_targeted_crawl": "plan_targeted_crawl"},
        )
        graph.add_edge("answer_existing", END)
        graph.add_edge("plan_targeted_crawl", "discover")
        graph.add_edge("discover", "extract")
        graph.add_edge("extract", "index")
        graph.add_edge("index", "answer")
        graph.add_edge("answer", END)
        return graph.compile()

    def _retrieve_existing(self, state: QueryCrawlState) -> QueryCrawlState:
        request = state["request"]
        answer = self._answer_from_vector_store(request)
        return {
            "existing_answer": answer,
            "trace": [*state.get("trace", []), "retrieve_existing"],
        }

    def _assess_sufficiency(self, state: QueryCrawlState) -> QueryCrawlState:
        request = state["request"]
        citations = state["existing_answer"].citations
        sufficient = bool(citations) and _citations_relevant(request.question, citations, request.company)
        return {
            "existing_sufficient": sufficient,
            "trace": [*state.get("trace", []), "assess_sufficiency"],
        }

    def _answer_existing(self, state: QueryCrawlState) -> QueryCrawlState:
        report = CrawlReport(used_existing_evidence=True)
        return {
            "answer": state["existing_answer"],
            "report": report,
            "trace": [*state.get("trace", []), "answer_existing"],
        }

    def _plan_targeted_crawl(self, state: QueryCrawlState) -> QueryCrawlState:
        return {
            "trace": [*state.get("trace", []), "plan_targeted_crawl"],
        }

    def _discover(self, state: QueryCrawlState) -> QueryCrawlState:
        request = state["request"]
        budget = request.budget
        provider_calls = state.get("provider_call_count", 0)
        fallback_count = state.get("fallback_count", 0)

        discovered = []
        if provider_calls < budget.max_provider_calls and hasattr(self._provider, "search"):
            discovered.extend(_urls_from_search(self._provider.search(_search_query(request, self._source), limit=budget.max_discovered_urls)))
            provider_calls += 1

        if not discovered and provider_calls < budget.max_provider_calls and fallback_count < budget.max_fallback_attempts:
            discovered.extend(self._provider.discover(self._source.url, search=None, limit=budget.max_discovered_urls))
            provider_calls += 1
            fallback_count += 1

        scoped = source_scoped_urls(self._source.url, discovered)[: budget.max_discovered_urls]
        return {
            "discovered_urls": discovered,
            "candidate_urls": scoped,
            "provider_call_count": provider_calls,
            "fallback_count": fallback_count,
            "trace": [*state.get("trace", []), "discover"],
        }

    def _extract(self, state: QueryCrawlState) -> QueryCrawlState:
        request = state["request"]
        budget = request.budget
        accepted: list[Document] = []
        skipped = list(state.get("skipped_reasons", []))
        failed = list(state.get("failed_reasons", []))
        provider_calls = state.get("provider_call_count", 0)
        scraped_count = 0
        budget_exhausted = state.get("budget_exhausted", False)

        for url in state.get("candidate_urls", []):
            if scraped_count >= budget.max_scraped_urls or provider_calls >= budget.max_provider_calls:
                budget_exhausted = True
                break
            try:
                scraped = self._provider.scrape(ScrapeRequest(company=self._source.company, url=url))
                provider_calls += 1
                scraped_count += 1
            except Exception as error:  # noqa: BLE001 - report crawl failures.
                provider_calls += 1
                failed.append(f"{url}: {error}")
                continue

            relevant, reason = _document_relevant(request.question, scraped.text, scraped.title, request.company)
            if not relevant:
                skipped.append(f"{url}: {reason}")
                continue
            accepted.append(
                Document(
                    company=scraped.company,
                    source_url=scraped.source_url,
                    title=scraped.title,
                    text=scraped.text,
                )
            )

        if state.get("candidate_urls") and scraped_count < len(state["candidate_urls"]):
            budget_exhausted = True

        return {
            "accepted_documents": accepted,
            "skipped_reasons": skipped,
            "failed_reasons": failed,
            "provider_call_count": provider_calls,
            "budget_exhausted": budget_exhausted,
            "trace": [*state.get("trace", []), "extract"],
        }

    def _index(self, state: QueryCrawlState) -> QueryCrawlState:
        documents = state.get("accepted_documents", [])
        if documents:
            self._corpus_store.upsert_many(documents)
            self._vector_store.add_documents(documents)
        return {
            "trace": [*state.get("trace", []), "index"],
        }

    def _answer(self, state: QueryCrawlState) -> QueryCrawlState:
        request = state["request"]
        answer = self._answer_from_vector_store(request)
        report = CrawlReport(
            used_existing_evidence=False,
            discovered_count=len(state.get("discovered_urls", [])),
            candidate_count=len(state.get("candidate_urls", [])),
            accepted_count=len(state.get("accepted_documents", [])),
            skipped_count=len(state.get("skipped_reasons", [])),
            failed_count=len(state.get("failed_reasons", [])),
            provider_call_count=state.get("provider_call_count", 0),
            fallback_count=state.get("fallback_count", 0),
            budget_exhausted=state.get("budget_exhausted", False),
            skipped_reasons=state.get("skipped_reasons", []),
            failed_reasons=state.get("failed_reasons", []),
        )
        return {
            "answer": answer,
            "report": report,
            "trace": [*state.get("trace", []), "answer"],
        }

    def _answer_from_vector_store(self, request: QueryCrawlRequest) -> CoachAnswer:
        agent = LangGraphInterviewAgent(self._vector_store, chat_provider=self._chat_provider)
        return agent.answer(request.question, limit=request.limit, company=request.company)


def _search_query(request: QueryCrawlRequest, source: Source) -> str:
    company = request.company or source.company
    return f"{company} engineering {request.question}"


def _urls_from_search(results: list[dict[str, object]]) -> list[str]:
    urls = []
    for result in results:
        url = result.get("url")
        if isinstance(url, str):
            urls.append(url)
    return urls


def _citations_relevant(question: str, citations, company: str | None) -> bool:
    for citation in citations:
        if company and citation.company.lower() != company.lower():
            continue
        relevant, _reason = _document_relevant(question, citation.snippet, citation.title, company)
        if relevant:
            return True
    return False


def _document_relevant(question: str, text: str, title: str, company: str | None) -> tuple[bool, str]:
    if len(text.strip()) < 40:
        return False, "content too short"
    query_terms = _meaningful_terms(question, company)
    haystack = f"{title} {text}".lower()
    overlap = [term for term in query_terms if term in haystack]
    if len(overlap) < max(1, min(2, len(query_terms))):
        return False, "insufficient query-term overlap"
    return True, "accepted"


def _meaningful_terms(question: str, company: str | None) -> list[str]:
    stopwords = {
        "how",
        "does",
        "should",
        "discuss",
        "handle",
        "what",
        "why",
        "the",
        "and",
        "for",
        "with",
        "from",
        "about",
        "company",
        "engineering",
    }
    if company:
        stopwords.add(company.lower())
    terms = []
    for term in re.findall(r"[a-z0-9]+", question.lower()):
        if len(term) < 4 or term in stopwords:
            continue
        terms.append(term)
    return terms or [term for term in re.findall(r"[a-z0-9]+", question.lower()) if len(term) >= 4]
