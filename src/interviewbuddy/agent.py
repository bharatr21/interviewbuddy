from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, StateGraph

from interviewbuddy.coach import CandidateProfile
from interviewbuddy.rag import Citation, CoachAnswer, SearchStore


@dataclass(frozen=True)
class AgentInput:
    question: str
    company: str | None = None
    limit: int = 4


@dataclass(frozen=True)
class AgentResult:
    message: str
    citations: list[Citation]
    trace: list[str] = field(default_factory=list)


class AgentState(TypedDict, total=False):
    question: str
    company: str | None
    limit: int
    profile: CandidateProfile
    citations: list[Citation]
    answer: str
    drills: list[str]
    trace: list[str]


class LangGraphInterviewAgent:
    """Interview-prep agent built as an explicit LangGraph workflow."""

    def __init__(self, search_store: SearchStore, profile: CandidateProfile | None = None, chat_provider=None) -> None:
        self._search_store = search_store
        self._profile = profile or CandidateProfile()
        self._chat_provider = chat_provider
        self._graph = self._build_graph()

    def run(self, agent_input: AgentInput) -> AgentResult:
        state = self._graph.invoke(
            {
                "question": agent_input.question,
                "company": agent_input.company,
                "limit": agent_input.limit,
                "profile": self._profile,
                "trace": [],
            }
        )
        return AgentResult(
            message=_render_message(state),
            citations=state.get("citations", []),
            trace=state.get("trace", []),
        )

    def answer(self, question: str, limit: int = 4, company: str | None = None) -> CoachAnswer:
        result = self.run(AgentInput(question=question, company=company, limit=limit))
        return CoachAnswer(message=result.message, citations=result.citations)

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("no_context", self._no_context)
        graph.add_node("synthesize", self._synthesize)
        graph.add_node("drills", self._drills)
        graph.set_entry_point("retrieve")
        graph.add_conditional_edges(
            "retrieve",
            lambda state: "synthesize" if state.get("citations") else "no_context",
            {"synthesize": "synthesize", "no_context": "no_context"},
        )
        graph.add_edge("synthesize", "drills")
        graph.add_edge("drills", END)
        graph.add_edge("no_context", END)
        return graph.compile()

    def _retrieve(self, state: AgentState) -> AgentState:
        results = self._search_store.search(
            state["question"],
            limit=state.get("limit", 4),
            company=state.get("company"),
        )
        return {
            "citations": [result.citation for result in results],
            "trace": [*state.get("trace", []), "retrieve"],
        }

    def _no_context(self, state: AgentState) -> AgentState:
        return {
            "answer": (
                "I could not find a strong grounded source for that question. "
                "Try adding a company name, system design topic, or more specific architecture keyword."
            ),
            "trace": [*state.get("trace", []), "no_context"],
        }

    def _synthesize(self, state: AgentState) -> AgentState:
        profile = state["profile"]
        citations = state.get("citations", [])
        if self._chat_provider:
            base_answer = self._chat_provider.answer(state["question"], citations, company=state.get("company"))
            answer = f"{base_answer}\n\n{_profile_block(profile, state.get('company'))}"
        else:
            source_lines = "\n".join(f"- {citation.company}: {citation.snippet}" for citation in citations)
            answer = (
                "Grounded answer:\n"
                f"{source_lines}\n\n"
                f"{_profile_block(profile, state.get('company'))}"
            )
        return {
            "answer": answer,
            "trace": [*state.get("trace", []), "synthesize"],
        }

    def _drills(self, state: AgentState) -> AgentState:
        drills = [
            "Give a 90-second architecture summary using one cited company example.",
            "Name two bottlenecks, one reliability risk, and one operational metric.",
            "Practice one tradeoff answer that includes estimation and failure modes.",
        ]
        return {
            "drills": drills,
            "trace": [*state.get("trace", []), "drills"],
        }


def _render_message(state: AgentState) -> str:
    if not state.get("drills"):
        return state.get("answer", "")
    drill_lines = "\n".join(f"- {drill}" for drill in state["drills"])
    return f"{state.get('answer', '')}\n\nFollow-up drills:\n{drill_lines}"


def _profile_block(profile: CandidateProfile, company: str | None) -> str:
    return (
        "Personalized interview coach plan:\n"
        f"- Target role: {profile.target_role}\n"
        f"- Target companies: {', '.join(profile.target_companies) or company or 'not specified'}\n"
        f"- Focus areas: {', '.join(profile.focus_areas)}\n"
        f"- Watch-outs: {', '.join(profile.weak_spots) or 'state assumptions, quantify scale, and explain tradeoffs'}"
    )
