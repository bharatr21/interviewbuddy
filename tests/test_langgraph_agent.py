from interviewbuddy.agent import AgentInput, LangGraphInterviewAgent
from interviewbuddy.coach import CandidateProfile
from interviewbuddy.rag import Citation, SearchResult, StaticSearchStore


def test_langgraph_agent_retrieves_synthesizes_and_generates_drills():
    store = StaticSearchStore(
        [
            Citation(
                title="DoorDash Reliability",
                company="DoorDash",
                url="https://example.com/reliability",
                snippet="DoorDash reliability depends on routing retries and observability.",
                score=0.91,
            )
        ]
    )
    agent = LangGraphInterviewAgent(
        search_store=store,
        profile=CandidateProfile(
            target_role="Staff Backend Engineer",
            target_companies=["DoorDash"],
            focus_areas=["system design", "reliability"],
            weak_spots=["capacity estimation"],
        ),
    )

    result = agent.run(
        AgentInput(
            question="How should I discuss DoorDash reliability?",
            company="DoorDash",
            limit=3,
        )
    )

    assert "Staff Backend Engineer" in result.message
    assert "capacity estimation" in result.message
    assert "Follow-up drills" in result.message
    assert result.citations[0].title == "DoorDash Reliability"
    assert result.trace == ["retrieve", "synthesize", "drills"]


def test_langgraph_agent_returns_no_context_guidance_when_retrieval_is_empty():
    agent = LangGraphInterviewAgent(search_store=StaticSearchStore([]))

    result = agent.run(AgentInput(question="unknown topic"))

    assert "I could not find" in result.message
    assert result.citations == []
    assert result.trace == ["retrieve", "no_context"]


def test_langgraph_agent_can_use_model_provider_for_synthesis():
    class FakeChatProvider:
        def answer(self, question: str, citations: list[Citation], company: str | None = None) -> str:
            return f"Model answer for {company}: {citations[0].title}"

    agent = LangGraphInterviewAgent(
        search_store=StaticSearchStore(
            [
                Citation(
                    title="DoorDash Reliability",
                    company="DoorDash",
                    url="https://example.com/reliability",
                    snippet="DoorDash reliability snippet.",
                    score=0.91,
                )
            ]
        ),
        chat_provider=FakeChatProvider(),
    )

    result = agent.run(AgentInput(question="How should I discuss reliability?", company="DoorDash"))

    assert "Model answer for DoorDash" in result.message
    assert "Follow-up drills" in result.message
