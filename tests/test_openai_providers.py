from interviewbuddy.llm import OpenAIChatProvider
from interviewbuddy.openai_providers import OpenAIEmbeddingProvider
from interviewbuddy.rag import Citation, GroundedCoach, StaticSearchStore


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls = []

    def create(self, model: str, input: str):
        self.calls.append({"model": model, "input": input})
        return type("EmbeddingResponse", (), {"data": [type("EmbeddingData", (), {"embedding": [0.1, 0.2]})()]})()


class FakeChatCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, model: str, messages: list[dict[str, str]], temperature: float):
        self.calls.append({"model": model, "messages": messages, "temperature": temperature})
        message = type("Message", (), {"content": "Use DoorDash reliability as a concrete scaling story."})()
        choice = type("Choice", (), {"message": message})()
        return type("ChatResponse", (), {"choices": [choice]})()


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddings()
        self.chat = type("Chat", (), {"completions": FakeChatCompletions()})()


def test_openai_embedding_provider_calls_embedding_api():
    client = FakeOpenAIClient()
    provider = OpenAIEmbeddingProvider(client=client, model="text-embedding-3-small")

    embedding = provider.embed("DoorDash reliability")

    assert embedding == [0.1, 0.2]
    assert client.embeddings.calls[0]["model"] == "text-embedding-3-small"


def test_grounded_coach_uses_chat_provider_when_configured():
    chat_client = FakeOpenAIClient()
    llm = OpenAIChatProvider(client=chat_client, model="gpt-4.1-mini")
    store = StaticSearchStore(
        [
            Citation(
                title="DoorDash Reliability",
                company="DoorDash",
                url="https://example.com",
                snippet="DoorDash reliability snippet",
                score=0.9,
            )
        ]
    )
    coach = GroundedCoach(store, chat_provider=llm)

    answer = coach.answer("How should I discuss reliability?", company="DoorDash")

    assert "Use DoorDash reliability" in answer.message
    assert answer.citations[0].title == "DoorDash Reliability"
    assert chat_client.chat.completions.calls[0]["model"] == "gpt-4.1-mini"
