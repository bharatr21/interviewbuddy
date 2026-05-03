from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from interviewbuddy.rag import Citation


class ChatProvider(Protocol):
    def answer(self, question: str, citations: list[Citation], company: str | None = None) -> str:
        """Synthesize a grounded interview-coach answer from citations."""


@dataclass(frozen=True)
class OpenAIChatProvider:
    client: object
    model: str = "gpt-4.1-mini"

    def answer(self, question: str, citations: list[Citation], company: str | None = None) -> str:
        context = "\n\n".join(
            f"[{index}] {citation.title} ({citation.company})\nURL: {citation.url}\nSnippet: {citation.snippet}"
            for index, citation in enumerate(citations, start=1)
        )
        system = (
            "You are Interview Buddy, a grounded interview coach for FAANG-tier system design prep. "
            "Use only the provided source snippets. Explain interview relevance, concrete tradeoffs, "
            "and how the candidate should frame the topic. Cite sources inline as [1], [2]."
        )
        user = (
            f"Company filter: {company or 'none'}\n"
            f"Question: {question}\n\n"
            f"Retrieved source snippets:\n{context}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
