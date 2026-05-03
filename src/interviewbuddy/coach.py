from __future__ import annotations

from dataclasses import dataclass, field

from interviewbuddy.rag import CoachAnswer, GroundedCoach


@dataclass(frozen=True)
class CandidateProfile:
    target_role: str = "Software Engineer"
    target_companies: list[str] = field(default_factory=list)
    focus_areas: list[str] = field(default_factory=lambda: ["system design"])
    weak_spots: list[str] = field(default_factory=list)


class InterviewCoach:
    def __init__(self, grounded_coach: GroundedCoach, profile: CandidateProfile | None = None) -> None:
        self._grounded_coach = grounded_coach
        self._profile = profile or CandidateProfile()

    def answer(self, question: str, limit: int = 4, company: str | None = None) -> CoachAnswer:
        grounded = self._grounded_coach.answer(question, limit=limit, company=company)
        if not grounded.citations:
            return grounded

        message = (
            f"{grounded.message}\n\n"
            "Personalized interview coach plan:\n"
            f"- Target role: {self._profile.target_role}\n"
            f"- Target companies: {', '.join(self._profile.target_companies) or company or 'not specified'}\n"
            f"- Focus areas: {', '.join(self._profile.focus_areas)}\n"
            f"- Watch-outs: {', '.join(self._profile.weak_spots) or 'state assumptions, quantify scale, and explain tradeoffs'}\n\n"
            "Follow-up drills:\n"
            "- Give a 90-second architecture summary using one cited company example.\n"
            "- Name two bottlenecks, one reliability risk, and one operational metric.\n"
            "- Practice one tradeoff answer that includes estimation and failure modes."
        )
        return CoachAnswer(message=message, citations=grounded.citations)
