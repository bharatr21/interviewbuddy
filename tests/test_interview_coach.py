from interviewbuddy.coach import CandidateProfile, InterviewCoach
from interviewbuddy.rag import Citation, CoachAnswer


class FakeGroundedCoach:
    def answer(self, question: str, limit: int = 4, company: str | None = None) -> CoachAnswer:
        return CoachAnswer(
            message="Grounded answer from sources.",
            citations=[
                Citation(
                    title="DoorDash Reliability",
                    company="DoorDash",
                    url="https://example.com",
                    snippet="DoorDash reliability snippet.",
                    score=0.9,
                )
            ],
        )


def test_interview_coach_adds_personalized_plan_and_followups():
    coach = InterviewCoach(
        grounded_coach=FakeGroundedCoach(),
        profile=CandidateProfile(
            target_role="Senior Backend Engineer",
            target_companies=["DoorDash", "OpenAI"],
            focus_areas=["system design", "reliability"],
            weak_spots=["estimation"],
        ),
    )

    answer = coach.answer("How should I discuss DoorDash reliability?", company="DoorDash")

    assert "Senior Backend Engineer" in answer.message
    assert "estimation" in answer.message
    assert "Follow-up drills" in answer.message
    assert answer.citations[0].title == "DoorDash Reliability"
