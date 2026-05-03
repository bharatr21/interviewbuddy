from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4


class JobStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class IngestionJob:
    id: str
    source_slug: str
    company: str
    status: JobStatus = JobStatus.RUNNING
    discovered_count: int = 0
    candidate_count: int = 0
    ingested_count: int = 0
    failed_count: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, IngestionJob] = {}

    def start(self, source_slug: str, company: str) -> IngestionJob:
        job = IngestionJob(id=str(uuid4()), source_slug=source_slug, company=company)
        self._jobs[job.id] = job
        return job

    def update(self, job: IngestionJob) -> None:
        self._jobs[job.id] = job

    def get(self, job_id: str) -> IngestionJob | None:
        return self._jobs.get(job_id)

    def list(self) -> list[IngestionJob]:
        return sorted(self._jobs.values(), key=lambda job: job.started_at, reverse=True)


DEFAULT_JOB_STORE = InMemoryJobStore()
