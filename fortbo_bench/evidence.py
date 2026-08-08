"""Metric, device-placement, and deterministic worker evidence records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import shutil
from typing import Iterable


@dataclass(frozen=True)
class MetricRow:
    evaluation: int
    objective_value: float | None
    feasible: bool | None
    constraint_violation: float | None
    cost: float
    acquisition_evaluations: int
    gradient_evaluations: int
    model_fits: int
    ess: float | None
    memory_bytes: int | None
    transfers_bytes: int | None
    wall_seconds: float | None
    lane: str = "cpu"
    status: str = "measured"
    refusal_reason: str | None = None


class MetricRecorder:
    """Record optimization metrics without treating unavailable values as zero."""

    def __init__(self, optimum: float | None = None) -> None:
        self.optimum = optimum
        self.rows: list[MetricRow] = []

    def add(self, row: MetricRow) -> None:
        if row.evaluation != len(self.rows):
            raise ValueError("evaluation rows must be appended in run order")
        if row.status == "refused" and row.wall_seconds is not None:
            raise ValueError("a refused row cannot carry a timing")
        if row.status == "refused" and not row.refusal_reason:
            raise ValueError("a refusal needs a reason")
        self.rows.append(row)

    def summary(self) -> dict:
        measured = [r for r in self.rows if r.status == "measured" and r.objective_value is not None]
        feasible = [r for r in measured if r.feasible is not False]
        best = min((r.objective_value for r in feasible), default=None)
        simple_regret = None if self.optimum is None or best is None else best-self.optimum
        cumulative = None
        if self.optimum is not None:
            cumulative = sum(r.objective_value-self.optimum for r in feasible)
        return {
            "rows": len(self.rows),
            "measured": len(measured),
            "refused": sum(r.status == "refused" for r in self.rows),
            "best_feasible_value": best,
            "simple_regret": simple_regret,
            "cumulative_regret": cumulative,
            "constraint_violations": sum((r.constraint_violation or 0.0) > 0.0 for r in measured),
            "objective_evaluations": len(measured),
            "acquisition_evaluations": sum(r.acquisition_evaluations for r in measured),
            "gradient_evaluations": sum(r.gradient_evaluations for r in measured),
            "model_fits": sum(r.model_fits for r in measured),
            "ess": [r.ess for r in measured if r.ess is not None],
            "memory_bytes": [r.memory_bytes for r in measured if r.memory_bytes is not None],
            "transfers_bytes": [r.transfers_bytes for r in measured if r.transfers_bytes is not None],
            "wall_seconds": sum(r.wall_seconds or 0.0 for r in measured),
        }

    def as_dicts(self) -> list[dict]:
        return [asdict(row) for row in self.rows]

    def checkpoint(self) -> dict:
        return {"optimum": self.optimum, "rows": self.as_dicts()}

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "MetricRecorder":
        recorder = cls(checkpoint.get("optimum"))
        for row in checkpoint.get("rows", []):
            recorder.add(MetricRow(**row))
        return recorder


def device_lanes() -> list[dict]:
    """Return explicit CPU/GPU lanes; no missing GPU is represented as zero."""
    rows = [{
        "lane": "cpu", "status": "measured", "device": "host",
        "toolchain": "numpy", "timing_includes_transfers": False,
        "reason": None,
    }]
    has_nvidia = shutil.which("nvidia-smi") is not None
    for lane, device, transfer in (
        ("openacc", "openacc-device", False),
        ("cuda-transfer-inclusive", "cuda", True),
        ("cuda-resident", "cuda", False),
    ):
        rows.append({
            "lane": lane,
            "status": "unavailable" if not has_nvidia else "refused",
            "device": device,
            "toolchain": "not detected" if not has_nvidia else "runtime not exercised",
            "timing_includes_transfers": transfer,
            "reason": ("no NVIDIA runtime detected; this is unavailable, not zero"
                       if not has_nvidia else
                       "device lane requires an explicit FortBO GPU run"),
        })
    return rows


@dataclass(frozen=True)
class WorkerEvent:
    time: int
    worker: int
    sequence: int
    job: str
    value: float | None
    cost: float
    failed: bool = False
    retry: bool = False


def replay_workers(events: Iterable[WorkerEvent], retry_limit: int = 1,
                   duplicate_policy: str = "refuse") -> dict:
    """Replay worker completions deterministically with bounded retries.

    Failed attempts remain in the cost ledger and never become objective
    values. Duplicate pending jobs are rejected before the event queue runs.
    """
    if retry_limit < 0:
        raise ValueError("retry_limit must be nonnegative")
    if duplicate_policy not in ("refuse", "deduplicate"):
        raise ValueError("duplicate_policy must be refuse or deduplicate")
    queue = [(e.time, e.worker, e.sequence, e) for e in events]
    heapq.heapify(queue)
    pending: set[str] = set()
    completed_jobs: set[str] = set()
    attempts: dict[str, int] = {}
    completed: list[dict] = []
    failures: list[dict] = []
    duplicates: list[str] = []
    total_cost = 0.0
    while queue:
        _, _, _, event = heapq.heappop(queue)
        total_cost += event.cost
        if event.job in pending:
            if not event.retry:
                if duplicate_policy == "refuse":
                    raise ValueError(f"duplicate pending job {event.job}")
                duplicates.append(event.job)
                continue
            pending.remove(event.job)
        elif event.job in completed_jobs:
            raise ValueError(f"duplicate completed job {event.job}")
        if event.failed:
            count = attempts.get(event.job, 0) + 1
            attempts[event.job] = count
            failures.append({"job": event.job, "attempt": count, "cost": event.cost})
            if count <= retry_limit:
                pending.add(event.job)
                # A retry is a new deterministic event, not an implicit value.
                heapq.heappush(queue, (event.time+1, event.worker, event.sequence+1,
                                       WorkerEvent(event.time+1, event.worker,
                                                   event.sequence+1, event.job,
                                                   event.value, event.cost, False, True)))
            continue
        completed_jobs.add(event.job)
        completed.append({"job": event.job, "value": event.value,
                          "attempt": attempts.get(event.job, 0)+1})
    return {"completed": completed, "failures": failures, "duplicates": duplicates,
            "total_attempt_cost": total_cost}
