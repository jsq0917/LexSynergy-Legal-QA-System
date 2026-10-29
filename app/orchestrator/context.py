"""TaskContext：一次任务的共享黑板。

Agent 只读 facts / 只写自己的 artifact；证据统一进账本；预算与回路计数在这里。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.evidence.ledger import EvidenceLedger
from app.orchestrator.models import Plan, StepResult, TriageResult
from config import Budget


@dataclass
class TaskContext:
    query: str
    session_id: str = ""
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    budget: Budget = field(default_factory=Budget)
    triage: TriageResult | None = None
    plan: Plan | None = None
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)
    results: dict[str, StepResult] = field(default_factory=dict)
    loop: int = 0
    llm_calls: int = 0
    events: list[tuple[str, dict]] = field(default_factory=list)
    #: SSE 实时推送钩子（每 emit 一次就回调一次）；为 None 时只记在 events 里
    sink: Any = None
    started: float = field(default_factory=time.perf_counter)
    #: 请求级硬超时（绝对时刻）。0 = 不设限。
    #: 与 `budget.deadline_s` 的区别：那个只罩住 DAG，撰稿/校验/回路都在它之外。
    deadline_at: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)

    def emit(self, event: str, data: dict) -> None:
        """记录事件（SSE 序列化的唯一来源，保证"看到的就是发生的"）。"""
        self.events.append((event, data))
        if self.sink is not None:
            self.sink(event, data)

    @property
    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)

    def remaining_s(self) -> float:
        return max(0.0, self.budget.deadline_s - (time.perf_counter() - self.started))

    def expired(self) -> bool:
        """请求级时间用尽了吗（含 0 = 不设限）。"""
        return self.deadline_at > 0 and time.perf_counter() >= self.deadline_at
