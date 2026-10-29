"""编排层数据模型：Triage / Plan / Step / 结果 / 最终答案。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

AGENTS = ("statute", "calc", "compliance", "verifier")

SHORT_CIRCUITS = ("follow_up", "greeting", "no_data")


@dataclass
class TriageResult:
    """Triage 的产物：意图 + 角色 + 法域 + 事实槽位 + 缺失槽位。

    一次 LLM 调用产出全部字段（"一次调用、结构化输出"），规则层可按同样结构产出。
    """

    query: str
    intent: str = "general"          # labor_law | civil_code | general | greeting | other
    role: str = "employee"           # employee | employer | unknown
    complexity: str = "medium"       # simple | medium | complex
    rewrite_type: str = "direct"     # direct | history | hyde | simplify | decompose
    law_hints: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    needs_calc: list[str] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    follow_up: str = ""
    as_of: date | None = None
    source: str = "rules"            # rules | llm | cache

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "intent": self.intent,
            "role": self.role,
            "complexity": self.complexity,
            "rewrite_type": self.rewrite_type,
            "law_hints": list(self.law_hints),
            "sub_queries": list(self.sub_queries),
            "facts": dict(self.facts),
            "needs_calc": list(self.needs_calc),
            "missing_slots": list(self.missing_slots),
            "follow_up": self.follow_up,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "source": self.source,
        }


@dataclass(frozen=True)
class Step:
    id: str
    agent: str
    deps: tuple[str, ...] = ()
    payload: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.agent not in AGENTS:
            raise ValueError(f"未知 agent：{self.agent}")


@dataclass(frozen=True)
class Plan:
    steps: tuple[Step, ...] = ()
    reason: str = ""
    short_circuit: str | None = None
    short_circuit_text: str = ""
    #: 缺槽位时先给依据再追问：这里记要补问哪些槽位
    ask_slots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.short_circuit and self.short_circuit not in SHORT_CIRCUITS:
            raise ValueError(f"未知短路类型：{self.short_circuit}")

    @property
    def fingerprint(self) -> str:
        raw = "|".join(f"{s.id}:{s.agent}" for s in self.steps)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]

    def step(self, step_id: str) -> Step | None:
        return next((s for s in self.steps if s.id == step_id), None)

    def to_dict(self) -> dict:
        return {
            "reason": self.reason,
            "short_circuit": self.short_circuit,
            "ask_slots": list(self.ask_slots),
            "fingerprint": self.fingerprint,
            "steps": [
                {"id": s.id, "agent": s.agent, "deps": list(s.deps), "status": "pending"}
                for s in self.steps
            ],
        }


@dataclass
class StepResult:
    step_id: str
    agent: str
    status: str                       # ok | no_data | input_required | failed | timeout
    artifact: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)
    elapsed_ms: int = 0
    llm_calls: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass
class PlanResult:
    done: dict[str, StepResult] = field(default_factory=dict)
    degraded_reason: str | None = None

    def of(self, agent: str) -> list[StepResult]:
        return [r for r in self.done.values() if r.agent == agent and r.ok]

    @property
    def calc_values(self) -> list[Decimal]:
        out: list[Decimal] = []
        for res in self.of("calc"):
            for item in res.artifact.get("results", []):
                try:
                    out.append(Decimal(str(item.get("value"))))
                except Exception:  # noqa: BLE001 - 非法值直接跳过
                    continue
        return out

    def to_dicts(self) -> list[dict]:
        return [
            {
                "step_id": r.step_id,
                "agent": r.agent,
                "status": r.status,
                "elapsed_ms": r.elapsed_ms,
                "evidence": len(r.evidence),
                "llm_calls": r.llm_calls,
                "error": r.error,
            }
            for r in self.done.values()
        ]


@dataclass
class Answer:
    text: str = ""
    route: str = ""
    mode: str = ""                   # fast | full（成本感知路由的档位）
    timings: dict = field(default_factory=dict)   # 分段耗时（ms），供时间线与延迟分析
    intent: str = ""
    sources: list[dict] = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    plan: Plan | None = None
    steps: list[dict] = field(default_factory=list)
    calc: list[dict] = field(default_factory=list)
    verdict: dict | None = None
    loops: int = 0
    llm_calls: int = 0
    elapsed_ms: int = 0
    cached: bool = False
    degraded: str | None = None
    events: list[tuple[str, dict]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "route": self.route,
            "mode": self.mode,
            "timings": self.timings,
            "intent": self.intent,
            "sources": self.sources,
            "facts": self.facts,
            "plan": self.plan.to_dict() if self.plan else None,
            "steps": self.steps,
            "calc": self.calc,
            "verdict": self.verdict,
            "loops": self.loops,
            "llm_calls": self.llm_calls,
            "elapsed_ms": self.elapsed_ms,
            "cached": self.cached,
            "degraded": self.degraded,
        }
