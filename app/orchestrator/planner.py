"""Planner：规则优先，按需 LLM，产出经过校验的 DAG。

三条硬约束（借鉴 supervisor 模式的生产实践）：
1. Planner 只返回结构化 Plan，不做任何自然语言解释；
2. LLM 产出的图必须过校验（唯一 id / 依赖存在 / 无环 / 步数上限），不过就回落规则图；
3. 规则命中即短路，不为"看起来更智能"多花一次 LLM。
"""

from __future__ import annotations

from typing import Callable

from app.core.errors import PlanError
from app.core.logger import get_logger
from app.orchestrator.models import Plan, Step, TriageResult
from app.orchestrator.rules import route

logger = get_logger("planner")

LLMPlanner = Callable[[TriageResult], Plan | dict]


class Planner:
    def __init__(self, llm_planner: LLMPlanner | None = None, *, max_steps: int = 8) -> None:
        self._llm = llm_planner
        self._max_steps = max_steps

    def plan(self, triage: TriageResult) -> Plan:
        base = route(triage)
        if base.short_circuit or self._llm is None or triage.complexity != "complex":
            return base
        try:
            raw = self._llm(triage)
            plan = self._validate(raw)
            logger.info("LLM 规划生效 steps=%s", [s.id for s in plan.steps])
            return plan
        except Exception as exc:  # noqa: BLE001 - 规划失败必须能回落，不能断链路
            logger.warning("LLM 规划不可用，回落规则图：%s", str(exc)[:160])
            return base

    # ------------------------------------------------------------------
    def _validate(self, raw: Plan | dict | None) -> Plan:
        plan = raw if isinstance(raw, Plan) else _coerce(raw)
        if not plan.steps:
            raise PlanError("空计划")
        if len(plan.steps) > self._max_steps:
            raise PlanError(f"步数 {len(plan.steps)} 超过上限 {self._max_steps}")
        ids = [s.id for s in plan.steps]
        if len(set(ids)) != len(ids):
            raise PlanError("step id 重复")
        for step in plan.steps:
            for dep in step.deps:
                if dep not in ids:
                    raise PlanError(f"依赖不存在：{step.id} -> {dep}")
        if _has_cycle(plan.steps):
            raise PlanError("计划存在环")
        return plan


def _coerce(raw: dict | None) -> Plan:
    if not isinstance(raw, dict):
        raise PlanError("计划不是 dict/Plan")
    steps = tuple(
        Step(
            id=str(item["id"]),
            agent=str(item["agent"]),
            deps=tuple(str(d) for d in item.get("deps", ())),
            payload=dict(item.get("payload", {})),
        )
        for item in raw.get("steps", ())
    )
    return Plan(steps=steps, reason=str(raw.get("reason", "")))


def _has_cycle(steps: tuple[Step, ...]) -> bool:
    indeg = {s.id: len(s.deps) for s in steps}
    children: dict[str, list[str]] = {s.id: [] for s in steps}
    for step in steps:
        for dep in step.deps:
            children.setdefault(dep, []).append(step.id)
    queue = [sid for sid, deg in indeg.items() if deg == 0]
    seen = 0
    while queue:
        sid = queue.pop()
        seen += 1
        for child in children.get(sid, ()):
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
    return seen != len(steps)
