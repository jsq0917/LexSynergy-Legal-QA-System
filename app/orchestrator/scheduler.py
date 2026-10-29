"""DAG 调度器：拓扑就绪即并行，单步超时不拖垮整体，每步落 checkpoint。

三道闸（借鉴 LangGraph supervisor 的生产约束）：
* ``max_steps``：总步数上限，防止"越查越多"；
* ``step_timeout``：单步超时，一个专家挂掉不影响其余；
* ``deadline``：全局 deadline，到点返回已有结果（部分降级而非整体失败）。
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

from app.core.errors import PlanError
from app.core.logger import get_logger
from app.orchestrator.context import TaskContext
from app.orchestrator.models import Plan, PlanResult, Step, StepResult

logger = get_logger("scheduler")

Dispatch = Callable[[Step, TaskContext], Awaitable[StepResult]]
Checkpoint = Callable[[Step, StepResult], None]


class Scheduler:
    def __init__(
        self,
        dispatch: Dispatch,
        *,
        max_steps: int = 8,
        step_timeout: float = 20.0,
        deadline_s: float = 25.0,
        checkpoint: Checkpoint | None = None,
    ) -> None:
        self._dispatch = dispatch
        self._max_steps = max_steps
        self._step_timeout = step_timeout
        self._deadline_s = deadline_s
        self._checkpoint = checkpoint

    async def run(self, plan: Plan, ctx: TaskContext) -> PlanResult:
        if not plan.steps:
            return PlanResult()
        deadline = time.perf_counter() + self._deadline_s
        pending = list(plan.steps)
        done: dict[str, StepResult] = {}
        degraded: str | None = None

        while pending:
            if len(done) >= self._max_steps:
                degraded = "max_steps"
                break
            if time.perf_counter() > deadline:
                degraded = "deadline"
                break
            ready = [s for s in pending if all(dep in done for dep in s.deps)]
            # 本轮最多再跑 (max_steps - 已完成) 步，避免一次并发就冲破上限
            ready = ready[: max(0, self._max_steps - len(done))]
            if not ready:
                raise PlanError("依赖无法满足（可能存在环或缺失依赖）")

            outcomes = await asyncio.gather(
                *(self._run_one(step, ctx) for step in ready), return_exceptions=True
            )
            for step, outcome in zip(ready, outcomes):
                result = (
                    outcome
                    if isinstance(outcome, StepResult)
                    else StepResult(
                        step_id=step.id,
                        agent=step.agent,
                        status="failed",
                        error=f"{type(outcome).__name__}: {outcome}",
                    )
                )
                done[step.id] = result
                ctx.results[step.id] = result
                pending.remove(step)
                ctx.emit(
                    "agent",
                    {
                        "step_id": step.id,
                        "agent": step.agent,
                        "status": result.status,
                        "elapsed_ms": result.elapsed_ms,
                        "evidence": len(result.evidence),
                        "llm_calls": result.llm_calls,
                        "error": result.error,
                    },
                )
                logger.info(
                    "step %s(%s) -> %s %dms 证据=%d",
                    step.id,
                    step.agent,
                    result.status,
                    result.elapsed_ms,
                    len(result.evidence),
                )
                if self._checkpoint is not None:
                    self._checkpoint(step, result)

        return PlanResult(done=done, degraded_reason=degraded)

    async def _run_one(self, step: Step, ctx: TaskContext) -> StepResult:
        timeout = min(self._step_timeout, max(1.0, ctx.remaining_s()))
        started = time.perf_counter()
        try:
            return await asyncio.wait_for(self._dispatch(step, ctx), timeout=timeout)
        except asyncio.TimeoutError:
            return StepResult(
                step_id=step.id,
                agent=step.agent,
                status="timeout",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                error=f"超时（>{timeout:.0f}s）",
            )
        except Exception as exc:  # noqa: BLE001 - 单个专家失败必须被隔离
            logger.warning("step %s 失败：%s", step.id, str(exc)[:160])
            return StepResult(
                step_id=step.id,
                agent=step.agent,
                status="failed",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )
