"""DAG 调度器：拓扑顺序、并行、失败隔离、超时、步数上限、checkpoint。"""

from __future__ import annotations

import asyncio

import pytest

from app.core.errors import PlanError
from app.orchestrator.context import TaskContext
from app.orchestrator.models import Plan, Step, StepResult
from app.orchestrator.scheduler import Scheduler
from config import Budget


def ctx(**kw) -> TaskContext:
    return TaskContext(query="q", budget=Budget(**kw) if kw else Budget())


def ok(step: Step) -> StepResult:
    return StepResult(step_id=step.id, agent=step.agent, status="ok")


@pytest.mark.asyncio
async def test_linear_order_respects_deps():
    order: list[str] = []

    async def dispatch(step, _ctx):
        order.append(step.id)
        return ok(step)

    plan = Plan(steps=(Step("a", "statute"), Step("b", "calc", ("a",)), Step("c", "compliance", ("b",))))
    result = await Scheduler(dispatch).run(plan, ctx())
    assert order == ["a", "b", "c"]
    assert len(result.done) == 3


@pytest.mark.asyncio
async def test_independent_steps_run_in_parallel():
    async def dispatch(step, _ctx):
        await asyncio.sleep(0.06)
        return ok(step)

    plan = Plan(steps=(Step("a", "statute"), Step("b", "calc")))
    started = asyncio.get_event_loop().time()
    await Scheduler(dispatch).run(plan, ctx())
    elapsed = asyncio.get_event_loop().time() - started
    assert elapsed < 0.11          # 串行需要 0.12s


@pytest.mark.asyncio
async def test_failure_is_isolated():
    async def dispatch(step, _ctx):
        if step.id == "b":
            raise RuntimeError("专家挂了")
        return ok(step)

    plan = Plan(steps=(Step("a", "statute"), Step("b", "calc")))
    result = await Scheduler(dispatch).run(plan, ctx())
    assert result.done["a"].status == "ok"
    assert result.done["b"].status == "failed"
    assert "专家挂了" in (result.done["b"].error or "")


@pytest.mark.asyncio
async def test_step_timeout():
    async def dispatch(step, _ctx):
        await asyncio.sleep(0.3)
        return ok(step)

    plan = Plan(steps=(Step("a", "statute"),))
    result = await Scheduler(dispatch, step_timeout=0.05).run(plan, ctx())
    assert result.done["a"].status == "timeout"


@pytest.mark.asyncio
async def test_max_steps_guard():
    async def dispatch(step, _ctx):
        return ok(step)

    plan = Plan(steps=(Step("a", "statute"), Step("b", "calc"), Step("c", "compliance")))
    result = await Scheduler(dispatch, max_steps=2).run(plan, ctx())
    assert len(result.done) == 2
    assert result.degraded_reason == "max_steps"


@pytest.mark.asyncio
async def test_deadline_guard():
    async def dispatch(step, _ctx):
        await asyncio.sleep(0.05)
        return ok(step)

    plan = Plan(steps=(Step("a", "statute"), Step("b", "calc", ("a",)), Step("c", "compliance", ("b",))))
    result = await Scheduler(dispatch, deadline_s=0.08).run(plan, ctx())
    assert result.degraded_reason == "deadline"


@pytest.mark.asyncio
async def test_checkpoint_called_per_step():
    async def dispatch(step, _ctx):
        return ok(step)

    seen: list[str] = []
    plan = Plan(steps=(Step("a", "statute"), Step("b", "calc")))
    await Scheduler(dispatch, checkpoint=lambda s, r: seen.append(f"{s.id}:{r.status}")).run(plan, ctx())
    assert seen == ["a:ok", "b:ok"]


@pytest.mark.asyncio
async def test_missing_dep_raises_plan_error():
    async def dispatch(step, _ctx):
        return ok(step)

    plan = Plan(steps=(Step("a", "statute", ("ghost",)),))
    with pytest.raises(PlanError):
        await Scheduler(dispatch).run(plan, ctx())


@pytest.mark.asyncio
async def test_events_emitted_per_step():
    async def dispatch(step, _ctx):
        return ok(step)

    c = ctx()
    await Scheduler(dispatch).run(Plan(steps=(Step("a", "statute"),)), c)
    assert [e for e, _ in c.events] == ["agent"]
