"""超时治理：超时不重试、请求级总闸会降级但不降质。"""

from __future__ import annotations

import time
from decimal import Decimal

import httpx
import pytest

from app.agents.calc_agent import CalcAgent
from app.agents.compliance import ComplianceAgent
from app.agents.statute import StatuteAgent
from app.agents.verifier_agent import VerifierAgent
from app.generation.llm_client import DashScopeClient, LLMTimeout
from app.orchestrator.pipeline import Pipeline, template_synthesizer
from app.orchestrator.rules import triage as rule_triage
from app.retrieval.port import InMemoryStatuteSearch
from app.runtime import demo_corpus, rule_based_auditor
from app.verifier.service import Verifier
from config import Settings

FACTS = {"monthly_wage": 12000, "start_date": "2021-03-01", "end_date": "2026-10-20"}


# ---------------------------------------------------------------- 重试策略
def test_timeout_is_not_retried(monkeypatch):
    """超时重试会把 35s 变成 70s，用户早走了 —— 默认必须一枪就降级。"""
    calls = {"n": 0}

    class TimeoutClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            calls["n"] += 1
            raise httpx.ReadTimeout("simulated")

    monkeypatch.setattr(httpx, "Client", TimeoutClient)
    settings = Settings(
        llm_api_key="k", llm_base_url="http://x", llm_model="m",
        llm_timeout_s=0.1, llm_max_retries=2, llm_retry_on_timeout=False,
    )
    with pytest.raises(LLMTimeout):
        DashScopeClient(settings).chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1, f"超时被重试了 {calls['n']} 次"


def test_timeout_can_be_retried_when_explicitly_enabled(monkeypatch):
    calls = {"n": 0}

    class TimeoutClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            calls["n"] += 1
            raise httpx.ReadTimeout("simulated")

    monkeypatch.setattr(httpx, "Client", TimeoutClient)
    settings = Settings(
        llm_api_key="k", llm_base_url="http://x", llm_model="m",
        llm_timeout_s=0.1, llm_max_retries=1, llm_retry_on_timeout=True,
    )
    with pytest.raises(LLMTimeout):
        DashScopeClient(settings).chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 2          # 显式打开才重试（老行为可复现）


# ---------------------------------------------------------------- 请求级总闸
def _pipeline(**settings_kwargs) -> Pipeline:
    agents = {
        "statute": StatuteAgent(InMemoryStatuteSearch(demo_corpus()), topk=5),
        "calc": CalcAgent(),
        "compliance": ComplianceAgent(auditor=rule_based_auditor),
        "verifier": VerifierAgent(Verifier(tolerance=Decimal("0.01"))),
    }
    return Pipeline(
        agents,
        triager=rule_triage,
        synthesizer=template_synthesizer,
        settings=Settings(**settings_kwargs),
    )


@pytest.mark.asyncio
async def test_request_deadline_degrades_to_template():
    """时间用尽 → 不调模型，用模板交答案，但仍然过确定性四道闸。"""
    answer = await _pipeline(request_deadline_s=0.0001).run("被裁员能拿多少经济补偿", facts=FACTS)
    assert answer.degraded == "timeout"
    assert answer.text.strip()
    assert answer.llm_calls == 0
    assert (answer.verdict or {}).get("timeout") is True
    assert "G4_misgrounded" in (answer.verdict or {}).get("skipped_gates", [])


@pytest.mark.asyncio
async def test_no_deadline_means_normal_path():
    answer = await _pipeline(request_deadline_s=0).run("被裁员能拿多少经济补偿", facts=FACTS)
    assert answer.degraded != "timeout"
    assert "72000.00" in answer.text


@pytest.mark.asyncio
async def test_deadline_does_not_truncate_answer_text():
    """降级 ≠ 截断：模板答案必须完整（结论/依据/计算齐全），不能只吐半句。"""
    answer = await _pipeline(request_deadline_s=0.0001).run("被裁员能拿多少经济补偿", facts=FACTS)
    for marker in ("【结论】", "【依据】", "【计算】"):
        assert marker in answer.text, f"模板答案缺 {marker}"


# ---------------------------------------------------------------- 故障 vs 知识边界
class _TimeoutStatuteAgent:
    """检索一律超时：用来验证"系统故障"不会被说成"没有相关规定"。"""

    name = "statute"

    async def run(self, payload: dict):
        from app.agents.base import AgentResult

        return AgentResult(agent="statute", status="timeout", error="超时（>20s）")


@pytest.mark.asyncio
async def test_retrieval_failure_is_not_reported_as_no_data():
    agents = {
        "statute": _TimeoutStatuteAgent(),
        "calc": CalcAgent(),
        "compliance": ComplianceAgent(auditor=rule_based_auditor),
        "verifier": VerifierAgent(Verifier(tolerance=Decimal("0.01"))),
    }
    pipeline = Pipeline(
        agents, triager=rule_triage, synthesizer=template_synthesizer, settings=Settings()
    )
    answer = await pipeline.run("试用期最长可以约定多久？")

    assert answer.degraded == "retrieval_failed"
    assert "本次检索没能完成" in answer.text
    assert "未覆盖" not in answer.text        # 绝不能把故障说成知识边界


@pytest.mark.asyncio
async def test_slow_synth_is_actually_interrupted():
    """必须**真的**截断，而不是"跑完再判断"。

    早先版本只检查 `ctx.expired()`（在每个步骤开始前），对正在飞的那次调用无能为力：
    deadline 设 2s、撰稿要 8s，端到端仍是 8.0s —— 总闸形同虚设。
    这条测试就是盯住那个退化。
    """
    import asyncio
    import time

    async def slow(ctx, ledger, calc):
        await asyncio.sleep(8)
        return "慢撰稿结果"

    agents = {
        "statute": StatuteAgent(InMemoryStatuteSearch(demo_corpus()), topk=5),
        "calc": CalcAgent(),
        "compliance": ComplianceAgent(auditor=rule_based_auditor),
        "verifier": VerifierAgent(Verifier(tolerance=Decimal("0.01"))),
    }
    pipeline = Pipeline(
        agents, triager=rule_triage, synthesizer=slow, settings=Settings(request_deadline_s=2)
    )
    started = time.perf_counter()
    answer = await pipeline.run("被裁员能拿多少经济补偿", facts=FACTS)
    took = time.perf_counter() - started

    assert took < 5, f"总闸没截断：实际 {took:.1f}s"
    assert answer.degraded == "timeout"
    for marker in ("【结论】", "【依据】", "【计算】"):
        assert marker in answer.text, f"降级后缺 {marker}"
