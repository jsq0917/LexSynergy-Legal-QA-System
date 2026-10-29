"""成本感知路由：档位怎么定、快速通道真的省了哪一次调用。"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.agents.calc_agent import CalcAgent
from app.agents.compliance import ComplianceAgent
from app.agents.statute import StatuteAgent
from app.agents.verifier_agent import VerifierAgent
from app.orchestrator.pipeline import Pipeline, template_synthesizer
from app.orchestrator.rules import route as rule_route
from app.orchestrator.rules import triage as rule_triage
from app.orchestrator.routing import FAST, FULL, choose_mode, fast_plan_of
from app.retrieval.port import InMemoryStatuteSearch
from app.runtime import demo_corpus, rule_based_auditor
from app.verifier.service import Verifier
from config import Settings

FACTS = {"monthly_wage": 12000, "start_date": "2021-03-01", "end_date": "2026-10-20"}


def _agents() -> dict:
    return {
        "statute": StatuteAgent(InMemoryStatuteSearch(demo_corpus()), topk=5),
        "calc": CalcAgent(),
        "compliance": ComplianceAgent(auditor=rule_based_auditor),
        "verifier": VerifierAgent(Verifier(tolerance=Decimal("0.01"))),
    }


def build(*, triager=None, policy: str = "auto") -> Pipeline:
    return Pipeline(
        _agents(),
        triager=triager or rule_triage,
        synthesizer=template_synthesizer,
        settings=Settings(routing_policy=policy),
    )


# ---------------------------------------------------------------- 定档规则
def _mode(query: str, facts: dict | None = None, policy: str = "auto") -> str:
    t = rule_triage(query, facts=facts)
    return choose_mode(t, rule_route(t), policy=policy).mode


def test_single_hop_goes_fast():
    assert _mode("试用期最长可以约定多久？") == FAST
    assert _mode("我不想干了，辞职要提前多久跟公司说？") == FAST


def test_money_kind_escalates_even_without_slots():
    """宽进严出：规则只要判出"钱相关"的测算类型就先升级，
    槽位够不够由全编排里的严格规则再定（否则会形成"不算钱→走快速→永不抽槽"的死循环）。"""
    assert _mode("竞业限制最长能约定几年？") == FULL


def test_calc_goes_full():
    assert _mode("被裁员能拿多少经济补偿", facts=FACTS) == FULL


def test_multi_hop_goes_full():
    assert _mode("裁员补偿怎么算，年假怎么折算") == FULL


def test_employer_compliance_goes_full():
    assert _mode("我们公司的员工手册里试用期条款要审查一下合规吗") == FULL


def test_policy_override():
    assert _mode("试用期最长可以约定多久？", policy="full") == FULL
    assert _mode("被裁员能拿多少经济补偿", FACTS, policy="fast") == FAST


def test_decision_carries_reason():
    t = rule_triage("被裁员能拿多少经济补偿", facts=FACTS)
    decision = choose_mode(t, rule_route(t))
    assert decision.mode == FULL
    assert any("测算" in r for r in decision.reasons)


def test_fast_plan_keeps_single_statute_step():
    t = rule_triage("试用期最长可以约定多久？")
    plan = fast_plan_of(rule_route(t))
    assert len(plan.steps) == 1
    assert plan.steps[0].agent == "statute"


# ---------------------------------------------------------------- 快速通道行为
@pytest.mark.asyncio
async def test_fast_path_skips_llm_triage_and_g4():
    """快速通道的核心承诺：不调 LLM 分诊、不跑 G4（都要钱）。"""

    def boom(*args, **kwargs):  # pragma: no cover - 被调用即失败
        raise AssertionError("快速通道不应调用 LLM 分诊")

    answer = await build(triager=boom).run("试用期最长可以约定多久？")

    assert answer.mode == FAST
    assert answer.route == "fast"
    assert answer.llm_calls == 0                      # 模板撰稿 + 零分诊
    assert len(answer.steps) == 1                     # 只跑一路检索
    assert "G4_misgrounded" in (answer.verdict or {}).get("skipped_gates", [])
    assert answer.verdict["passed"] is True


@pytest.mark.asyncio
async def test_full_path_does_call_triager():
    calls: list[str] = []

    def counting(query, **kwargs):
        calls.append(query)
        return rule_triage(query, **kwargs)

    answer = await build(triager=counting).run("被裁员能拿多少经济补偿", facts=FACTS)
    assert answer.mode == FULL
    assert calls, "全编排必须经过 LLM 分诊（这里用规则替代以保持离线）"
    assert answer.route == "calc"
    assert "72000.00" in answer.text


@pytest.mark.asyncio
async def test_fast_path_still_runs_deterministic_gates():
    """便宜的档位不等于不校验：G1/G2/G3/G5 仍然跑。"""
    answer = await build().run("试用期最长可以约定多久？")
    skipped = (answer.verdict or {}).get("skipped_gates", [])
    assert skipped == ("G4_misgrounded",) or list(skipped) == ["G4_misgrounded"]
    assert (answer.verdict or {}).get("checked", 0) > 0


@pytest.mark.asyncio
async def test_cache_does_not_share_between_modes():
    """同一个问题在 fast/full 两档下的裁决不同，缓存键必须分开。"""
    from app.orchestrator.cache import cache_key
    from app.orchestrator.models import Plan

    plan = Plan()
    assert cache_key("q", "employee", plan, None, "fast") != cache_key(
        "q", "employee", plan, None, "full"
    )


def test_fast_plan_drops_rule_based_law_filter():
    """规则猜的法域不准，快速通道宁宽不窄：不做过滤。"""
    t = rule_triage("老板入职时扣了我身份证还收了两千押金，能要回来吗？")
    plan = fast_plan_of(rule_route(t))
    assert plan.steps
    assert "law_hints" not in plan.steps[0].payload
