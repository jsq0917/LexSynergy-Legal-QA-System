"""端到端：编排 → 检索 → 计算 → 汇总 → 校验 → 回路 → 缓存。

全部离线：检索用内存假实现、审查用规则假实现、校验用真实现（这正是我们要测的）。
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.agents.calc_agent import CalcAgent
from app.agents.compliance import ComplianceAgent
from app.agents.statute import StatuteAgent
from app.agents.verifier_agent import VerifierAgent
from app.orchestrator.cache import AnswerCache
from app.orchestrator.pipeline import Pipeline, template_synthesizer
from app.retrieval.port import InMemoryStatuteSearch
from app.runtime import demo_corpus, rule_based_auditor
from app.verifier.service import Verifier
from config import Settings

LCL = "中华人民共和国劳动合同法"
FACTS = {"monthly_wage": 12000, "start_date": "2021-03-01", "end_date": "2026-10-20"}


def build(*, synthesizer=None, cache=None) -> Pipeline:
    agents = {
        "statute": StatuteAgent(InMemoryStatuteSearch(demo_corpus()), topk=5),
        "calc": CalcAgent(),
        "compliance": ComplianceAgent(auditor=rule_based_auditor),
        "verifier": VerifierAgent(Verifier(tolerance=Decimal("0.01"))),
    }
    return Pipeline(agents, synthesizer=synthesizer, cache=cache, settings=Settings())


@pytest.mark.asyncio
async def test_calc_route_end_to_end():
    answer = await build().run("被裁员能拿多少经济补偿", facts=FACTS)
    assert answer.route == "calc"
    assert "72000.00" in answer.text
    assert "第四十七条" in answer.text
    assert answer.verdict and answer.verdict["passed"] is True
    assert answer.verdict["action"] == "pass"
    assert answer.sources and answer.sources[0]["eid"].startswith("E")
    assert [s["agent"] for s in answer.steps] == ["statute", "calc"]


@pytest.mark.asyncio
async def test_llm_calls_counted_from_agents():
    answer = await build().run("被裁员能拿多少经济补偿", facts=FACTS)
    # 两个专家都不含 LLM：整条链路 0 次 LLM
    assert answer.llm_calls == 0


@pytest.mark.asyncio
async def test_missing_slots_gives_basis_then_asks():
    answer = await build().run("被裁员能拿多少经济补偿")
    assert answer.text.startswith("为了给你准确的结论")
    assert "月工资" in answer.text
    assert "第四十七条" in answer.text        # 先给依据，不是干巴巴要参数
    assert answer.sources


@pytest.mark.asyncio
async def test_greeting_short_circuit():
    answer = await build().run("你好")
    assert answer.route == "greeting"
    assert "LexSynergy" in answer.text
    assert answer.steps == []


@pytest.mark.asyncio
async def test_out_of_scope_short_circuit():
    answer = await build().run("今天天气不错")
    assert answer.route == "no_data"
    assert "暂未在现有资料中" in answer.text


@pytest.mark.asyncio
async def test_verifier_failure_triggers_one_refine_loop():
    calls = {"n": 0}

    def flaky(ctx, ledger, calc):
        calls["n"] += 1
        base = template_synthesizer(ctx, ledger, calc)
        if calls["n"] == 1:      # 第一次故意引一个不存在的条号
            return base + f"\n《{LCL}》第九十九条：“不存在的条文。”"
        return base

    answer = await build(synthesizer=flaky).run("被裁员能拿多少经济补偿", facts=FACTS)
    assert answer.loops == 1
    assert calls["n"] == 2
    assert answer.verdict and answer.verdict["passed"] is True
    # 补检索确实多跑了一次 statute
    assert len([s for s in answer.steps if s["agent"] == "statute"]) == 2


@pytest.mark.asyncio
async def test_numeric_error_degrades_without_loop():
    def wrong_amount(ctx, ledger, calc):
        return template_synthesizer(ctx, ledger, calc) + "\n按上述标准应付 66000 元。"

    answer = await build(synthesizer=wrong_amount).run("被裁员能拿多少经济补偿", facts=FACTS)
    assert answer.loops == 0
    assert answer.verdict["action"] == "downgrade"
    assert "校验未通过" in answer.text
    assert "G5_numeric" in answer.text


@pytest.mark.asyncio
async def test_compliance_route_for_employer():
    answer = await build().run("我们公司的员工手册写着自愿放弃社保，合规吗")
    assert answer.route == "compliance"
    compliance_steps = [s for s in answer.steps if s["agent"] == "compliance"]
    assert compliance_steps and compliance_steps[0]["status"] in {"ok", "failed"}


@pytest.mark.asyncio
async def test_multi_hop_produces_two_statute_steps():
    answer = await build().run("裁员补偿怎么算，年假怎么折算", facts={**FACTS, "annual_leave_days": 5})
    assert answer.route == "multi_hop"
    assert len([s for s in answer.steps if s["agent"] == "statute"]) == 2


@pytest.mark.asyncio
async def test_cache_hit_skips_everything():
    pipeline = build(cache=AnswerCache())
    first = await pipeline.run("被裁员能拿多少经济补偿", facts=FACTS)
    second = await pipeline.run("被裁员能拿多少经济补偿", facts=FACTS)
    assert first.cached is False
    assert second.cached is True
    assert second.text == first.text
    assert second.steps == []          # 缓存命中不再派发


@pytest.mark.asyncio
async def test_no_cache_for_partial_answer():
    """带追问的半成品不写缓存：它依赖用户后续补的参数。"""
    cache = AnswerCache()
    await build(cache=cache).run("被裁员能拿多少经济补偿")   # 缺槽位 → 追问
    assert len(cache) == 0


def test_cache_key_includes_role_and_plan():
    from app.orchestrator.cache import cache_key
    from app.orchestrator.models import Plan, Step

    plan = Plan(steps=(Step("s1", "statute"),))
    other = Plan(steps=(Step("s1", "statute"), Step("c1", "calc")))
    assert cache_key("q", "employee", plan, None) != cache_key("q", "employer", plan, None)
    assert cache_key("q", "employee", plan, None) != cache_key("q", "employee", other, None)
    assert cache_key("q", "employee", plan, None) == cache_key("q", "employee", plan, None)


@pytest.mark.asyncio
async def test_events_order_is_stable():
    answer = await build().run("被裁员能拿多少经济补偿", facts=FACTS)
    names = [e for e, _ in answer.events]
    assert names[0] == "meta" and names[1] == "plan"
    assert names[-2:] == ["delta", "done"]
    assert "verify" in names and "sources" in names


@pytest.mark.asyncio
async def test_event_sink_receives_stream():
    seen: list[str] = []
    await build().run("被裁员能拿多少经济补偿", facts=FACTS, event_sink=lambda e, d: seen.append(e))
    assert seen[0] == "meta" and seen[-1] == "done"


@pytest.mark.asyncio
async def test_as_of_filters_expired_article():
    answer = await build().run("被裁员能拿多少经济补偿", facts=FACTS, as_of=date(2026, 10, 20))
    assert answer.verdict["passed"] is True


def test_only_missing_evidence_classifier():
    """区分"缺证据"与"写错了"：只有 G1 才走"补证据不重写"。"""
    from app.orchestrator.pipeline import _only_missing_evidence

    assert _only_missing_evidence({"fails": [{"gate": "G1_not_found"}]})
    assert _only_missing_evidence({"fails": [{"gate": "G1_not_found"}, {"gate": "G1_not_found"}]})
    assert not _only_missing_evidence({"fails": [{"gate": "G1_not_found"}, {"gate": "G2_misquote"}]})
    assert not _only_missing_evidence({"fails": [{"gate": "G4_misgrounded"}]})
    assert not _only_missing_evidence({"fails": []})
