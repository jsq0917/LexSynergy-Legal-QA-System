"""规则路由与 Planner 的校验/回落。"""

from __future__ import annotations

from datetime import date

import pytest

from app.orchestrator.models import Plan, Step
from app.orchestrator.planner import Planner
from app.orchestrator.rules import (
    infer_calc_kinds,
    infer_intent,
    infer_role,
    split_sub_queries,
    triage,
)

FACTS = {"monthly_wage": 12000, "start_date": "2021-03-01", "end_date": "2026-10-20"}


def test_intent_greeting():
    assert infer_intent("你好") == "greeting"
    assert infer_intent("你能做什么？") == "greeting"


def test_intent_labor_and_civil():
    assert infer_intent("公司裁员怎么补偿") == "labor_law"
    assert infer_intent("离婚时房产怎么分") == "civil_code"


def test_role_employer_by_keywords():
    assert infer_role("我们公司的员工手册这条合规吗") == "employer"
    assert infer_role("我被公司辞退了") == "employee"


@pytest.mark.parametrize(
    "query,kinds",
    [
        ("被裁员能拿多少经济补偿", ["severance"]),
        ("公司违法解除，能要赔偿金吗", ["illegal"]),
        ("没签劳动合同可以主张二倍工资吗", ["double_wage"]),
        ("加班费怎么算", ["overtime"]),
        ("年假没休怎么折算", ["annual_leave"]),
        ("试用期最长多久", ["probation"]),
        ("无关键词的普通问题", []),
    ],
)
def test_infer_calc_kinds(query, kinds):
    assert infer_calc_kinds(query) == kinds


def test_illegal_excludes_severance():
    assert infer_calc_kinds("违法解除的经济补偿和赔偿金") == ["illegal"]


def test_split_sub_queries():
    assert len(split_sub_queries("裁员补偿怎么算，年假怎么折算")) == 2
    assert split_sub_queries("裁员补偿怎么算") == []


def test_fact_fragments_are_not_sub_queries():
    """带逗号但只有一句是问句 → 不拆。事实片段不是独立检索单元。"""
    assert split_sub_queries("公司裁员，我 2021 年 3 月入职，月薪 12000，能拿多少经济补偿？") == []
    assert split_sub_queries("我们公司的员工手册规定试用期 6 个月，合规吗") == []


# ---------------------------------------------------------------- 路由
def test_route_greeting_short_circuits():
    plan = triage("你好")
    from app.orchestrator.rules import route

    assert route(plan).short_circuit == "greeting"
    assert route(plan).steps == ()


def test_route_other_short_circuits():
    from app.orchestrator.rules import route

    assert route(triage("今天天气不错")).short_circuit == "no_data"


def test_route_missing_slots_still_retrieves_then_asks():
    """缺槽位不短路：先给法条依据与公式，再追问缺什么。"""
    from app.orchestrator.rules import route

    t = triage("被裁员能拿多少经济补偿")   # 没给月薪/起止日期
    plan = route(t)
    assert plan.short_circuit is None
    assert [s.agent for s in plan.steps] == ["statute", "calc"]
    assert set(plan.ask_slots) == {"monthly_wage", "start_date", "end_date"}
    assert "缺槽位" in plan.reason


def test_route_calc_parallel_with_statute():
    from app.orchestrator.rules import route

    t = triage("被裁员能拿多少经济补偿", facts=FACTS)
    plan = route(t)
    agents = [s.agent for s in plan.steps]
    assert agents == ["statute", "calc"]
    calc = next(s for s in plan.steps if s.agent == "calc")
    assert calc.deps == ()          # 与检索并行，不等它
    assert calc.payload["kinds"] == ["severance"]


def test_route_employer_adds_compliance_after_statute():
    from app.orchestrator.rules import route

    t = triage("我们公司的员工手册规定试用期 6 个月，合规吗")
    plan = route(t)
    agents = [s.agent for s in plan.steps]
    assert "compliance" in agents
    compliance = next(s for s in plan.steps if s.agent == "compliance")
    assert compliance.deps == ("s1",)   # 审查依赖证据，必须串行


def test_route_decompose_multi_statute():
    from app.orchestrator.rules import route

    t = triage("裁员补偿怎么算，年假怎么折算")
    plan = route(t)
    assert len([s for s in plan.steps if s.agent == "statute"]) == 2


# ---------------------------------------------------------------- Planner
def test_planner_uses_rules_when_simple():
    called = []
    planner = Planner(lambda t: called.append(t) or Plan())
    plan = planner.plan(triage("被裁员能拿多少经济补偿", facts=FACTS))
    assert not called                 # 规则能定就不问 LLM
    assert [s.agent for s in plan.steps] == ["statute", "calc"]


def test_planner_llm_only_for_complex():
    seen = []
    planner = Planner(lambda t: seen.append(t) or Plan(steps=(Step(id="x1", agent="statute"),), reason="llm"))
    t = triage("裁员补偿怎么算，年假怎么折算，另外竞业限制有没有补偿")
    t.complexity = "complex"
    plan = planner.plan(t)
    assert seen and plan.reason == "llm"


def test_planner_rejects_cycle_and_falls_back():
    planner = Planner(lambda t: {"steps": [
        {"id": "a", "agent": "statute", "deps": ["b"]},
        {"id": "b", "agent": "calc", "deps": ["a"]},
    ]})
    t = triage("被裁员能拿多少经济补偿", facts=FACTS)
    t.complexity = "complex"
    plan = planner.plan(t)
    assert [s.id for s in plan.steps] == ["s1", "c1"]   # 回落规则图


def test_planner_rejects_duplicate_id():
    planner = Planner(lambda t: {"steps": [
        {"id": "a", "agent": "statute"},
        {"id": "a", "agent": "calc"},
    ]})
    t = triage("被裁员能拿多少经济补偿", facts=FACTS)
    t.complexity = "complex"
    assert [s.id for s in planner.plan(t).steps] == ["s1", "c1"]


def test_planner_rejects_missing_dep():
    planner = Planner(lambda t: {"steps": [{"id": "a", "agent": "statute", "deps": ["zzz"]}]})
    t = triage("被裁员能拿多少经济补偿", facts=FACTS)
    t.complexity = "complex"
    assert planner.plan(t).reason.startswith("规则路由")


def test_planner_accepts_valid_llm_plan():
    planner = Planner(lambda t: {
        "steps": [
            {"id": "a", "agent": "statute", "payload": {"sub_query": "裁员补偿"}},
            {"id": "b", "agent": "calc", "deps": ["a"], "payload": {"kinds": ["severance"]}},
        ],
        "reason": "先取证再算",
    })
    t = triage("被裁员能拿多少经济补偿", facts=FACTS)
    t.complexity = "complex"
    plan = planner.plan(t)
    assert [s.id for s in plan.steps] == ["a", "b"]


# ---------------------------------------------------------------- 指纹
def test_fingerprint_stable_and_distinct():
    a = Plan(steps=(Step(id="s1", agent="statute"),))
    b = Plan(steps=(Step(id="s1", agent="statute"),), reason="不同理由")
    c = Plan(steps=(Step(id="s1", agent="statute"), Step(id="c1", agent="calc")))
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint


def test_as_of_passed_through():
    t = triage("被裁员能拿多少经济补偿", facts=FACTS, as_of=date(2026, 1, 1))
    assert t.as_of == date(2026, 1, 1)


def test_calc_dropped_when_rule_question_without_slots():
    """问规则（几倍/几天/是否有效）且没给数据 → 不算钱，直接讲规则。"""
    from app.orchestrator.rules import _calc_kinds, wants_specific_amount

    t = triage("平时加班、休息日加班、法定节假日加班，加班费分别按几倍工资支付？")
    t.needs_calc = ["overtime"]
    assert not wants_specific_amount(t.query)
    assert _calc_kinds(t) == []          # 否则会变成"请提供月工资"


def test_calc_kept_when_amount_requested():
    from app.orchestrator.rules import _calc_kinds

    t = triage("我工作 3 年 7 个月被违法辞退，月薪 2 万，能拿多少赔偿？")
    t.needs_calc = ["illegal"]
    t.facts = {"monthly_wage": 20000}
    assert _calc_kinds(t) == ["illegal"]   # 要具体金额 → 缺日期就追问


def test_calc_kept_when_slots_complete_even_without_wording():
    from app.orchestrator.rules import _calc_kinds

    t = triage("我在 2019 年 7 月 15 日入职，2026 年 10 月 20 日被裁员，月平均工资 8000 元，经济补偿怎么算？")
    t.needs_calc = ["severance"]
    t.facts = {"monthly_wage": 8000, "start_date": "2019-07-15", "end_date": "2026-10-20"}
    assert _calc_kinds(t) == ["severance"]
