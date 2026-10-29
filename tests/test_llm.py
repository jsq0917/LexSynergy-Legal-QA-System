"""LLM 三件（Triage / 撰稿 / G4）的离线单测：全部用假客户端，不碰网络。"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.generation.llm_client import (
    ChatResult,
    DashScopeClient,
    LLMAuthError,
    LLMBadRequest,
    LLMQuotaError,
    LLMRateLimit,
    LLMServerError,
    LLMResponseError,
    Usage,
    _strip_fences,
)
from app.generation.prompts import synthesis_messages, triage_messages
from app.generation.synthesizer import LLMSynthesizer
from app.orchestrator.context import TaskContext
from app.orchestrator.pipeline import DISCLAIMER, template_synthesizer
from app.orchestrator.rules import triage as rule_triage
from app.orchestrator.triage_llm import LLMTriager
from app.verifier.support_llm import LLMSupportChecker
from config import Settings

SETTINGS = Settings(
    llm_api_key="test-key",
    llm_model="deepseek-v4.1-flash",
    llm_base_url="https://example.invalid/v1",
    evidence_budget_chars=2000,
)


class FakeClient:
    """可编程假客户端：只实现被调用的两个方法。"""

    def __init__(self, *, data=None, text="", raises: Exception | None = None) -> None:
        self.data = data or {}
        self.text = text
        self.raises = raises
        self.calls: list[dict] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"kind": "chat", "messages": messages, **kwargs})
        if self.raises:
            raise self.raises
        return ChatResult(text=self.text, model="fake", usage=Usage(10, 20))

    def json_chat(self, messages, **kwargs):
        self.calls.append({"kind": "json", "messages": messages, **kwargs})
        if self.raises:
            raise self.raises
        return self.data, Usage(10, 20)


# ---------------------------------------------------------------- Triage
@pytest.mark.asyncio
async def test_triage_uses_llm_facts_and_calc():
    client = FakeClient(
        data={
            "intent": "labor_law",
            "role": "employee",
            "complexity": "medium",
            "rewrite_type": "direct",
            "law_hints": ["中华人民共和国劳动合同法"],
            "sub_queries": ["经济性裁员的经济补偿计算"],
            "facts": {"monthly_wage": 12000, "start_date": "2021-03-01"},
            "needs_calc": ["severance"],
        }
    )
    triage = await LLMTriager(client, SETTINGS)("公司裁员，2021年3月入职，月薪12000，能拿多少补偿")
    assert triage.source == "llm"
    assert triage.facts["monthly_wage"] == 12000
    assert triage.needs_calc == ["severance"]
    assert triage.missing_slots == ["end_date"]        # 缺离职日 → 会追问
    assert triage.sub_queries[0].startswith("经济性裁员")


@pytest.mark.asyncio
async def test_triage_falls_back_when_llm_fails():
    triage = await LLMTriager(FakeClient(raises=LLMRateLimit("429")), SETTINGS)(
        "公司裁员，我 2021 年 3 月入职，月薪 12000"
    )
    assert triage.source == "rules"
    assert triage.intent == "labor_law"


@pytest.mark.asyncio
async def test_triage_rejects_invalid_enums():
    client = FakeClient(
        data={"intent": "nonsense", "role": "alien", "complexity": "huge", "needs_calc": ["magic"]}
    )
    triage = await LLMTriager(client, SETTINGS)("被裁员了怎么补偿")
    base = rule_triage("被裁员了怎么补偿")
    assert triage.intent == base.intent
    assert triage.role == base.role
    assert triage.needs_calc == []          # 非法 kind 被丢掉


@pytest.mark.asyncio
async def test_triage_explicit_facts_win_over_llm():
    client = FakeClient(data={"facts": {"monthly_wage": 8000, "start_date": "2020-01-01"}})
    triage = await LLMTriager(client, SETTINGS)(
        "被裁员能拿多少补偿", facts={"monthly_wage": 12000}
    )
    assert triage.facts["monthly_wage"] == 12000   # 用户写的为准
    assert triage.facts["start_date"] == "2020-01-01"


@pytest.mark.asyncio
async def test_triage_greeting_skips_llm():
    client = FakeClient(data={"intent": "labor_law"})
    triage = await LLMTriager(client, SETTINGS)("你好")
    assert triage.intent == "greeting"
    assert client.calls == []


# ---------------------------------------------------------------- 撰稿
@pytest.mark.asyncio
async def test_synthesizer_uses_llm_and_appends_disclaimer():
    from app.evidence.ledger import EvidenceLedger
    from app.evidence.models import Evidence

    ctx = TaskContext(query="裁员补偿", triage=rule_triage("裁员补偿"))
    ledger = EvidenceLedger()
    ledger.add(
        [Evidence(law="中华人民共和国劳动合同法", article_no="第四十七条", quote="经济补偿按年限计算。")]
    )
    client = FakeClient(text="**1. 结论**\n按第四十七条计算。")
    text = await LLMSynthesizer(client, SETTINGS)(ctx, ledger, [])
    assert "按第四十七条计算" in text
    assert DISCLAIMER in text
    assert ctx.llm_calls == 1
    assert client.calls[0]["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_synthesizer_falls_back_to_template_on_error():
    from app.evidence.ledger import EvidenceLedger
    from app.evidence.models import Evidence

    ctx = TaskContext(query="裁员补偿", triage=rule_triage("裁员补偿"))
    ledger = EvidenceLedger()
    ledger.add(
        [Evidence(law="中华人民共和国劳动合同法", article_no="第四十七条", quote="经济补偿按年限计算。")]
    )
    text = await LLMSynthesizer(FakeClient(raises=LLMServerError("500")), SETTINGS)(ctx, ledger, [])
    assert text == template_synthesizer(ctx, ledger, [])
    assert "【结论】" in text


@pytest.mark.asyncio
async def test_synthesizer_without_evidence_uses_template():
    ctx = TaskContext(query="随便问问", triage=rule_triage("随便问问"))
    client = FakeClient(text="不该被调用")
    text = await LLMSynthesizer(client, SETTINGS)(ctx, ctx.ledger, [])
    assert "【结论】" in text
    assert client.calls == []


# ---------------------------------------------------------------- G4
@pytest.mark.asyncio
async def test_support_checker_aligns_back_to_original_claims():
    claims = ["用人单位应当支付经济补偿", "你可以直接查封公司账户"]
    client = FakeClient(data={"unsupported": ["你可以直接查封公司账户。"]})
    flagged = await LLMSupportChecker(client, SETTINGS)(claims, "证据：经济补偿按年限支付。")
    assert flagged == ["你可以直接查封公司账户"]


@pytest.mark.asyncio
async def test_support_checker_drops_invented_sentences():
    """模型可能"复述"出原文里没有的句子，不能直接当作失败项。"""
    claims = ["用人单位应当支付经济补偿"]
    client = FakeClient(data={"unsupported": ["这句话根本不在输入里"]})
    flagged = await LLMSupportChecker(client, SETTINGS)(claims, "证据：经济补偿按年限支付。")
    assert flagged == []


@pytest.mark.asyncio
async def test_support_checker_empty_claims():
    client = FakeClient(data={"unsupported": ["x"]})
    assert await LLMSupportChecker(client, SETTINGS)([], "ctx") == []
    assert client.calls == []


# ---------------------------------------------------------------- 客户端纯逻辑
def test_strip_fences():
    assert _strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_fences('前言 {"a": 1} 后记') == '{"a": 1}'


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (401, "unauthorized", LLMAuthError),
        (403, "forbidden", LLMAuthError),
        (429, "too many", LLMRateLimit),
        (500, "boom", LLMServerError),
        (400, "Free quota exhausted", LLMQuotaError),
        (400, "bad model", LLMBadRequest),
    ],
)
def test_error_classification(status, body, expected):
    client = DashScopeClient(SETTINGS)
    with pytest.raises(expected):
        client._raise_for(status, body)


def test_empty_content_is_retriable_error():
    exc = LLMResponseError("空返回")
    assert exc.retriable is True


def test_has_llm_requires_key():
    assert SETTINGS.has_llm is True
    assert Settings(llm_api_key="").has_llm is False


# ---------------------------------------------------------------- Prompt 契约
def test_triage_prompt_lists_allowed_calc_kinds_and_fact_fields():
    text = triage_messages("裁员补偿")[0]["content"]
    assert "severance" in text and "double_wage" in text
    assert "monthly_wage" in text and "end_date" in text
    assert "不要输出解释" in text


def test_synthesis_prompt_has_money_discipline():
    text = synthesis_messages("q", "资料", [])[0]["content"]
    assert "只能出现【计算结果】里给出的金额" in text
    assert "逐字照抄" in text
    assert "严禁编造法条号" in text


def test_synthesis_prompt_includes_calc_block():
    messages = synthesis_messages(
        "q",
        "资料",
        [{"kind": "severance", "formula": "N × 基数", "substitution": "6 × 12000", "value": "72000.00", "unit": "元"}],
    )
    assert "【计算结果】（必须原样使用，不得改动数字）" in messages[1]["content"]
    assert "72000.00" in messages[1]["content"]


def test_as_of_is_injected():
    text = synthesis_messages("q", "资料", [], as_of=date(2026, 10, 20))[0]["content"]
    assert "2026-10-20" in text
