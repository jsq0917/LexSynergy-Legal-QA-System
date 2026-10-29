"""阶段 8：撰稿流式化。

零网络、零容器：流式客户端用假实现，SSE 用 ASGITransport。
这里盯四件事：
1. 分片必须**逐段**回调（不是攒满再发）—— 否则首字延迟等于总时长；
2. 返回值仍是完整文本 —— 流式不能绕过引文对位与五道闸；
3. 流中断必须退回模板，且终稿整体替换已流出的草稿（不留半句）；
4. 未过闸时告示在答案**上方**（流式下用户可能先读完答案）。
"""

from __future__ import annotations

import json
from decimal import Decimal

import httpx
import pytest

from app.agents.calc_agent import CalcAgent
from app.agents.compliance import ComplianceAgent
from app.agents.statute import StatuteAgent
from app.agents.verifier_agent import VerifierAgent
from app.api.main import create_app
from app.evidence.ledger import EvidenceLedger
from app.evidence.models import Evidence
from app.generation.llm_client import DashScopeClient, LLMServerError, Usage
from app.generation.synthesizer import LLMSynthesizer
from app.orchestrator.context import TaskContext
from app.orchestrator.models import Answer
from app.orchestrator.pipeline import DISCLAIMER, Pipeline, template_synthesizer
from app.orchestrator.rules import triage as rule_triage
from app.retrieval.port import InMemoryStatuteSearch
from app.runtime import demo_corpus, rule_based_auditor
from app.verifier.service import Verifier
from config import Settings

SETTINGS = Settings(
    llm_api_key="test-key",
    llm_model="deepseek-v4.1-flash",
    llm_base_url="https://example.invalid/v1",
    evidence_budget_chars=2000,
)
FACTS = {"monthly_wage": 12000, "start_date": "2021-03-01", "end_date": "2026-10-20"}


class FakeStreamClient:
    """只实现流式的假客户端：非流式 chat 一旦被调用就失败，确保走的是流式路径。"""

    def __init__(self, chunks, *, usage: Usage | None = None, raises: Exception | None = None):
        self.chunks = list(chunks)
        self.usage = usage or Usage(prompt_tokens=10, completion_tokens=5)
        self.raises = raises
        self.calls: list[dict] = []

    def chat(self, *args, **kwargs):  # pragma: no cover - 正常路径不会走到
        raise AssertionError("流式路径不应调用非流式 chat")

    def chat_stream(self, messages, *, max_tokens=None):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        for chunk in self.chunks:
            yield chunk, None
        if self.raises is not None:
            raise self.raises
        yield "", self.usage


def _ctx_with_evidence() -> tuple[TaskContext, EvidenceLedger]:
    ctx = TaskContext(query="裁员补偿", triage=rule_triage("裁员补偿"))
    ledger = EvidenceLedger()
    ledger.add(
        [Evidence(law="中华人民共和国劳动合同法", article_no="第四十七条", quote="经济补偿按年限计算。")]
    )
    return ctx, ledger


# ---------------------------------------------------------------- 撰稿器
@pytest.mark.asyncio
async def test_stream_calls_back_each_chunk_and_returns_full_text():
    ctx, ledger = _ctx_with_evidence()
    seen: list[str] = []
    client = FakeStreamClient(["【结论】", "按第四十七条", "支付经济补偿。"])

    text = await LLMSynthesizer(client, SETTINGS).stream(ctx, ledger, [], seen.append)

    assert seen == ["【结论】", "按第四十七条", "支付经济补偿。"]   # 逐段而不是攒满
    assert text.startswith("【结论】") and text.endswith(DISCLAIMER)
    assert ctx.llm_calls == 1
    assert ctx.trace["synthesis_usage"] == {"prompt_tokens": 10, "completion_tokens": 5}
    assert client.calls[0]["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_stream_failure_falls_back_to_template():
    """断流时不能把半句留给用户：返回模板，上游终稿会整体覆盖。"""
    ctx, ledger = _ctx_with_evidence()
    seen: list[str] = []
    client = FakeStreamClient(["半句"], raises=LLMServerError("连接中断"))

    text = await LLMSynthesizer(client, SETTINGS).stream(ctx, ledger, [], seen.append)

    assert seen == ["半句"]                       # 已流出的部分保留在客户端
    assert text == template_synthesizer(ctx, ledger, [])
    assert ctx.trace["synth_stream_error"] == "LLMServerError"


@pytest.mark.asyncio
async def test_stream_without_evidence_uses_template_and_never_calls_model():
    ctx = TaskContext(query="随便问问", triage=rule_triage("随便问问"))
    client = FakeStreamClient(["不该出现"])
    seen: list[str] = []

    text = await LLMSynthesizer(client, SETTINGS).stream(ctx, ctx.ledger, [], seen.append)

    assert "【结论】" in text
    assert seen == [] and client.calls == []


@pytest.mark.asyncio
async def test_stream_empty_content_falls_back_to_template():
    """上游只回 usage、没有正文时不能答空。"""
    ctx, ledger = _ctx_with_evidence()
    client = FakeStreamClient([])
    text = await LLMSynthesizer(client, SETTINGS).stream(ctx, ledger, [], lambda _c: None)
    assert text == template_synthesizer(ctx, ledger, [])


# ---------------------------------------------------------------- 客户端解析
def test_chat_stream_parses_sse_and_usage(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"content":"你"}}]}',
        'data: {"choices":[{"delta":{"content":"好"}}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}',
        "data: [DONE]",
        "",
    ]
    captured: dict = {}

    class FakeResp:
        status_code = 200

        def iter_lines(self):
            yield from lines

        def iter_text(self):
            return ""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeHTTP:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, method, url, headers=None, json=None):
            captured.update(json or {})
            return FakeResp()

    monkeypatch.setattr(httpx, "Client", FakeHTTP)
    items = list(DashScopeClient(SETTINGS).chat_stream([{"role": "user", "content": "hi"}]))

    assert "".join(text for text, _ in items) == "你好"
    assert items[-1][1].completion_tokens == 2
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
    assert captured["enable_thinking"] is False      # 思考模式必须显式关闭


# ---------------------------------------------------------------- 流水线
def build(*, synthesizer=None) -> Pipeline:
    agents = {
        "statute": StatuteAgent(InMemoryStatuteSearch(demo_corpus()), topk=5),
        "calc": CalcAgent(),
        "compliance": ComplianceAgent(auditor=rule_based_auditor),
        "verifier": VerifierAgent(Verifier(tolerance=Decimal("0.01"))),
    }
    return Pipeline(
        agents,
        triager=rule_triage,
        synthesizer=synthesizer or template_synthesizer,
        settings=Settings(),
    )


class ChunkingSynth:
    """按字符切三段流式输出；`mutate` 用来制造未过闸的答案。"""

    def __init__(self, mutate=None) -> None:
        self.mutate = mutate

    async def stream(self, ctx, ledger, calc, on_delta):
        full = template_synthesizer(ctx, ledger, calc)
        if self.mutate:
            full = self.mutate(full)
        step = max(1, len(full) // 3)
        for i in range(0, len(full), step):
            on_delta(full[i : i + step])
        return full


@pytest.mark.asyncio
async def test_pipeline_streams_chunks_then_replaces_with_verified_text():
    answer = await build(synthesizer=ChunkingSynth()).run(
        "被裁员能拿多少经济补偿", facts=FACTS
    )
    deltas = [data for event, data in answer.events if event == "delta"]

    assert deltas[0] == {"text": "", "append": False}          # 开流前清屏
    assert len(deltas) > 2
    assert all(d.get("append") is True for d in deltas[1:-1])  # 中间全是分片
    final = deltas[-1]
    assert final["append"] is False and final["passed"] is True
    assert final["text"] == answer.text                        # 终稿=校验后的答案
    assert "".join(d["text"] for d in deltas[1:-1]) == answer.text
    assert answer.timings.get("first_token_ms") is not None
    assert answer.timings["synth_first_token_ms"] < 1000       # 假客户端应瞬时出首字


@pytest.mark.asyncio
async def test_pipeline_marks_rejected_answer_and_puts_banner_on_top():
    def wrong_amount(text: str) -> str:
        return text + "\n按上述标准应付 66000 元。"

    answer = await build(synthesizer=ChunkingSynth(wrong_amount)).run(
        "被裁员能拿多少经济补偿", facts=FACTS
    )
    final = [data for event, data in answer.events if event == "delta"][-1]

    assert final["passed"] is False
    assert "G5_numeric" in answer.text
    # 流式下用户先看到答案，风险提示必须在其上方
    assert answer.text.index("【校验未通过") < answer.text.index("【结论】")


@pytest.mark.asyncio
async def test_pipeline_without_stream_method_still_emits_single_final_delta():
    """模板撰稿器没有 stream，事件契约不能变。"""
    answer = await build().run("被裁员能拿多少经济补偿", facts=FACTS)
    deltas = [data for event, data in answer.events if event == "delta"]
    assert len(deltas) == 1 and deltas[0]["text"] == answer.text


# ---------------------------------------------------------------- SSE 透传
class _ScriptPipeline:
    _agents = {"statute": object()}

    async def run(self, query, *, event_sink=None, **_):
        event_sink("meta", {"intent": "labor_law", "mode": "full", "route": "statute"})
        event_sink("delta", {"text": "【结论】", "append": True})
        event_sink("delta", {"text": "【结论】完整", "append": False, "passed": False})
        event_sink("done", {"latency_ms": 10})
        return Answer(text="【结论】完整")


async def test_sse_forwards_stream_flags():
    app = create_app(pipeline=_ScriptPipeline())
    transport = httpx.ASGITransport(app=app)
    events: list[str] = []
    payloads: list[dict] = []
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/api/chat/stream", json={"query": "裁员补偿"}) as res:
            raw = "".join([chunk async for chunk in res.aiter_text()])

    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        name, data = "", ""
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        if name:
            events.append(name)
            payloads.append(json.loads(data) if data else {})

    assert events == ["meta", "delta", "delta", "done"]
    assert payloads[1]["append"] is True
    assert payloads[2]["append"] is False and payloads[2]["passed"] is False
