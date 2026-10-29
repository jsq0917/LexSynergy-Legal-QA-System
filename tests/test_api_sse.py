"""SSE 接入层契约：事件顺序、帧格式、终止性、异常兜底。

这里用一个**脚本化的假流水线**，只测接入层本身（帧格式/顺序/终止/异常），
不启模型、不调 LLM。流水线自己的事件顺序由 `tests/test_pipeline.py::test_events_order_is_stable` 覆盖。
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.api.main import create_app
from app.orchestrator.models import Answer

SCRIPT = [
    ("meta", {"intent": "labor_law", "mode": "full", "reasons": ["需要测算（severance）"], "route": "calc"}),
    ("plan", {"steps": [{"id": "s1", "agent": "statute"}, {"id": "c1", "agent": "calc"}]}),
    ("agent", {"step_id": "s1", "agent": "statute", "status": "ok", "elapsed_ms": 12, "evidence": 5, "llm_calls": 0}),
    ("agent", {"step_id": "c1", "agent": "calc", "status": "ok", "elapsed_ms": 1, "evidence": 0, "llm_calls": 0}),
    ("sources", {"sources": [{"eid": "E1", "law": "中华人民共和国劳动合同法", "article_no": "第四十七条", "score": 0.9}]}),
    ("verify", {"passed": True, "action": "pass", "checked": 2, "fails": [], "repaired": [{"from": "a#1", "to": "a#2", "quote": "x"}]}),
    ("delta", {"text": "【结论】经济补偿约为 72000.00 元。"}),
    ("done", {"latency_ms": 1234, "llm_calls": 3, "timings": {"triage_ms": 900, "dag_ms": 5, "synth_ms": 300}, "mode": "full"}),
]


class FakePipeline:
    """按脚本发事件；`boom=True` 时抛异常，用来验证 error 兜底。"""

    _agents = {"statute": object(), "calc": object(), "verifier": object()}

    def __init__(self, *, boom: bool = False) -> None:
        self.boom = boom

    async def run(self, query, *, session_id="", facts=None, as_of=None, event_sink=None, **_):
        if self.boom:
            raise RuntimeError("模拟链路异常")
        for event, data in SCRIPT:
            event_sink(event, data)
        return Answer(text="ok", mode="full")


async def _collect(pipeline) -> tuple[list[str], list[dict], str]:
    app = create_app(pipeline=pipeline)
    transport = httpx.ASGITransport(app=app)
    events: list[str] = []
    payloads: list[dict] = []
    raw_all = ""
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/api/chat/stream", json={"query": "裁员补偿怎么算"}) as res:
            assert res.status_code == 200
            assert res.headers["content-type"].startswith("text/event-stream")
            assert res.headers.get("x-accel-buffering") == "no"
            async for chunk in res.aiter_text():
                raw_all += chunk
    for block in raw_all.split("\n\n"):
        if not block.strip():
            continue
        name = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        if name:
            events.append(name)
            payloads.append(json.loads(data) if data else {})
    return events, payloads, raw_all


@pytest.mark.asyncio
async def test_sse_event_order_and_termination():
    events, payloads, _ = await _collect(FakePipeline())
    assert events == [e for e, _ in SCRIPT]          # 顺序与流水线一致
    assert events[-1] == "done"                      # 必须能终止（否则前端一直转圈）
    meta = payloads[0]
    assert meta["mode"] == "full" and meta["reasons"]


@pytest.mark.asyncio
async def test_sse_frames_are_well_formed():
    _, _, raw = await _collect(FakePipeline())
    for block in [b for b in raw.split("\n\n") if b.strip()]:
        assert block.startswith("event: "), block[:40]
        assert "\ndata: " in block


@pytest.mark.asyncio
async def test_sse_verify_payload_carries_repairs():
    """前端要靠 repaired/no_rewrite 展示"校验改了什么"，字段不能在半路丢。"""
    events, payloads, _ = await _collect(FakePipeline())
    verify = payloads[events.index("verify")]
    assert verify["passed"] is True
    assert verify["repaired"][0]["to"] == "a#2"


@pytest.mark.asyncio
async def test_sse_reports_error_and_stops():
    """链路炸了也必须有终止事件，否则前端永远等下去。"""
    events, payloads, _ = await _collect(FakePipeline(boom=True))
    assert events == ["error"]
    assert payloads[0]["code"] == "internal_error"


@pytest.mark.asyncio
async def test_index_serves_frontend():
    app = create_app(pipeline=FakePipeline())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/")
        assert res.status_code == 200
        assert "LexSynergy" in res.text
        static = await client.get("/static/app.js")
        assert static.status_code == 200
