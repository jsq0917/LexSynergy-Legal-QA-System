"""协议层：AgentCard 可发现、DataPart 往返不丢字段、Executor 真能跑完一次任务。"""

from __future__ import annotations

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import DataPart, Message, MessageSendParams, Part, Role, TaskState

from app.agents.base import AgentResult
from app.agents.calc_agent import CalcAgent
from app.evidence.models import Evidence
from app.protocol.executor import A2AExecutor, payload_of, result_of_parts

URL = "http://127.0.0.1:5102"


class RecordingQueue(EventQueue):
    """只记录不消费：避免测试里跟 SDK 的队列生命周期纠缠。"""

    def __init__(self) -> None:
        super().__init__()
        self.events: list = []

    async def enqueue_event(self, event) -> None:  # type: ignore[override]
        self.events.append(event)


def _message(payload: dict | None = None, text: str = "") -> Message:
    parts = (
        [Part(root=DataPart(data=payload))] if payload else [Part(text=text)]
    )
    return Message(message_id="m-1", role=Role.user, parts=parts)


def _context(message: Message) -> RequestContext:
    return RequestContext(
        request=MessageSendParams(message=message), task_id="t-1", context_id="c-1"
    )


def test_card_exposes_skills_and_examples():
    card = CalcAgent().card(URL)
    assert card.name == "calc"
    assert card.url == URL
    assert card.skills and card.skills[0].examples
    assert card.default_input_modes and card.default_output_modes


def test_payload_from_data_part():
    payload = payload_of(_context(_message({"kinds": ["severance"]})))
    assert payload == {"kinds": ["severance"]}


def test_payload_from_text_falls_back_to_sub_query():
    assert payload_of(_context(_message(text="裁员补偿怎么算"))) == {"sub_query": "裁员补偿怎么算"}


def test_payload_from_json_text():
    assert payload_of(_context(_message(text='{"sub_query": "试用期"}'))) == {"sub_query": "试用期"}


def test_result_roundtrip_preserves_evidence():
    original = AgentResult.success(
        "calc",
        artifact={"results": [{"kind": "severance", "value": "72000.00"}]},
        evidence=[Evidence(law="中华人民共和国劳动合同法", article_no="第四十七条", quote="经济补偿…")],
        llm_calls=0,
    )
    part = Part(root=DataPart(data=original.to_dict()))
    restored = result_of_parts([part])
    assert restored is not None
    assert restored.artifact["results"][0]["value"] == "72000.00"
    assert restored.evidence[0].article_no == "第四十七条"


@pytest.mark.asyncio
async def test_executor_runs_task_and_emits_artifact():
    agent = CalcAgent()
    queue = RecordingQueue()
    ctx = _context(
        _message(
            {
                "kinds": ["severance"],
                "facts": {
                    "monthly_wage": 12000,
                    "start_date": "2021-03-01",
                    "end_date": "2026-10-20",
                },
            }
        )
    )
    await A2AExecutor(agent).execute(ctx, queue)

    artifacts = [e.artifact for e in queue.events if hasattr(e, "artifact")]
    assert artifacts, "没有产出 artifact"
    restored = result_of_parts(artifacts[0].parts)
    assert restored is not None
    assert restored.artifact["results"][0]["value"] == "72000.00"


@pytest.mark.asyncio
async def test_executor_reports_input_required():
    agent = CalcAgent()
    queue = RecordingQueue()
    ctx = _context(_message({"kinds": ["severance"], "facts": {}}))
    await A2AExecutor(agent).execute(ctx, queue)

    states = [e.status.state for e in queue.events if hasattr(e, "status")]
    assert TaskState("input-required") in states
