"""A2A 协议适配：约 40 行，把业务 Agent 挂到官方 SDK 上。

刻意保持极薄：AgentExecutor 里**不做任何业务判断**，只做
「解析 payload → 调 agent.run() → 把 AgentResult 塞进 DataPart」。
因此换协议（A2A/MCP/HTTP）只需换这一层，业务代码一行不动。
"""

from __future__ import annotations

import json
import uuid

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Part, TaskState
from a2a.utils import get_data_parts, get_message_text, new_agent_text_message

from app.agents.base import AgentResult, BaseAgent
from app.core.logger import get_logger

logger = get_logger("protocol")

_COMPLETED = TaskState("completed")
_INPUT_REQUIRED = TaskState("input-required")
_FAILED = TaskState("failed")


def payload_of(context: RequestContext) -> dict:
    """Message → dict：优先取 DataPart，其次把文本当 sub_query。"""
    message = context.message
    if message is None:
        return {}
    # get_data_parts 已把 DataPart 解成 dict 列表
    for data in get_data_parts(message.parts) or []:
        if isinstance(data, dict):
            return data
    text = (get_message_text(message) or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"sub_query": text}
    except json.JSONDecodeError:
        return {"sub_query": text}


class A2AExecutor(AgentExecutor):
    def __init__(self, agent: BaseAgent) -> None:
        self._agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or uuid.uuid4().hex
        context_id = context.context_id or uuid.uuid4().hex
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()

        try:
            payload = payload_of(context)
        except Exception as exc:  # noqa: BLE001 - 协议层不做业务，但要把错说清楚
            await updater.failed(message=new_agent_text_message(f"请求解析失败：{exc}", context_id, task_id))
            return

        try:
            result = await self._agent.run(payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s 执行异常", self._agent.name)
            await updater.failed(
                message=new_agent_text_message(f"{type(exc).__name__}: {exc}", context_id, task_id)
            )
            return

        await updater.add_artifact(
            parts=[Part(root=DataPart(data=result.to_dict()))],
            name=f"{self._agent.name}-result",
        )
        if result.status == "input_required":
            await updater.update_status(
                _INPUT_REQUIRED,
                message=new_agent_text_message(
                    f"缺少必要信息：{'、'.join(result.missing_slots)}", context_id, task_id
                ),
            )
        elif result.status in {"failed", "timeout"}:
            await updater.update_status(
                _FAILED, message=new_agent_text_message(result.error or "执行失败", context_id, task_id)
            )
        else:
            await updater.update_status(_COMPLETED)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("本项目的专家 Agent 均为短任务，不支持取消")


def result_of_parts(parts: list) -> AgentResult | None:
    """从响应 parts 里还原 AgentResult（客户端侧用）。"""
    for part in parts or []:
        root = getattr(part, "root", part)
        if isinstance(root, DataPart) and isinstance(root.data, dict):
            data = root.data
            if data.get("schema") == "agent-result/1" or "status" in data:
                return AgentResult.from_dict(data)
    return None
