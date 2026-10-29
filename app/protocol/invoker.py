"""两种调用方式：进程内直调（测试/单机）与 HTTP A2A（多服务/演示）。

同一个 `Pipeline` 通过注入 Invoker 切换传输，业务代码零改动——这就是"协议边界"
与"可测性"能同时拿到的原因。
"""

from __future__ import annotations

import uuid
from typing import Protocol

from app.agents.base import AgentResult, BaseAgent
from app.core.errors import AgentFailed
from app.core.logger import get_logger

logger = get_logger("invoker")


class Invoker(Protocol):
    async def call(self, agent: str, payload: dict) -> AgentResult: ...


class InProcInvoker:
    """进程内直调：单测与 CI 走这条，不启任何容器、不碰网络。"""

    def __init__(self, agents: dict[str, BaseAgent]) -> None:
        self._agents = dict(agents)

    async def call(self, agent: str, payload: dict) -> AgentResult:
        target = self._agents.get(agent)
        if target is None:
            return AgentResult.failed(agent, "未注册的 agent")
        return await target.run(payload)


class A2AHttpInvoker:
    """HTTP A2A：服务化部署时用。全 async，不存在同步框架的事件循环冲突。"""

    def __init__(self, base_urls: dict[str, str], *, timeout_s: float = 30.0) -> None:
        self._urls = dict(base_urls)
        self._timeout = timeout_s

    async def call(self, agent: str, payload: dict) -> AgentResult:
        url = self._urls.get(agent)
        if not url:
            return AgentResult.failed(agent, f"未配置 {agent} 的服务地址")

        import httpx
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import (
            DataPart,
            Message,
            MessageSendParams,
            Part,
            Role,
            SendMessageRequest,
        )

        from app.protocol.executor import result_of_parts

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                card = await A2ACardResolver(client, url).get_agent_card()
                a2a = A2AClient(client, agent_card=card)
                message = Message(
                    message_id=uuid.uuid4().hex,
                    role=Role.user,
                    parts=[Part(root=DataPart(data=payload))],
                )
                response = await a2a.send_message(
                    SendMessageRequest(id=uuid.uuid4().hex, params=MessageSendParams(message=message))
                )
                return self._extract(agent, response, result_of_parts)
        except Exception as exc:  # noqa: BLE001 - 网络/协议错误统一收敛成 AgentFailed
            logger.warning("调用 %s(%s) 失败：%s", agent, url, str(exc)[:160])
            raise AgentFailed(f"{agent} 调用失败：{exc}") from exc

    @staticmethod
    def _extract(agent: str, response, parser) -> AgentResult:
        root = getattr(response, "root", response)
        result = getattr(root, "result", None)
        if result is None:
            return AgentResult.failed(agent, "A2A 响应为空")
        parts = getattr(result, "parts", None)
        if parts is None:  # Task 形态：从 artifacts 里捞
            parts = [p for artifact in (getattr(result, "artifacts", None) or []) for p in artifact.parts]
        parsed = parser(parts)
        return parsed or AgentResult.failed(agent, "响应中不含结构化产物")
