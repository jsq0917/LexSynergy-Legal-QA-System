"""把 Agent 挂成 A2A 服务：一个 Agent 一个进程，端口由配置决定。

启动后可用 `curl http://127.0.0.1:5101/.well-known/agent-card.json` 自证能力可发现。
"""

from __future__ import annotations

from app.agents.base import BaseAgent
from app.core.logger import get_logger
from app.protocol.executor import A2AExecutor

logger = get_logger("protocol.server")


def build_app(agent: BaseAgent, *, url: str):
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore

    handler = DefaultRequestHandler(
        agent_executor=A2AExecutor(agent),
        task_store=InMemoryTaskStore(),
    )
    return A2AStarletteApplication(agent_card=agent.card(url), http_handler=handler).build()


def serve(agent: BaseAgent, *, port: int, host: str = "127.0.0.1") -> None:
    import uvicorn

    url = f"http://{host}:{port}"
    logger.info("启动 %s 于 %s", agent.name, url)
    uvicorn.run(build_app(agent, url=url), host=host, port=port, log_level="info")
