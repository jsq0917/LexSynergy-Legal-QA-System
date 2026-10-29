"""HTTP 接入层：SSE 流式 + 非流式问答 + Agent 能力发现。

SSE 事件与 Pipeline 内部事件一一对应（meta → plan → agent×N → sources → verify → delta → done），
前端靠 `agent` 事件画"协作时间线"，靠 `verify` 事件标红未通过项。
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.core.logger import get_logger
from app.runtime import build_pipeline
from config import get_settings

logger = get_logger("api")

#: 前端静态目录（零构建：一个 html + 一个 js + 一个 css，浏览器直接跑）
WEB_DIR = Path(__file__).resolve().parents[2] / "web"

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def sse_frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    facts: dict | None = Field(default=None, description="结构化事实，如月工资/起止日期")
    as_of: str | None = Field(default=None, description="事发日期 YYYY-MM-DD，用于时效校验")
    session_id: str = ""


async def _warmup(pipeline: Any) -> None:
    """服务启动后台预热检索模型。

    **不做这件事的代价是真实的**：bge-m3 冷启动要 ~6-20s，第一次请求会把这笔时间
    算进检索步的 20s 超时里，于是首个用户拿到"检索超时"——把基础设施的冷启动
    伪装成"没有相关资料"，是最不该出现的一类错误。
    只预热检索（零 LLM 调用），不动生成链路。
    """
    agent = getattr(pipeline, "_agents", {}).get("statute")
    if agent is None:
        return
    started = time.perf_counter()
    try:
        await agent.run({"sub_query": "试用期最长可以约定多久"})
        logger.info("检索预热完成 %.1fs", time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 - 预热失败不影响服务启动
        logger.warning("检索预热失败（不影响启动）：%s", str(exc)[:120])


def create_app(pipeline: Any | None = None) -> FastAPI:
    """`pipeline` 可注入：测试用一个脚本化的假流水线验证 SSE 契约，不启模型。"""
    settings = get_settings()
    pipeline = pipeline or build_pipeline()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = asyncio.create_task(_warmup(pipeline))
        yield
        if not task.done():
            task.cancel()

    app = FastAPI(
        title="LexSynergy",
        version="0.1.0",
        description="法律合规多智能体专家问答系统",
        lifespan=lifespan,
    )

    @app.get("/")
    async def index() -> Any:
        page = WEB_DIR / "index.html"
        if not page.exists():
            return {"message": "web/index.html 不存在"}
        return FileResponse(page)

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok", "agents": sorted(pipeline._agents), "transport": settings.agent_transport}  # noqa: SLF001

    @app.get("/api/agents")
    async def agents() -> list[dict]:
        return [
            agent.card(settings.agent_url(name)).model_dump(mode="json")
            for name, agent in pipeline._agents.items()  # noqa: SLF001
        ]

    def _as_of(raw: str | None) -> date | None:
        return date.fromisoformat(raw) if raw else None

    @app.post("/api/chat")
    async def chat(req: ChatRequest) -> dict:
        answer = await pipeline.run(
            req.query, session_id=req.session_id, facts=req.facts, as_of=_as_of(req.as_of)
        )
        return answer.to_dict()

    @app.post("/api/chat/stream")
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        queue: asyncio.Queue[tuple[str, dict] | None] = asyncio.Queue()

        def sink(event: str, data: dict) -> None:
            queue.put_nowait((event, data))

        async def run() -> None:
            try:
                await pipeline.run(
                    req.query,
                    session_id=req.session_id,
                    facts=req.facts,
                    as_of=_as_of(req.as_of),
                    event_sink=sink,
                )
            except Exception as exc:  # noqa: BLE001 - 顶层兜底：SSE 必须有终止事件
                logger.exception("问答链路异常")
                sink("error", {"code": "internal_error", "message": str(exc)[:200], "retriable": False})

        async def frames() -> AsyncIterator[str]:
            task = asyncio.create_task(run())
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    event, data = item
                    yield sse_frame(event, data)
                    if event in {"done", "error"}:
                        break
            finally:
                if not task.done():
                    task.cancel()

        return StreamingResponse(frames(), media_type="text/event-stream", headers=SSE_HEADERS)

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    return app


def __getattr__(name: str) -> Any:
    """惰性构造：`import app.api.main` 不该顺手建流水线（会拉起检索/模型依赖）。

    `uvicorn app.api.main:app` 走属性访问，仍然拿得到；测试直接调 `create_app(pipeline=...)`，
    导入成本为零。
    """
    if name == "app":
        return create_app()
    raise AttributeError(name)
