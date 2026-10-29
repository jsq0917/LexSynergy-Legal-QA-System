"""LLM 版撰稿器：把已核实的证据组织成六段式答案。

与模板版的关系不是"替换"而是"可切换"：两者签名相同、都从 EvidenceLedger 取数，
所以评测跑的是同一条链路，只是撰稿器换人。LLM 失败时**自动退回模板版**，
保证链路不因一次模型抽风而输出空答案。

阶段 8 起支持流式：`stream()` 每收到一段就回调 on_delta，返回值仍是完整文本，
供上游做引文对位与五道闸。**流式只改变呈现时机，不改变校验口径** ——
最终 `delta` 事件会用校验后的完整文本覆盖已流出的草稿。
"""

from __future__ import annotations

import asyncio
import threading
from typing import AsyncIterator, Callable

from app.core.logger import get_logger
from app.evidence.ledger import EvidenceLedger
from app.generation.llm_client import DashScopeClient, Usage, get_llm_client
from app.generation.prompts import synthesis_messages
from app.orchestrator.context import TaskContext
from app.orchestrator.pipeline import DISCLAIMER, template_synthesizer
from config import Settings, get_settings

logger = get_logger("synthesizer")

#: 文本增量回调（由 Pipeline 接到 `delta` SSE 事件）
DeltaSink = Callable[[str], None]


class LLMSynthesizer:
    def __init__(
        self,
        client: DashScopeClient | None = None,
        settings: Settings | None = None,
        *,
        max_tokens: int = 1200,
    ) -> None:
        self._s = settings or get_settings()
        self._client = client or get_llm_client(self._s)
        self._max_tokens = max_tokens

    async def __call__(
        self, ctx: TaskContext, ledger: EvidenceLedger, calc: list[dict]
    ) -> str:
        """非流式入口：CLI / 单测 / 不支持流式的客户端走这里。"""
        return await self._generate(ctx, ledger, calc, on_delta=None)

    async def stream(
        self,
        ctx: TaskContext,
        ledger: EvidenceLedger,
        calc: list[dict],
        on_delta: DeltaSink,
    ) -> str:
        """流式入口：逐段回调文本增量，返回值仍是完整文本（供校验）。"""
        return await self._generate(ctx, ledger, calc, on_delta=on_delta)

    # ------------------------------------------------------------------
    async def _generate(
        self,
        ctx: TaskContext,
        ledger: EvidenceLedger,
        calc: list[dict],
        *,
        on_delta: DeltaSink | None,
    ) -> str:
        evidence_block = ledger.for_prompt(
            as_of=ctx.triage.as_of if ctx.triage else None,
            budget_chars=self._s.evidence_budget_chars,
        )
        if not evidence_block and not calc:
            return template_synthesizer(ctx, ledger, calc)

        messages = synthesis_messages(
            ctx.query or (ctx.triage.query if ctx.triage else ""),
            evidence_block,
            calc,
            as_of=ctx.triage.as_of if ctx.triage else None,
        )
        if on_delta is not None and hasattr(self._client, "chat_stream"):
            return await self._stream(ctx, ledger, calc, messages, on_delta)
        return await self._chat(ctx, ledger, calc, messages)

    async def _chat(
        self,
        ctx: TaskContext,
        ledger: EvidenceLedger,
        calc: list[dict],
        messages: list[dict],
    ) -> str:
        try:
            result = await asyncio.to_thread(
                self._client.chat,
                messages,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - 撰稿失败退回模板，绝不空答
            logger.warning("LLM 撰稿失败，回退模板：%s", str(exc)[:160])
            return template_synthesizer(ctx, ledger, calc)

        self._record_usage(ctx, result.usage)
        return self._finish(result.text)

    async def _stream(
        self,
        ctx: TaskContext,
        ledger: EvidenceLedger,
        calc: list[dict],
        messages: list[dict],
        on_delta: DeltaSink,
    ) -> str:
        parts: list[str] = []
        usage = Usage()
        emitted = 0
        try:
            async for delta, chunk_usage in _stream_chunks(
                self._client, messages, self._max_tokens
            ):
                if chunk_usage is not None:
                    usage.merge(chunk_usage)
                if delta:
                    parts.append(delta)
                    emitted += len(delta)
                    on_delta(delta)
        except Exception as exc:  # noqa: BLE001 - 流中断也退回模板，终稿会覆盖已流出内容
            logger.warning(
                "LLM 流式撰稿失败（已输出 %d 字），回退模板：%s", emitted, str(exc)[:160]
            )
            ctx.trace["synth_stream_error"] = type(exc).__name__
            return template_synthesizer(ctx, ledger, calc)

        self._record_usage(ctx, usage)
        text = "".join(parts).strip()
        if not text:
            logger.warning("LLM 流式撰稿返回空，回退模板")
            return template_synthesizer(ctx, ledger, calc)
        return self._finish(text)

    def _record_usage(self, ctx: TaskContext, usage: Usage) -> None:
        ctx.llm_calls += 1
        ctx.trace["synthesis_usage"] = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        }

    @staticmethod
    def _finish(text: str) -> str:
        text = text.strip()
        if DISCLAIMER not in text:
            text = f"{text}\n\n{DISCLAIMER}"
        return text


async def _stream_chunks(
    client: DashScopeClient,
    messages: list[dict],
    max_tokens: int,
) -> AsyncIterator[tuple[str, Usage | None]]:
    """把同步 SSE 迭代器搬进事件循环。

    为什么必须开线程：httpx 的 `iter_lines()` 是同步阻塞的，直接 await 会卡住
    整个事件循环（其它请求、心跳全部停摆）。这里用一个守护线程读流，
    再通过线程安全队列交给 asyncio —— 每来一段就立刻上抛，首字延迟才不会被
    "攒满再发"吃掉。

    取消语义：等待方（`asyncio.wait_for`）被取消时线程可能仍在读，但它受
    `llm_timeout_s` 约束、且不再有人消费队列，不会让请求继续变慢。
    """

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    errors: list[BaseException] = []

    def _push(item: object) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:
            # 事件循环已关闭（请求被取消 / 服务退出）：无人消费，直接丢弃
            pass

    def _produce() -> None:
        try:
            for item in client.chat_stream(messages, max_tokens=max_tokens):
                _push(item)
        except BaseException as exc:  # noqa: BLE001 - 线程内异常原样抛回事件循环
            errors.append(exc)
        finally:
            _push(None)

    threading.Thread(target=_produce, name="llm-stream", daemon=True).start()

    while True:
        item = await queue.get()
        if item is None:
            break
        yield item
    if errors:
        raise errors[0]
