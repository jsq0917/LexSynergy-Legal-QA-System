"""法条检索专家：**不含 LLM**，只交证据、不写结论。

为什么无 LLM 也算 Agent：判据是"能力边界 + 可独立部署 + 稳定契约"，不是"有没有大模型"。
它只做一件确定的事——把查询变成 Evidence[]，因此永远不会编故事。
"""

from __future__ import annotations

import time
from datetime import date

from app.agents.base import AgentResult, BaseAgent
from app.core.logger import get_logger
from app.retrieval.port import StatuteSearchPort

logger = get_logger("agent.statute")


class StatuteAgent(BaseAgent):
    name = "statute"
    description = "法条检索专家：按子查询检索中国法律条文，返回法名、条号与原文，不做结论。"
    skills = (
        {
            "id": "search_statute",
            "name": "检索法条",
            "description": "输入自然语言子查询与法域提示，返回最相关法条的原文与出处",
            "tags": ["legal", "retrieval"],
            "examples": [
                "裁员的经济补偿怎么算",
                "试用期最长可以约定多久",
                "未签书面劳动合同的二倍工资",
            ],
        },
    )

    def __init__(self, search: StatuteSearchPort, *, topk: int = 5) -> None:
        self._search = search
        self._topk = topk

    async def run(self, payload: dict) -> AgentResult:
        started = time.perf_counter()
        queries = list(payload.get("sub_queries") or [])
        if not queries and payload.get("sub_query"):
            queries = [str(payload["sub_query"])]
        # **改写是追加，不是替换**：只检索改写后的子查询会丢召回
        # （实测 q23/q25/q33/q34 丢掉了原问题本来能命中的条）。
        # 原问题一并检索，合并去重后仍按 top-k 截断 —— 扩大候选集而不改变计划形状。
        original = str(payload.get("query") or "").strip()
        if original and original not in queries:
            queries.append(original)
        if not queries:
            return AgentResult.input_required(self.name, ["sub_query"])

        hints = payload.get("law_hints") or None
        as_of_raw = payload.get("as_of")
        as_of = date.fromisoformat(as_of_raw) if as_of_raw else None
        topk = int(payload.get("topk") or self._topk)

        evidence = await self._search.search_many(
            queries, law_hints=hints, as_of=as_of, topk=topk
        )
        elapsed = int((time.perf_counter() - started) * 1000)
        if not evidence:
            logger.info("无命中 queries=%s hints=%s", queries, hints)
            return AgentResult.no_data(self.name, "未检索到相关法条")

        logger.info("命中 %d 条 queries=%s", len(evidence), queries)
        return AgentResult.success(
            self.name,
            evidence=evidence,
            artifact={"count": len(evidence), "queries": queries},
            elapsed_ms=elapsed,
            llm_calls=0,
        )
