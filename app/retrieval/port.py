"""法条检索的端口（Port）与一个可直接跑通的内存实现。

真实实现是 LawAnswer 的混合检索（bge-m3 双向量 + RRF + reranker + 父块回填），
在 P0 阶段通过 `LawAnswerSearchAdapter` 接进来；此处只定义契约 + 离线替身，
保证编排层与 Agent 可以完全脱离 Milvus 开发与测试。
"""

from __future__ import annotations

import re
from datetime import date
from typing import Protocol, Sequence, runtime_checkable

from app.evidence.models import Evidence

_GRAM = re.compile(r"[\u4e00-\u9fa5]{2}")


def _grams(text: str) -> set[str]:
    return set(_GRAM.findall(text or ""))


@runtime_checkable
class StatuteSearchPort(Protocol):
    async def search_many(
        self,
        sub_queries: Sequence[str],
        *,
        law_hints: Sequence[str] | None = None,
        kb_ids: Sequence[str] | None = None,
        as_of: date | None = None,
        topk: int = 5,
    ) -> list[Evidence]: ...


class InMemoryStatuteSearch:
    """字符二元组重叠打分：确定性、零依赖，用来跑通链路与写单测。

    它不是"简化版 RAG"，而是一个**假实现**：接口一样、行为可预测，
    真实检索的召回质量由 LawAnswer 的评测数据负责。
    """

    def __init__(self, corpus: Sequence[Evidence], *, min_score: float = 0.18) -> None:
        self._corpus = list(corpus)
        self._min_score = min_score

    async def search_many(
        self,
        sub_queries: Sequence[str],
        *,
        law_hints: Sequence[str] | None = None,
        kb_ids: Sequence[str] | None = None,
        as_of: date | None = None,
        topk: int = 5,
    ) -> list[Evidence]:
        hints = list(law_hints or [])
        scored: list[Evidence] = []
        for query in sub_queries:
            q_grams = _grams(query)
            if not q_grams:
                continue
            for item in self._corpus:
                if hints and item.law not in hints:
                    continue
                if not item.effective_at(as_of):
                    continue
                target = _grams(f"{item.law}{item.article_no}{item.quote}")
                overlap = len(q_grams & target)
                score = overlap / max(1, len(q_grams))
                if score >= self._min_score:
                    scored.append(
                        Evidence(
                            law=item.law,
                            article_no=item.article_no,
                            quote=item.quote,
                            chunk_id=item.chunk_id,
                            chapter_path=item.chapter_path,
                            score=round(score, 4),
                            from_agent="statute",
                            effective_from=item.effective_from,
                            effective_to=item.effective_to,
                            priority=item.priority,
                        )
                    )

        # 同一法条多子查询命中：取最高分，按分降序
        best: dict[str, Evidence] = {}
        for item in scored:
            prev = best.get(item.key)
            if prev is None or item.score > prev.score:
                best[item.key] = item
        ranked = sorted(best.values(), key=lambda e: (-e.priority, -e.score))
        return ranked[: max(1, topk)]


class LawAnswerSearchAdapter:
    """P0 接线位：把 LawAnswer 的 retrieval 包成本端口。

    真实实现应调用 ``app.retrieval.service.retrieve`` / ``retrieve_multi``，
    再把 ``Hit`` 映射成 ``Evidence``（quote 取父块全文，保证 G2 逐字比对有基准）。
    当前保留显式 NotImplemented，避免"看起来接上了其实没接"。
    """

    def __init__(self, *, collection: str = "law_chunks") -> None:
        self._collection = collection

    async def search_many(
        self,
        sub_queries: Sequence[str],
        *,
        law_hints: Sequence[str] | None = None,
        kb_ids: Sequence[str] | None = None,
        as_of: date | None = None,
        topk: int = 5,
    ) -> list[Evidence]:
        raise NotImplementedError(
            "P0：从 LawAnswer 复制 ingest/retrieval 后，在此调用 retrieve_multi() 并映射为 Evidence"
        )
