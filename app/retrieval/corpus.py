"""语料索引：加载 chunks.jsonl，建 BM25，对外提供 `StatuteSearchPort` 实现。

检索单元是**子块**（精度），返回的证据正文是**父块整条**（完整性）——
这是 LawAnswer 父子分块的核心取舍，在这里原样保留：
子块只用来定位，进 Prompt 和进 G2 逐字比对的必须是整条法条。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from app.core.logger import get_logger
from app.evidence.models import Evidence
from app.retrieval.bm25 import BM25

logger = get_logger("retrieval.corpus")

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"


@dataclass
class Corpus:
    children: list[dict]
    parents: dict[str, dict]
    bm25: BM25
    manifest: dict

    @property
    def laws(self) -> list[str]:
        return sorted({c["law_name"] for c in self.children})

    @property
    def article_count(self) -> int:
        return len(self.parents)

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "Corpus":
        return _load(str(directory or DEFAULT_DIR))

    # ---------------- 命中 → 证据 ----------------
    def evidence_at(self, index: int, score: float) -> Evidence | None:
        """第 index 个子块 → 证据（正文取父块整条，G2 的逐字基准）。"""
        child = self.children[index]
        parent = self.parents.get(child["parent_id"])
        if parent is None:
            return None
        return Evidence(
            law=child["law_name"],
            article_no=child["article_no"],
            quote=parent["content"],
            chunk_id=child["chunk_id"],
            chapter_path=child.get("chapter_path", ""),
            score=round(float(score), 4),
            from_agent="statute",
            priority=20 if "劳动合同法" in child["law_name"] else 10,
        )

    def merge_hits(
        self,
        hits: Sequence[tuple[int, float]],
        *,
        topk: int,
        law_hints: Sequence[str] = (),
    ) -> list[Evidence]:
        """多路命中合并：按 `法名#条号` 去重取最高分，法域过滤后截断。"""
        hints = {h for h in law_hints if h}
        best: dict[str, Evidence] = {}
        for index, score in hits:
            if index < 0 or index >= len(self.children):
                continue
            child = self.children[index]
            if hints and child["law_name"] not in hints:
                continue
            item = self.evidence_at(index, score)
            if item is None:
                continue
            prev = best.get(item.key)
            if prev is None or item.score > prev.score:
                best[item.key] = item
        # 以相关度为主序，特别法优先只作为同分时的微弱 tie-break。
        # 早期版本把 priority 放在首位，导致跨法查询里"劳动合同法 0.3 分"压过
        # "民法典 0.6 分"，民法典题的召回被系统性拉低 —— 这是个真 bug，不是取舍。
        return sorted(best.values(), key=lambda e: (-e.score, -e.priority))[:topk]

    def search(self, query: str, *, topk: int = 5, law_hints: Sequence[str] = ()) -> list[Evidence]:
        """BM25 单路检索（保留给基线/消融用）。"""
        hits = self.bm25.search(query, topk=topk * 4) if query.strip() else []
        if not hits:
            return []
        max_score = hits[0][1] or 1.0
        normalized = [(idx, score / max_score) for idx, score in hits]
        return self.merge_hits(normalized, topk=topk, law_hints=law_hints)


@lru_cache(maxsize=4)
def _load(directory: str) -> Corpus:
    base = Path(directory)
    chunks_path = base / "chunks.jsonl"
    manifest_path = base / "manifest.json"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"缺少语料索引：{chunks_path}；先跑 python -m scripts.build_corpus"
        )
    children: list[dict] = []
    parents: dict[str, dict] = {}
    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["chunk_type"] == "parent":
                parents[row["chunk_id"]] = row
            else:
                children.append(row)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    docs = [f"{c['title']}\n{c['content']}" for c in children]
    corpus = Corpus(children=children, parents=parents, bm25=BM25(docs), manifest=manifest)
    logger.info(
        "语料加载完成：%d 条法条 / %d 个子块 / %d 部法律",
        corpus.article_count,
        len(children),
        len(corpus.laws),
    )
    return corpus


class Bm25StatuteSearch:
    """`StatuteSearchPort` 的离线实现 —— 真实语料 + 词项检索（可选 Query 扩展）。"""

    def __init__(
        self,
        corpus: Corpus | None = None,
        *,
        min_score: float = 0.05,
        use_expansion: bool = True,
    ) -> None:
        self._corpus = corpus or Corpus.load()
        self._min_score = min_score
        self._use_expansion = use_expansion

    async def search_many(
        self,
        sub_queries: Sequence[str],
        *,
        law_hints: Sequence[str] | None = None,
        kb_ids: Sequence[str] | None = None,
        as_of: date | None = None,
        topk: int = 5,
    ) -> list[Evidence]:
        from app.retrieval.expand import query_variants

        hints = list(law_hints or [])
        merged: dict[str, Evidence] = {}
        for query in sub_queries:
            variants = query_variants(query) if self._use_expansion else [query]
            for variant in variants:
                for item in self._corpus.search(variant, topk=topk, law_hints=hints):
                    if item.score < self._min_score:
                        continue
                    prev = merged.get(item.key)
                    if prev is None or item.score > prev.score:
                        merged[item.key] = item
        return sorted(merged.values(), key=lambda e: (-e.priority, -e.score))[:topk]
