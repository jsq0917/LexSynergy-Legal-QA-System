"""稠密检索：bge-m3 (1024 维, COSINE) + 精确余弦；以及 dense ⊕ BM25 的混合检索。

为什么不用 Milvus：语料只有 1,470 个子块，向量矩阵 6MB，np 精确点积 <5ms，
且**结果完全可复现**、零基础设施、能进 CI。上到十万级再换 Milvus，接口不用改。

混合策略用 RRF（k=60）而不是分数加权：余弦相似度与 BM25 分数不同量纲，
加权需要归一化常数，换语料就失效；RRF 只看名次，跨语料稳定。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np

from app.core.logger import get_logger
from app.evidence.models import Evidence
from app.retrieval.corpus import DEFAULT_DIR, Corpus
from app.retrieval.embedder import BgeM3Embedder

logger = get_logger("retrieval.dense")

RRF_K = 60


def rrf_fuse(
    rankings: Sequence[Sequence[int]],
    *,
    weights: Sequence[float] | None = None,
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """多路名次融合：score = Σ w_i / (k + rank)。

    为什么要支持权重：本语料实测等权 RRF（dense 1.0 + bm25 1.0）R@5 0.5786，
    比**纯稠密** 0.6381 还低 —— 弱词项路的噪声会把稠密排好的名次冲散。
    降到 0.3 后回到 0.6167。这说明"混合一定更好"是错的，权重必须按语料标定。
    """
    ws = list(weights) if weights else [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, ws):
        for rank, index in enumerate(ranking, start=1):
            scores[index] = scores.get(index, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


@dataclass
class DenseIndex:
    vectors: np.ndarray
    meta: dict

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "DenseIndex":
        base = Path(directory or DEFAULT_DIR)
        path = base / "dense.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"缺少稠密向量：{path}；先跑 python -m scripts.build_embeddings"
            )
        vectors = np.load(path)
        meta_path = base / "dense_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(vectors=vectors, meta=meta)

    def search(self, query_vec: np.ndarray, *, topk: int = 20) -> list[tuple[int, float]]:
        if self.vectors.size == 0:
            return []
        sims = self.vectors @ query_vec.reshape(-1)
        k = min(topk, sims.shape[0])
        idx = np.argpartition(-sims, k - 1)[:k]
        ranked = sorted(idx.tolist(), key=lambda i: -sims[i])
        return [(i, float(sims[i])) for i in ranked]


class DenseStatuteSearch:
    """`StatuteSearchPort` 的稠密实现。"""

    def __init__(
        self,
        corpus: Corpus | None = None,
        index: DenseIndex | None = None,
        embedder: BgeM3Embedder | None = None,
        *,
        min_score: float = 0.30,
    ) -> None:
        self._corpus = corpus or Corpus.load()
        self._index = index or DenseIndex.load()
        self._embedder = embedder or BgeM3Embedder()
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
        merged: dict[str, Evidence] = {}
        for query in sub_queries:
            vector = await asyncio.to_thread(self._embedder.encode, [query])
            hits = self._index.search(vector[0], topk=topk * 4)
            hits = [(i, s) for i, s in hits if s >= self._min_score]
            for item in self._corpus.merge_hits(hits, topk=topk, law_hints=hints):
                prev = merged.get(item.key)
                if prev is None or item.score > prev.score:
                    merged[item.key] = item
        return sorted(merged.values(), key=lambda e: (-e.priority, -e.score))[:topk]


class HybridStatuteSearch:
    """dense ⊕ BM25，RRF 融合。两路各自召回，名次融合，再回表取父块。"""

    def __init__(
        self,
        corpus: Corpus | None = None,
        index: DenseIndex | None = None,
        embedder: BgeM3Embedder | None = None,
        *,
        candidates: int = 20,
        min_dense: float = 0.30,
        dense_weight: float = 1.0,
        sparse_weight: float = 0.3,
    ) -> None:
        self._corpus = corpus or Corpus.load()
        self._index = index or DenseIndex.load()
        self._embedder = embedder or BgeM3Embedder()
        self._candidates = candidates
        self._min_dense = min_dense
        self._weights = (dense_weight, sparse_weight)

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
            variants = query_variants(query)
            dense_rank: list[int] = []
            sparse_rank: list[int] = []
            sims: dict[int, float] = {}
            for variant in variants:
                vector = await asyncio.to_thread(self._embedder.encode, [variant])
                for i, s in self._index.search(vector[0], topk=self._candidates):
                    if s >= self._min_dense and i not in sims:
                        dense_rank.append(i)
                        sims[i] = s
                for i, _score in self._corpus.bm25.search(variant, topk=self._candidates):
                    if i not in sparse_rank:
                        sparse_rank.append(i)
            fused = rrf_fuse([dense_rank, sparse_rank], weights=self._weights)
            for item in self._corpus.merge_hits(fused, topk=topk, law_hints=hints):
                prev = merged.get(item.key)
                if prev is None or item.score > prev.score:
                    merged[item.key] = item
        return sorted(merged.values(), key=lambda e: (-e.priority, -e.score))[:topk]
