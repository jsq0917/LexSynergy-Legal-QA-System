"""检索层：分词 / BM25 / Query 扩展 / 语料合并 / RRF 融合。全部离线。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.evidence.models import Evidence
from app.retrieval.bm25 import BM25, tokenize
from app.retrieval.corpus import Corpus
from app.retrieval.dense import DenseIndex, rrf_fuse
from app.retrieval.expand import expand_terms, query_variants, strip_question_noise

LCL = "中华人民共和国劳动合同法"
CIVIL = "中华人民共和国民法典"


# ---------------------------------------------------------------- 分词 / BM25
def test_tokenize_cjk_bigrams_and_alnum():
    tokens = tokenize("试用期 3 个月 N+1")
    assert "试用" in tokens and "用期" in tokens
    assert "3" in tokens
    assert all(len(t) <= 2 for t in tokens if t.isascii() is False)


def test_bm25_ranks_matching_doc_first():
    docs = ["试用期不得超过一个月", "经济补偿按工作年限计算", "竞业限制期限不得超过二年"]
    bm25 = BM25(docs)
    hits = bm25.search("竞业限制几年", topk=3)
    assert hits and hits[0][0] == 2


def test_bm25_empty_query():
    assert BM25(["abc"]).search("") == []


# ---------------------------------------------------------------- Query 扩展
def test_strip_question_noise():
    assert "多久" not in strip_question_noise("试用期最长可以约定多久？")
    assert "试用期" in strip_question_noise("试用期最长可以约定多久？")


def test_expand_terms_maps_colloquial_to_statute():
    assert "解除劳动合同" in expand_terms("老板把我辞退了")
    assert "延长工作时间" in expand_terms("加班费怎么算")
    assert "二年" in expand_terms("竞业限制最长几年")


def test_query_variants_keeps_original_first():
    variants = query_variants("老板把我辞退了怎么办")
    assert variants[0] == "老板把我辞退了怎么办"
    assert len(variants) >= 2
    assert len(set(variants)) == len(variants)


def test_query_variants_no_expansion_returns_original_only():
    assert query_variants("甲方乙方") == ["甲方乙方"]


# ---------------------------------------------------------------- 语料合并
@pytest.fixture()
def tiny_corpus(tmp_path: Path) -> Corpus:
    rows = [
        {
            "chunk_id": "d1:p19", "parent_id": "d1:p19", "doc_id": "d1", "kb_id": "kb_labor",
            "chunk_type": "parent", "article_no": "第十九条", "chapter_path": "第二章",
            "law_name": LCL, "title": f"{LCL}·第二章·第十九条",
            "content": "劳动合同期限三个月以上不满一年的，试用期不得超过一个月。",
        },
        {
            "chunk_id": "d1:p19:c0", "parent_id": "d1:p19", "doc_id": "d1", "kb_id": "kb_labor",
            "chunk_type": "child", "article_no": "第十九条", "chapter_path": "第二章",
            "law_name": LCL, "title": f"{LCL}·第二章·第十九条",
            "content": "试用期不得超过一个月。",
        },
        {
            "chunk_id": "d2:p705", "parent_id": "d2:p705", "doc_id": "d2", "kb_id": "kb_civil",
            "chunk_type": "parent", "article_no": "第七百零五条", "chapter_path": "第十四章",
            "law_name": CIVIL, "title": f"{CIVIL}·第十四章·第七百零五条",
            "content": "租赁期限不得超过二十年。",
        },
        {
            "chunk_id": "d2:p705:c0", "parent_id": "d2:p705", "doc_id": "d2", "kb_id": "kb_civil",
            "chunk_type": "child", "article_no": "第七百零五条", "chapter_path": "第十四章",
            "law_name": CIVIL, "title": f"{CIVIL}·第十四章·第七百零五条",
            "content": "租赁期限不得超过二十年。",
        },
    ]
    with (tmp_path / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    return Corpus.load(str(tmp_path))


def test_corpus_quote_is_parent_article(tiny_corpus: Corpus):
    evidence = tiny_corpus.evidence_at(0, 0.9)
    assert evidence is not None
    assert evidence.quote.startswith("劳动合同期限三个月以上")   # 父块整条，不是子块
    assert evidence.article_no == "第十九条"


def test_merge_hits_sorts_by_score_not_by_priority(tiny_corpus: Corpus):
    """回归测试：早期实现按"特别法优先"排在相关度之前，跨法查询被系统性压分。"""
    merged = tiny_corpus.merge_hits([(0, 0.30), (1, 0.60)], topk=2)
    assert merged[0].law == CIVIL          # 0.60 的民法典必须排在 0.30 的劳动合同法之前
    assert [e.article_no for e in merged] == ["第七百零五条", "第十九条"]


def test_merge_hits_keeps_highest_score_per_article(tiny_corpus: Corpus):
    """同一条被多路命中（同一索引出现两次）时保留高分。"""
    merged = tiny_corpus.merge_hits([(0, 0.4), (0, 0.8)], topk=5)
    assert len(merged) == 1
    assert merged[0].score == 0.8


def test_merge_hits_law_filter(tiny_corpus: Corpus):
    merged = tiny_corpus.merge_hits([(0, 0.9), (1, 0.8)], topk=5, law_hints=[CIVIL])
    assert [e.law for e in merged] == [CIVIL]


# ---------------------------------------------------------------- RRF / 稠密索引
def test_rrf_prefers_consensus():
    fused = rrf_fuse([[1, 2, 3], [2, 1, 3]])
    assert fused[0][0] in (1, 2)
    assert fused[-1][0] == 3          # 两路都排最后


def test_rrf_weight_can_flip_order():
    equal = dict(rrf_fuse([[1], [2]]))
    weighted = dict(rrf_fuse([[1], [2]], weights=[1.0, 0.0]))
    assert equal[1] == equal[2]
    assert weighted[1] > weighted[2] == 0.0


def test_dense_index_exact_search(tmp_path: Path):
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
    np.save(tmp_path / "dense.npy", vecs)
    (tmp_path / "dense_meta.json").write_text('{"dim": 2}', encoding="utf-8")
    index = DenseIndex.load(tmp_path)
    hits = index.search(np.array([0.9, 0.1], dtype=np.float32), topk=2)
    assert hits[0][0] == 0
    assert hits[0][1] > hits[1][1]


# ---------------------------------------------------------------- 端口契约
@pytest.mark.asyncio
async def test_bm25_search_returns_evidence_with_quote(tiny_corpus: Corpus):
    from app.retrieval.corpus import Bm25StatuteSearch

    search = Bm25StatuteSearch(tiny_corpus, use_expansion=False)
    found = await search.search_many(["租赁期限"], topk=2)
    assert found
    assert found[0].law == CIVIL
    assert isinstance(found[0], Evidence)
    assert found[0].quote.endswith("二十年。")
