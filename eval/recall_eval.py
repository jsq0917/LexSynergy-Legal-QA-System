"""检索评测：40 条标注样本，条级 Recall@K / MRR，分题型拆解。

口径与 LawAnswer 的 REPORT 对齐（便于横向对比）：
* 条级召回：命中任一子块即该条命中，用 `法名#条号` 唯一标识（跨法同号不混淆）；
* 无期望法条的样本（no_answer）不计入召回分母；
* topK 固定 8，再按 @1/@3/@5/@8 截断。

    python -m eval.recall_eval
    python -m eval.recall_eval --hints        # 叠加规则层预测的法域过滤
    python -m eval.recall_eval --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import time

from app.orchestrator.rules import infer_intent, infer_law_hints
from app.retrieval.corpus import Bm25StatuteSearch, Corpus

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "eval" / "qa_test_cases.jsonl"
OUT_DIR = ROOT / "eval" / "results"
KS = (1, 3, 5, 8)

#: LawAnswer 已发布的对照数字（同 40 条样本，口径一致）
BASELINE = {
    "L1 纯稠密（HNSW）": {"recall@5": 0.7405, "mrr": 0.7324},
    "L3 混合+重排": {"recall@5": 0.7024, "mrr": 0.6426},
    "L6 等权RRF+法域过滤": {"recall@5": 0.7071, "mrr": 0.6886},
}


def load_samples() -> list[dict]:
    rows: list[dict] = []
    with SAMPLES.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def recall_at_k(hits: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return float("nan")
    got = set(hits[:k])
    return len(got & expected) / len(expected)


def mrr(hits: list[str], expected: set[str]) -> float:
    if not expected:
        return float("nan")
    for rank, key in enumerate(hits, start=1):
        if key in expected:
            return 1.0 / rank
    return 0.0


def build_search(backend: str, corpus: Corpus, *, use_expansion: bool = True):
    if backend == "bm25":
        return Bm25StatuteSearch(corpus, use_expansion=use_expansion)
    if backend == "dense":
        from app.retrieval.dense import DenseStatuteSearch

        return DenseStatuteSearch(corpus)
    if backend == "hybrid":
        from app.retrieval.dense import HybridStatuteSearch

        return HybridStatuteSearch(corpus)
    raise ValueError(f"未知 backend：{backend}")


async def run(
    *, use_hints: bool = False, backend: str = "bm25", use_expansion: bool = True
) -> dict:
    samples = load_samples()
    corpus = Corpus.load()
    search = build_search(backend, corpus, use_expansion=use_expansion)
    per_sample: list[dict] = []

    for row in samples:
        question = row["question"]
        expected = set(row.get("expected_articles") or [])
        hints = []
        if use_hints:
            hints = infer_law_hints(question, infer_intent(question))
        started = time.perf_counter()
        found = await search.search_many([question], law_hints=hints or None, topk=max(KS))
        latency = int((time.perf_counter() - started) * 1000)
        keys = [f"{e.law}#{e.article_no}" for e in found]
        per_sample.append(
            {
                "id": row["id"],
                "type": row["type"],
                "question": question,
                "expected": sorted(expected),
                "retrieved": keys,
                "hits": keys,
                "recall": {k: recall_at_k(keys, expected, k) for k in KS},
                "mrr": mrr(keys, expected),
                "latency_ms": latency,
            }
        )

    scored = [s for s in per_sample if s["expected"]]
    summary = {
        "samples": len(per_sample),
        "scored": len(scored),
        "recall@" + str(KS[0]): round(statistics.mean(s["recall"][KS[0]] for s in scored), 4),
        "recall@" + str(KS[1]): round(statistics.mean(s["recall"][KS[1]] for s in scored), 4),
        "recall@" + str(KS[2]): round(statistics.mean(s["recall"][KS[2]] for s in scored), 4),
        "recall@" + str(KS[3]): round(statistics.mean(s["recall"][KS[3]] for s in scored), 4),
        "mrr": round(statistics.mean(s["mrr"] for s in scored), 4),
        "latency_ms": round(statistics.mean(s["latency_ms"] for s in per_sample), 1),
        "config": f"{backend}{'+hints' if use_hints else ''}"
        + ("" if use_expansion or backend != "bm25" else "（无扩展）"),
    }

    by_type: dict[str, dict] = {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in scored:
        groups[s["type"]].append(s)
    for name, rows in groups.items():
        entry: dict = {"n": len(rows)}
        entry.update(
            {f"recall@{k}": round(statistics.mean(r["recall"][k] for r in rows), 4) for k in KS}
        )
        entry["mrr"] = round(statistics.mean(r["mrr"] for r in rows), 4)
        by_type[name] = entry

    return {"summary": summary, "by_type": by_type, "samples": per_sample}


def _print(report: dict, *, use_hints: bool) -> None:
    s = report["summary"]
    print(f"配置：{s['config']}（{s['scored']}/{s['samples']} 条计入召回，no_answer 不计）")
    print("=" * 62)
    print(f"{'指标':<12}{'实测':>10}{'LawAnswer 最佳':>16}{'差值':>10}")
    best = BASELINE["L1 纯稠密（HNSW）"]
    print(f"{'Recall@5':<12}{s['recall@5']:>10.4f}{best['recall@5']:>16.4f}"
          f"{s['recall@5'] - best['recall@5']:>+10.4f}")
    print(f"{'MRR':<12}{s['mrr']:>10.4f}{best['mrr']:>16.4f}{s['mrr'] - best['mrr']:>+10.4f}")
    print(
        f"Recall@1 {s['recall@1']:.4f} | @3 {s['recall@3']:.4f} | "
        f"@5 {s['recall@5']:.4f} | @8 {s['recall@8']:.4f}"
    )
    print("-" * 62)
    print(f"{'题型':<14}{'n':>4}{'@1':>8}{'@3':>8}{'@5':>8}{'@8':>8}{'MRR':>8}")
    for name, row in sorted(report["by_type"].items()):
        print(
            f"{name:<14}{row['n']:>4}{row['recall@1']:>8.3f}{row['recall@3']:>8.3f}"
            f"{row['recall@5']:>8.3f}{row['recall@8']:>8.3f}{row['mrr']:>8.3f}"
        )
    print("-" * 62)
    misses = [s for s in report["samples"] if s["expected"] and s["recall"][5] == 0]
    print(f"Recall@5 全未命中的样本：{len(misses)} 条")
    for miss in misses[:5]:
        print(f"  - [{miss['id']}/{miss['type']}] {miss['question'][:40]}")
        print(f"    期望 {miss['expected']} -> 检索 {miss['retrieved'][:4]}")


def _ablate(as_json: bool) -> int:
    """四组配置对比：BM25 / BM25+扩展 / 稠密 / 混合。"""
    configs = [
        ("L0 BM25", {"backend": "bm25", "use_expansion": False}),
        ("L1 BM25+扩展", {"backend": "bm25", "use_expansion": True}),
        ("L2 稠密", {"backend": "dense"}),
        ("L3 混合(RRF)", {"backend": "hybrid"}),
    ]
    rows = []
    for name, kwargs in configs:
        report = asyncio.run(run(**kwargs))
        rows.append((name, report["summary"]))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"ablation_{name.split()[0]}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if as_json:
        print(json.dumps({n: s for n, s in rows}, ensure_ascii=False, indent=2))
        return 0

    print(f"{'配置':<16}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'R@8':>7}{'MRR':>7}{'延迟ms':>9}")
    print("-" * 60)
    for name, s in rows:
        print(
            f"{name:<16}{s['recall@1']:>7.4f}{s['recall@3']:>7.4f}{s['recall@5']:>7.4f}"
            f"{s['recall@8']:>7.4f}{s['mrr']:>7.4f}{s['latency_ms']:>9.1f}"
        )
    print("-" * 60)
    print("LawAnswer 同口径对照：L1 纯稠密 R@5 0.7405 / MRR 0.7324；L6 混合+法域过滤 0.7071 / 0.6886")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.recall_eval")
    parser.add_argument("--hints", action="store_true", help="叠加规则层预测的法域过滤")
    parser.add_argument("--backend", default="bm25", choices=("bm25", "dense", "hybrid"))
    parser.add_argument("--no-expand", action="store_true", help="关闭 BM25 的 Query 扩展")
    parser.add_argument("--ablate", action="store_true", help="跑 4 组消融并输出对比表")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.ablate:
        return _ablate(args.json)

    report = asyncio.run(
        run(use_hints=args.hints, backend=args.backend, use_expansion=not args.no_expand)
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"recall_{report['summary']['config'].replace('+', '_')}.json"
    (OUT_DIR / name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    else:
        _print(report, use_hints=args.hints)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
