"""把点估计变成"带区间的结论"：逐题指标的 bootstrap 置信区间 + 配对差异检验。

**为什么必须配对**：四个配置跑的是**同一批题**。A 答对的题和 C 答对的题高度相关，
把它们当成两组独立样本算置信区间会低估精度。所以差异用**同一批题重采样**（paired bootstrap），
这也是我在 REPORT 里只敢写"差 3pt"而不敢写"显著提升"的原因——现在可以写了，但要带区间。

    python -m eval.compare_stats                      # 默认 4 档 × 法条集
    python -m eval.compare_stats --set calc
    python -m eval.compare_stats --samples eval/qa_test_cases.jsonl   # 原始 40 条
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import fmean

import numpy as np

from app.retrieval.corpus import Corpus
from app.verifier.gates import extract_citations

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "eval" / "results" / "raw"
DEFAULT_SAMPLES = ROOT / "eval" / "qa_set_100.jsonl"
LEGACY_SAMPLES = ROOT / "eval" / "qa_test_cases.jsonl"
CONFIGS = ("A_baseline", "B_multi", "C_verified", "D_auto")

BOOT = 4000


# ---------------------------------------------------------------- 数据
def load_raw(kind: str, config: str) -> dict[str, dict]:
    path = RAW / f"{kind}_{config}.jsonl"
    if not path.exists():
        return {}
    return {
        json.loads(line)["sample_id"]: json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def resolve_law(law: str, laws: list[str]) -> str:
    if law in laws:
        return law
    for full in laws:
        if full.endswith(law) or law.endswith(full):
            return full
    return law


# ---------------------------------------------------------------- 指标
def law_metrics(case: dict, expected: set[str], corpus_keys: set[str], laws: list[str]) -> dict:
    cited = {
        f"{resolve_law(c.law, laws)}#{c.article_no}"
        for c in extract_citations(case.get("text") or "")
    }
    ranked = sorted(case.get("sources") or [], key=lambda e: -(e.get("score") or 0))[:5]
    retrieved = {f"{e['law']}#{e['article_no']}" for e in ranked}
    evidence = {f"{e['law']}#{e['article_no']}" for e in case.get("sources") or []}

    hit = len(cited & expected) / len(expected) if expected else float("nan")
    recall = len(retrieved & expected) / len(expected) if expected else float("nan")
    valid = len([k for k in cited if k in corpus_keys]) / len(cited) if cited else float("nan")
    trace = len([k for k in cited if k in evidence]) / len(cited) if cited else float("nan")
    return {"recall": recall, "cite_hit": hit, "cite_valid": valid, "cite_trace": trace}


# ---------------------------------------------------------------- 统计
def bootstrap_ci(values: list[float], *, n: int = BOOT, seed: int = 7) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v == v], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(
    a: list[float], b: list[float], *, n: int = BOOT, seed: int = 7
) -> tuple[float, float, float]:
    """同一批题上 b − a 的均值差与 95% 区间（逐题重采样，保持配对）。"""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(arr_a) | np.isnan(arr_b))
    arr_a, arr_b = arr_a[mask], arr_b[mask]
    if arr_a.size == 0:
        return float("nan"), float("nan"), float("nan")
    diff = arr_b - arr_a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n, diff.size))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt(x: float) -> str:
    return "  —  " if x != x else f"{x:.3f}"


def fmt_diff(mean: float, lo: float, hi: float) -> str:
    if mean != mean:
        return "—"
    star = "*" if (lo > 0 or hi < 0) else " "
    return f"{mean:+.3f}{star}[{lo:+.3f},{hi:+.3f}]"


# ---------------------------------------------------------------- 报告
def report_law(samples_path: Path, configs: tuple[str, ...]) -> None:
    corpus = Corpus.load()
    keys = {f"{p['law_name']}#{p['article_no']}" for p in corpus.parents.values()}
    laws = sorted({p["law_name"] for p in corpus.parents.values()}, key=len)
    samples = {
        json.loads(l)["id"]: json.loads(l)
        for l in samples_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    }

    per: dict[str, dict[str, dict]] = {}
    for cfg in configs:
        raw = load_raw("law", cfg)
        if not raw:
            continue
        per[cfg] = {
            sid: law_metrics(raw[sid], set(s["expected_articles"] or []), keys, laws)
            for sid, s in samples.items()
            if sid in raw
        }
    if not per:
        print("没有可用的 raw 结果，先跑 eval.compare_agents")
        return

    ids = sorted(set.intersection(*(set(v) for v in per.values())))
    n = len(ids)
    print(f"法条问答：{n} 条（样本集 {samples_path.name}）\n")

    metrics = (
        ("R@5", "recall"),
        ("引用命中", "cite_hit"),
        ("引注有效", "cite_valid"),
        ("引注可溯", "cite_trace"),
    )
    head = f"{'指标':<10}" + "".join(f"{c.split('_')[0]:>18}" for c in per)
    print(head)
    print("-" * len(head.encode("gbk", errors="replace")))
    for label, key in metrics:
        row = f"{label:<10}"
        for cfg in per:
            vals = [per[cfg][i][key] for i in ids]
            lo, hi = bootstrap_ci(vals)
            row += f"{fmt(fmean([v for v in vals if v == v])):>6} [{lo:.2f},{hi:.2f}]".rjust(18)
        print(row)

    print("\n配对差异（同一批题重采样，* 表示 95% 区间不含 0）")
    # C−B 是**最干净的一对**：拓扑完全相同，只差一个校验器 ——
    # 多 Agent 的收益到底来自编排还是来自校验，只能靠这一对回答。
    pairs = [
        ("B_multi", "A_baseline"),
        ("C_verified", "B_multi"),
        ("C_verified", "A_baseline"),
        ("D_auto", "A_baseline"),
        ("D_auto", "C_verified"),
    ]
    for hi_cfg, lo_cfg in pairs:
        if hi_cfg not in per or lo_cfg not in per:
            continue
        print(f"\n{hi_cfg} − {lo_cfg}")
        for label, key in metrics:
            a = [per[lo_cfg][i][key] for i in ids]
            b = [per[hi_cfg][i][key] for i in ids]
            mean, lo, hi = paired_diff_ci(a, b)
            print(f"  {label:<10}{fmt_diff(mean, lo, hi)}")

    report_by_type(samples, ids, per)


#: 分题型名称与包含的 type —— 多跳与复杂合并一行，是因为扩样后两者都是"跨条题"，
#: 分开看 n 仍然偏小；但报告里必须同时给出分开的两行。
TYPE_GROUPS = (
    ("单跳", ("single_hop",)),
    ("多跳", ("multi_hop",)),
    ("复杂", ("complex",)),
    ("多跳+复杂", ("multi_hop", "complex")),
)


def report_by_type(samples: dict[str, dict], ids: list[str], per: dict[str, dict[str, dict]]) -> None:
    """按题型重算：这是"多 Agent 在复杂问题上更值"的证据所在，也是扩样的直接目的。

    只给点估计等于没给结论 —— 题型样本量小，必须带区间，且差异用配对 bootstrap。
    """
    kinds = {sid: samples[sid].get("type") for sid in samples}
    print("\n分题型（引用命中，单配置 95% 区间）")
    head = f"{'题型':<10}{'n':>4}" + "".join(f"{c.split('_')[0]:>18}" for c in per)
    print(head)
    print("-" * len(head.encode("gbk", errors="replace")))
    for name, tset in TYPE_GROUPS:
        gids = [i for i in ids if kinds.get(i) in tset]
        if not gids:
            continue
        row = f"{name:<10}{len(gids):>4}"
        for cfg in per:
            vals = [per[cfg][i]["cite_hit"] for i in gids]
            clean = [v for v in vals if v == v]
            lo, hi = bootstrap_ci(vals)
            row += f"{fmt(fmean(clean) if clean else float('nan')):>6} [{lo:.2f},{hi:.2f}]".rjust(18)
        print(row)

    print("\n分题型配对差异（引用命中，* = 区间不含 0）")
    pair_head = f"{'题型':<10}{'n':>4}{'C−A':>26}{'C−B':>26}{'D−A':>26}"
    print(pair_head)
    for name, tset in TYPE_GROUPS:
        gids = [i for i in ids if kinds.get(i) in tset]
        if not gids:
            continue
        row = f"{name:<10}{len(gids):>4}"
        for hi_cfg, lo_cfg in (("C_verified", "A_baseline"), ("C_verified", "B_multi"), ("D_auto", "A_baseline")):
            if hi_cfg not in per or lo_cfg not in per:
                row += f"{'—':>26}"
                continue
            a = [per[lo_cfg][i]["cite_hit"] for i in gids]
            b = [per[hi_cfg][i]["cite_hit"] for i in gids]
            mean, lo, hi = paired_diff_ci(a, b)
            row += f"{fmt_diff(mean, lo, hi):>26}"
        print(row)
    print(
        "\n注：题型样本量小，点估计不构成结论；只有区间不含 0 才能说\"可区分\"。"
        "\n    多跳+复杂合并行是扩样后主看的一行，分开的两行仍按原口径保留。"
    )


def report_calc(samples_path: Path, configs: tuple[str, ...]) -> None:
    path = ROOT / "eval" / "calc_cases.jsonl"
    samples = {
        json.loads(l)["id"]: json.loads(l)
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    per: dict[str, dict[str, float]] = {}
    for cfg in configs:
        raw = load_raw("calc", cfg)
        if not raw:
            continue
        per[cfg] = {}
        for sid in samples:
            if sid not in raw:
                continue
            got = raw[sid].get("amount")
            gold = float(samples[sid]["expected_value"])
            per[cfg][sid] = 1.0 if got is not None and abs(float(got) - gold) <= 1 else 0.0

    ids = sorted(set.intersection(*(set(v) for v in per.values())))
    print(f"\n金额测算：{len(ids)} 条")
    print(f"{'配置':<12}{'准确率':>8}{'95%CI':>16}")
    for cfg, vals in per.items():
        arr = [vals[i] for i in ids]
        lo, hi = bootstrap_ci(arr)
        print(f"{cfg:<12}{fmean(arr):>8.3f}   [{lo:.2f},{hi:.2f}]")
    if len(ids) < 30:
        print(f"\n注意：{len(ids)} 条样本，单题权重 {1/len(ids):.0%}，"
              "准确率的区间基本没有判别力——金额部分只能当定性结论读。")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="配对 bootstrap：点估计 → 区间")
    parser.add_argument("--set", default="all", choices=("all", "law", "calc"))
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--configs", default=",".join(CONFIGS))
    args = parser.parse_args(argv)

    configs = tuple(c.strip() for c in args.configs.split(",") if c.strip())
    samples = Path(args.samples)
    if not samples.exists():
        samples = LEGACY_SAMPLES
    if args.set in ("all", "law"):
        report_law(samples, configs)
    if args.set in ("all", "calc"):
        report_calc(samples, configs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
