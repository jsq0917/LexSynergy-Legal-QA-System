"""三配置对照评测：单 Agent 基线 / 多 Agent / 多 Agent + 五道闸。

**这个脚本是简历上所有数字的唯一来源。** 设计上的三条约束：

1. **同检索、同样本、同 Prompt 素材**：三配置共用同一个检索器与 top-k，
   基线不"放水"（它的证据就是多 Agent 第一跳拿到的那些），差值才能归因于编排与校验；
2. **六列里五列是确定性指标**：Recall / 引注有效率 / 引注可溯率 / 引用命中率 / 金额准确率 /
   LLM 调用数 / 延迟，全都不依赖模型自评。只有 faithfulness 需要 judge，
   因此单独用 `--faithfulness` 开关，且报告里必须标注"同族模型自评，仅作相对比较"；
3. **可复现**：原始输出落 `eval/results/raw/{config}.jsonl`，`--reuse` 可跳过已跑样本，
   任何一次报告都能追到逐题原文。

    python -m eval.compare_agents --limit 8          # 先小样本验链路
    python -m eval.compare_agents                    # 全量（法条 40 + 测算 9）
    python -m eval.compare_agents --faithfulness     # 追加语义支撑判定（多 1 次/题）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from app.core.logger import get_logger
from app.generation.llm_client import get_llm_client
from app.orchestrator.pipeline import Pipeline
from app.retrieval.corpus import Corpus
from app.verifier.gates import claims_of, extract_citations
from config import get_settings

logger = get_logger("eval.compare")

ROOT = Path(__file__).resolve().parents[1]
#: 默认用 100 条集（40 条原始 + 60 条生成）；`--law-set` 可切回原始 40 条以复现历史数字
LAW_SET = ROOT / "eval" / "qa_set_100.jsonl"
LEGACY_SET = ROOT / "eval" / "qa_test_cases.jsonl"
CALC_SET = ROOT / "eval" / "calc_cases.jsonl"
RAW_DIR = ROOT / "eval" / "results" / "raw"
OUT_DIR = ROOT / "eval" / "results"

CONFIGS = {
    "A_baseline": "单 Agent（1 次检索 + 1 次生成，复刻 LawAnswer 链路）",
    "B_multi": "多 Agent（分诊 + 检索 + 测算 + 生成，无校验）",
    "C_verified": "多 Agent + 五道闸 + 补检索回路（完整系统，强制全编排）",
    "D_auto": "成本感知路由（单跳走快速通道，测算/多跳/合规才升级全编排）",
}

#: B/C 是"强制全编排"的对照组，必须钉死 policy=full；
#: 否则它们会跟着默认值走成 auto，对照组就消失了。
POLICY = {"B_multi": "full", "C_verified": "full", "D_auto": "auto"}


# ---------------------------------------------------------------- 三档装配
def build_multi(*, verify: bool, policy: str = "full") -> Pipeline:
    """B/C 只差一个校验器；D 额外放开路由策略。其余拓扑完全相同（隔离变量）。"""
    import os

    from config import reset_settings

    os.environ["ROUTING_POLICY"] = policy
    reset_settings()

    from app.runtime import build_pipeline

    return build_pipeline(use_llm=True, verify=verify)


def build_search():
    """与线上链路同一个检索器（默认 dense），保证对照的检索条件一致。"""
    from app.runtime import build_search as _build

    return _build(get_settings())


# ---------------------------------------------------------------- 指标（纯函数）
@dataclass
class LawScore:
    recall: float = float("nan")
    cite_n: int = 0
    cite_valid: int = 0          # 条号在语料中真实存在
    cite_traceable: int = 0      # 引用的是本次检索到的证据（不是凭记忆）
    cite_hit: float = float("nan")   # 期望法条被引用的比例


def _law_index(corpus: Corpus) -> tuple[set[str], list[str]]:
    keys = {f"{p['law_name']}#{p['article_no']}" for p in corpus.parents.values()}
    laws = sorted({p["law_name"] for p in corpus.parents.values()}, key=len)
    return keys, laws


def resolve_law(law: str, laws: list[str]) -> str:
    """把《劳动合同法》这类简称对齐到语料里的全称（按后缀匹配，取最短命中）。"""
    if law in laws:
        return law
    for full in laws:
        if full.endswith(law) or law.endswith(full):
            return full
    return law


def score_law(
    text: str,
    sources: list[dict],
    expected: set[str],
    corpus_keys: set[str],
    laws: list[str],
    *,
    topk: int = 5,
) -> LawScore:
    ranked = sorted(
        sources, key=lambda e: -(e.get("score") or 0.0)
    )[:topk]
    got = {f"{e['law']}#{e['article_no']}" for e in ranked}
    evidence_keys = {f"{e['law']}#{e['article_no']}" for e in sources}

    citations = extract_citations(text)
    resolved = [f"{resolve_law(c.law, laws)}#{c.article_no}" for c in citations]
    valid = [k for k in resolved if k in corpus_keys]
    traceable = [k for k in resolved if k in evidence_keys]

    recall = len(got & expected) / len(expected) if expected else float("nan")
    hit = len(set(resolved) & expected) / len(expected) if expected else float("nan")
    return LawScore(
        recall=round(recall, 4) if expected else float("nan"),
        cite_n=len(resolved),
        cite_valid=len(valid),
        cite_traceable=len(traceable),
        cite_hit=round(hit, 4) if expected else float("nan"),
    )


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, int(round(0.95 * (len(ordered) - 1))))
    return ordered[idx]


def mean(values: list[float]) -> float:
    clean = [v for v in values if v == v]  # 去 NaN
    return round(statistics.fmean(clean), 4) if clean else float("nan")


# ---------------------------------------------------------------- 单次执行
@dataclass
class Case:
    """一次 (配置, 样本) 的执行结果 —— 落盘格式即报告口径。"""

    config: str
    sample_id: str
    question: str
    text: str = ""
    sources: list[dict] = field(default_factory=list)
    calc: list[dict] = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    amount: str | None = None
    llm_calls: int = 0
    elapsed_ms: int = 0
    degraded: str | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "sample_id": self.sample_id,
            "question": self.question,
            "text": self.text,
            "sources": self.sources,
            "calc": self.calc,
            "facts": self.facts,
            "amount": self.amount,
            "llm_calls": self.llm_calls,
            "elapsed_ms": self.elapsed_ms,
            "degraded": self.degraded,
            "error": self.error,
            "extra": self.extra,
        }


async def run_baseline(sample: dict, agent) -> Case:
    ans = await agent.answer(sample["question"])
    return Case(
        config="A_baseline",
        sample_id=sample["id"],
        question=sample["question"],
        text=ans.text,
        sources=ans.sources,
        amount=str(ans.amount) if ans.amount is not None else None,
        llm_calls=ans.llm_calls,
        elapsed_ms=ans.elapsed_ms,
        error=ans.error,
        extra={"tokens_in": ans.tokens_in, "tokens_out": ans.tokens_out},
    )


async def run_pipeline(sample: dict, pipeline: Pipeline, config: str) -> Case:
    ans = await pipeline.run(sample["question"])
    amount = None
    if ans.calc:
        value = ans.calc[0].get("value")
        amount = str(value) if value is not None else None
    return Case(
        config=config,
        sample_id=sample["id"],
        question=sample["question"],
        text=ans.text,
        sources=ans.sources,
        calc=ans.calc,
        facts=dict(ans.facts or {}),
        amount=amount,
        llm_calls=ans.llm_calls,
        elapsed_ms=ans.elapsed_ms,
        degraded=ans.degraded,
        extra={
            "route": ans.route,
            "mode": ans.mode,
            "loops": ans.loops,
            "verdict": ans.verdict,
        },
    )


# ---------------------------------------------------------------- 主流程
async def run_config(
    config: str,
    samples: list[dict],
    *,
    kind: str,
    concurrency: int,
    reuse: dict[str, Case],
) -> list[Case]:
    todo = [s for s in samples if f"{config}:{s['id']}" not in reuse]
    if not todo:
        return [reuse[f"{config}:{s['id']}"] for s in samples]

    agent = None
    pipeline = None
    if config == "A_baseline":
        from eval.baseline_agent import BaselineSingleAgent

        agent = BaselineSingleAgent(build_search(), topk=get_settings().retrieval_topk)
    else:
        pipeline = build_multi(verify=(config != "B_multi"), policy=POLICY.get(config, "full"))

    sem = asyncio.Semaphore(max(1, concurrency))
    results: list[Case] = []

    async def one(sample: dict) -> Case:
        async with sem:
            try:
                if config == "A_baseline":
                    return await run_baseline(sample, agent)
                return await run_pipeline(sample, pipeline, config)
            except Exception as exc:  # noqa: BLE001 - 单题失败不能中断整批
                logger.warning("[%s] %s 失败：%s", config, sample["id"], str(exc)[:160])
                return Case(
                    config=config,
                    sample_id=sample["id"],
                    question=sample["question"],
                    error=f"{type(exc).__name__}: {str(exc)[:160]}",
                )

    for coro in asyncio.as_completed([one(s) for s in todo]):
        results.append(await coro)

    ordered = _ordered_cases(samples, config, results, reuse)
    _append_raw(config, kind, results)
    return ordered


def _ordered_cases(
    samples: list[dict], config: str, results: list[Case], reuse: dict[str, Case]
) -> list[Case]:
    """按样本顺序返回**本次新跑 + 复用**的全部用例。

    早先只把 `results`（本次新跑）放进 `done`，于是 `--reuse` 补样时汇总表只覆盖了
    新补的那部分，却在标题里写着全量样本数 —— 30 题被当成 130 题报（2026-10-28 扩样时抓到）。
    复用池里可能有其他配置的用例（`reuse` 按 `config:sample_id` 建键），这里只按本配置取值。
    """
    done: dict[str, Case] = {f"{c.config}:{c.sample_id}": c for c in reuse.values()}
    done.update({f"{c.config}:{c.sample_id}": c for c in results})
    return [done[f"{config}:{s['id']}"] for s in samples if f"{config}:{s['id']}" in done]


def _append_raw(config: str, kind: str, cases: list[Case]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{kind}_{config}.jsonl"
    existing: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["sample_id"]] = row
    for case in cases:
        existing[case.sample_id] = case.to_dict()
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in existing.values()) + "\n",
        encoding="utf-8",
    )


def load_reuse(kind: str) -> dict[str, Case]:
    out: dict[str, Case] = {}
    if not RAW_DIR.exists():
        return out
    for path in RAW_DIR.glob(f"{kind}_*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            case = Case(**{k: v for k, v in row.items() if k in Case.__dataclass_fields__})
            out[f"{case.config}:{case.sample_id}"] = case
    return out


# ---------------------------------------------------------------- faithfulness
FAITH_SYSTEM = """你是严格的法律事实核查员。给定【参考资料】与【回答】，找出回答中**无法由参考资料支持**的断言。

只输出一个 JSON 对象，不要解释：
{"unsupported": ["断言原文", "..."]}

判定口径：
- 与参考资料同义、可推出的命题 → 支持，不列入；
- 复述用户问题里的事实、复述算术代入过程 → 不列入；
- 程序性建议（"可申请劳动仲裁"）、过渡句、免责声明 → 不列入；
- 与参考资料矛盾、或参考资料完全没提到该法律规则 → 列入。"""


async def judge_faithfulness(cases: list[Case], *, concurrency: int = 4) -> dict[str, float]:
    """按断言级支持率打分：1 - 无依据断言数 / 断言总数。"""
    s = get_settings()
    client = get_llm_client(s)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(case: Case) -> tuple[str, float]:
        claims = claims_of(case.text)
        if not claims:
            return f"{case.config}:{case.sample_id}", float("nan")
        evidence = "\n\n".join(
            f"【{e.get('law')}·{e.get('article_no')}】{e.get('quote', '')[:600]}"
            for e in case.sources[:6]
        )
        user = f"【参考资料】\n{evidence}\n\n【回答】\n{case.text[:2500]}"
        async with sem:
            try:
                data, _ = await asyncio.to_thread(
                    client.json_chat,
                    [{"role": "system", "content": FAITH_SYSTEM}, {"role": "user", "content": user}],
                    model=s.fast_model,
                    max_tokens=700,
                )
                bad = data.get("unsupported") or []
                return f"{case.config}:{case.sample_id}", round(
                    1 - len(bad) / len(claims), 4
                )
            except Exception as exc:  # noqa: BLE001 - judge 失败标 NaN，不污染均值
                logger.warning("judge 失败 %s：%s", case.sample_id, str(exc)[:120])
                return f"{case.config}:{case.sample_id}", float("nan")

    out: dict[str, float] = {}
    for coro in asyncio.as_completed([one(c) for c in cases]):
        key, value = await coro
        out[key] = value
    return out


# ---------------------------------------------------------------- 报告
def report_law(
    by_config: dict[str, list[Case]],
    corpus: Corpus,
    faith: dict,
    samples_path: Path | None = None,
) -> tuple[str, list[dict]]:
    keys, laws = _law_index(corpus)
    samples = {s["id"]: s for s in _load(samples_path or LAW_SET)}
    rows = []
    for config, cases in by_config.items():
        recalls, hits, valids, trace, lat, calls, degraded = [], [], [], [], [], [], 0
        judge = []
        for case in cases:
            sample = samples[case.sample_id]
            expected = set(sample.get("expected_articles") or [])
            sc = score_law(case.text, case.sources, expected, keys, laws)
            if expected:
                recalls.append(sc.recall)
                hits.append(sc.cite_hit)
            if sc.cite_n:
                valids.append(sc.cite_valid / sc.cite_n)
                trace.append(sc.cite_traceable / sc.cite_n)
            lat.append(case.elapsed_ms)
            calls.append(case.llm_calls)
            verdict = (case.extra or {}).get("verdict") or {}
            if case.degraded or verdict.get("passed") is False:
                degraded += 1
            f = faith.get(f"{config}:{case.sample_id}")
            if f is not None and f == f:
                judge.append(f)
        rows.append(
            {
                "config": config,
                "n": len(cases),
                "recall@5": mean(recalls),
                "cite_valid": mean(valids),
                "cite_trace": mean(trace),
                "cite_hit": mean(hits),
                "faith": mean(judge) if judge else float("nan"),
                "llm": mean([float(c) for c in calls]),
                "p95_ms": p95([float(x) for x in lat]),
                "degraded": degraded,
            }
        )

    head = (
        f"{'配置':<14}{'R@5':>8}{'引注有效':>9}{'引注可溯':>9}{'引用命中':>9}"
        f"{'支撑率':>8}{'LLM/题':>8}{'P95ms':>8}{'未过闸':>7}"
    )
    lines = [head, "-" * 72]
    for r in rows:
        lines.append(
            f"{r['config']:<14}{_fmt(r['recall@5']):>8}{_fmt(r['cite_valid']):>9}"
            f"{_fmt(r['cite_trace']):>9}{_fmt(r['cite_hit']):>8}{_fmt(r['faith']):>8}"
            f"{r['llm']:>8.2f}{r['p95_ms']:>8.0f}{r['degraded']:>7}"
        )
    return "\n".join(lines), rows


def report_calc(by_config: dict[str, list[Case]]) -> tuple[str, list[dict]]:
    samples = {s["id"]: s for s in _load(CALC_SET)}
    rows = []
    for config, cases in by_config.items():
        correct, slot_ok, lat, calls, degraded = 0, [], [], [], 0
        detail = []
        for case in cases:
            sample = samples[case.sample_id]
            gold = Decimal(sample["expected_value"])
            got = Decimal(case.amount) if case.amount else None
            ok = got is not None and abs(got - gold) <= Decimal("1")
            correct += ok
            gold_facts = sample["facts"]
            if config == "A_baseline":
                # 单 Agent 没有抽槽环节，这一列对它不适用（记 NaN → 报告显示"—"）
                slot_ok.append(float("nan"))
            elif case.facts:
                hit = sum(1 for k, v in gold_facts.items() if _same(case.facts.get(k), v))
                slot_ok.append(hit / len(gold_facts))
            else:
                slot_ok.append(0.0)
            lat.append(case.elapsed_ms)
            calls.append(case.llm_calls)
            verdict = (case.extra or {}).get("verdict") or {}
            if case.degraded or verdict.get("passed") is False:
                degraded += 1
            detail.append((case.sample_id, str(got), sample["expected_value"], ok))
        rows.append(
            {
                "config": config,
                "n": len(cases),
                "amount_acc": round(correct / len(cases), 4) if cases else float("nan"),
                "slot_acc": mean(slot_ok),
                "llm": mean([float(c) for c in calls]),
                "p95_ms": p95([float(x) for x in lat]),
                "degraded": degraded,
                "detail": detail,
            }
        )

    head = f"{'配置':<14}{'金额准确率':>10}{'抽槽准确率':>11}{'LLM/题':>8}{'P95ms':>8}{'降级':>6}"
    lines = [head, "-" * 56]
    for r in rows:
        lines.append(
            f"{r['config']:<14}{_fmt(r['amount_acc']):>10}{_fmt(r['slot_acc']):>11}"
            f"{r['llm']:>8.2f}{r['p95_ms']:>8.0f}{r['degraded']:>6}"
        )
    return "\n".join(lines), rows


def _same(a, b) -> bool:
    if a is None:
        return False
    if isinstance(b, bool):
        return bool(a) is b
    if isinstance(b, (int, float)):
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (TypeError, ValueError):
            return False
    return str(a)[:10] == str(b)[:10]


def _fmt(v: float) -> str:
    return "—" if v != v else f"{v:.3f}"


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------- 入口
async def main_async(args) -> int:
    if not get_settings().has_llm:
        print("未配置 LLM_API_KEY —— 本对照必须真调模型，否则 A 档无意义。")
        return 2

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in CONFIGS:
            print(f"未知配置 {c}，可选：{', '.join(CONFIGS)}")
            return 2

    want_law = args.set in ("all", "law")
    want_calc = args.set in ("all", "calc")
    law_path = Path(args.law_set) if args.law_set else LAW_SET
    if not law_path.exists():
        print(f"样本集不存在：{law_path}，回退 {LEGACY_SET.name}")
        law_path = LEGACY_SET
    law_samples = _load(law_path)[: args.limit or None] if want_law else []
    calc_samples = _load(CALC_SET)[: args.limit or None] if want_calc else []
    corpus = Corpus.load()

    reuse_law = load_reuse("law") if args.reuse else {}
    reuse_calc = load_reuse("calc") if args.reuse else {}

    law_results: dict[str, list[Case]] = {}
    calc_results: dict[str, list[Case]] = {}
    for config in configs:
        if want_law:
            print(f"[{config}] 法条集 {len(law_samples)} 题…", flush=True)
            law_results[config] = await run_config(
                config, law_samples, kind="law", concurrency=args.concurrency, reuse=reuse_law
            )
        if want_calc:
            print(f"[{config}] 测算集 {len(calc_samples)} 题…", flush=True)
            calc_results[config] = await run_config(
                config, calc_samples, kind="calc", concurrency=args.concurrency, reuse=reuse_calc
            )

    faith: dict = {}
    faith_path = OUT_DIR / "faith.json"
    if args.faithfulness:
        flat = [c for cases in law_results.values() for c in cases]
        print(f"语义支撑判定：{len(flat)} 题…", flush=True)
        judged = await judge_faithfulness(flat, concurrency=args.concurrency)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        merged = (
            json.loads(faith_path.read_text(encoding="utf-8")) if faith_path.exists() else {}
        )
        merged.update(judged)                      # 增量合并，避免局部重跑冲掉其他配置
        faith_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        faith = merged
    elif faith_path.exists():
        faith = json.loads(faith_path.read_text(encoding="utf-8"))
        print(f"复用已有支撑率：{faith_path.name}（{len(faith)} 条）", flush=True)

    law_table, law_rows = (
        report_law(law_results, corpus, faith, law_path) if law_results else ("（本次未跑）", [])
    )
    calc_table, calc_rows = report_calc(calc_results) if calc_results else ("（本次未跑）", [])

    print(f"\n=== 表 1：法条问答（{len(law_samples)} 条，{law_path.name}）===")
    print(law_table)
    print(f"\n=== 表 2：金额测算（{len(calc_samples)} 条，含真实日期的工作年限折算）===")
    print(calc_table)
    if args.faithfulness:
        print("\n注：支撑率由同族模型判定，绝对值不可跨系统比较，仅用于各配置横向对照。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "compare_report.json"
    merged = merge_report(
        report_path,
        law=law_rows,
        calc=calc_rows,
        meta={
            "limit": args.limit,
            "law_samples": len(law_samples),
            "calc_samples": len(calc_samples),
            "law_set": law_path.name if want_law else None,
            "faithfulness": bool(args.faithfulness),
        },
    )
    print(
        f"\n机器可读结果：{report_path}"
        f"（law {len(merged.get('law') or [])} 段 / calc {len(merged.get('calc') or [])} 段）"
    )
    return 0


def merge_report(path: Path, *, law: list[dict], calc: list[dict], meta: dict) -> dict:
    """按段合并写报告：**只覆盖本次跑了的段**，另一段原样保留。

    为什么不能整体覆写：报告是"当前所有结论的快照"，而 `--set calc` 只重算一半。
    整体覆写会让"只补跑了金额"看起来像"法条结论没了"——验收时我就是被这个
    假退步误导过一次。每段自带 `generated_at`，才能看出哪半是新的。
    """
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    out = dict(existing)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    if law:
        out["law"] = law
        out["law_meta"] = {**meta, "generated_at": now, "samples": meta.get("law_samples")}
    if calc:
        out["calc"] = calc
        out["calc_meta"] = {**meta, "generated_at": now, "samples": meta.get("calc_samples")}
    out["configs"] = CONFIGS
    out["updated_at"] = now
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="单 Agent / 多 Agent / 多 Agent+校验 三配置对照")
    parser.add_argument("--configs", default=",".join(CONFIGS), help="逗号分隔，默认三档全跑")
    parser.add_argument("--limit", type=int, default=0, help="每个集合只跑前 N 题（调试用）")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--set", default="all", choices=("all", "law", "calc"), help="只跑某个样本集")
    parser.add_argument(
        "--law-set", default=None, help="法条样本集路径（默认 qa_set_100.jsonl；传 qa_test_cases.jsonl 复现旧数字）"
    )
    parser.add_argument("--faithfulness", action="store_true", help="追加语义支撑判定（每题 1 次 LLM）")
    parser.add_argument("--reuse", action="store_true", help="复用 eval/results/raw 里已跑的样本")
    args = parser.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
