"""延迟分阶段探针：把"端到端 P95"拆成"哪一段慢"。

只报总延迟没有用 —— 用户问"为什么慢"，答案必须能落到具体阶段
（分诊 / 检索 / 撰稿 / 校验 / 回路）。这些耗时由 `Answer.timings` 提供，
和前端时间线用的是同一份数据，因此**看到的与量到的一致**。

    python -m eval.latency_probe                  # 默认 12 题
    python -m eval.latency_probe --limit 30 --policy fast
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from app.runtime import build_pipeline
from config import get_settings

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "eval" / "qa_set_100.jsonl"
CALC = ROOT / "eval" / "calc_cases.jsonl"

STAGES = ("triage_ms", "dag_ms", "synth_ms", "verify_ms", "total_ms")


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="分阶段延迟探针")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--policy", default=None, choices=("auto", "fast", "full"))
    parser.add_argument("--mix", action="store_true", help="加入测算题（默认只跑法条题）")
    args = parser.parse_args(argv)

    if not get_settings().has_llm:
        print("需要 LLM_API_KEY")
        return 2

    import os

    from config import reset_settings

    if args.policy:
        os.environ["ROUTING_POLICY"] = args.policy
        reset_settings()

    rows = [json.loads(l) for l in SAMPLES.read_text(encoding="utf-8").splitlines() if l.strip()]
    questions = [(r["id"], r["question"]) for r in rows[: args.limit]]
    if args.mix:
        calc = [json.loads(l) for l in CALC.read_text(encoding="utf-8").splitlines() if l.strip()]
        questions += [(r["id"], r["question"]) for r in calc[: max(1, args.limit // 3)]]

    pipeline = build_pipeline()

    async def run_all() -> list[dict]:
        out = []
        for sid, question in questions:
            ans = await pipeline.run(question)
            out.append({"id": sid, "mode": ans.mode, "degraded": ans.degraded, **ans.timings})
            print(
                f"  {sid} mode={ans.mode or '—':<4} total={ans.elapsed_ms:>6}ms "
                f"first={ans.timings.get('first_token_ms', '—')!s:>5} "
                f"triage={ans.timings.get('triage_ms', 0):>5} "
                f"dag={ans.timings.get('dag_ms', 0):>5} "
                f"synth={ans.timings.get('synth_ms', 0):>5} "
                f"verify={ans.timings.get('verify_ms', 0):>5} loops={ans.loops}"
            )
        return out

    print(f"跑 {len(questions)} 题（policy={get_settings().routing_policy}）…")
    results = asyncio.run(run_all())

    print("\n阶段        P50      P95     最大    占比")
    print("-" * 44)
    total_p95 = pct([r.get("total_ms", 0) for r in results], 0.95) or 1
    for stage in STAGES:
        values = [float(r.get(stage, 0) or 0) for r in results]
        share = pct(values, 0.95) / total_p95
        print(
            f"{stage[:-3]:<10}{pct(values, 0.5):>7.0f}{pct(values, 0.95):>9.0f}"
            f"{max(values):>9.0f}{share:>9.0%}"
        )
    print("-" * 44)

    # 首字延迟单独报：它是"体感"，与总时长不是一回事。两个口径都保留：
    # first_token_ms = 请求开始 → 首个 delta；synth_first_token_ms = 撰稿调用开始 → 首个 delta。
    first = [float(r["first_token_ms"]) for r in results if r.get("first_token_ms") is not None]
    synth_first = [
        float(r["synth_first_token_ms"])
        for r in results
        if r.get("synth_first_token_ms") is not None
    ]
    if first:
        print(
            f"首字(请求) P50 {pct(first, 0.5):.0f}ms  P95 {pct(first, 0.95):.0f}ms  "
            f"最大 {max(first):.0f}ms  n={len(first)}"
        )
    if synth_first:
        print(
            f"首字(撰稿) P50 {pct(synth_first, 0.5):.0f}ms  "
            f"P95 {pct(synth_first, 0.95):.0f}ms  最大 {max(synth_first):.0f}ms"
        )
    if not first:
        print("首字：本次没有流式输出（检查 LLM 是否启用 / 稿件是否走模板）")

    degraded = [r for r in results if r.get("degraded")]
    loops = [r for r in results if (r.get("loops") or 0) > 0]
    print(f"降级 {len(degraded)}/{len(results)}，走回路 {len(loops)}/{len(results)}")
    if degraded:
        print("降级原因：", {r["id"]: r["degraded"] for r in degraded})

    out = ROOT / "eval" / "results" / "latency_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
