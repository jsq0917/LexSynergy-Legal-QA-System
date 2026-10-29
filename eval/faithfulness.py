"""阶段 9：支撑率评测（独立 judge + 可复现）。

**为什么不能再用 `compare_agents --faithfulness` 那一列**：判官与被判答案同族
（都用 deepseek-v4.1-flash），Prompt 没有版本号，`temperature` 取默认 0.1，
同一批答案重跑漂移 ±0.02 —— 所以 REPORT 里这一列长期标 ⚠、禁止引用。

本脚本不改任何答案，只改"谁来判、怎么判、判几次"：

1. **判官异构**：默认 `qwen3.7-plus`（非 deepseek 族），与生成模型不同源；
2. **参数钉死**：`temperature=0`、`enable_thinking=false`，Prompt 带版本号
   （`PROMPT_VERSION`）。改 Prompt 必须升版本，否则新旧结果不可比；
3. **重复测量**：同一批答案跑 N 次（默认 3），报告每次支撑率与**最大漂移**；
   漂移 ≥ `DRIFT_LIMIT` 就判 ❌，这一列继续标 ⚠。

评分口径（断言级，与 G4 同一套断言切分）：

    support = 1 - 匹配到的"无依据断言"数 / 断言总数

判官返回的条目先与断言做归一化匹配；匹配不上的记为 `stray`，**不计入分子**
（判官偶尔会把纯引文、过渡句列回来，那是 Prompt 偏差，不该算成答案的错）。

    python -m eval.faithfulness --limit 10      # 先小样本验链路
    python -m eval.faithfulness                 # 全量 × 3 次
    python -m eval.faithfulness --repeats 5     # 追加重复测量
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import statistics
import sys
import time
from pathlib import Path

from app.core.logger import get_logger
from app.generation.llm_client import get_llm_client
from app.verifier.gates import claims_of, normalize
from config import get_settings

logger = get_logger("eval.faith")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "eval" / "results" / "raw"
OUT_PATH = ROOT / "eval" / "results" / "faithfulness.json"

CONFIGS = ("A_baseline", "B_multi", "C_verified", "D_auto")

#: Prompt 版本号：任何措辞改动都必须升版本，否则历史数字失去可比性
PROMPT_VERSION = "faith-v1.0"
#: 漂移阈值（验收标准）：最大成对差值必须严格小于它
DRIFT_LIMIT = 0.01
#: 判官：非 deepseek 族（关键）。实测 3.7 系免费额度已耗尽，3.8-27b 可用
DEFAULT_JUDGE_MODEL = "qwen3.8-27b"
#: 判官失败率上限：超过就判"数据不足"，不能拿残缺样本的 0.0000 漂移当通过
MAX_FAIL_RATE = 0.10

FAITH_SYSTEM = f"""你是严格的法律事实核查员。给定【用户问题】【参考资料】【回答】，找出回答中**无法由参考资料支持**的断言。

只输出一个 JSON 对象，不要解释、不要用 Markdown 代码块：
{{"unsupported": ["断言原文", "..."]}}

判定口径：
- 与参考资料同义、可由其推出的命题 → 支持，不列入；
- 复述【用户问题】里用户自己给出的事实（入职时间、月薪、离职事由）→ 不列入；
- 复述算术代入过程（"6 × 12000 = 72000"）、复述系统给出的计算结果 → 不列入；
- 程序性建议（"可申请劳动仲裁"）、过渡句、免责声明 → 不列入；
- 与参考资料矛盾、或参考资料完全没提到该法律规则 → 列入。

判定器版本：{PROMPT_VERSION}
"""


# ---------------------------------------------------------------- Prompt
def judge_messages(question: str, answer: str, sources: list[dict]) -> list[dict]:
    """构造判官输入。**确定性**：同一输入必须产生逐字相同的 messages。"""
    evidence = "\n\n".join(
        f"【{e.get('law')}·{e.get('article_no')}】{(e.get('quote') or '')[:600]}"
        for e in (sources or [])[:6]
    )
    user = (
        f"【用户问题】\n{(question or '').strip()[:600]}\n\n"
        f"【参考资料】\n{evidence or '（无）'}\n\n"
        f"【回答】\n{(answer or '').strip()[:2500]}"
    )
    return [
        {"role": "system", "content": FAITH_SYSTEM},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------- 评分
def match_claim(item: str, claims: list[str], *, threshold: float = 0.6) -> int | None:
    """把判官返回的一条"无依据断言"对到具体断言上；对不上返回 None。

    三级匹配：包含 → 相似度。判官常把断言截短或改写标点，所以不能只做全等。
    """
    target = normalize(item)
    if not target:
        return None
    for idx, claim in enumerate(claims):
        text = normalize(claim)
        if not text:
            continue
        if target in text or text in target:
            return idx
    best, best_ratio = None, 0.0
    for idx, claim in enumerate(claims):
        ratio = difflib.SequenceMatcher(None, target, normalize(claim)).ratio()
        if ratio > best_ratio:
            best, best_ratio = idx, ratio
    return best if best_ratio >= threshold else None


def support_score(
    claims: list[str], unsupported: list[str], *, threshold: float = 0.6
) -> tuple[float, int]:
    """返回 (支撑率, stray 数)。无断言时返回 NaN（不参与均值）。"""
    if not claims:
        return float("nan"), 0
    hit: set[int] = set()
    stray = 0
    for item in unsupported:
        idx = match_claim(item, claims, threshold=threshold)
        if idx is None:
            stray += 1
        else:
            hit.add(idx)
    return round(max(0.0, 1 - len(hit) / len(claims)), 4), stray


def parse_unsupported(text: str) -> list[str]:
    """容错解析判官输出：代码块、前后缀噪声、字段名变形都试一遍。"""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3]
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    items = data.get("unsupported")
    if not isinstance(items, list):
        return []
    return [str(x) for x in items if isinstance(x, (str, int, float)) and str(x).strip()]


# ---------------------------------------------------------------- 漂移
def drift_of(per_run: dict[str, list[float]]) -> tuple[float, str | None]:
    """所有配置上、任意两次之间支撑率差的最大值；同时返回最不稳的配置。"""
    worst, worst_cfg = 0.0, None
    for config, values in per_run.items():
        clean = [v for v in values if v == v]
        for i in range(len(clean)):
            for j in range(i + 1, len(clean)):
                gap = abs(clean[i] - clean[j])
                if gap > worst:
                    worst, worst_cfg = gap, config
    return round(worst, 4), worst_cfg


def mean(values: list[float]) -> float:
    clean = [v for v in values if v == v]
    return round(statistics.fmean(clean), 4) if clean else float("nan")


def _fmt(value: float) -> str:
    return "  -   " if value != value else f"{value:.4f}"


# ---------------------------------------------------------------- 执行
def load_cases(configs: list[str], *, limit: int = 0) -> list[dict]:
    cases: list[dict] = []
    for config in configs:
        path = RAW_DIR / f"law_{config}.jsonl"
        if not path.exists():
            logger.warning("缺少原始输出：%s（先跑 compare_agents）", path.name)
            continue
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if limit:
            rows = rows[:limit]
        for row in rows:
            row.setdefault("config", config)
            cases.append(row)
    return cases


async def judge_all(
    cases: list[dict],
    client,
    *,
    model: str,
    concurrency: int = 8,
) -> tuple[list[dict], int, int]:
    """判一遍。返回 (逐题行, 失败数, 总 completion tokens)。"""
    sem = asyncio.Semaphore(max(1, concurrency))
    tokens = 0

    async def one(case: dict) -> dict:
        nonlocal tokens
        claims = claims_of(case.get("text") or "")
        key = f"{case['config']}:{case['sample_id']}"
        if not claims:
            return {"key": key, "config": case["config"], "score": float("nan"),
                    "claims": 0, "stray": 0, "error": "no_claims"}
        messages = judge_messages(
            case.get("question") or "", case.get("text") or "", case.get("sources") or []
        )
        async with sem:
            try:
                result = await asyncio.to_thread(
                    client.chat,
                    messages,
                    model=model,
                    temperature=0.0,
                    max_tokens=700,
                    json_mode=True,
                    enable_thinking=False,
                )
                tokens += getattr(result.usage, "completion_tokens", 0)
                unsupported = parse_unsupported(result.text)
                score, stray = support_score(claims, unsupported)
                return {"key": key, "config": case["config"], "score": score,
                        "claims": len(claims), "stray": stray,
                        "unsupported": unsupported, "error": None}
            except Exception as exc:  # noqa: BLE001 - 单题失败标 NaN，不污染均值
                logger.warning("判官失败 %s：%s", key, str(exc)[:120])
                return {"key": key, "config": case["config"], "score": float("nan"),
                        "claims": len(claims), "stray": 0, "error": type(exc).__name__}

    rows = await asyncio.gather(*(one(case) for case in cases))
    failures = sum(1 for row in rows if row["error"])
    return rows, failures, tokens


def summarise(rows: list[dict], configs: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for config in configs:
        out[config] = mean([r["score"] for r in rows if r["config"] == config])
    return out


def report(per_run: dict[str, list[float]], meta: dict) -> tuple[str, str]:
    """返回 (表格文本, 判定)。判定三态：ok / insufficient / fail。

    **0.0000 漂移不等于通过**：判官全挂时每次都是 NaN，差值也是 0。
    所以先查覆盖率，再谈漂移 —— 这是本脚本最容易自欺的一处。
    """
    runs = len(next(iter(per_run.values()))) if per_run else 0
    head = f"{'配置':<14}" + "".join(f"{f'第{i + 1}次':>8}" for i in range(runs))
    head += f"{'均值':>9}{'漂移':>8}"
    lines = [head, "-" * len(head)]
    for config, values in per_run.items():
        line = f"{config:<14}" + "".join(f"{_fmt(v):>8}" for v in values)
        line += f"{_fmt(mean(values)):>9}{_fmt(_drift_cfg(values)):>8}"
        lines.append(line)

    drift, worst_cfg = drift_of(per_run)
    fail_rate = meta["failures"] / max(1, meta["cases"] * max(1, runs))
    covered = runs >= 2 and all(
        len([v for v in values if v == v]) >= 2 for values in per_run.values()
    )
    lines.append("")
    lines.append(
        f"判官 {meta['judge_model']} · prompt {PROMPT_VERSION} · temperature=0 · "
        f"{meta['repeats']} 次 · {meta['cases']} 题 · 失败 {meta['failures']}"
    )
    if not covered or fail_rate > MAX_FAIL_RATE:
        status = "insufficient"
        lines.append(
            f"覆盖率不足（失败率 {fail_rate:.1%}，阈值 {MAX_FAIL_RATE:.0%}）→ "
            "⏳ 无法判定，不能拿 0 漂移当通过"
        )
    elif drift < DRIFT_LIMIT:
        status = "ok"
        lines.append(
            f"最大漂移 {drift:.4f}（{worst_cfg}）< {DRIFT_LIMIT:.2f} → ✅ 可复现，这一列可以引用"
        )
    else:
        status = "fail"
        lines.append(
            f"最大漂移 {drift:.4f}（{worst_cfg}）≥ {DRIFT_LIMIT:.2f} → ❌ 不可复现，继续标 ⚠"
        )
    return "\n".join(lines), status


def _drift_cfg(values: list[float]) -> float:
    clean = [v for v in values if v == v]
    if len(clean) < 2:
        return 0.0
    return round(max(clean) - min(clean), 4)


async def run(args: argparse.Namespace) -> int:
    s = get_settings()
    if not s.has_llm and not args.dry_run:
        print("未配置 LLM_API_KEY，无法调用判官（可用 --dry-run 检查 Prompt）", file=sys.stderr)
        return 2

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    cases = load_cases(configs, limit=args.limit)
    if not cases:
        print("没有可判的答案：先跑 `python -m eval.compare_agents`", file=sys.stderr)
        return 2

    if args.dry_run:
        sample = next((c for c in cases if claims_of(c.get("text") or "")), cases[0])
        messages = judge_messages(
            sample.get("question") or "", sample.get("text") or "", sample.get("sources") or []
        )
        print(f"cases={len(cases)}  prompt={PROMPT_VERSION}  model={args.judge_model}")
        print(messages[1]["content"][:600])
        return 0

    client = get_llm_client(s)
    started = time.perf_counter()
    per_run: dict[str, list[float]] = {}
    run_rows: list[dict] = []
    failures = 0
    tokens = 0
    for i in range(args.repeats):
        print(f"第 {i + 1}/{args.repeats} 次判定（{len(cases)} 题，并发 {args.concurrency}）…", flush=True)
        rows, fails, used = await judge_all(
            cases, client, model=args.judge_model, concurrency=args.concurrency
        )
        failures += fails
        tokens += used
        summary = summarise(rows, configs)
        for config, score in summary.items():
            per_run.setdefault(config, []).append(score)
        run_rows.append({"run": i + 1, "scores": summary})

    text, status = report(
        per_run,
        {
            "judge_model": args.judge_model,
            "repeats": args.repeats,
            "cases": len(cases),
            "failures": failures,
        },
    )
    print("\n=== 支撑率（独立 judge）===")
    print(text)

    drift, worst_cfg = drift_of(per_run)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "judge_model": args.judge_model,
        "prompt_version": PROMPT_VERSION,
        "temperature": 0.0,
        "drift_limit": DRIFT_LIMIT,
        "drift": drift,
        "worst_config": worst_cfg,
        "status": status,
        "passed": status == "ok",
        "repeats": args.repeats,
        "cases": len(cases),
        "failures": failures,
        "completion_tokens": tokens,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "per_config": per_run,
        "runs": run_rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n机器可读结果：{OUT_PATH}")
    return 0 if status == "ok" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="支撑率评测（独立 judge，可复现）")
    parser.add_argument("--configs", default=",".join(CONFIGS), help="要评的配置（逗号分隔）")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--repeats", type=int, default=3, help="同一批答案重复判定次数")
    parser.add_argument("--limit", type=int, default=0, help="每个配置只取前 N 题（调试用）")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="只打印 Prompt，不调用判官")
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
