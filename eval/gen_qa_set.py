"""扩充评测集：给一条法条，让模型写"当事人会问的口语问题"，二次校验后入库。

**为什么不用手写**：60 条带金标条号的题，手写等于 60 次记忆性断言，错一条就污染指标。
这里让金标直接来自语料本身——问题是为某条法条生成的，答案就是那条法条的原文。

**两个必须防的偏差**：

1. **题面泄漏原文**（"经济补偿按劳动者在本单位工作的年限……怎么算"）——
   这样检索变成词组匹配，召回虚高。所以强制要求：不得照抄原文，且程序化检查
   字符二元组重合率，超过阈值的直接丢弃。
2. **一条题其实需要别的法条**（生成的题问超出该条范围）——所以第二次调用专门判
   "这条法条能否直接回答这个问题"，判否的直接丢弃。

    python -m eval.gen_qa_set --per-law 劳动合同法=24 劳动法=12 民法典=24 --seed 20261028
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from pathlib import Path

from app.core.logger import get_logger
from app.generation.llm_client import get_llm_client
from app.retrieval.corpus import Corpus
from config import get_settings

logger = get_logger("eval.gen")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "eval" / "generated"
EXISTING = ROOT / "eval" / "qa_test_cases.jsonl"

GEN_SYSTEM = """你在为法律问答系统编写评测题。给定一条法条，写一个**普通当事人会亲口问出来**的问题。

硬性要求：
1. 不得出现条号（"第X条"）、不得出现法律名称（"劳动合同法"等）；
2. 不得照抄法条原文，最多引用 6 个字的短语；要用生活化说法（"老板""公司""我""房东"）；
3. 这个问题**必须能由给定法条直接回答**，不需要再查别的条；
4. 一句话，不超过 40 字，像真人在咨询。
5. 场景化优先：给一个具体情境，而不是泛泛地问"法律规定是什么"。

只输出 JSON：{"question": "...", "scenario": "一句话说明你为什么这么问"}"""

CHECK_SYSTEM = """你是评测集质检员。判断【问题】是否**主要依靠【法条】就能回答**。

判 ok=false 的情形：
- 问题问的事情这条法条没规定，需要别的条文；
- 问题包含多个子问题，答案跨多条；
- 问题里的具体数字/期限与该条不一致，会造成误导。

只输出 JSON：{"ok": true/false, "reason": "简短理由"}"""

BAD_QUESTION = re.compile(r"第[一二三四五六七八九十百千零〇两0-9]{1,8}条|《|》")
#: 立法目的条、施行日期条、术语定义条：写不出"当事人会问"的问题，抽到就是噪声
TRIVIAL = re.compile(
    r"制定本法|本法自.*施行|施行之日|本法所称|依据本法|以上.*以下.*以内.*届满|"
    r"^为了(保护|维护|规范|完善|保障)"
)


def grams(text: str) -> set[str]:
    clean = re.sub(r"\s+", "", text)
    return {clean[i : i + 2] for i in range(len(clean) - 1)}


def overlap(question: str, article: str) -> float:
    """问题与原文的二元组重合率（相对问题长度）：越高越像抄原文。"""
    q, a = grams(question), grams(article)
    return len(q & a) / max(1, len(q))


def load_used() -> set[str]:
    """已经用过的条文：原 40 条 + 已生成的候选，避免生成重复题。"""
    used: set[str] = set()
    files = [EXISTING, OUT_DIR / "qa_candidates.jsonl"]
    for path in files:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            used |= set(row.get("expected_articles") or [])
            if row.get("law_name") and row.get("article_no"):
                used.add(f"{row['law_name']}#{row['article_no']}")
    return used


def pick_articles(corpus: Corpus, per_law: dict[str, int], seed: int) -> list[dict]:
    used = load_used()
    rng = random.Random(seed)
    picked: list[dict] = []
    for law, n in per_law.items():
        pool = [
            p
            for p in corpus.parents.values()
            if p["law_name"] == law
            and f"{law}#{p['article_no']}" not in used
            and 60 <= len(p["content"]) <= 600
            and not TRIVIAL.search(p["content"])
        ]
        pool.sort(key=lambda p: p["article_no"])
        take = rng.sample(pool, min(n, len(pool)))
        picked.extend(sorted(take, key=lambda p: p["article_no"]))
        logger.info("%s：候选 %d，抽取 %d", law, len(pool), len(take))
    return picked


async def gen_one(article: dict, client, sem: asyncio.Semaphore) -> dict | None:
    law, no, content = article["law_name"], article["article_no"], article["content"]
    async with sem:
        try:
            data, _ = await asyncio.to_thread(
                client.json_chat,
                [
                    {"role": "system", "content": GEN_SYSTEM},
                    {"role": "user", "content": f"【法条】{content}"},
                ],
                max_tokens=400,
            )
            question = str(data.get("question") or "").strip()
            if not question or BAD_QUESTION.search(question):
                return None
            score = overlap(question, content)
            if score > 0.45:                     # 抄得太狠，检索会变成词组匹配
                logger.info("丢弃（重合 %.2f）：%s", score, question[:30])
                return None

            check, _ = await asyncio.to_thread(
                client.json_chat,
                [
                    {"role": "system", "content": CHECK_SYSTEM},
                    {"role": "user", "content": f"【法条】{content}\n\n【问题】{question}"},
                ],
                max_tokens=200,
            )
            if not check.get("ok"):
                logger.info("丢弃（不匹配：%s）：%s", check.get("reason", "")[:40], question[:30])
                return None
        except Exception as exc:  # noqa: BLE001 - 单条失败不中断整批
            logger.warning("生成失败 %s#%s：%s", law, no, str(exc)[:100])
            return None

    return {
        "question": question,
        "law_name": law,
        "article_no": no,
        "article": content,
        "overlap": round(score, 3),
        "scenario": str(data.get("scenario") or "")[:120],
    }


async def main_async(args) -> int:
    settings = get_settings()
    if not settings.has_llm:
        print("未配置 LLM_API_KEY")
        return 2

    per_law = {}
    for item in args.per_law:
        name, n = item.split("=")
        per_law[name.strip()] = int(n)

    corpus = Corpus.load()
    articles = pick_articles(corpus, per_law, args.seed)
    print(f"待生成 {len(articles)} 条", flush=True)

    client = get_llm_client(settings)
    sem = asyncio.Semaphore(args.concurrency)
    tasks = [gen_one(a, client, sem) for a in articles]
    results = []
    for done, coro in enumerate(asyncio.as_completed(tasks), start=1):
        row = await coro
        if row:
            results.append(row)
        if done % 10 == 0:
            print(f"进度 {done}/{len(tasks)}，已通过 {len(results)}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "qa_candidates.jsonl"
    # **合并而不是覆盖**：分批补题时，上一批的结果必须保留
    # （第一次写这个脚本时直接覆盖，54 条白跑了一轮）
    merged: dict[tuple[str, str], dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                merged[(row["law_name"], row["article_no"])] = row
    for row in results:
        merged[(row["law_name"], row["article_no"])] = row
    rows = sorted(merged.values(), key=lambda r: (r["law_name"], r["article_no"]))
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print(f"\n本轮通过 {len(results)}/{len(articles)}，累计 {len(rows)} 条，写入 {path}")
    if results:
        ov = sorted(r["overlap"] for r in results)
        print(f"题面重合率 中位 {ov[len(ov)//2]:.2f} / 最大 {ov[-1]:.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从法条生成评测题（含二次校验）")
    parser.add_argument(
        "--per-law",
        nargs="+",
        default=["中华人民共和国劳动合同法=24", "中华人民共和国劳动法=12", "中华人民共和国民法典=24"],
        help="每部法律生成多少条，格式 法名=数量",
    )
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args(argv)
    started = time.time()
    code = asyncio.run(main_async(args))
    print(f"耗时 {int(time.time() - started)}s")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
