"""生成多跳/跨条评测题：先指定 2–3 条法条的组合，再让模型写"必须同时用到这几条"的问题。

**为什么不沿用 `gen_qa_set.py` 的单条出题**：那个脚本按一条法条出题，天然只能造单跳题；
多跳题的金标必须是"几条法条的组合"，而条号组合不能靠模型自由发挥——一旦模型多写或少写
一条，金标就错了，指标会静默失真。所以这里反过来：**组合由人指定，模型只负责把组合写成
一个当事人会问的问题**。

两道机器校验（与 `gen_qa_set` 同构，但针对跨条场景）：

1. **题面泄漏原文**：问题与组合内全部法条原文的字符二元组重合率 > 0.45 直接丢弃；
2. **组合并非"都必须"**：逐条问"去掉这条后问题还能完整回答吗"，不能则从金标里剔除；
   剔除后不足 2 条 → 该题不构成多跳，整条丢弃。

    python -m eval.gen_multihop --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from app.core.logger import get_logger
from app.generation.llm_client import get_llm_client
from app.retrieval.corpus import Corpus
from config import get_settings

logger = get_logger("eval.multihop")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "eval" / "generated"
OUT = OUT_DIR / "multihop_candidates.jsonl"
EXISTING = ROOT / "eval" / "qa_set_100.jsonl"

LCL = "中华人民共和国劳动合同法"
LABOR = "中华人民共和国劳动法"
CIVIL = "中华人民共和国民法典"

#: 固定事实：测算类问题统一用这组日期与工资。
#: 固定而不是随机，是为了让"同一事实、不同法条组合"的题可以直接横向看，
#: 也避免模型编出跨不过百分点精度的日期。
CALC_FACTS = "2021 年 3 月 1 日入职、2026 年 10 月 20 日离职、月薪 12000 元"

#: (主题, [(法名, 条号), ...], flavor)。flavor=`calc` 时问题里必须带具体日期与工资。
#: 组合由人指定：模型只写题面，不决定金标。顺序即候选编号顺序（可复现）。
COMBOS: list[tuple[str, list[tuple[str, str]], str]] = [
    ("试用期不符合录用条件辞退是否有补偿", [(LCL, "第三十九条"), (LCL, "第四十六条")], "reason"),
    ("三期女职工合同到期能否终止", [(LCL, "第四十二条"), (LCL, "第四十五条")], "reason"),
    ("试用期内解除合同的法定限制", [(LCL, "第十九条"), (LCL, "第二十一条")], "reason"),
    ("违法约定试用期的赔偿", [(LCL, "第十九条"), (LCL, "第八十三条")], "reason"),
    ("未签书面劳动合同的二倍工资", [(LCL, "第十条"), (LCL, "第八十二条")], "reason"),
    ("竞业限制的经济补偿与期限", [(LCL, "第二十三条"), (LCL, "第二十四条")], "reason"),
    ("竞业限制违约金与补偿的关系", [(LCL, "第二十三条"), (LCL, "第二十五条")], "reason"),
    ("服务期违约金是否合法", [(LCL, "第二十二条"), (LCL, "第二十五条")], "reason"),
    ("拖欠工资的行政责任与加付赔偿", [(LCL, "第三十条"), (LCL, "第八十五条")], "reason"),
    ("强制 996 的工时上限", [(LCL, "第三十一条"), (LABOR, "第四十一条")], "reason"),
    ("劳务派遣员工被退回后能否解除", [(LCL, "第五十八条"), (LCL, "第六十五条")], "reason"),
    ("劳务派遣的岗位范围与自设派遣", [(LCL, "第六十六条"), (LCL, "第六十七条")], "reason"),
    ("单方解除未通知工会的后果", [(LCL, "第四十三条"), (LCL, "第八十七条")], "reason"),
    ("违法解除后能否要求继续履行", [(LCL, "第四十八条"), (LCL, "第八十七条")], "reason"),
    ("非全日制用工的终止与结算", [(LCL, "第六十八条"), (LCL, "第七十二条")], "reason"),
    ("连续工作满十年能否要求无固定期限", [(LCL, "第十四条"), (LCL, "第八十二条")], "reason"),
    ("劳动者主动辞职是否要赔公司", [(LCL, "第三十七条"), (LCL, "第九十条")], "reason"),
    ("劳动合同无效后怎么处理", [(LCL, "第二十六条"), (LCL, "第三十九条")], "reason"),
    ("解除合同后的手续与补偿", [(LCL, "第四十六条"), (LCL, "第五十条")], "reason"),
    ("规章制度违法的后果", [(LCL, "第四条"), (LCL, "第八十条")], "reason"),
    ("试用期工资能低于最低工资吗", [(LCL, "第十九条"), (LCL, "第二十条")], "reason"),
    ("试用期以不能胜任为由解除", [(LCL, "第四十条"), (LCL, "第二十一条")], "reason"),
    ("调岗协商与不能胜任解除", [(LCL, "第三十五条"), (LCL, "第四十条")], "reason"),
    ("公司不缴社保员工能否解除要补偿", [(LABOR, "第七十二条"), (LCL, "第三十八条")], "reason"),
    ("孕期女职工的保护范围", [(LABOR, "第二十九条"), (LCL, "第四十二条")], "reason"),
    ("离职时工资与手续怎么结", [(LABOR, "第五十条"), (LCL, "第五十条")], "reason"),
    ("休息休假与加班安排", [(LABOR, "第三十八条"), (LCL, "第三十一条")], "reason"),
    ("招用未解除合同员工的连带责任", [(LABOR, "第九十一条"), (LCL, "第九十四条")], "reason"),
    ("劳动争议的解决途径", [(LABOR, "第七十七条"), (LCL, "第七十九条")], "reason"),
    ("违法解除的后果与赔偿", [(LABOR, "第四十八条"), (LCL, "第八十七条")], "reason"),
    ("房东拒不退押金的违约责任", [(CIVIL, "第五百七十七条"), (CIVIL, "第七百零八条")], "reason"),
    ("违约金过高能否要求调低", [(CIVIL, "第五百八十五条"), (LCL, "第二十五条")], "reason"),
    ("婚前房产婚后共同还贷怎么分", [(CIVIL, "第一千零六十三条"), (CIVIL, "第一千零八十七条")], "reason"),
    ("借款逾期不还的利息", [(CIVIL, "第六百七十五条"), (CIVIL, "第六百七十六条")], "reason"),
    ("口头租房协议是否有效", [(CIVIL, "第七百零三条"), (CIVIL, "第七百零七条")], "reason"),
    ("员工执行工作任务致害谁担责", [(CIVIL, "第一千一百六十五条"), (CIVIL, "第一千一百九十一条")], "reason"),
    ("遗嘱与法定继承的关系", [(CIVIL, "第一千一百二十七条"), (CIVIL, "第一千一百三十三条")], "reason"),
    ("合同解除后的责任承担", [(CIVIL, "第五百六十三条"), (CIVIL, "第五百六十六条")], "reason"),
    ("离婚时无过错方能否索赔", [(CIVIL, "第一千零七十九条"), (CIVIL, "第一千零九十一条")], "reason"),
    ("履行原则与违约责任", [(CIVIL, "第五百零九条"), (CIVIL, "第五百七十七条")], "reason"),
    ("裁员的经济补偿怎么算", [(LCL, "第四十一条"), (LCL, "第四十六条"), (LCL, "第四十七条")], "calc"),
    ("没提前通知辞退的补偿与代通知金", [(LCL, "第四十条"), (LCL, "第四十六条"), (LCL, "第四十七条")], "calc"),
    ("拖欠工资被迫离职的补偿", [(LCL, "第三十八条"), (LCL, "第四十六条"), (LCL, "第四十七条")], "calc"),
    ("孕期被违法辞退的赔偿", [(LCL, "第四十二条"), (LCL, "第四十七条"), (LCL, "第八十七条")], "calc"),
    ("2008 年前入职的经济补偿分段", [(LCL, "第四十七条"), (LCL, "第九十七条")], "calc"),
    ("未签合同的二倍工资能要多少", [(LCL, "第十条"), (LCL, "第八十二条")], "calc"),
    ("加班费的倍数与欠付加罚", [(LABOR, "第四十四条"), (LCL, "第八十五条")], "calc"),
    ("违法解除后继续履行还是赔钱", [(LCL, "第四十八条"), (LCL, "第四十七条"), (LCL, "第八十七条")], "calc"),
]

GEN_SYSTEM = """你在为法律问答系统编写**多跳**评测题。给定 2–3 条法条，写一个普通当事人会亲口问出来的问题。

硬性要求：
1. 这个问题**必须同时用到给出的全部法条**才能完整回答；去掉任何一条都答不全；
2. 不得出现条号（"第X条"）、不得出现法律名称（"劳动合同法""民法典"等）、不得出现书名号；
3. 不得照抄法条原文，最多引用 6 个字的短语；用生活化说法（"老板""公司""我""房东"）；
4. 一句话，不超过 60 字；
5. 给一条问题后，再给每条法条对应的子问题（法条用语、≤20 字、顺序与法条一致），
   供检索分路使用。

只输出 JSON：{"question": "...", "sub_questions": ["...", "..."]}"""

GEN_SYSTEM_CALC = """你在为法律问答系统编写**多跳且涉及金额**的评测题。给定 2–3 条法条，写一个普通当事人会亲口问的问题。

硬性要求：
1. 问题里必须出现具体事实：{facts}；
2. 问题必须**同时问两件事**：一是定性（"公司这样做合法吗 / 该不该给"），二是定量（"具体能拿多少"）；
   只问金额会退化成单条题，必须两条都要；
3. 问题必须同时用到给出的全部法条才能完整回答；
4. 不得出现条号（"第X条"）、不得出现法律名称、不得出现书名号；不得照抄原文超过 6 个字；
5. 一句话，不超过 80 字；
6. 再给每条法条对应的子问题（法条用语、≤20 字、顺序与法条一致）。

只输出 JSON：{"question": "...", "sub_questions": ["...", "..."]}"""

CHECK_SYSTEM = """你是评测集质检员。给定【问题】与若干条【法条】（编号 A/B/C），判断回答问题**必须用到哪几条**。

判定口径：
- 逐条问："如果没有这一条，这个问题还能完整回答吗？" 能 → 不列入 needed；
- 只要问题要求的结论依赖该条的规则，就列入；
- 若 needed 少于 2 条，说明它不是多跳题，sufficient 记 false；
- 若给出的法条合起来仍答不全问题（需要别的条），sufficient 记 false。

只输出 JSON：{"needed": ["A", "B"], "sufficient": true, "reason": "简短理由"}"""

BAD_QUESTION = re.compile(r"第[一二三四五六七八九十百千零〇两0-9]{1,8}条|《|》")


def grams(text: str) -> set[str]:
    clean = re.sub(r"\s+", "", text)
    return {clean[i : i + 2] for i in range(len(clean) - 1)}


def overlap(question: str, article_text: str) -> float:
    q, a = grams(question), grams(article_text)
    return len(q & a) / max(1, len(q))


def load_existing_questions() -> list[str]:
    if not EXISTING.exists():
        return []
    rows = [json.loads(l) for l in EXISTING.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r["question"] for r in rows if r.get("question")]


def nearest_overlap(question: str, existing: list[str]) -> float:
    """与已有题面的最高二元组 Jaccard —— 挡掉"换个说法重复出题"。"""
    q = grams(question)
    best = 0.0
    for other in existing:
        o = grams(other)
        union = len(q | o)
        if union:
            best = max(best, len(q & o) / union)
    return best


def article_texts(corpus: Corpus, articles: list[tuple[str, str]]) -> list[dict]:
    index = {(p["law_name"], p["article_no"]): p for p in corpus.parents.values()}
    out = []
    for law, no in articles:
        parent = index.get((law, no))
        if parent is None:
            raise KeyError(f"语料里没有 {law}#{no} —— 组合写错了，先修 COMBOS")
        out.append({"law": law, "article_no": no, "content": parent["content"]})
    return out


async def gen_one(combo, corpus: Corpus, client, sem: asyncio.Semaphore, existing: list[str]) -> dict | None:
    topic, articles, flavor = combo
    law_names = "/".join(sorted({law for law, _ in articles}))
    texts = article_texts(corpus, articles)
    payload = "\n\n".join(f"【{t['law']}·{t['article_no']}】\n{t['content']}" for t in texts)
    system = GEN_SYSTEM if flavor == "reason" else GEN_SYSTEM_CALC.replace("{facts}", CALC_FACTS)

    async with sem:
        try:
            data, _ = await asyncio.to_thread(
                client.json_chat,
                [{"role": "system", "content": system}, {"role": "user", "content": payload}],
                max_tokens=500,
            )
            question = str(data.get("question") or "").strip()
            subs = [str(s).strip() for s in (data.get("sub_questions") or []) if str(s).strip()]
            if not question or BAD_QUESTION.search(question):
                logger.info("丢弃（题面不合规）：%s", topic)
                return None
            score = overlap(question, "\n".join(t["content"] for t in texts))
            if score > 0.45:
                logger.info("丢弃（重合 %.2f）：%s", score, question[:40])
                return None
            near = nearest_overlap(question, existing)
            if near > 0.70:
                logger.info("丢弃（与已有题面近似 %.2f）：%s", near, question[:40])
                return None

            labels = [chr(ord("A") + i) for i in range(len(texts))]
            check_payload = (
                f"【问题】{question}\n\n"
                + "\n\n".join(
                    f"【法条 {label}】{t['law']}{t['article_no']}：{t['content']}"
                    for label, t in zip(labels, texts)
                )
            )
            check, _ = await asyncio.to_thread(
                client.json_chat,
                [{"role": "system", "content": CHECK_SYSTEM}, {"role": "user", "content": check_payload}],
                max_tokens=200,
            )
            need_labels = [str(x).strip().upper() for x in (check.get("needed") or [])]
            needed = [i for i, label in enumerate(labels) if label in need_labels]
            if not check.get("sufficient") or len(needed) < 2:
                logger.info(
                    "丢弃（不构成多跳，needed=%s）：%s", need_labels, question[:40]
                )
                return None
        except Exception as exc:  # noqa: BLE001 - 单条失败不中断整批
            logger.warning("生成失败 %s：%s", topic, str(exc)[:120])
            return None

    gold = [texts[i] for i in needed]
    return {
        "topic": topic,
        "flavor": flavor,
        "law_area": law_names,
        "question": question,
        "sub_questions": subs[: len(gold)] or ["该条规则如何适用" for _ in gold],
        "expected_articles": [f"{t['law']}#{t['article_no']}" for t in gold],
        "expected_laws": sorted({t["law"] for t in gold}),
        "article_texts": [{"law": t["law"], "article_no": t["article_no"], "content": t["content"]} for t in gold],
        "dropped_articles": [
            f"{texts[i]['law']}#{texts[i]['article_no']}" for i in range(len(texts)) if i not in needed
        ],
        "overlap": round(score, 3),
        "near_duplicate": round(near, 3),
        "check_reason": str(check.get("reason") or "")[:160],
        "combo_size": len(articles),
        "source": "generated_multihop",
    }


async def main_async(args) -> int:
    settings = get_settings()
    if not settings.has_llm:
        print("未配置 LLM_API_KEY")
        return 2

    corpus = Corpus.load()
    existing = load_existing_questions()
    client = get_llm_client(settings)
    sem = asyncio.Semaphore(args.concurrency)

    combos = COMBOS
    if args.only:
        wanted = {t.strip() for t in args.only.split(",") if t.strip()}
        combos = [c for c in COMBOS if c[0] in wanted]
        unknown = wanted - {c[0] for c in combos}
        if unknown:
            print(f"未知主题：{sorted(unknown)}")
            return 2

    print(f"候选组合 {len(combos)} 组，开始出题…", flush=True)
    tasks = [gen_one(c, corpus, client, sem, existing) for c in combos]
    results = []
    for done, coro in enumerate(asyncio.as_completed(tasks), start=1):
        row = await coro
        if row:
            results.append(row)
        if done % 8 == 0:
            print(f"进度 {done}/{len(tasks)}，通过 {len(results)}", flush=True)

    # 按 COMBOS 顺序落盘，保证同一种子下产物可复现
    order = {topic: i for i, (topic, _, _) in enumerate(COMBOS)}
    results.sort(key=lambda r: order.get(r["topic"], 999))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # **按主题合并而不是覆盖**：分批补题时上一批必须保留（`gen_qa_set` 踩过覆写的坑）
    merged: dict[str, dict] = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                merged[row["topic"]] = row
    for row in results:
        merged[row["topic"]] = row
    rows = sorted(merged.values(), key=lambda r: order.get(r["topic"], 999))
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    calc = sum(1 for r in rows if r["flavor"] == "calc")
    print(f"\n本轮通过 {len(results)}/{len(combos)}，累计 {len(rows)} 条（calc {calc} / reason {len(rows) - calc}）")
    if results:
        ov = sorted(r["overlap"] for r in results)
        print(f"本轮题面重合率 中位 {ov[len(ov)//2]:.2f} / 最大 {ov[-1]:.2f}")
    print(f"写入 {OUT}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从指定条组合生成多跳评测题")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--only", default=None, help="只重跑指定主题（逗号分隔的主题名），结果按主题合并"
    )
    args = parser.parse_args(argv)
    started = time.time()
    code = asyncio.run(main_async(args))
    print(f"耗时 {int(time.time() - started)}s")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
