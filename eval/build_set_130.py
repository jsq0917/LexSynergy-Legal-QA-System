"""合成 130 条评测集：原 100 条 + 30 条多跳/复杂题（`eval/qa_set_130.jsonl`）。

**为什么不覆盖 `qa_set_100.jsonl`**：100 条集已发布过数字（REPORT §9.1–§9.3），
覆盖会让历史结论无法复现。130 条是**新集合**，报告里两套并存、分别标注样本量。

选题口径（必须写进报告，否则数字会被误读）：

1. 只收**跨条必要性机器校验通过**的候选（`eval/gen_multihop.py` 的 needed ≥ 2）；
2. 人工复核条组合（3 条被剔除：派遣"随时走人"、离职补偿时点、包工头连带责任——
   题面问的规则超出所给法条，金标不完整）；
3. 数量分配：测算类（3 条法条 + 具体金额）归 `complex`，其余归 `multi_hop`；
   为保留法域多样性，多跳题按 **劳动 17 / 民事 8** 取，而不是按文件顺序截断。

    python -m eval.build_set_130
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "eval" / "qa_set_100.jsonl"
CANDIDATES = ROOT / "eval" / "generated" / "multihop_candidates.jsonl"
OUT = ROOT / "eval" / "qa_set_130.jsonl"

KB = {
    "中华人民共和国劳动合同法": "kb_labor",
    "中华人民共和国劳动法": "kb_labor",
    "中华人民共和国民法典": "kb_civil",
}

#: 人工复核后剔除的候选：题面问的规则超出所给法条，金标不完整（保留记录，不静默丢弃）
EXCLUDED_TOPICS = {
    "劳务派遣员工被退回后能否解除",       # 题面问"能不能随时走人"，需第三十七条，未给
    "离职时工资与手续怎么结",             # 题面问"能马上要补偿吗"，需第四十六条，未给
    "招用未解除合同员工的连带责任",       # 题面是包工头，需第九十四条；给的第九十一条不对题
}

#: 多跳题按法域配额取，避免"文件顺序"把民事题全部截掉（17 + 8 = 25 条多跳）
QUOTA = {"kb_labor": 17, "kb_civil": 8}


def build_reference(row: dict) -> str:
    """金标参考答案 = 逐条原文（不是模型复述）—— 供人工核对，不参与自动打分。"""
    parts = [
        f"《{t['law'].replace('中华人民共和国', '')}》{t['article_no']}：{t['content']}"
        for t in row["article_texts"]
    ]
    return "\n".join(parts)


def main() -> int:
    base = [json.loads(l) for l in BASE.read_text(encoding="utf-8").splitlines() if l.strip()]
    cand = [
        json.loads(l)
        for l in CANDIDATES.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    calc = [r for r in cand if r["flavor"] == "calc" and r["topic"] not in EXCLUDED_TOPICS]
    reason = [r for r in cand if r["flavor"] != "calc" and r["topic"] not in EXCLUDED_TOPICS]

    picked: list[dict] = list(calc)
    used = Counter()
    for row in reason:
        kb = KB.get(row["expected_laws"][0], "kb_labor")
        if used[kb] >= QUOTA.get(kb, 99):
            continue
        used[kb] += 1
        picked.append(row)

    target = 30
    if len(picked) > target:                 # 配额之和应正好 30；多出来时按原顺序截断
        picked = picked[:target]
    if len(picked) < target:
        print(f"⚠ 可用候选只有 {len(picked)} 条（目标 {target}），先补跑 eval.gen_multihop")

    next_id = len(base) + 1
    rows = list(base)
    for item in picked:
        laws = item["expected_laws"]
        rows.append(
            {
                "id": f"q{next_id:03d}",
                "type": "complex" if item["flavor"] == "calc" or len(item["expected_articles"]) >= 3 else "multi_hop",
                "difficulty": "hard",
                "colloquial": True,
                "kb_ids": sorted({KB.get(law, "kb_labor") for law in laws}),
                "question": item["question"],
                "reference_answer": build_reference(item),
                "expected_articles": item["expected_articles"],
                "expected_laws": laws,
                "sub_questions": item["sub_questions"],
                "source": "generated_multihop",
                "topic": item["topic"],
                "overlap": item.get("overlap"),
                "gold_review": "necessity_checked+manual_combo",
            }
        )
        next_id += 1

    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )

    types = Counter(r["type"] for r in rows)
    sources = Counter(r.get("source", "original") for r in rows)
    print(f"写入 {OUT}：{len(rows)} 条（原有 {len(base)}，新增 {len(picked)}）")
    print("题型：", dict(types))
    print("来源：", dict(sources))
    print("新增题法域：", dict(used))
    print("新增题金标条数分布：", dict(Counter(len(r['expected_articles']) for r in picked)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
