"""把原 40 条与生成的候选合成一份 100 条的评测集（`eval/qa_set_100.jsonl`）。

**为什么不直接改原文件**：`qa_test_cases.jsonl` 是已发布数字的来源（REPORT §3/§9 的 0.741 等），
改动它会让历史结论无法复现。新集合另存，报告里两套并存、分别标注样本量。

生成题的口径（必须写在报告里，否则数字会被误读）：
* 全部是 `single_hop`，因此 100 条集的题型分布比原 40 条更偏单跳；
* 金标是"这条法条就是问题的答案来源"，由生成时的二次校验保证，不是人工逐条复核；
* `colloquial=True`（题面是口语），`difficulty` 只按题面与原文的重合率粗分，不作分析用。

    python -m eval.build_set
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "eval" / "qa_test_cases.jsonl"
CANDIDATES = ROOT / "eval" / "generated" / "qa_candidates.jsonl"
OUT = ROOT / "eval" / "qa_set_100.jsonl"

KB = {
    "中华人民共和国劳动合同法": "kb_labor",
    "中华人民共和国劳动法": "kb_labor",
    "中华人民共和国民法典": "kb_civil",
}


def difficulty_of(overlap: float) -> str:
    if overlap <= 0.08:
        return "medium"          # 题面几乎不含原文用词，靠语义匹配
    if overlap <= 0.20:
        return "easy"
    return "easy"


def main() -> int:
    base = [json.loads(l) for l in BASE.read_text(encoding="utf-8").splitlines() if l.strip()]
    cand = [
        json.loads(l) for l in CANDIDATES.read_text(encoding="utf-8").splitlines() if l.strip()
    ]

    rows = list(base)
    seen = {a for r in base for a in r.get("expected_articles") or []}
    next_id = len(base) + 1

    for item in cand:
        law, no = item["law_name"], item["article_no"]
        key = f"{law}#{no}"
        if key in seen:                       # 同一条不重复出题
            continue
        seen.add(key)
        rows.append(
            {
                "id": f"q{next_id:02d}",
                "type": "single_hop",
                "difficulty": difficulty_of(item.get("overlap", 0.0)),
                "colloquial": True,
                "kb_ids": [KB.get(law, "kb_labor")],
                "question": item["question"],
                "reference_answer": f"《{law}》{no}：{item['article']}",
                "expected_articles": [key],
                "expected_laws": [law],
                "sub_questions": [],
                "source": "generated",
                "overlap": item.get("overlap"),
            }
        )
        next_id += 1

    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )

    types = Counter(r["type"] for r in rows)
    laws = Counter((r.get("expected_laws") or ["—"])[0] for r in rows)
    gen = sum(1 for r in rows if r.get("source") == "generated")
    print(f"写入 {OUT}：{len(rows)} 条（原有 {len(base)}，生成 {gen}）")
    print("题型：", dict(types))
    print("法域：", {k.replace("中华人民共和国", ""): v for k, v in laws.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
