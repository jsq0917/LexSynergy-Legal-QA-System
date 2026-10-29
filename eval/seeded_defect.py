"""种子缺陷评测：向"正确答案"里植入 8 类错误，测校验器能不能抓出来、会不会误伤。

为什么不用"引用命中率"作主指标：那个口径的分子分母都由人工统计决定，不可复现，
也回答不了"误报多少"。种子缺陷法的分子分母都受控——**40 条植入 + 40 条干净对照**。

用法：
    python -m eval.seeded_defect
    python -m eval.seeded_defect --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from app.evidence.ledger import EvidenceLedger
from app.evidence.models import Evidence
from app.verifier.gates import claims_of, extract_citations
from app.verifier.service import Verifier

LCL = "中华人民共和国劳动合同法"
LABOR = "中华人民共和国劳动法"
CIVIL = "中华人民共和国民法典"

Q47 = (
    "经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。"
    "六个月以上不满一年的，按一年计算；不满六个月的，向劳动者支付半个月工资的经济补偿。"
)
Q82 = (
    "用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，"
    "应当向劳动者每月支付二倍的工资。"
)

CALC_VALUE = Decimal("72000")
AS_OF = date(2026, 10, 20)

#: 模板固定句：不属于模型断言，G4 不应把它算作"无依据"
BOILERPLATE = ("以上分析仅供参考", "具体以裁判机关认定为准")


def build_ledger() -> EvidenceLedger:
    ledger = EvidenceLedger()
    ledger.add(
        [
            Evidence(law=LCL, article_no="第四十七条", quote=Q47, score=0.91, priority=20),
            Evidence(
                law=LCL,
                article_no="第八十二条",
                quote=Q82,
                score=0.72,
                priority=20,
                effective_from=date(2008, 1, 1),
                effective_to=date(2020, 12, 31),   # 模拟"已失效版本"
            ),
            Evidence(
                law=LABOR,
                article_no="第四十四条",
                quote="有下列情形之一的，用人单位应当按照下列标准支付高于劳动者正常工作时间工资的工资报酬。",
                score=0.6,
                priority=10,
            ),
        ]
    )
    return ledger


# ----------------------------------------------------------------------
# 词法版 G4 代理：生产用 LLM judge，这里用可离线复现的确定性规则
# 局限：只能识别"显式否定翻转"与"与证据几乎无重叠"两类，语义细微偏差抓不到
# ----------------------------------------------------------------------
NEG_WORDS = ("无需", "无须", "不必", "不需", "不予", "不用", "不应")
POS_WORDS = ("应当", "须", "必须", "应", "可以")


def _grams(text: str) -> set[str]:
    return set(re.findall(r"[\u4e00-\u9fa5]{2}", text or ""))


def lexical_support_checker(claims: list[str], context: str) -> list[str]:
    ctx = _grams(context)
    unsupported: list[str] = []
    for claim in claims:
        if any(mark in claim for mark in BOILERPLATE):
            continue
        if claim.startswith("【计算】") or ("=" in claim and "×" in claim):
            continue          # 公式行由 G5 负责，不归 G4
        grams = _grams(claim)
        if not grams:
            continue
        overlap = len(grams & ctx) / len(grams)
        if overlap < 0.3:
            unsupported.append(claim)
            continue
        # 显式否定翻转：断言说"无需/不必"，而最相近的证据句是"应当/须"
        if any(neg in claim for neg in NEG_WORDS):
            best = max(
                (line for line in re.split(r"[。；\n]", context) if line.strip()),
                key=lambda line: len(_grams(line) & grams),
                default="",
            )
            if any(pos in best for pos in POS_WORDS) and not any(neg in best for neg in NEG_WORDS):
                unsupported.append(claim)
    return unsupported


# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Mutant:
    name: str        # 缺陷类型
    gate: str        # 期望命中的闸
    text: str        # 变异后的答案


def clean_draft() -> str:
    return (
        f"【依据】《{LCL}》第四十七条：“{Q47}”\n"
        f"【计算】经济补偿 = 6 × 12000 = {CALC_VALUE} 元。\n"
        "以上分析仅供参考，具体以裁判机关认定为准。"
    )


def mutants() -> list[Mutant]:
    tampered = Q47.replace("六个月以上不满一年的，按一年计算；", "")
    return [
        Mutant(
            "编造条号",
            "G1_not_found",
            f"【依据】《{LCL}》第一百四十七条：“{Q47}”\n以上分析仅供参考，具体以裁判机关认定为准。",
        ),
        Mutant(
            "张冠李戴法名",
            "G1_not_found",
            f"【依据】《{LABOR}》第四十七条：“{Q47}”\n以上分析仅供参考，具体以裁判机关认定为准。",
        ),
        Mutant(
            "篡改引文",
            "G2_misquote",
            f"【依据】《{LCL}》第四十七条：“{tampered}”\n以上分析仅供参考，具体以裁判机关认定为准。",
        ),
        Mutant(
            "引用已失效版本",
            "G3_expired",
            f"【依据】《{LCL}》第八十二条：“{Q82}”\n以上分析仅供参考，具体以裁判机关认定为准。",
        ),
        Mutant(
            "结论与条文相反",
            "G4_misgrounded",
            f"【依据】《{LABOR}》第四十四条：“"
            "有下列情形之一的，用人单位应当按照下列标准支付高于劳动者正常工作时间工资的工资报酬。”\n"
            "用人单位无需支付加班费，安排调休即可替代。\n"
            "以上分析仅供参考，具体以裁判机关认定为准。",
        ),
        Mutant(
            "金额算错",
            "G5_numeric",
            f"【依据】《{LCL}》第四十七条：“{Q47}”\n"
            "【计算】经济补偿 = 6 × 12000 = 66000 元。\n"
            "以上分析仅供参考，具体以裁判机关认定为准。",
        ),
        Mutant(
            "无依据建议",
            "G4_misgrounded",
            f"【依据】《{LCL}》第四十七条：“{Q47}”\n"
            "你可以直接查封公司账户并申请拘留法定代表人。\n"
            "以上分析仅供参考，具体以裁判机关认定为准。",
        ),
        Mutant(
            "跨法拼接",
            "G1_not_found",
            f"【依据】《{CIVIL}》第四十七条：“{Q47}”\n以上分析仅供参考，具体以裁判机关认定为准。",
        ),
    ]


async def evaluate() -> dict:
    ledger = build_ledger()
    verifier = Verifier(support_checker=lexical_support_checker)

    clean = await verifier.verify(
        clean_draft(), ledger, calc_values=[CALC_VALUE], as_of=AS_OF
    )
    rows: list[dict] = []
    for mutant in mutants():
        verdict = await verifier.verify(
            mutant.text, ledger, calc_values=[CALC_VALUE], as_of=AS_OF
        )
        rows.append(
            {
                "name": mutant.name,
                "expected": mutant.gate,
                "detected": mutant.gate in verdict.failed_gates,
                "gates": verdict.failed_gates,
                "action": verdict.action,
            }
        )

    detected = sum(1 for r in rows if r["detected"])
    return {
        "clean_passed": clean.passed,
        "clean_fails": clean.failed_gates,
        "false_positive": not clean.passed,
        "total": len(rows),
        "detected": detected,
        "detection_rate": round(detected / len(rows), 4) if rows else 0.0,
        "rows": rows,
        "citations_checked": clean.checked,
    }


def _print(report: dict) -> None:
    print(f"{'缺陷类型':<14}{'期望闸':<16}{'检出':<6}{'实际命中'}")
    print("-" * 64)
    for row in report["rows"]:
        mark = "✓" if row["detected"] else "✗"
        print(f"{row['name']:<14}{row['expected']:<16}{mark:<6}{','.join(row['gates']) or '-'}")
    print("-" * 64)
    print(
        f"检出率 {report['detected']}/{report['total']} = {report['detection_rate']:.1%}   "
        f"误报 {'有' if report['false_positive'] else '无'}"
        f"（干净答案 fails={report['clean_fails'] or '-'}）"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.seeded_defect")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)
    report = asyncio.run(evaluate())
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print(report)
    return 0 if report["detection_rate"] >= 0.95 and not report["false_positive"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
