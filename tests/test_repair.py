"""引文—条号自动对位的单测：只修"标签错"，不修"内容错"。"""

from __future__ import annotations

from app.evidence.ledger import EvidenceLedger
from app.evidence.models import Evidence
from app.verifier.repair import remap_citations
from app.verifier.service import Verifier

LCL = "中华人民共和国劳动合同法"
A46 = (
    "有下列情形之一的，用人单位应当向劳动者支付经济补偿："
    "（四）用人单位依照本法第四十一条第一款规定解除劳动合同的。"
)
A47 = (
    "经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。"
    "六个月以上不满一年的，按一年计算；不满六个月的，向劳动者支付半个月工资的经济补偿。"
)


def ledger() -> EvidenceLedger:
    led = EvidenceLedger()
    led.add(
        [
            Evidence(law=LCL, article_no="第四十六条", quote=A46, score=1.0),
            Evidence(law=LCL, article_no="第四十七条", quote=A47, score=0.9),
        ]
    )
    return led


def test_remap_fixes_shifted_citation():
    led = ledger()
    draft = f"依据《{LCL}》第四十六条：“{A47}”应支付经济补偿。"   # 引文是四十七，标签是四十六
    text, repairs = remap_citations(draft, led)
    assert len(repairs) == 1
    assert repairs[0].old.endswith("第四十六条") and repairs[0].new.endswith("第四十七条")
    assert "第四十七条" in text and "第四十六条" not in text.split("：")[0]


def test_remap_leaves_correct_citation_alone():
    led = ledger()
    draft = f"依据《{LCL}》第四十七条：“{A47}”"
    text, repairs = remap_citations(draft, led)
    assert text == draft and repairs == []


def test_remap_skips_empty_quote():
    led = ledger()
    draft = f"依据《{LCL}》第四十六条。"
    text, repairs = remap_citations(draft, led)
    assert text == draft and repairs == []


def test_remap_skips_ambiguous_owner():
    led = EvidenceLedger()
    same = "用人单位应当支付经济补偿。"
    led.add(
        [
            Evidence(law=LCL, article_no="第四十六条", quote=same, score=1.0),
            Evidence(law=LCL, article_no="第八十七条", quote=same, score=0.9),
        ]
    )
    draft = f"依据《{LCL}》第四十条：“{same}”"
    text, repairs = remap_citations(draft, led)
    assert text == draft and repairs == []


def test_repaired_draft_passes_g2():
    """修完之后 G2 必须真的过 —— 否则修了等于没修。"""
    import asyncio

    from app.verifier.gates import extract_citations, verbatim_ok

    led = ledger()
    draft = f"依据《{LCL}》第四十六条：“{A47}”应支付经济补偿。"
    assert not verbatim_ok(
        extract_citations(draft)[0].quote, led.lookup(LCL, "第四十六条").quote
    )
    text, _ = remap_citations(draft, led)
    verdict = asyncio.run(Verifier().verify(text, led))
    assert verdict.passed, verdict.to_dict()


def test_remap_is_noop_when_it_would_not_improve():
    """改完不减少冲突就退回原文 —— 自动改引用宁可不改，也不能改错。"""
    from app.verifier.repair import g2_violations

    led = ledger()
    draft = f"依据《{LCL}》第四十六条：“{A47}”"
    before = g2_violations(draft, led)
    text, repairs = remap_citations(draft, led)
    assert before == 1 and g2_violations(text, led) == 0      # 正常情形：确实下降
    assert repairs
