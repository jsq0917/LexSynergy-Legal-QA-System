"""五道闸的单测：G1/G2/G3/G5 必须确定性，G4 用注入的假检查器验证契约。"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.evidence.ledger import EvidenceLedger
from app.evidence.models import Evidence
from app.verifier.gates import (
    claims_of,
    extract_amounts,
    extract_citations,
    normalize,
    numeric_mismatches,
    verbatim_ok,
)
from app.verifier.service import Verifier

LCL = "中华人民共和国劳动合同法"
Q47 = (
    "经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。"
    "六个月以上不满一年的，按一年计算；不满六个月的，向劳动者支付半个月工资的经济补偿。"
)


@pytest.fixture()
def ledger() -> EvidenceLedger:
    led = EvidenceLedger()
    led.add(
        [
            Evidence(law=LCL, article_no="第四十七条", quote=Q47, score=0.9, priority=20),
            Evidence(
                law=LCL,
                article_no="第八十七条",
                quote="用人单位违反本法规定解除或者终止劳动合同的，应当依照本法第四十七条规定的经济补偿标准的二倍向劳动者支付赔偿金。",
                score=0.8,
                priority=20,
            ),
            Evidence(
                law=LCL,
                article_no="第八十二条",
                quote="用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。",
                score=0.7,
                effective_to=date(2020, 12, 31),
                priority=20,
            ),
        ]
    )
    return led


def good_draft() -> str:
    return f"【依据】《{LCL}》第四十七条：“{Q47}”"


# ---------------------------------------------------------------- 纯函数
def test_normalize_strips_punctuation_only():
    assert normalize("经济补偿，按年限。") == normalize("经济补偿按年限")


@pytest.mark.parametrize(
    "quote,expected",
    [
        (Q47, True),
        (Q47.replace("，", ","), True),                    # 标点差异不算篡改
        (Q47.replace("六个月以上不满一年的，按一年计算；", ""), False),  # 删句子 = 篡改
        ("经济补偿按劳动者在本单位工作的年限……不满六个月的，向劳动者支付半个月工资的经济补偿。", True),
        ("经济补偿按劳动者……并不存在的句子……", False),
        ("", True),                                        # 没摘录原文就不判
    ],
)
def test_verbatim(quote, expected):
    assert verbatim_ok(quote, Q47) is expected


def test_extract_citations_with_chapter_path():
    cites = extract_citations("《中华人民共和国劳动合同法》·第二章·第四十七条：“经济补偿……”")
    assert len(cites) == 1
    assert cites[0].article_no == "第四十七条"
    assert cites[0].quote.startswith("经济补偿")


def test_extract_amounts_units():
    amounts = extract_amounts("补偿 72000 元，或 7.2 万元")
    assert [v for _, v in amounts] == [Decimal("72000"), Decimal("72000")]


def test_numeric_mismatch():
    bad = numeric_mismatches("应付 66000 元", [Decimal("72000")])
    assert bad and bad[0][0] == "66000 元"
    assert numeric_mismatches("应付 72000 元", [Decimal("72000")]) == []


def test_claims_split():
    assert len(claims_of("第一句话足够长可以算作断言。第二句同样是足够长的断言内容。")) == 2
    assert claims_of("太短。") == []          # 过短片段不算断言，避免噪声


# ---------------------------------------------------------------- 闸门
@pytest.mark.asyncio
async def test_pass(ledger):
    verdict = await Verifier().verify(good_draft(), ledger)
    assert verdict.passed
    assert verdict.action == "pass"
    assert verdict.skipped_gates == ("G4_misgrounded",)


@pytest.mark.asyncio
async def test_g1_fabricated_article(ledger):
    draft = f"《{LCL}》第一百四十七条：“经济补偿按劳动者在本单位工作的年限”"
    verdict = await Verifier().verify(draft, ledger)
    assert not verdict.passed
    assert verdict.failed_gates == ["G1_not_found"]
    assert verdict.action == "refine_retrieval"


@pytest.mark.asyncio
async def test_g2_misquote(ledger):
    tampered = Q47.replace("六个月以上不满一年的，按一年计算；", "")
    draft = f"《{LCL}》第四十七条：“{tampered}”"
    verdict = await Verifier().verify(draft, ledger)
    assert verdict.failed_gates == ["G2_misquote"]


@pytest.mark.asyncio
async def test_g3_expired(ledger):
    draft = (
        f"《{LCL}》第八十二条：“用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，"
        "应当向劳动者每月支付二倍的工资。”"
    )
    verdict = await Verifier().verify(draft, ledger, as_of=date(2026, 10, 1))
    assert verdict.failed_gates == ["G3_expired"]


@pytest.mark.asyncio
async def test_g5_numeric_downgrades(ledger):
    draft = good_draft() + "\n应付 66000 元。"
    verdict = await Verifier().verify(draft, ledger, calc_values=[Decimal("72000")])
    assert verdict.failed_gates == ["G5_numeric"]
    assert verdict.action == "downgrade"


@pytest.mark.asyncio
async def test_g5_allows_facts_amounts(ledger):
    draft = good_draft() + "\n月工资 12000 元，应付 72000 元。"
    verdict = await Verifier().verify(
        draft, ledger, calc_values=[Decimal("72000")], facts_amounts=[Decimal("12000")]
    )
    assert verdict.passed


@pytest.mark.asyncio
async def test_g4_unsupported_claim(ledger):
    async def checker(claims: list[str], context: str) -> list[str]:
        return [c for c in claims if "可以直接查封" in c]

    v = Verifier(support_checker=checker)
    draft = good_draft() + "\n你可以直接查封公司账户。"
    verdict = await v.verify(draft, ledger)
    assert verdict.failed_gates == ["G4_misgrounded"]
    assert verdict.action == "refine_retrieval"
    assert any("查封" in h for h in verdict.missing_hints)


@pytest.mark.asyncio
async def test_g4_exception_is_skipped_not_fatal(ledger):
    def boom(claims, context):
        raise RuntimeError("judge 挂了")

    verdict = await Verifier(support_checker=boom).verify(good_draft(), ledger)
    assert verdict.passed
    assert "G4_misgrounded" in verdict.skipped_gates


@pytest.mark.asyncio
async def test_no_citation_skips_g4(ledger):
    def should_not_run(claims, context):
        raise AssertionError("无引用时不应调用 G4")

    verdict = await Verifier(support_checker=should_not_run).verify("这是一句没有引用的回答。", ledger)
    assert verdict.passed


@pytest.mark.asyncio
async def test_multiple_fails_sorted(ledger):
    draft = (
        f"《{LCL}》第一百四十七条：“不存在的条文”\n"
        f"《{LCL}》第四十七条：“{Q47.replace('，', '')}篡改”\n"
        "金额 1 元。"
    )
    verdict = await Verifier().verify(draft, ledger, calc_values=[Decimal("72000")])
    gates = verdict.failed_gates
    assert gates == sorted(gates)
    assert "G5_numeric" in gates


# ---------------------------------------------------------------- 回归：引文归属
# 这两个 bug 都是在真实问答里暴露的，不是构造出来的：
# 早先版本在 ±(120/40) 窗口里找"最近的引号"，密集引用段落中会抓到**邻居条文**的引文，
# G2 把正确引用判成篡改（q01 一次误报 8 条、每题多烧 2 次 LLM 走补检索回路）。
def test_quote_binds_to_its_own_citation_in_dense_block():
    text = (
        f"- 《{LCL}》·第四章·第四十七条（E1）：“引文四十七”\n"
        f"- 《{LCL}》·第二章·第二十三条（E2）：“引文二十三”\n"
        f"- 《{LCL}》·第二章·第十九条（E3）：“引文十九”"
    )
    got = {c.article_no: c.quote for c in extract_citations(text)}
    assert got == {"第四十七条": "引文四十七", "第二十三条": "引文二十三", "第十九条": "引文十九"}


def test_quote_does_not_steal_from_previous_citation():
    text = f"依据《{LCL}》第二十条。又依据《中华人民共和国劳动法》第四十八条：“最低工资标准”"
    cites = extract_citations(text)
    assert cites[0].article_no == "第二十条"
    assert cites[0].quote == ""          # 中间隔了正文 → 不认领
    assert cites[1].quote == "最低工资标准"


def test_quote_preceding_citation_in_parentheses():
    text = f"“经济补偿按劳动者在本单位工作的年限”（《{LCL}》第四十七条）"
    assert extract_citations(text)[0].quote.startswith("经济补偿按劳动者")


def test_verbatim_accepts_partial_quote():
    """截取片段是合法引用；只有增删改才判篡改。"""
    assert verbatim_ok("经济补偿按劳动者在本单位工作的年限", Q47)
    assert not verbatim_ok("经济补偿按劳动者在本单位工作的年限，每满两年支付一个月工资", Q47)


def test_claims_of_skips_section_labels_and_meta():
    text = (
        "**1. 结论** 您问的是试用期法定上限。\n"
        "**2. 依据** 同一用人单位与同一劳动者只能约定一次试用期。\n"
        "**3. 提示** 具体以当地裁审口径为准。"
    )
    claims = claims_of(text)
    assert all("结论" not in c and "提示" not in c for c in claims)
    assert any("只能约定一次试用期" in c for c in claims)


@pytest.mark.asyncio
async def test_support_false_skips_g4(ledger):
    """补证据后复查：只跑四道确定性闸，不重复烧 G4 那次调用。"""
    ledger.add([Evidence(law=LCL, article_no="第四十七条", quote=Q47, score=1.0)])
    verdict = await Verifier().verify(
        f"《{LCL}》第四十七条：“{Q47}”", ledger, support=False
    )
    assert "G4_misgrounded" in verdict.skipped_gates
    assert verdict.passed


def test_ledger_resolves_law_alias():
    """《劳动合同法》是中文习惯写法，不能因为不是全称就判 G1 不存在。"""
    led = EvidenceLedger()
    led.add([Evidence(law=LCL, article_no="第四十一条", quote="用人单位可以裁减人员。", score=1.0)])
    assert led.canonical_law("劳动合同法") == LCL
    assert led.lookup("劳动合同法", "第四十一条") is not None
    assert led.canonical_law("民法典") is None        # 无此证据 → 不猜


@pytest.mark.asyncio
async def test_g1_accepts_short_law_name():
    led = EvidenceLedger()
    led.add([Evidence(law=LCL, article_no="第四十一条", quote="用人单位可以裁减人员。", score=1.0)])
    verdict = await Verifier().verify("《劳动合同法》第四十一条：“用人单位可以裁减人员。”", led)
    assert verdict.passed, verdict.to_dict()
