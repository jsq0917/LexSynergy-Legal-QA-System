"""证据账本：编号、去重、预算装配、时效过滤。"""

from __future__ import annotations

from datetime import date

import pytest

from app.evidence.ledger import EvidenceLedger
from app.evidence.models import Evidence

LCL = "中华人民共和国劳动合同法"


def ev(article: str = "第四十七条", *, score: float = 0.8, quote: str = "经济补偿按劳动者在本单位工作的年限……", **kw) -> Evidence:
    return Evidence(law=kw.pop("law", LCL), article_no=article, quote=quote, score=score, **kw)


def test_eid_is_sequential_and_stable():
    ledger = EvidenceLedger()
    added = ledger.add([ev("第四十七条"), ev("第八十七条")])
    assert [e.eid for e in added] == ["E1", "E2"]
    assert ledger.by_eid("E2").article_no == "第八十七条"


def test_dedupe_keeps_higher_score_and_original_eid():
    ledger = EvidenceLedger()
    ledger.add([ev("第四十七条", score=0.4)])
    ledger.add([ev("第四十七条", score=0.9)])
    assert len(ledger) == 1
    only = ledger.all()[0]
    assert only.score == 0.9
    assert only.eid == "E1"          # 编号不因覆盖而变化，答案里的引用才不会串


def test_dedupe_keeps_higher_even_if_later_lower():
    ledger = EvidenceLedger()
    ledger.add([ev("第四十七条", score=0.9)])
    ledger.add([ev("第四十七条", score=0.1)])
    assert ledger.all()[0].score == 0.9


def test_lookup_by_law_and_article():
    ledger = EvidenceLedger()
    ledger.add([ev()])
    assert ledger.lookup(LCL, "第四十七条") is not None
    assert ledger.lookup("中华人民共和国劳动法", "第四十七条") is None


def test_lookup_loose_only_when_unique():
    ledger = EvidenceLedger()
    ledger.add([ev("第三十九条")])
    assert ledger.lookup_loose("第三十九条").law == LCL
    ledger.add([ev("第三十九条", law="中华人民共和国劳动法")])
    assert ledger.lookup_loose("第三十九条") is None   # 跨法同号 → 必须由上游补法名


def test_effective_at_filters_expired():
    ledger = EvidenceLedger()
    ledger.add([ev("第八十七条", effective_from=date(2008, 1, 1), effective_to=date(2020, 12, 31))])
    assert ledger.all()[0].effective_at(date(2019, 6, 1)) is True
    assert ledger.all()[0].effective_at(date(2026, 1, 1)) is False


def test_for_prompt_skips_expired_and_respects_priority():
    ledger = EvidenceLedger()
    ledger.add(
        [
            ev("第四十七条", score=0.9, priority=20),
            ev("第四十四条", score=0.95, priority=10, law="中华人民共和国劳动法"),
            ev("第八十七条", score=0.99, effective_to=date(2020, 1, 1)),
        ]
    )
    block = ledger.for_prompt(as_of=date(2026, 10, 1), budget_chars=4000)
    assert "第四十七条" in block
    assert "第四十四条" in block
    assert "第八十七条" not in block          # 已失效，不能进上下文
    assert block.index("第四十七条") < block.index("第四十四条")   # 特别法优先


def test_for_prompt_budget_truncates():
    ledger = EvidenceLedger()
    ledger.add([ev("第四十七条", quote="条文" * 400)])
    block = ledger.for_prompt(budget_chars=200)
    assert len(block) <= 260


def test_sources_roundtrip():
    ledger = EvidenceLedger()
    ledger.add([ev(effective_from=date(2008, 1, 1))])
    payload = ledger.to_dicts()
    again = Evidence.from_dict(payload[0])
    assert again == ledger.all()[0]


@pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
def test_score_preserved(score):
    ledger = EvidenceLedger()
    ledger.add([ev(score=score)])
    assert ledger.all()[0].score == score
