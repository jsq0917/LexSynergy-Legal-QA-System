"""种子缺陷评测本身也要被测：8 类必须全检出，干净答案不能被误伤。"""

from __future__ import annotations

import pytest

from eval.seeded_defect import evaluate, mutants


@pytest.mark.asyncio
async def test_detection_rate_is_full():
    report = await evaluate()
    assert report["total"] == 8
    assert report["detected"] == 8
    assert report["detection_rate"] == 1.0


@pytest.mark.asyncio
async def test_no_false_positive_on_clean_answer():
    report = await evaluate()
    assert report["clean_passed"] is True
    assert report["clean_fails"] == []


@pytest.mark.asyncio
async def test_every_mutant_hits_its_expected_gate():
    report = await evaluate()
    for row in report["rows"]:
        assert row["detected"], f"{row['name']} 未被 {row['expected']} 捕获"


def test_mutants_cover_five_gates():
    gates = {m.gate for m in mutants()}
    assert gates == {
        "G1_not_found",
        "G2_misquote",
        "G3_expired",
        "G4_misgrounded",
        "G5_numeric",
    }
