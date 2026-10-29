"""报告合并语义：局部重跑不能吃掉另一半结论。

背景：`compare_report.json` 是"所有结论的快照"，但 `--set calc` 只重算金额那一半。
早先整体覆写，导致"只补跑金额"之后 `law` 段变成空列表 —— 验收时我自己就被这个
假退步误导过一次。这组测试盯住它。
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.compare_agents import Case, _ordered_cases, merge_report


def _law_rows(n: int = 4) -> list[dict]:
    return [{"config": c, "n": 100, "recall@5": 0.75} for c in
            ("A_baseline", "B_multi", "C_verified", "D_auto")][:n]


def _calc_rows(n: int = 4) -> list[dict]:
    return [{"config": c, "n": 30, "amount_acc": 1.0} for c in
            ("A_baseline", "B_multi", "C_verified", "D_auto")][:n]


def test_calc_only_run_keeps_law_section(tmp_path: Path):
    """V1：只跑 calc，law 段必须原样保留。"""
    path = tmp_path / "compare_report.json"
    merge_report(path, law=_law_rows(), calc=_calc_rows(), meta={"law_samples": 100, "calc_samples": 30})
    first = json.loads(path.read_text(encoding="utf-8"))
    assert len(first["law"]) == 4 and len(first["calc"]) == 4

    # 只重算 calc
    merge_report(path, law=[], calc=_calc_rows(), meta={"calc_samples": 30})
    second = json.loads(path.read_text(encoding="utf-8"))
    assert len(second["law"]) == 4, "law 段被清空了 —— 这就是要防的回归"
    assert len(second["calc"]) == 4


def test_law_only_run_keeps_calc_section(tmp_path: Path):
    """V2：只跑 law，calc 段必须原样保留。"""
    path = tmp_path / "compare_report.json"
    merge_report(path, law=_law_rows(), calc=_calc_rows(), meta={"law_samples": 100, "calc_samples": 30})

    merge_report(path, law=_law_rows(), calc=[], meta={"law_samples": 100})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["calc"]) == 4, "calc 段被清空了"
    assert len(data["law"]) == 4


def test_each_section_carries_its_own_timestamp_and_size(tmp_path: Path):
    """V3：每段带自己的生成时间与样本量，才能看出哪半是新的。"""
    path = tmp_path / "compare_report.json"
    merge_report(path, law=_law_rows(), calc=[], meta={"law_samples": 100, "law_set": "qa_set_100.jsonl"})
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["law_meta"]["samples"] == 100
    assert data["law_meta"]["law_set"] == "qa_set_100.jsonl"
    assert data["law_meta"]["generated_at"]
    assert "calc_meta" not in data          # 没跑就不编造
    assert data["updated_at"]


def test_corrupted_report_does_not_crash(tmp_path: Path):
    """报告被人手改坏时，应当重建而不是抛异常。"""
    path = tmp_path / "compare_report.json"
    path.write_text("{ 这不是 JSON", encoding="utf-8")

    merged = merge_report(path, law=_law_rows(), calc=[], meta={"law_samples": 100})
    assert len(merged["law"]) == 4
    assert json.loads(path.read_text(encoding="utf-8"))["law"]


def _case(config: str, sample_id: str, note: str) -> Case:
    return Case(config=config, sample_id=sample_id, question=f"问{sample_id}", text=note)


def _samples(*ids: str) -> list[dict]:
    return [{"id": i, "question": f"问{i}"} for i in ids]


def test_reuse_and_new_results_are_both_counted():
    """R1：`--reuse` 补样时，复用样本必须进汇总 —— 否则 30 题被当成 130 题报。"""
    samples = _samples("q1", "q2", "q3")
    reuse = {"C_verified:q1": _case("C_verified", "q1", "旧"), "C_verified:q2": _case("C_verified", "q2", "旧")}
    results = [_case("C_verified", "q3", "新")]

    ordered = _ordered_cases(samples, "C_verified", results, reuse)

    assert [c.sample_id for c in ordered] == ["q1", "q2", "q3"]
    assert [c.text for c in ordered] == ["旧", "旧", "新"]


def test_ordered_cases_follows_sample_order_not_finish_order():
    """R2：并发跑完顺序是乱的，输出必须按样本顺序，否则逐题比对会对错行。"""
    samples = _samples("q1", "q2", "q3")
    results = [_case("C_verified", "q3", "3"), _case("C_verified", "q1", "1")]

    ordered = _ordered_cases(samples, "C_verified", results, {})

    assert [c.sample_id for c in ordered] == ["q1", "q3"]


def test_ordered_cases_skips_missing_samples():
    """R3：复用池与本次结果都没有的样本要跳过，不能编造空行。"""
    samples = _samples("q1", "q2")
    ordered = _ordered_cases(samples, "C_verified", [_case("C_verified", "q1", "1")], {})

    assert [c.sample_id for c in ordered] == ["q1"]


def test_ordered_cases_does_not_mix_configs():
    """R4：reuse 按 `config:sample_id` 建键，别的配置的用例不得串进来。"""
    samples = _samples("q1")
    reuse = {"A_baseline:q1": _case("A_baseline", "q1", "A"), "C_verified:q1": _case("C_verified", "q1", "C")}

    ordered = _ordered_cases(samples, "C_verified", [], reuse)

    assert [(c.config, c.text) for c in ordered] == [("C_verified", "C")]


def test_new_result_overrides_reuse_for_same_key():
    """R5：同一配置同一样本重跑时以新结果为准，不能留下两条。"""
    samples = _samples("q1")
    reuse = {"C_verified:q1": _case("C_verified", "q1", "旧")}

    ordered = _ordered_cases(samples, "C_verified", [_case("C_verified", "q1", "新")], reuse)

    assert len(ordered) == 1 and ordered[0].text == "新"


def test_merge_preserves_identical_numbers(tmp_path: Path):
    """V5：合并不得改动数字本身。"""
    path = tmp_path / "compare_report.json"
    law, calc = _law_rows(), _calc_rows()
    merge_report(path, law=law, calc=calc, meta={})
    first = json.loads(path.read_text(encoding="utf-8"))

    merge_report(path, law=[], calc=calc, meta={})
    second = json.loads(path.read_text(encoding="utf-8"))

    assert first["law"] == second["law"]
    assert first["calc"] == second["calc"]