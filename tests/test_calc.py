"""测算引擎用例：30+ 条，重点是边界（半年线、月末、3 倍封顶、12 年上限）。

这些用例是"公式没有歧义"的证据 —— 先写用例再写实现，改公式必须先改用例。
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.calc.engine import (
    add_months,
    annual_leave,
    compute,
    compute_many,
    double_wage,
    illegal_termination,
    non_compete,
    notice_pay,
    overtime,
    probation,
    probation_limit,
    service_years,
    severance,
    ymd_between,
)
from app.calc.models import Facts
from app.core.errors import MissingSlots

D = Decimal


def facts(**kw) -> Facts:
    base = dict(monthly_wage=D("12000"), start_date=date(2021, 3, 1), end_date=date(2026, 10, 20))
    base.update(kw)
    return Facts(**base)


# ---------------------------------------------------------------- 日历差
@pytest.mark.parametrize(
    "start,end,expected",
    [
        (date(2021, 3, 1), date(2026, 3, 1), (5, 0, 0)),
        (date(2021, 1, 31), date(2021, 3, 1), (0, 1, 1)),   # 月末夹取，不能出现负数天
        (date(2021, 2, 28), date(2024, 2, 29), (3, 0, 1)),  # 闰年
        (date(2026, 3, 1), date(2021, 3, 1), (-5, 0, 0)),   # 反向
    ],
)
def test_ymd_between(start, end, expected):
    assert ymd_between(start, end) == expected


def test_add_months_clamps_month_end():
    assert add_months(date(2021, 1, 31), 1) == date(2021, 2, 28)
    assert add_months(date(2021, 1, 31), 13) == date(2022, 2, 28)


# ---------------------------------------------------------------- 年限折算
@pytest.mark.parametrize(
    "start,end,months",
    [
        (date(2021, 3, 1), date(2026, 3, 1), "5"),      # 整 5 年
        (date(2021, 3, 1), date(2026, 8, 31), "5.5"),   # 余 5 个月 30 天 → 不满半年
        (date(2021, 3, 1), date(2026, 9, 1), "6"),      # 余 6 个月整 → 按一年
        (date(2026, 1, 1), date(2026, 4, 1), "0.5"),    # 不满 6 个月
        (date(2026, 1, 1), date(2026, 1, 20), "0.5"),   # 不足 1 个月
        (date(2018, 7, 1), date(2026, 6, 30), "8"),     # 7 年 11 个月 29 天 → 8
    ],
)
def test_service_years(start, end, months):
    assert service_years(start, end) == D(months)


# ---------------------------------------------------------------- 经济补偿
def test_severance_basic():
    result = severance(facts())
    assert result.kind == "severance"
    assert result.value == D("72000.00")
    assert result.substitution == "6 × 12000"
    assert "劳动合同法#第四十七条" in result.basis_articles[0]


def test_severance_half_year_branch():
    result = severance(facts(end_date=date(2026, 7, 20)))
    assert result.value == D("66000.00")   # 5.5 个月


def test_severance_exact_years():
    result = severance(facts(end_date=date(2026, 3, 1)))
    assert result.value == D("60000.00")


def test_severance_six_month_boundary():
    """余 6 个月整必须进位到一年，差一天不退位。"""
    assert severance(facts(end_date=date(2026, 9, 1))).value == D("72000.00")
    assert severance(facts(end_date=date(2026, 8, 31))).value == D("66000.00")


def test_severance_under_half_year():
    result = severance(facts(start_date=date(2026, 1, 1), end_date=date(2026, 4, 1)))
    assert result.value == D("6000.00")


def test_severance_three_times_cap():
    result = severance(facts(monthly_wage=D("40000"), region_cap=D("10000")))
    assert result.value == D("180000.00")   # 30000 × 6
    assert any("3 倍" in note for note in result.assumptions)


def test_severance_cap_and_twelve_year_limit():
    result = severance(
        facts(
            monthly_wage=D("40000"),
            region_cap=D("10000"),
            start_date=date(2008, 1, 1),
            end_date=date(2026, 1, 1),
        )
    )
    assert result.value == D("360000.00")   # 30000 × 12（封顶后年限不得超过 12 年）


def test_severance_no_cap_means_no_twelve_year_limit():
    result = severance(facts(monthly_wage=D("8000"), start_date=date(2008, 1, 1), end_date=date(2026, 1, 1)))
    assert result.value == D("144000.00")   # 8000 × 18


# ---------------------------------------------------------------- 2N / 代通知金
def test_illegal_is_double():
    result = illegal_termination(facts())
    assert result.value == D("144000.00")
    assert result.substitution.startswith("2 ×")


def test_notice_pay_when_not_notified():
    assert notice_pay(facts()).value == D("12000.00")


def test_notice_pay_when_notified():
    assert notice_pay(facts(notice_given=True)).value == D("0.00")


# ---------------------------------------------------------------- 加班费
def test_overtime_three_rates():
    result = overtime(
        facts(
            monthly_wage=D("8700"),   # 8700 / 21.75 / 8 = 50 元/小时
            weekday_overtime_hours=D("10"),
            restday_overtime_hours=D("8"),
            holiday_overtime_hours=D("8"),
        )
    )
    assert result.value == D("2750.00")   # 750 + 800 + 1200
    assert "第四十四条" in result.basis_articles[0]


def test_overtime_zero_hours():
    assert overtime(facts()).value == D("0.00")


# ---------------------------------------------------------------- 二倍工资
def test_double_wage_capped_at_eleven_months():
    result = double_wage(
        facts(start_date=date(2023, 1, 1), end_date=date(2025, 1, 1), signed_contract=False)
    )
    assert result.value == D("132000.00")   # 11 × 12000


def test_double_wage_partial_period():
    result = double_wage(
        facts(start_date=date(2025, 1, 1), end_date=date(2025, 8, 1), signed_contract=False)
    )
    assert result.value == D("72000.00")    # 首月不计，6 个月


def test_double_wage_none_when_signed():
    assert double_wage(facts()).value == D("0.00")


# ---------------------------------------------------------------- 年假 / 竞业 / 试用期
def test_annual_leave_three_hundred_percent():
    result = annual_leave(facts(monthly_wage=D("8700"), annual_leave_days=D("5")))
    assert result.value == D("6000.00")     # 400 × 5 × 3


def test_non_compete_thirty_percent():
    assert non_compete(facts(monthly_wage=D("10000"))).value == D("3000.00")


def test_non_compete_floor_by_min_wage():
    result = non_compete(facts(monthly_wage=D("10000"), local_min_wage=D("3500")))
    assert result.value == D("3500.00")


@pytest.mark.parametrize(
    "contract_months,limit",
    [(2, 0), (3, 1), (11, 1), (12, 2), (35, 2), (36, 6), (60, 6)],
)
def test_probation_limits(contract_months, limit):
    assert probation_limit(contract_months) == limit
    assert probation(facts(contract_months=contract_months)).value == D(limit)


# ---------------------------------------------------------------- 缺槽位
def test_missing_slots_reports_all():
    with pytest.raises(MissingSlots) as err:
        compute("severance", Facts())
    assert set(err.value.slots) == {"monthly_wage", "start_date", "end_date"}


def test_compute_many_merges_missing():
    with pytest.raises(MissingSlots) as err:
        compute_many(["severance", "annual_leave"], Facts(monthly_wage=D("9000")))
    assert "start_date" in err.value.slots and "annual_leave_days" in err.value.slots


def test_compute_many_ok():
    results = compute_many(["severance", "notice"], facts())
    assert [r.kind for r in results] == ["severance", "notice"]


def test_unknown_kind():
    with pytest.raises(KeyError):
        compute("nope", facts())
