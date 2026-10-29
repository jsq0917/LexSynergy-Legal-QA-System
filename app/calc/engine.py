"""金额测算引擎：纯函数，零 LLM，零 IO。

所有金额用 Decimal 计算并 ROUND_HALF_UP 到分；年限折算用"满年 + 余月"口径。
每个函数都显式声明缺失槽位（抛 MissingSlots），由编排层转成追问。
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from app.calc.models import CalcResult, Facts
from app.core.errors import MissingSlots

CENT = Decimal("0.01")
#: 月计薪天数（《关于职工全年月平均工作时间和工资折算问题的通知》）
PAYABLE_DAYS_PER_MONTH = Decimal("21.75")
HOURS_PER_DAY = Decimal("8")

LCL = "中华人民共和国劳动合同法"
LABOR = "中华人民共和国劳动法"

_REQUIRED: dict[str, tuple[str, ...]] = {
    "severance": ("monthly_wage", "start_date", "end_date"),
    "illegal": ("monthly_wage", "start_date", "end_date"),
    "notice": ("monthly_wage",),
    "overtime": ("monthly_wage",),
    "double_wage": ("monthly_wage", "start_date", "end_date"),
    "annual_leave": ("monthly_wage", "annual_leave_days"),
    "non_compete": ("monthly_wage",),
    "probation": ("contract_months",),
}

#: 对外暴露：编排层用它算"缺哪些槽位"，避免两处各写一份必然跑偏
REQUIRED_SLOTS = _REQUIRED

#: 辅助槽位：**不是必填**，但缺了会按默认口径算（例如默认"已提前通知"→ 代通知金 0）。
#: 这类槽位必须让分诊器去抽，否则用户说了"没提前通知"系统也听不见 —— 实测就是 0 元。
AUX_SLOTS: dict[str, tuple[str, ...]] = {
    "overtime": ("weekday_overtime_hours", "restday_overtime_hours", "holiday_overtime_hours"),
    "double_wage": ("signed_contract",),
    "notice": ("notice_given",),
}

SLOT_LABELS = {
    "monthly_wage": "离职前 12 个月平均工资",
    "start_date": "入职日期",
    "end_date": "离职日期",
    "annual_leave_days": "应享未休年假天数",
    "contract_months": "劳动合同期限（月）",
    "weekday_overtime_hours": "工作日延时加班小时数",
    "restday_overtime_hours": "休息日加班小时数（未安排补休）",
    "holiday_overtime_hours": "法定节假日加班小时数",
    "signed_contract": "是否已签书面劳动合同（true/false）",
    "notice_given": "是否已提前 30 日书面通知（true/false）",
}


def money(value: Decimal) -> Decimal:
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def _require(facts: Facts, kind: str) -> None:
    missing = [name for name in _REQUIRED[kind] if getattr(facts, name) in (None, "")]
    if missing:
        raise MissingSlots(missing)


def add_months(day: date, months: int) -> date:
    """日历月推进（月末夹取）：1-31 + 1 月 → 2-28。"""
    total = day.month - 1 + months
    year = day.year + total // 12
    month = total % 12 + 1
    # 目标月的最后一天
    if month == 12:
        last = 31
    else:
        last = (date(year, month + 1, 1) - timedelta(days=1)).day
    return date(year, month, min(day.day, last))


def ymd_between(start: date, end: date) -> tuple[int, int, int]:
    """(年, 月, 日) 差。

    用"逐月推进"而不是直接借位：后者在 1-31 → 3-01 这类月末场景会算出负数天。
    """
    if end < start:
        y, m, d = ymd_between(end, start)
        return -y, -m, -d
    months = 0
    cursor = start
    while add_months(cursor, 1) <= end:
        cursor = add_months(cursor, 1)
        months += 1
    years, months = divmod(months, 12)
    return years, months, (end - cursor).days


def service_years(start: date, end: date) -> Decimal:
    """经济补偿的工作年限折算（返回"补偿月数"）。

    每满 1 年计 1 个月；余 6 个月以上不满 1 年按 1 年计；余不满 6 个月计半个月。
    """
    years, months, days = ymd_between(start, end)
    base = Decimal(years)
    if months >= 6:
        return base + 1
    if months >= 1 or days > 0:
        return base + Decimal("0.5")
    return base


def _base_wage(facts: Facts) -> tuple[Decimal, Decimal, list[str], bool]:
    """返回 (月工资基数, 年限上限, 追加假设, 是否触发三倍封顶)。"""
    assert facts.monthly_wage is not None
    wage = facts.monthly_wage
    notes: list[str] = []
    capped = False
    if facts.region_cap is not None and wage > facts.region_cap * 3:
        wage = facts.region_cap * 3
        capped = True
        notes.append("月工资高于当地上年度职工月平均工资 3 倍，按 3 倍计，年限最高 12 年")
    return wage, Decimal("12") if capped else Decimal("999"), notes, capped


def severance(facts: Facts) -> CalcResult:
    _require(facts, "severance")
    assert facts.start_date and facts.end_date
    n = service_years(facts.start_date, facts.end_date)
    wage, cap, notes, _ = _base_wage(facts)
    if n > cap:
        n = cap
    value = money(wage * n)
    assumptions = [
        "N = 工作年限（每满 1 年计 1 个月；余 6 个月以上按 1 年计；余不满 6 个月计半个月）",
        "月工资基数 = 离职前 12 个月平均工资",
        *notes,
    ]
    return CalcResult(
        kind="severance",
        formula="N × 月工资基数",
        substitution=f"{_plain(n)} × {_plain(wage)}",
        value=value,
        basis_articles=(f"{LCL}#第四十七条",),
        assumptions=tuple(assumptions),
    )


def illegal_termination(facts: Facts) -> CalcResult:
    """违法解除赔偿金 = 2 × 经济补偿（《劳动合同法》第八十七条）。"""
    base = severance(facts)
    doubled = money(base.value * 2)
    return CalcResult(
        kind="illegal",
        formula="2 × N × 月工资基数",
        substitution=f"2 × {base.substitution}",
        value=doubled,
        basis_articles=(f"{LCL}#第八十七条", f"{LCL}#第四十七条"),
        assumptions=tuple(base.assumptions) + ("违法解除赔偿金与经济补偿不并用，取赔偿金",),
    )


def notice_pay(facts: Facts) -> CalcResult:
    """代通知金：未提前 30 日书面通知时额外支付 1 个月工资（按上月工资标准）。"""
    _require(facts, "notice")
    assert facts.monthly_wage is not None
    value = money(facts.monthly_wage) if not facts.notice_given else Decimal("0.00")
    return CalcResult(
        kind="notice",
        formula="1 × 上月工资（仅未提前 30 日书面通知时支付）",
        substitution=f"1 × {_plain(facts.monthly_wage)}"
        + ("（已提前通知，无需支付）" if facts.notice_given else ""),
        value=value,
        basis_articles=(f"{LCL}#第四十条",),
        assumptions=("代通知金按劳动者上一个月的工资标准确定",),
    )


def overtime(facts: Facts) -> CalcResult:
    _require(facts, "overtime")
    assert facts.monthly_wage is not None
    hourly = facts.monthly_wage / PAYABLE_DAYS_PER_MONTH / HOURS_PER_DAY
    weekday = hourly * Decimal("1.5") * facts.weekday_overtime_hours
    restday = hourly * Decimal("2") * facts.restday_overtime_hours
    holiday = hourly * Decimal("3") * facts.holiday_overtime_hours
    value = money(weekday + restday + holiday)
    hours_total = (
        facts.weekday_overtime_hours + facts.restday_overtime_hours + facts.holiday_overtime_hours
    )
    return CalcResult(
        kind="overtime",
        formula="小时工资 × 倍数 × 小时数；小时工资 = 月工资 ÷ 21.75 ÷ 8",
        substitution=(
            f"{_plain(hourly.quantize(CENT, rounding=ROUND_HALF_UP))} × "
            f"(1.5×{_plain(facts.weekday_overtime_hours)} + "
            f"2×{_plain(facts.restday_overtime_hours)} + "
            f"3×{_plain(facts.holiday_overtime_hours)})"
        ),
        value=value,
        basis_articles=(f"{LCL}#第四十四条", f"{LABOR}#第四十四条"),
        assumptions=(
            "工作日延时 1.5 倍；休息日不能安排补休 2 倍；法定节假日 3 倍",
            f"月计薪天数按 21.75 天、日工作 8 小时折算；合计加班 {_plain(hours_total)} 小时",
        ),
    )


def double_wage(facts: Facts) -> CalcResult:
    """未签书面劳动合同的二倍工资差额：满 1 个月的次日起至满 1 年前一日，最多 11 个月。"""
    _require(facts, "double_wage")
    assert facts.start_date and facts.end_date and facts.monthly_wage is not None
    if facts.signed_contract:
        return CalcResult(
            kind="double_wage",
            formula="已签书面劳动合同，无二倍工资差额",
            substitution="0",
            value=Decimal("0.00"),
            basis_articles=(f"{LCL}#第八十二条",),
            assumptions=("用人单位自用工之日起一个月内已订立书面劳动合同",),
        )
    years, months, days = ymd_between(facts.start_date, facts.end_date)
    elapsed_months = years * 12 + months + (1 if days > 0 else 0)
    billable = max(0, min(11, elapsed_months - 1))  # 首月不计
    value = money(facts.monthly_wage * billable)
    return CalcResult(
        kind="double_wage",
        formula="月工资 × 应付月份（最多 11 个月）",
        substitution=f"{_plain(facts.monthly_wage)} × {billable}",
        value=value,
        basis_articles=(f"{LCL}#第八十二条",),
        assumptions=(
            "自用工之日起满一个月的次日起算，至满一年的前一日止",
            "最多 11 个月；已发工资不冲抵，另支付一倍",
        ),
    )


def annual_leave(facts: Facts) -> CalcResult:
    """未休年假：按日工资 300% 支付（含正常工作期间工资 100%）。"""
    _require(facts, "annual_leave")
    assert facts.monthly_wage is not None
    daily = facts.monthly_wage / PAYABLE_DAYS_PER_MONTH
    value = money(daily * facts.annual_leave_days * 3)
    return CalcResult(
        kind="annual_leave",
        formula="日工资 × 未休天数 × 300%",
        substitution=f"{_plain(daily.quantize(CENT, rounding=ROUND_HALF_UP))} × "
        f"{_plain(facts.annual_leave_days)} × 3",
        value=value,
        basis_articles=("《职工带薪年休假条例》#第五条",),
        assumptions=(
            "300% 中含正常工作期间工资 100%，用人单位应额外支付 200%",
            "折算后不足 1 整天的部分不支付",
        ),
    )


def non_compete(facts: Facts) -> CalcResult:
    """竞业限制补偿：月平均工资的 30%，且不低于劳动合同履行地最低工资标准。"""
    _require(facts, "non_compete")
    assert facts.monthly_wage is not None
    raw = facts.monthly_wage * Decimal("0.3")
    note: list[str] = []
    value = raw
    if facts.local_min_wage is not None and raw < facts.local_min_wage:
        value = facts.local_min_wage
        note.append("30% 低于当地最低工资标准，按最低工资计")
    return CalcResult(
        kind="non_compete",
        formula="月平均工资 × 30%（不低于当地最低工资）",
        substitution=f"{_plain(facts.monthly_wage)} × 30%",
        value=money(value),
        unit="元/月",
        basis_articles=("最高人民法院关于审理劳动争议案件适用法律问题的解释（一）#第三十六条",),
        assumptions=("按月支付；未约定补偿标准时按 30% 认定", *note),
    )


#: 试用期上限（月）：合同期 <3 个月不得约定；3 个月~1 年 → 1；1~3 年 → 2；3 年以上/无固定 → 6
def probation_limit(contract_months: int) -> int:
    if contract_months < 3:
        return 0
    if contract_months < 12:
        return 1
    if contract_months < 36:
        return 2
    return 6


def probation(facts: Facts) -> CalcResult:
    _require(facts, "probation")
    assert facts.contract_months is not None
    limit = probation_limit(facts.contract_months)
    return CalcResult(
        kind="probation",
        formula="按合同期限查上限：<3 月不得约定 / 3 月~1 年 1 个月 / 1~3 年 2 个月 / 3 年以上 6 个月",
        substitution=f"合同期限 {facts.contract_months} 个月 → 上限 {limit} 个月",
        value=Decimal(limit),
        unit="月",
        basis_articles=(f"{LCL}#第十九条",),
        assumptions=("同一用人单位与同一劳动者只能约定一次试用期",),
    )


_DISPATCH = {
    "severance": severance,
    "illegal": illegal_termination,
    "notice": notice_pay,
    "overtime": overtime,
    "double_wage": double_wage,
    "annual_leave": annual_leave,
    "non_compete": non_compete,
    "probation": probation,
}


def compute(kind: str, facts: Facts) -> CalcResult:
    if kind not in _DISPATCH:
        raise KeyError(f"未知测算类型：{kind}；可选 {sorted(_DISPATCH)}")
    return _DISPATCH[kind](facts)


def compute_many(kinds: list[str], facts: Facts) -> list[CalcResult]:
    """按顺序计算多个类别；任一缺槽位直接抛 MissingSlots（合并所有缺失项）。"""
    missing: list[str] = []
    for kind in kinds:
        missing += [m for m in _REQUIRED.get(kind, ()) if getattr(facts, m) in (None, "")]
    if missing:
        raise MissingSlots(sorted(set(missing)))
    return [compute(kind, facts) for kind in kinds]


def _plain(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    return text
