"""金额测算的输入模型。

设计约束：**只接受结构化参数**，不接受自然语言。自然语言到槽位的抽取归 Triage 的 LLM，
计算本身一行 LLM 都不碰——法律场景算错赔偿金是事故，不是"偶尔不准"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

#: 与 CalcResult.kind 一一对应
KINDS = (
    "severance",      # 经济补偿 N
    "illegal",        # 违法解除赔偿金 2N
    "notice",         # 代通知金（N+1 的 +1）
    "overtime",       # 加班费
    "double_wage",    # 未签书面劳动合同二倍工资
    "annual_leave",   # 未休年假折算
    "non_compete",    # 竞业限制补偿
    "probation",      # 试用期上限校验
)


@dataclass(frozen=True)
class Facts:
    """一次测算所需的全部事实。字段全部可选，缺什么由引擎显式报 missing。"""

    monthly_wage: Decimal | None = None          # 离职前 12 个月平均工资（元）
    start_date: date | None = None               # 入职日
    end_date: date | None = None                 # 离职日
    cause: str = "layoff"                        # layoff/negotiated/expiry/illegal/resignation
    notice_given: bool = False                   # 是否提前 30 日书面通知
    region_cap: Decimal | None = None            # 当地上年度职工月平均工资（用于 3 倍封顶）
    local_min_wage: Decimal | None = None        # 当地最低工资（竞业限制兜底）
    weekday_overtime_hours: Decimal = Decimal("0")
    restday_overtime_hours: Decimal = Decimal("0")
    holiday_overtime_hours: Decimal = Decimal("0")
    contract_months: int | None = None           # 合同期限（月），用于试用期上限
    signed_contract: bool = True                 # 是否已签书面劳动合同
    annual_leave_days: Decimal | None = None     # 应享未休年假天数（None = 用户未提供）
    non_compete: bool = False                    # 是否约定竞业限制

    @staticmethod
    def from_dict(raw: dict) -> "Facts":
        """从 JSON（A2A DataPart / Triage 输出）构造，做类型收敛。"""

        def _d(value) -> Decimal | None:
            if value is None or value == "":
                return None
            return Decimal(str(value))

        def _date(value) -> date | None:
            if not value:
                return None
            return date.fromisoformat(str(value))

        return Facts(
            monthly_wage=_d(raw.get("monthly_wage")),
            start_date=_date(raw.get("start_date")),
            end_date=_date(raw.get("end_date")),
            cause=str(raw.get("cause") or "layoff"),
            notice_given=bool(raw.get("notice_given", False)),
            region_cap=_d(raw.get("region_cap")),
            local_min_wage=_d(raw.get("local_min_wage")),
            weekday_overtime_hours=_d(raw.get("weekday_overtime_hours")) or Decimal("0"),
            restday_overtime_hours=_d(raw.get("restday_overtime_hours")) or Decimal("0"),
            holiday_overtime_hours=_d(raw.get("holiday_overtime_hours")) or Decimal("0"),
            contract_months=(
                int(raw["contract_months"]) if raw.get("contract_months") not in (None, "") else None
            ),
            signed_contract=bool(raw.get("signed_contract", True)),
            annual_leave_days=_d(raw.get("annual_leave_days")),
            non_compete=bool(raw.get("non_compete", False)),
        )


@dataclass(frozen=True)
class CalcResult:
    kind: str
    formula: str
    substitution: str
    value: Decimal
    unit: str = "元"
    basis_articles: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def text(self) -> str:
        return f"{self.value}".rstrip("0").rstrip(".") if "." in f"{self.value}" else f"{self.value}"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "formula": self.formula,
            "substitution": self.substitution,
            "value": str(self.value),
            "unit": self.unit,
            "basis_articles": list(self.basis_articles),
            "assumptions": list(self.assumptions),
        }
