"""成本感知路由：决定这一题走「快速通道」还是「全编排」。

**为什么要它**：实测（REPORT §9）表明，全编排在单跳规则问答上只比单 Agent 高 3pt 引用命中，
却要 3.6 倍 LLM 与 3 倍延迟；收益真正集中的场景是"要算钱 / 多跳串条 / 企业方合规风险"。

所以默认策略不该是"永远全编排"，而应该是**按风险升级**：

| 判据 | 走向 | 理由 |
|---|---|---|
| 需要金额测算 | 全编排 | 只有计算专家能保证金额正确 |
| 多跳（拆出 ≥2 路子查询） | 全编排 | 单跳检索覆盖不到跨条推理 |
| 企业方 / 工单式审查 | 全编排 | 合规风险高，值得多花两次调用 |
| 其余（单跳规则问答） | 快速通道 | 一次检索 + 一次生成就够 |

**判据只来自规则分诊（零 LLM）**。这点很关键：如果先用 LLM 分诊再决定走不走廉价通道，
省下来的钱又被分诊吃回去了——快速通道的意义就是**连分诊都不调模型**。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.logger import get_logger
from app.orchestrator.models import Plan, Step, TriageResult

logger = get_logger("routing")

#: auto（按风险升级）| fast（强制快速）| full（强制全编排）
POLICIES = ("auto", "fast", "full")

FAST = "fast"
FULL = "full"


@dataclass(frozen=True)
class RouteDecision:
    mode: str
    reasons: tuple[str, ...] = ()

    @property
    def is_fast(self) -> bool:
        return self.mode == FAST

    def to_dict(self) -> dict:
        return {"mode": self.mode, "reasons": list(self.reasons)}


def choose_mode(
    triage: TriageResult, plan: Plan, *, policy: str | None = None
) -> RouteDecision:
    """按风险给这次请求定档。policy 来自配置，默认 auto。"""
    policy = (policy or "auto").lower()
    if policy not in POLICIES:
        logger.warning("未知 routing policy=%s，按 auto 处理", policy)
        policy = "auto"

    if policy == FULL:
        return RouteDecision(FULL, ("配置强制全编排",))
    if policy == FAST:
        return RouteDecision(FAST, ("配置强制快速通道",))
    if plan.short_circuit:
        return RouteDecision(FAST, (f"短路：{plan.short_circuit}",))

    reasons: list[str] = []

    # 1) 可能要算钱 —— 唯一不能省掉计算专家的情形。
    # 注意这里看 `triage.needs_calc` 而不是 `plan`：规则分诊抽不出日期金额，
    # 于是 `_calc_kinds` 会因"缺槽位 + 没索要金额"把 calc 步骤摘掉；
    # 若档位判定也依赖这个结果，就会形成死循环 ——
    # 不算钱 → 走快速通道 → 不调 LLM 分诊 → 永远抽不到槽位 → 永远不算钱。
    # 所以档位判定用**宽口径**（可能涉及计算），算不算得成由全编排里的严格规则再定。
    money_kinds = [k for k in triage.needs_calc if k in MONEY_KINDS]
    if money_kinds or _looks_like_money_question(triage.query):
        reasons.append(f"需要测算（{'/'.join(money_kinds) or '疑似金额问题'}）")

    # 2) 多跳 —— 拆出了多路子查询，或计划里有多路检索
    statute_steps = [s for s in plan.steps if s.agent == "statute"]
    if len(statute_steps) >= 2:
        reasons.append(f"多跳（{len(statute_steps)} 路检索）")

    # 3) 合规审查 —— 企业方视角或制度类问题，错了代价高
    if any(step.agent == "compliance" for step in plan.steps):
        reasons.append("合规审查")

    if reasons:
        return RouteDecision(FULL, tuple(reasons))
    return RouteDecision(FAST, ("单跳规则问答：无需测算/多跳/合规审查",))


#: 与"钱"有关的测算类型。`probation` 不算：它判的是试用期上限，纯规则题，
#: 没有金额可算，放进全编排只会白花一次分诊。
MONEY_KINDS = frozenset(
    {"severance", "illegal", "notice", "overtime", "double_wage", "annual_leave", "non_compete"}
)

#: 索要金额 + 涉及钱 —— 规则没判出具体 kind 时也要升级（宽进严出）
_MONEY_RE = re.compile(r"(补偿|赔偿|工资|加班费|年假|竞业|试用期|工资|报酬|钱|费用)")


def _looks_like_money_question(query: str) -> bool:
    """问句在索要金额、且涉及钱：宁可多花一次分诊，也不能把该算的钱漏掉。"""
    return "多少" in (query or "") and bool(_MONEY_RE.search(query or ""))


def fast_plan_of(plan: Plan) -> Plan:
    """快速通道只跑第一路检索：保留原步骤 id，便于与全编排的轨迹对齐。

    **并且丢掉规则猜出来的 `law_hints`**。分诊的法域提示是把双刃剑：
    LLM 分诊给得准，可以提精度；规则分诊靠关键词，给错就直接把正确答案挡在门外。
    实测：问"入职扣身份证收押金"，"押金"命中民事词表 → 检索被锁进民法典，
    正确答案《劳动合同法》第八十四条根本进不了候选，单跳题引用命中因此掉 5pt。
    快速通道手上只有规则猜的法域，所以**宁宽不窄**：全库检索。
    """
    statute = next((s for s in plan.steps if s.agent == "statute"), None)
    if statute is None:
        return Plan(steps=(), reason="快速通道：无检索步骤")
    payload = {k: v for k, v in statute.payload.items() if k != "law_hints"}
    trimmed = Step(id=statute.id, agent=statute.agent, deps=statute.deps, payload=payload)
    return Plan(steps=(trimmed,), reason=f"快速通道：单路检索 {statute.id}（不做规则法域过滤）")
