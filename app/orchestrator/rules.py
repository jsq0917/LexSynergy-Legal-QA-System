"""规则层：能确定性判断的，一律不交给 LLM。

职责：
1. 从问句里抽角色 / 意图 / 法域 / 测算类别（规则版 Triage，也是 LLM 失败时的兜底）；
2. 按"缺槽位 → 短路 / 单检索 / 检索+计算 / 检索+审查"的规则表产出 DAG。

规则命中率目标 ~80%：命中即零 LLM 出计划，未命中才交给 Planner 的 LLM。
"""

from __future__ import annotations

import re
from datetime import date

from app.calc.engine import REQUIRED_SLOTS, SLOT_LABELS
from app.core.logger import get_logger
from app.orchestrator.models import Plan, Step, TriageResult

logger = get_logger("orchestrator.rules")

LCL = "中华人民共和国劳动合同法"
LABOR = "中华人民共和国劳动法"
CIVIL = "中华人民共和国民法典"

GREETING_RE = re.compile(r"^(你好|您好|hi|hello|在吗|谢谢|多谢|你是谁|你能做什么)[!！。？?~\s]*$", re.I)

#: 企业视角关键词 —— 触发 ComplianceAgent
EMPLOYER_WORDS = (
    "我们公司", "我公司", "本公司", "用人单位", "员工手册", "规章制度", "公司制度",
    "考勤制度", "薪酬制度", "制度", "条款", "审查", "合规", "招聘", "录用条件",
    "hr", "人事", "劳动合同模板", "隐私政策",
)

#: 测算类别关键词（按最长匹配优先，`illegal` 必须先于 `severance` 判断）
CALC_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("illegal", ("违法解除", "违法辞退", "2n", "二倍赔偿", "赔偿金")),
    ("severance", ("经济补偿", "补偿金", "裁员", "辞退", "离职补偿", "n+1", "n＋1")),
    ("notice", ("代通知金", "没有提前通知", "未提前通知")),
    ("overtime", ("加班费", "加班工资", "加班")),
    ("double_wage", ("二倍工资", "双倍工资", "没签合同", "未签劳动合同", "没签劳动合同")),
    ("annual_leave", ("年假", "年休假")),
    ("non_compete", ("竞业",)),
    ("probation", ("试用期",)),
)

CIVIL_WORDS = (
    "婚姻", "离婚", "继承", "遗嘱", "侵权", "物权", "借款", "租赁", "买卖合同",
    "房东", "租客", "押金", "遗产", "彩礼", "抚养", "赡养", "车库", "物业",
)

LABOR_WORDS = (
    "劳动", "工资", "加班", "辞退", "裁员", "社保", "工伤", "试用期", "年假",
    "竞业", "保密协议", "解除合同", "离职", "辞职", "解雇", "降薪", "拖欠", "报酬",
    "劳动者", "用人单位", "经济补偿", "赔偿金", "二倍工资", "劳动合同", "考勤", "调岗",
)

#: 明确"与法律无关"的信号。只有命中它才判 `other`。
#: 早先版本把 `other` 当兜底，结果"竞业限制不给补偿金还受约束吗"被判无关、
#: 直接走 no_data 一句话打发 —— 40 题里因此掉了 5 题的召回。**兜底必须是"继续检索"，不是"拒绝回答"。**
NON_LEGAL_RE = re.compile(
    r"(天气|笑话|菜谱|做饭|翻译|写代码|写诗|股票|彩票|星座|电影|游戏|唱歌|聊天|你是谁|在吗|讲个|推荐.*(餐厅|电影|书))"
)

DECOMPOSE_RE = re.compile(r"，|,|；|;|以及|另外|还有|分别")
HISTORY_RE = re.compile(r"那|这个|具体|继续|还有呢|上面|刚才")
#: 纯尾巴问句：切出来也不构成独立子查询，否则"……，合规吗"会被拆成两跳
TAIL_RE = re.compile(r"^(合规吗|怎么办|可以吗|对吗|如何|行吗|合法吗|有效吗|怎么算|是否合规)[？?。！!]*$")
#: 子查询必须像"问题"：事实片段（"我 2021 年 3 月入职"）不该被拆成独立检索
QUESTION_RE = re.compile(r"怎么|多少|如何|是否|什么|多久|能不能|可不可以|吗|？|\?")

FOLLOW_UP_TEMPLATE = "为了给你准确的结论，还需要你补充：{slots}。"


def infer_intent(query: str) -> str:
    q = (query or "").strip()
    if GREETING_RE.match(q):
        return "greeting"
    if not q:
        return "other"
    if any(word in q for word in CIVIL_WORDS):
        return "civil_code"
    if any(word in q for word in LABOR_WORDS):
        return "labor_law"
    if NON_LEGAL_RE.search(q):
        return "other"
    # 兜底是 general（照常检索、不按法域过滤），不是 other（直接不答）
    return "general"


def infer_role(query: str) -> str:
    q = (query or "").lower()
    return "employer" if any(word in q for word in EMPLOYER_WORDS) else "employee"


def infer_calc_kinds(query: str) -> list[str]:
    q = (query or "").lower()
    hits: list[str] = []
    for kind, words in CALC_WORDS:
        if any(word in q for word in words):
            hits.append(kind)
    # 补偿与赔偿互斥：判成违法解除就不要再算 N
    if "illegal" in hits and "severance" in hits:
        hits.remove("severance")
    return hits


def infer_law_hints(query: str, intent: str) -> list[str]:
    q = query or ""
    if intent == "greeting" or intent == "other":
        return []
    if intent == "civil_code":
        return [CIVIL]
    hints: list[str] = []
    if "劳动合同" in q or intent == "labor_law":
        hints.append(LCL)
    if "劳动法" in q and LCL not in hints:
        hints.append(LCL)
    if "劳动" in q and LABOR not in hints:
        hints.append(LABOR)
    return hints or ([LCL, LABOR] if intent == "labor_law" else [])


def infer_complexity(query: str, sub_queries: list[str]) -> str:
    q = query or ""
    if len(sub_queries) > 1 or len(q) > 40 or q.count("？") + q.count("?") >= 2:
        return "complex"
    if len(q) < 15:
        return "simple"
    return "medium"


def missing_slots_for(kinds: list[str], facts: dict) -> list[str]:
    missing: list[str] = []
    for kind in kinds:
        for slot in REQUIRED_SLOTS.get(kind, ()):
            if facts.get(slot) in (None, ""):
                missing.append(slot)
    return sorted(set(missing))


def follow_up_text(slots: list[str]) -> str:
    labels = [SLOT_LABELS.get(s, s) for s in slots]
    return FOLLOW_UP_TEMPLATE.format(slots="、".join(labels))


def split_sub_queries(query: str) -> list[str]:
    """只有"两段以上都是问句"才拆；事实片段（日期、金额）不构成子查询。"""
    if not DECOMPOSE_RE.search(query or ""):
        return []
    parts = [p.strip(" 。？?！!") for p in DECOMPOSE_RE.split(query)]
    parts = [p for p in parts if len(p) >= 6 and not TAIL_RE.match(p)]
    if sum(1 for p in parts if QUESTION_RE.search(p)) < 2:
        return []
    return parts if len(parts) >= 2 else []


def triage(
    query: str,
    *,
    facts: dict | None = None,
    as_of: date | None = None,
    history: list[dict] | None = None,
) -> TriageResult:
    """规则版 Triage：不需要 LLM 也能给出可用计划（也是 LLM 失败后的兜底）。"""
    query = (query or "").strip()
    facts = dict(facts or {})
    intent = infer_intent(query)
    role = infer_role(query)
    kinds = infer_calc_kinds(query) if role == "employee" or "试用期" in query else []
    subs = split_sub_queries(query)
    rewrite = "decompose" if subs else ("history" if HISTORY_RE.search(query) and history else "direct")
    missing = missing_slots_for(kinds, facts)
    return TriageResult(
        query=query,
        intent=intent,
        role=role,
        complexity=infer_complexity(query, subs),
        rewrite_type=rewrite,
        law_hints=infer_law_hints(query, intent),
        sub_queries=subs,
        facts=facts,
        needs_calc=kinds,
        missing_slots=missing,
        follow_up=follow_up_text(missing) if missing else "",
        as_of=as_of,
        source="rules",
    )


def wants_specific_amount(query: str) -> bool:
    """问题是否在**索要一个具体金额**（而不是"这个规则怎么算/几倍/有没有效"）。

    判据故意只用一个词"多少"：它覆盖了"能拿多少/赔多少钱/该给我多少补偿"，
    又不误伤"按几倍支付""年假能休几天""这条有效吗"这类**问规则**的问题。
    """
    return "多少" in (query or "")


def _calc_kinds(t: TriageResult) -> list[str]:
    """决定这一题是否真的需要走测算。

    规则：**要算钱，就得有数据或有意愿给数据**。
    若关键槽位缺失、而问题又不是在索要具体金额，说明用户问的是规则
    （"加班费按几倍""年假有几天""竞业限制三年有效吗"），
    此时弹一句"请提供月工资和起止日期"既不回答、又丢掉了整段法律依据——
    实测 40 题里有 15 题掉进这个坑。这类问题应当直接讲规则。
    """
    kinds = list(t.needs_calc)
    if not kinds:
        return []
    missing = missing_slots_for(kinds, t.facts)
    if missing and not wants_specific_amount(t.query):
        logger.info("测算降级为规则解释：缺 %s 且问题未索要具体金额", missing)
        return []
    return kinds


def route(t: TriageResult) -> Plan:
    """规则路由表：命中即出 DAG，零 LLM。

    缺槽位**不短路**：仍然派发法条检索（零 LLM、几百毫秒），先把法律依据和公式给出，
    再补一句追问。比"什么都不答直接要参数"体验好得多，成本几乎为零。
    """
    if t.intent == "greeting":
        return Plan(short_circuit="greeting", reason="问候/寒暄：固定话术，不派发任何专家")
    if t.intent == "other":
        return Plan(short_circuit="no_data", reason="与法律无关：引导话术，不派发任何专家")

    steps: list[Step] = []
    subs = t.sub_queries or [t.query]
    base_payload = {"law_hints": t.law_hints, "as_of": t.as_of.isoformat() if t.as_of else None}
    for idx, sub in enumerate(subs, start=1):
        steps.append(
            Step(id=f"s{idx}", agent="statute", payload={**base_payload, "sub_query": sub})
        )
    statute_ids = tuple(s.id for s in steps)

    kinds = _calc_kinds(t)
    if kinds:
        steps.append(
            Step(id="c1", agent="calc", payload={"kinds": kinds, "facts": dict(t.facts)})
        )

    wants_compliance = t.role == "employer" or any(
        word in t.query.lower() for word in ("制度", "条款", "审查", "合规", "员工手册")
    )
    if wants_compliance:
        steps.append(
            Step(
                id="k1",
                agent="compliance",
                deps=statute_ids,
                payload={"text": t.query, "role": t.role, "law_hints": t.law_hints},
            )
        )

    reason = f"规则路由：{len(statute_ids)} 路检索"
    if kinds:
        reason += f" + 测算({'/'.join(kinds)})"
    if wants_compliance:
        reason += " + 合规审查"
    if t.missing_slots:
        reason += f"（缺槽位 {t.missing_slots}，先给依据再追问）"
    return Plan(steps=tuple(steps), reason=reason, ask_slots=tuple(t.missing_slots))
