"""流水线：Triage → Plan → DAG 执行 → 汇总 → 校验 →（必要时）补检索一轮 → 交付。

这是全系统唯一的编排入口。SSE 事件与最终答案都从这里产出，
因此"用户看到的"与"实际发生的"永远一致（评测口径 = 线上链路）。
"""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from datetime import date
from decimal import Decimal
from typing import Callable

from app.agents.base import AgentResult, BaseAgent
from app.core.logger import get_logger
from app.evidence.ledger import EvidenceLedger
from app.verifier.repair import remap_citations
from app.orchestrator.cache import AnswerCache, cache_key
from app.orchestrator.context import TaskContext
from app.orchestrator.models import Answer, Plan, PlanResult, Step, StepResult, TriageResult
from app.orchestrator.planner import Planner
from app.orchestrator.routing import choose_mode, fast_plan_of
from app.orchestrator.rules import triage as rule_triage
from app.orchestrator.scheduler import Scheduler
from config import Settings, get_settings

logger = get_logger("pipeline")

GREETING_REPLY = (
    "你好，我是 LexSynergy 法律合规助手，可以回答劳动用工与民法典相关的合规问题，"
    "也能做经济补偿、加班费等金额测算。可以把你的情况说得具体些（入职离职时间、月工资、事由）。"
)
NO_DATA_REPLY = (
    "暂未在现有资料中找到与该问题直接相关的法条依据。为避免误导，我不做推测性回答。"
    "建议：1）补充关键事实（时间、金额、事由）；2）确认问题属于劳动用工还是民法典领域；"
    "3）向当地劳动监察部门或律师核实。"
)
DISCLAIMER = "以上分析仅供参考，具体以裁判机关认定为准。"
RETRY_REPLY = (
    "抱歉，本次检索没能完成（超时或服务暂时不可用），因此我没有足够依据回答这个问题。"
    "这不是「没有相关规定」，而是「这次没查到」。请稍后重试；若反复失败，可以补充更具体的事实再试。"
)

#: (上下文, 证据账本, 测算结果) -> 答案草稿
Synthesizer = Callable[[TaskContext, EvidenceLedger, list[dict]], str]

VERIFIER_AGENT = "verifier"

#: 数字 token（用于把公式中间量纳入 G5 的合法集合）
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


class Pipeline:
    def __init__(
        self,
        agents: dict[str, BaseAgent],
        *,
        triager: Callable[..., TriageResult] = rule_triage,
        planner: Planner | None = None,
        synthesizer: Synthesizer | None = None,
        cache: AnswerCache | None = None,
        settings: Settings | None = None,
        checkpoint: Callable[[Step, StepResult], None] | None = None,
    ) -> None:
        self._agents = dict(agents)
        self._triager = triager
        self._settings = settings or get_settings()
        self._planner = planner or Planner()
        self._synthesizer = synthesizer or template_synthesizer
        self._cache = cache
        budget = self._settings.budget
        self._scheduler = Scheduler(
            self._dispatch,
            max_steps=budget.max_steps,
            step_timeout=budget.step_timeout_s,
            deadline_s=budget.deadline_s,
            checkpoint=checkpoint,
        )

    # ------------------------------------------------------------------
    async def run(
        self,
        query: str,
        *,
        session_id: str = "",
        facts: dict | None = None,
        as_of: date | None = None,
        history: list[dict] | None = None,
        event_sink: Callable[[str, dict], None] | None = None,
    ) -> Answer:
        started = time.perf_counter()
        ctx = TaskContext(query=query, session_id=session_id, budget=self._settings.budget, sink=event_sink)
        # **请求级总闸**：scheduler 的 deadline 只管 DAG，撰稿 / G4 / 回路都在它外面。
        # 不设这道闸，一次 LLM 超时就能把请求拖到 70s（实测过）。
        deadline_s = float(getattr(self._settings, "request_deadline_s", 0) or 0)
        ctx.deadline_at = started + deadline_s if deadline_s > 0 else 0.0

        # ---- 定档：规则分诊（零 LLM）先跑，再决定值不值当调模型 ----
        # 顺序不能反：若先用 LLM 分诊再判断"要不要走廉价通道"，省下的钱又被分诊吃回去了。
        base_triage = rule_triage(query, facts=facts, as_of=as_of, history=history)
        base_plan = self._planner.plan(base_triage)
        decision = choose_mode(base_triage, base_plan, policy=self._settings.routing_policy)

        if decision.is_fast:
            triage = base_triage
            plan = base_plan if base_plan.short_circuit else fast_plan_of(base_plan)
            logger.info("快速通道：%s", "；".join(decision.reasons))
        else:
            triage = self._triager(query, facts=facts, as_of=as_of, history=history)
            if inspect.isawaitable(triage):
                triage = await triage
            plan = self._planner.plan(triage)
            logger.info("全编排：%s", "；".join(decision.reasons))
        ctx.triage = triage
        ctx.trace["triage_ms"] = int((time.perf_counter() - started) * 1000)
        if getattr(triage, "source", "") == "llm":
            ctx.llm_calls += 1          # 分诊也烧 token，必须计入成本
        ctx.plan = plan

        ans = Answer(
            intent=triage.intent,
            plan=plan,
            route=plan.short_circuit or ("fast" if decision.is_fast else self._route_of(plan, triage)),
            facts=dict(getattr(triage, "facts", {}) or {}),
            mode=decision.mode,
        )
        ctx.emit("meta", {**self._meta(ctx, plan, triage), **decision.to_dict()})
        ctx.emit("plan", plan.to_dict())

        # ---- 缓存前置：同问题 + 同角色 + 同 DAG 指纹 ----
        key = cache_key(query, triage.role, plan, as_of, decision.mode)
        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                ans.text, ans.sources, ans.verdict, ans.cached = (
                    hit.text,
                    hit.sources,
                    hit.verdict,
                    True,
                )
                ans.elapsed_ms = int((time.perf_counter() - started) * 1000)
                ctx.emit("sources", {"sources": hit.sources})
                ctx.emit("delta", {"text": hit.text})
                ctx.emit("done", self._done(ans, ctx))
                ans.events = list(ctx.events)
                logger.info("缓存命中 key=%s", key)
                return ans

        # ---- 短路：问候 / 无关 / 缺槽位 ----
        if plan.short_circuit:
            ans.text = self._short_circuit_text(plan)
            ans.degraded = plan.short_circuit if plan.short_circuit in {"greeting", "no_data"} else None
            ans.llm_calls = ctx.llm_calls
            ans.elapsed_ms = int((time.perf_counter() - started) * 1000)
            ctx.emit("delta", {"text": ans.text})
            ctx.emit("done", self._done(ans, ctx))
            ans.events = list(ctx.events)
            logger.info("短路 route=%s", plan.short_circuit)
            return ans

        # ---- 执行 DAG ----
        dag_started = time.perf_counter()
        result = await self._scheduler.run(plan, ctx)
        ctx.trace["dag_ms"] = int((time.perf_counter() - dag_started) * 1000)

        # ---- 某专家报"缺槽位" → 先给依据与公式，再补一句追问 ----
        # 检索整段超时/失败时，账本是空的 —— 但这不是"没有相关法条"。
        # 早先版本会答"现有资料未覆盖该问题"，把系统故障说成知识边界，属于欺骗性输出。
        if not len(ctx.ledger) and _all_steps_broken(result):
            ans.text = RETRY_REPLY
            ans.degraded = "retrieval_failed"
            ans.steps = result.to_dicts()
            ans.llm_calls = ctx.llm_calls
            ans.elapsed_ms = int((time.perf_counter() - started) * 1000)
            ctx.emit("sources", {"sources": []})
            ctx.emit("delta", {"text": ans.text})
            ctx.emit("done", self._done(ans, ctx))
            ans.events = list(ctx.events)
            logger.warning("检索整段失败（%s），按可重试错误返回", result.degraded_reason or "step failed")
            return ans

        missing = self._collect_missing(result)
        if missing:
            text = self._follow_up_for(missing)
            if len(ctx.ledger):
                # 这一支只解释法律依据，**必须用模板撰稿器**：
                # LLM 在缺数据时会自己假设一个日期/金额把算术做掉（实测会把"离职日"默认成今天），
                # 那正是本系统要避免的事。
                text = f"{text}\n\n{template_synthesizer(ctx, ctx.ledger, [])}"
            ans.text = text
            ans.sources = ctx.ledger.to_dicts()
            ans.steps = result.to_dicts()
            ans.llm_calls = ctx.llm_calls
            ans.elapsed_ms = int((time.perf_counter() - started) * 1000)
            ctx.emit("sources", {"sources": ans.sources})
            ctx.emit("delta", {"text": ans.text})
            ctx.emit("done", self._done(ans, ctx))
            ans.events = list(ctx.events)
            return ans

        draft, calc_dicts = await self._synthesize_and_verify(
            ctx, result, as_of, support=False if decision.is_fast else None
        )
        timeout_degraded = ctx.expired()

        # ---- 回路：只补检索 + 重写，最多一次（快速通道没有回路，这是它便宜的代价）----
        loop = 0
        while (
            not decision.is_fast
            and not timeout_degraded          # 时间用尽就不再赌回路
            and draft.verdict
            and draft.verdict.get("action") == "refine_retrieval"
            and loop < self._settings.budget.max_loop
        ):
            loop += 1
            ctx.loop = loop
            logger.info(
                "校验未通过，触发第 %d 次补检索 hints=%s",
                loop,
                draft.verdict.get("missing_hints"),
            )
            refine = self._refine_plan(draft.verdict.get("missing_hints") or [], as_of)
            if not refine.steps:
                break
            result2 = await self._scheduler.run(refine, ctx)
            result.done.update(result2.done)
            draft, calc_dicts = await self._synthesize_and_verify(ctx, result, as_of)

        text = draft.text
        if draft.verdict and not draft.verdict.get("passed"):
            text = self._degrade(text, draft.verdict)

        ans.text = text
        ans.sources = ctx.ledger.to_dicts()
        ans.calc = calc_dicts
        ans.verdict = draft.verdict
        ans.steps = result.to_dicts()
        ans.loops = loop
        ans.llm_calls = ctx.llm_calls
        ans.degraded = "timeout" if timeout_degraded else result.degraded_reason
        ans.elapsed_ms = int((time.perf_counter() - started) * 1000)
        ans.timings = {**ctx.trace, "total_ms": ans.elapsed_ms, "loops": loop}
        ctx.emit("sources", {"sources": ans.sources})
        ctx.emit("verify", draft.verdict or {})
        # 终稿必须用 replace 覆盖：流式草稿还没经过引文对位与五道闸，
        # 直接把它当答案等于"先生成后校验再拦"这道防线失效。
        ctx.emit(
            "delta",
            {
                "text": ans.text,
                "append": False,
                "passed": (draft.verdict or {}).get("passed"),
            },
        )
        ctx.emit("done", self._done(ans, ctx))
        ans.events = list(ctx.events)

        if self._cache is not None and not ans.degraded:
            self._cache.put(key, text=ans.text, sources=ans.sources, verdict=ans.verdict)
        return ans

    # ------------------------------------------------------------------
    async def _dispatch(self, step: Step, ctx: TaskContext) -> StepResult:
        agent = self._agents.get(step.agent)
        if agent is None:
            return StepResult(
                step_id=step.id, agent=step.agent, status="failed", error="未注册的 agent"
            )

        payload = dict(step.payload)
        if step.agent == "statute":
            payload.setdefault("query", ctx.query)
        if step.agent == "compliance":
            payload["context"] = ctx.ledger.for_prompt(
                as_of=self._as_of(ctx), budget_chars=self._settings.evidence_budget_chars
            )
        result: AgentResult = await agent.run(payload)
        if result.evidence:
            ctx.ledger.add(result.evidence, agent=step.agent)
        ctx.llm_calls += result.llm_calls

        artifact = dict(result.artifact)
        if result.status == "input_required":
            artifact["missing_slots"] = list(result.missing_slots)
        return StepResult(
            step_id=step.id,
            agent=step.agent,
            status=result.status,
            artifact=artifact,
            evidence=list(result.evidence),
            elapsed_ms=result.elapsed_ms,
            llm_calls=result.llm_calls,
            error=result.error,
        )

    async def _synthesize_with_deadline(
        self, ctx: TaskContext, calc_dicts: list[dict]
    ) -> str:
        """撰稿：优先流式；到点就放弃这次调用，用模板兜底。

        为什么必须 `wait_for` 而不是"跑完再判断 ctx.expired()"：
        后者只能阻止**后续**步骤启动，对正在飞的 8 秒调用无能为力 —— 实测
        deadline 设 2s、撰稿要 8s，端到端仍是 8.0s，那道总闸形同虚设。
        超时必须能**打断**正在等待的协程，否则写多少闸都拦不住尾延迟。

        流式与总闸的配合：`wait_for` 取消的是消费协程，已流出的分片保留在页面上；
        超时后用模板生成终稿，再由 `run()` 的终稿 `delta` 整体替换。
        """
        stream_fn = getattr(self._synthesizer, "stream", None)
        started = time.perf_counter()
        if callable(stream_fn):
            # 回路重写会再次流式输出；先把上一轮草稿清掉，否则第二轮会接在旧答案后面。
            ctx.emit("delta", {"text": "", "append": False})

            def on_delta(chunk: str) -> None:
                if "first_token_ms" not in ctx.trace:
                    # 两个口径：用户体感（从请求开始）与撰稿本身（从调用开始）
                    ctx.trace["first_token_ms"] = ctx.elapsed_ms
                    ctx.trace["synth_first_token_ms"] = int(
                        (time.perf_counter() - started) * 1000
                    )
                ctx.emit("delta", {"text": chunk, "append": True})

            produced = stream_fn(ctx, ctx.ledger, calc_dicts, on_delta)
        else:
            produced = self._synthesizer(ctx, ctx.ledger, calc_dicts)
        if not inspect.isawaitable(produced):
            return produced                       # 模板撰稿器是同步的，不受影响

        budget = 0.0
        if ctx.deadline_at > 0:
            budget = max(0.0, ctx.deadline_at - time.perf_counter())
        if budget <= 0:
            logger.warning("请求时间已用尽，撰稿直接用模板")
            return template_synthesizer(ctx, ctx.ledger, calc_dicts)
        try:
            return await asyncio.wait_for(produced, timeout=budget)
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("撰稿超时（预算 %.1fs），降级为模板", budget)
            ctx.trace["synth_timeout"] = True
            return template_synthesizer(ctx, ctx.ledger, calc_dicts)

    async def _synthesize_and_verify(
        self,
        ctx: TaskContext,
        result: PlanResult,
        as_of: date | None,
        *,
        support: bool | None = None,
    ) -> tuple["_Draft", list[dict]]:
        calc_dicts = [item for res in result.of("calc") for item in res.artifact.get("results", [])]
        if ctx.expired():
            # 时间已用尽：不再调模型，用模板把已有证据组织成答案（零额外延迟），
            # 但仍然过确定性四道闸 —— 宁可答得朴素，不可答得没有依据。
            logger.warning("请求超时，撰稿降级为模板（已用 %dms）", ctx.elapsed_ms)
            text = template_synthesizer(ctx, ctx.ledger, calc_dicts)
            verdict = await self._verify(text, ctx, result, as_of, support=False)
            if verdict is not None:
                verdict["timeout"] = True
            return _Draft(text=text, verdict=verdict), calc_dicts

        synth_started = time.perf_counter()
        text = await self._synthesize_with_deadline(ctx, calc_dicts)
        ctx.trace["synth_ms"] = int((time.perf_counter() - synth_started) * 1000)

        # 先做确定性对位，再校验：模型把"第四十七条的原文"挂到"第四十六条"上时，
        # 补检索是错的解法（证据本来就有），改标签才是。零 LLM、可审计。
        text, repairs = remap_citations(text, ctx.ledger)
        effective_support = False if (support is False or ctx.expired()) else support
        verify_started = time.perf_counter()
        verdict = await self._verify(text, ctx, result, as_of, support=effective_support)
        ctx.trace["verify_ms"] = int((time.perf_counter() - verify_started) * 1000)
        if repairs and verdict is not None:
            verdict["repaired"] = [r.to_dict() for r in repairs]
        return _Draft(text=text, verdict=verdict), calc_dicts

    async def _verify(
        self, text: str, ctx: TaskContext, result: PlanResult, as_of, *, support: bool | None = None
    ):
        agent = self._agents.get(VERIFIER_AGENT)
        if agent is None:
            return None
        facts = (ctx.triage.facts if ctx.triage else {}) or {}
        facts_amounts = [facts["monthly_wage"]] if facts.get("monthly_wage") not in (None, "") else []
        # G5 的合法金额 = 最终结果 + 代入过程里的中间量（小时工资、日工资…）。
        # 只给最终值会把"57.47 元/小时"这类正确中间量判成编造 —— 实测 c05/c06 就是这么误报的。
        intermediates = [
            Decimal(token)
            for res in result.of("calc")
            for item in res.artifact.get("results", [])
            for token in _NUMBER_RE.findall(str(item.get("substitution") or ""))
        ]
        payload = {
            "draft": text,
            "evidence": ctx.ledger.to_dicts(),
            "calc_values": [str(v) for v in [*result.calc_values, *intermediates]],
            "facts_amounts": [str(v) for v in facts_amounts],
            "as_of": as_of.isoformat() if as_of else None,
            "extra_context": self._facts_context(facts) + self._calc_context(result),
            "support": support,
        }
        verified = await agent.run(payload)
        ctx.llm_calls += verified.llm_calls
        return verified.artifact or None

    # ------------------------------------------------------------------
    @staticmethod
    def _facts_context(facts: dict) -> str:
        """用户自述的事实：复述它们不算"无依据"，但也不能当作法律依据。"""
        if not facts:
            return ""
        from app.calc.engine import SLOT_LABELS

        items = [f"- {SLOT_LABELS.get(k, k)}：{v}" for k, v in facts.items()]
        return "【用户提供的事实】（视为已知，复述不算无依据）\n" + "\n".join(items)

    @staticmethod
    def _calc_context(result: PlanResult) -> str:
        """计算结果对 G4 是**权威输入**：金额来自确定性引擎，judge 不该质疑它。"""
        rows = [item for res in result.of("calc") for item in res.artifact.get("results", [])]
        if not rows:
            return ""
        lines = [
            f"- {row.get('formula')} = {row.get('substitution')} = {row.get('value')} {row.get('unit')}"
            for row in rows
        ]
        return "\n\n【计算结果】（由确定性引擎给出，视为已知，不要判它无依据）\n" + "\n".join(lines)

    @staticmethod
    def _route_of(plan: Plan, triage: TriageResult) -> str:
        if plan.short_circuit:
            return plan.short_circuit
        agents = {s.agent for s in plan.steps}
        if "compliance" in agents:
            return "compliance"
        if sum(1 for s in plan.steps if s.agent == "statute") > 1:
            return "multi_hop"
        if "calc" in agents:
            return "calc"
        return "statute"

    @staticmethod
    def _meta(ctx: TaskContext, plan: Plan, triage: TriageResult) -> dict:
        return {
            "task_id": ctx.task_id,
            "intent": triage.intent,
            "role": triage.role,
            "complexity": triage.complexity,
            "rewrite_type": triage.rewrite_type,
            "route": Pipeline._route_of(plan, triage),
            "plan_fingerprint": plan.fingerprint,
            "mode": None,          # 由 decision.to_dict() 覆盖，占位保持字段顺序稳定
            "cached": False,
            "degraded": "triage_rules" if triage.source == "rules" and plan.short_circuit else None,
        }

    @staticmethod
    def _done(ans: Answer, ctx: TaskContext) -> dict:
        return {
            "latency_ms": ans.elapsed_ms,
            "timings": ans.timings,
            "mode": ans.mode,
            "llm_calls": ans.llm_calls,
            "sources_count": len(ans.sources),
            "loops": ans.loops,
            "cached": ans.cached,
            "degraded": ans.degraded,
        }

    @staticmethod
    def _short_circuit_text(plan: Plan) -> str:
        if plan.short_circuit == "greeting":
            return GREETING_REPLY
        if plan.short_circuit == "no_data":
            return NO_DATA_REPLY
        return plan.short_circuit_text

    @staticmethod
    def _collect_missing(result: PlanResult) -> list[str]:
        slots: list[str] = []
        for res in result.done.values():
            if res.status == "input_required":
                slots.extend(res.artifact.get("missing_slots") or [])
        return sorted(set(slots))

    @staticmethod
    def _follow_up_for(slots: list[str]) -> str:
        from app.calc.engine import SLOT_LABELS

        labels = "、".join(SLOT_LABELS.get(s, s) for s in slots)
        return f"为了给你准确的结论，还需要你补充：{labels}。"

    def _refine_plan(self, hints: list[str], as_of: date | None) -> Plan:
        steps: list[Step] = []
        for idx, hint in enumerate(hints[:2], start=1):
            steps.append(
                Step(
                    id=f"r{idx}",
                    agent="statute",
                    payload={
                        "sub_query": hint.replace("#", " "),
                        "as_of": as_of.isoformat() if as_of else None,
                    },
                )
            )
        return Plan(steps=tuple(steps), reason=f"补检索 {len(steps)} 路：{hints[:2]}")

    @staticmethod
    def _degrade(text: str, verdict: dict) -> str:
        """把未过闸告示**放在答案上方**（阶段 8 起）。

        流式下答案先于校验到达用户，若仍把告示追加在末尾，用户很可能读完答案
        就离开、看不到风险提示。横幅前置是唯一能保证"先看到风险"的位置。
        """
        lines = ["【校验未通过，以下内容需人工核对】"]
        for fail in (verdict.get("fails") or [])[:5]:
            target = f"（{fail['citation']}）" if fail.get("citation") else ""
            extra = f"｜{fail['extra'][:60]}" if fail.get("extra") else ""
            lines.append(f"- {fail.get('gate')}：{fail.get('detail')}{target}{extra}")
        if verdict.get("skipped_gates"):
            lines.append(f"- 已跳过：{'、'.join(verdict['skipped_gates'])}")
        return "\n".join([*lines, "", text])

    @staticmethod
    def _as_of(ctx: TaskContext) -> date | None:
        return ctx.triage.as_of if ctx.triage else None


class _Draft:
    __slots__ = ("text", "verdict")

    def __init__(self, text: str, verdict: dict | None) -> None:
        self.text = text
        self.verdict = verdict


_KIND_LABELS = {
    "severance": "经济补偿",
    "illegal": "违法解除赔偿金",
    "notice": "代通知金",
    "overtime": "加班费",
    "double_wage": "未签书面劳动合同二倍工资差额",
    "annual_leave": "未休年假折算",
    "non_compete": "竞业限制补偿",
    "probation": "试用期上限",
}


def _kind_label(kind: str) -> str:
    return _KIND_LABELS.get(kind, kind)


def _top_evidence(
    items: list, *, calc: list[dict] | None = None, limit: int = 3, min_ratio: float = 0.45
) -> list:
    """挑选要写进答案的法条：先保测算依据条，再按相关度补足。

    两条规则：
    1. **测算引用的条号优先** —— 计算结果的依据（如第四十七条）必须在依据里出现，
       否则用户看到"赔 16 万"却找不到算法出处；
    2. **低于首条 45% 分数的丢弃** —— 检索 top5 里常有凑数的近义条文，
       全塞进【依据】会让答案像法条堆砌，也给 G4 增加无谓负担。
    """
    if not items:
        return []
    basis: set[str] = set()
    for row in calc or []:
        for ref in row.get("basis_articles") or []:
            basis.add(ref.split("#")[-1])

    top = max(item.score for item in items) or 1.0
    keep = [item for item in items if item.score >= top * min_ratio or item.article_no in basis]
    keep.sort(key=lambda e: (e.article_no not in basis, -e.score))
    return keep[:limit]


def _all_steps_broken(result: PlanResult) -> bool:
    """所有已执行的步骤都没产出证据，且至少一个明确失败/超时。"""
    steps = list(result.done.values())
    if not steps:
        return False
    if any(step.ok and step.evidence for step in steps):
        return False
    return any(step.status in {"timeout", "failed"} for step in steps)


def _only_missing_evidence(verdict: dict | None) -> bool:
    """失败是否**只有**"引用了没检索到的条号"这一类。"""
    fails = (verdict or {}).get("fails") or []
    return bool(fails) and all(f.get("gate") == "G1_not_found" for f in fails)


def template_synthesizer(ctx: TaskContext, ledger: EvidenceLedger, calc: list[dict]) -> str:
    """离线/兜底撰稿器：把已核实的证据组织成固定结构。

    生产用 LLM 版本（prompts.SYNTHESIS_SYSTEM），但**结构与引用格式完全一致**——
    这样评测跑的是同一条链路，只是撰稿器换人。
    """
    laws = _top_evidence(ledger.all(), calc=calc, limit=3)
    lines: list[str] = []

    if calc:
        first = calc[0]
        lines.append(
            f"【结论】按你提供的事实，{_kind_label(first['kind'])}约为 {first['value']} {first['unit']}。"
        )
    elif laws:
        lines.append(f"【结论】与你的问题最相关的是{laws[0].title}，具体规定见下方依据。")
    else:
        lines.append("【结论】现有资料未覆盖该问题，无法给出确定结论。")

    if laws:
        lines += ["", "【依据】"]
        for item in laws:
            lines.append(f"- {item.title}（{item.eid}）：“{item.quote}”")

    if calc:
        lines += ["", "【计算】"]
        for item in calc:
            lines.append(
                f"- {_kind_label(item['kind'])}：{item['formula']} = {item['substitution']} "
                f"= {item['value']} {item['unit']}"
            )
            for note in item.get("assumptions") or []:
                lines.append(f"  - {note}")

    lines += ["", "【提示】以上结论以已核实的法条为依据；个案还需结合完整证据与当地裁审口径。", "", DISCLAIMER]
    return "\n".join(lines)
