"""LLM 版 Triage：规则先跑一遍做兜底，LLM 只负责"更准地抽事实"。

为什么要规则打底而不是"完全交给 LLM"：
1. LLM 失败（限流/超时/JSON 抽风）时必须立刻可用 —— 直接把规则结果返回，链路不断；
2. 规则结果同时充当**默认值容器**：LLM 某个字段非法时逐个回落，而不是整条作废；
3. 显式传入的 `facts`（用户在 CLI/API 里写的）**优先级最高**，不允许被模型改写。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace

from app.core.logger import get_logger
from app.generation.llm_client import DashScopeClient, get_llm_client
from app.generation.prompts import CALC_KINDS, FACT_NAMES, triage_messages
from app.orchestrator.models import TriageResult
from app.orchestrator.rules import follow_up_text, missing_slots_for
from app.orchestrator.rules import triage as _rule
from config import Settings, get_settings

logger = get_logger("triage")

INTENTS = ("labor_law", "civil_code", "general", "greeting", "other")
ROLES = ("employee", "employer")
COMPLEXITIES = ("simple", "medium", "complex")
REWRITES = ("direct", "history", "decompose")
LAWS = (
    "中华人民共和国劳动合同法",
    "中华人民共和国劳动法",
    "中华人民共和国民法典",
)
FACT_KEYS = FACT_NAMES


class LLMTriager:
    def __init__(
        self,
        client: DashScopeClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._s = settings or get_settings()
        self._client = client or get_llm_client(self._s)

    async def __call__(
        self,
        query: str,
        *,
        facts: dict | None = None,
        as_of=None,
        history: list[dict] | None = None,
    ) -> TriageResult:
        base = _rule(query, facts=facts, as_of=as_of, history=history)
        if not self._s.has_llm or base.intent == "greeting":
            return base

        started = time.perf_counter()
        try:
            data, usage = await asyncio.to_thread(
                self._client.json_chat,
                triage_messages(query, history),
                model=self._s.fast_model,
                max_tokens=400,
            )
        except Exception as exc:  # noqa: BLE001 - 分诊失败必须能降级
            logger.warning("LLM 分诊失败，回落规则：%s", str(exc)[:160])
            return base

        merged = self._merge(base, data, facts)
        logger.info(
            "LLM 分诊 intent=%s role=%s calc=%s facts=%s 耗时=%dms tokens=%d",
            merged.intent,
            merged.role,
            merged.needs_calc,
            sorted(merged.facts),
            int((time.perf_counter() - started) * 1000),
            usage.total,
        )
        return merged

    # ------------------------------------------------------------------
    def _merge(self, base: TriageResult, data: dict, explicit: dict | None) -> TriageResult:
        intent = data.get("intent") if data.get("intent") in INTENTS else base.intent
        role = data.get("role") if data.get("role") in ROLES else base.role
        complexity = (
            data.get("complexity") if data.get("complexity") in COMPLEXITIES else base.complexity
        )
        rewrite = data.get("rewrite_type") if data.get("rewrite_type") in REWRITES else base.rewrite_type

        laws = [l for l in (data.get("law_hints") or []) if l in LAWS][:3]
        if not laws:
            laws = base.law_hints

        subs = [s.strip() for s in (data.get("sub_queries") or []) if isinstance(s, str) and s.strip()]
        if not subs:
            subs = base.sub_queries
        if rewrite != "decompose":
            subs = subs[:1] or base.sub_queries

        kinds = [k for k in (data.get("needs_calc") or []) if k in CALC_KINDS]
        if intent in {"greeting", "other"}:
            kinds = []

        # 事实：LLM 抽的 → 显式传入的覆盖（用户写的为准）
        facts: dict = {}
        raw_facts = data.get("facts") or {}
        if isinstance(raw_facts, dict):
            for key in FACT_KEYS:
                value = raw_facts.get(key)
                if value not in (None, "", []):
                    facts[key] = value
        facts.update({k: v for k, v in (explicit or {}).items() if v not in (None, "")})

        missing = missing_slots_for(kinds, facts)
        return replace(
            base,
            intent=intent,
            role=role,
            complexity=complexity,
            rewrite_type=rewrite,
            law_hints=laws,
            sub_queries=subs or base.sub_queries,
            facts=facts,
            needs_calc=kinds,
            missing_slots=missing,
            follow_up=follow_up_text(missing) if missing else "",
            source="llm",
        )
