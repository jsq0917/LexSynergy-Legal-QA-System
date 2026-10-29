"""引用校验专家：把五道闸包成 A2A 可调用的能力。

跨进程时账本要序列化，所以 payload 里传 evidence 的 dict 列表，在这里重建账本
（eid 按顺序重排，保证与答案里的 E1..En 一致）。这顺带证明了协议边界是"真的"。
"""

from __future__ import annotations

import time
from datetime import date
from decimal import Decimal

from app.agents.base import AgentResult, BaseAgent
from app.evidence.ledger import EvidenceLedger
from app.evidence.models import Evidence
from app.verifier.service import Verifier


class VerifierAgent(BaseAgent):
    name = "verifier"
    description = "引用校验专家：核验条号存在性、引文逐字一致性、时效性、语义支撑与金额一致性。"
    skills = (
        {
            "id": "verify_citations",
            "name": "引用与断言校验",
            "description": "输入答案草稿与证据集，返回五道闸的裁决（pass/refine_retrieval/downgrade）",
            "tags": ["legal", "verification", "guardrail"],
            "examples": ["校验这段回答里的第X条引用是否真实且支撑结论"],
        },
    )

    def __init__(self, verifier: Verifier | None = None) -> None:
        self._verifier = verifier or Verifier()

    async def run(self, payload: dict) -> AgentResult:
        started = time.perf_counter()
        draft = payload.get("draft") or ""
        if not draft:
            return AgentResult.input_required(self.name, ["draft"])

        ledger = EvidenceLedger()
        ledger.add([Evidence.from_dict(item) for item in payload.get("evidence") or []])
        calc_values = [Decimal(str(v)) for v in payload.get("calc_values") or []]
        facts_amounts = [Decimal(str(v)) for v in payload.get("facts_amounts") or []]
        as_of_raw = payload.get("as_of")
        as_of = date.fromisoformat(as_of_raw) if as_of_raw else None

        verdict = await self._verifier.verify(
            draft,
            ledger,
            calc_values=calc_values,
            facts_amounts=facts_amounts,
            as_of=as_of,
            extra_context=str(payload.get("extra_context") or ""),
            support=payload.get("support"),
        )
        return AgentResult.success(
            self.name,
            artifact=verdict.to_dict(),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            llm_calls=(
                0
                if self._verifier._support is None or payload.get("support") is False
                else 1
            ),  # noqa: SLF001 - 计量用
        )
