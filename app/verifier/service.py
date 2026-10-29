"""校验服务：把五道闸串起来，产出带 action 的裁决。

action 语义（由编排层消费）：
* ``pass``               —— 直接交付；
* ``refine_retrieval``   —— 缺证据/引错条/支撑不足：补检索一轮后重写；
* ``downgrade``          —— 金额算错：不硬改，降级为"需人工核对"并标红。
"""

from __future__ import annotations

import inspect
from decimal import Decimal
from typing import Any, Awaitable, Callable

from app.core.logger import get_logger
from app.evidence.ledger import EvidenceLedger
from app.verifier.gates import claims_of, extract_citations, numeric_mismatches, verbatim_ok
from app.verifier.models import GateFail, Verdict

logger = get_logger("verifier")

SupportChecker = Callable[[list[str], str], list[str] | Awaitable[list[str]]]


class Verifier:
    def __init__(
        self,
        *,
        support_checker: SupportChecker | None = None,
        tolerance: Decimal = Decimal("0.01"),
    ) -> None:
        self._support = support_checker
        self._tolerance = tolerance

    async def verify(
        self,
        draft: str,
        ledger: EvidenceLedger,
        *,
        calc_values: list[Decimal] | None = None,
        facts_amounts: list[Decimal] | None = None,
        as_of=None,
        extra_context: str = "",
        support: bool | None = None,
    ) -> Verdict:
        """`support=False` 只跑四道确定性闸，跳过 G4（供"补证据后复查同一份草稿"用）。"""
        fails: list[GateFail] = []
        skipped: list[str] = []
        citations = extract_citations(draft)

        # ---- G1 存在性 / G2 逐字 / G3 时效 ----
        for cite in citations:
            evidence = ledger.lookup(cite.law, cite.article_no)
            if evidence is None:
                loose = ledger.lookup_loose(cite.article_no)
                detail = "法名与条号不匹配" if loose else "该条号不存在于已检索证据"
                fails.append(GateFail("G1_not_found", detail, cite))
                continue
            if not verbatim_ok(cite.quote, evidence.quote):
                fails.append(GateFail("G2_misquote", "引文与法条原文不一致", cite))
                continue
            if not evidence.effective_at(as_of):
                fails.append(GateFail("G3_expired", "该条在事发时点无效", cite))

        # ---- G5 数字一致（确定性） ----
        allowed = list(calc_values or []) + list(facts_amounts or [])
        if allowed:
            for raw, value in numeric_mismatches(draft, allowed, tolerance=self._tolerance):
                fails.append(
                    GateFail("G5_numeric", f"金额 {raw} 无法由测算结果推出", extra=str(value))
                )

        # ---- G4 语义支撑（唯一用 LLM 的闸） ----
        if support is False or self._support is None:
            skipped.append("G4_misgrounded")
        elif citations:
            claims = claims_of(draft)
            context = ledger.for_prompt(as_of=as_of, budget_chars=8000)
            if extra_context:
                context = f"{context}\n\n{extra_context}"
            try:
                result = self._support(claims, context)
                if inspect.isawaitable(result):
                    result = await result
                for claim in result or []:
                    fails.append(
                        GateFail("G4_misgrounded", "该断言未被任何证据支撑", extra=claim[:100])
                    )
            except Exception as exc:  # noqa: BLE001 - 校验失败不应让整条链路断
                logger.warning("G4 支撑检查异常，降级为跳过：%s", str(exc)[:160])
                skipped.append("G4_misgrounded")

        fails.sort(key=lambda f: f.gate)
        action = self._decide(fails)
        verdict = Verdict(
            passed=not fails,
            action=action,
            checked=len(citations),
            fails=tuple(fails),
            missing_hints=tuple(self._hints(fails)),
            skipped_gates=tuple(skipped),
        )
        logger.info(
            "校验完成 checked=%d passed=%s action=%s fails=%s skipped=%s",
            verdict.checked,
            verdict.passed,
            verdict.action,
            verdict.failed_gates or "-",
            verdict.skipped_gates or "-",
        )
        return verdict

    @staticmethod
    def _decide(fails: list[GateFail]) -> str:
        if not fails:
            return "pass"
        if any(f.gate == "G5_numeric" for f in fails):
            return "downgrade"
        return "refine_retrieval"

    @staticmethod
    def _hints(fails: list[GateFail]) -> list[str]:
        hints: list[str] = []
        for fail in fails:
            if fail.gate in {"G1_not_found", "G2_misquote", "G3_expired"} and fail.citation:
                hints.append(fail.citation.key)
            elif fail.gate == "G4_misgrounded" and fail.extra:
                hints.append(fail.extra)
        # 去重保序
        seen: set[str] = set()
        unique = [h for h in hints if not (h in seen or seen.add(h))]
        return unique[:5]
