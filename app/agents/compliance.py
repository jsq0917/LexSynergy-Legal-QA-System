"""合规审查专家（唯一真正用 LLM 判断的业务 Agent）。

输入：待审文本（制度/条款/事实描述）+ 已检索证据。
输出：issue list，每条必须挂 ``basis``（证据 key），没有依据就判 ``unknown`` 而不是猜。

审计器以可调用对象注入：生产是 LLM，测试是确定性假实现——这样"审查逻辑"和
"模型调用"解耦，能把 Prompt 变更对结果的影响单独测出来。
"""

from __future__ import annotations

import inspect
import time
from typing import Awaitable, Callable

from app.agents.base import AgentResult, BaseAgent
from app.core.logger import get_logger

logger = get_logger("agent.compliance")

#: (待审文本, 证据上下文) -> issue 列表
Auditor = Callable[[str, str], list[dict] | Awaitable[list[dict]]]

RISK_LEVELS = ("high", "medium", "low")
VERDICTS = ("violation", "risk", "compliant", "unknown")


class ComplianceAgent(BaseAgent):
    name = "compliance"
    description = "合规审查专家：对劳动用工制度与条款逐项判定合规性，给出风险等级、依据与整改建议。"
    skills = (
        {
            "id": "review_clause",
            "name": "制度条款合规审查",
            "description": "输入员工手册条款或事实描述，按 checklist 输出违规项与整改建议",
            "tags": ["legal", "compliance", "review"],
            "examples": [
                "我们公司规定试用期 6 个月，工资按 80% 发，合规吗",
                "员工手册写着自愿放弃社保，有效吗",
            ],
        },
    )

    def __init__(self, auditor: Auditor | None = None) -> None:
        self._auditor = auditor

    async def run(self, payload: dict) -> AgentResult:
        started = time.perf_counter()
        text = (payload.get("text") or "").strip()
        if not text:
            return AgentResult.input_required(self.name, ["text"])
        if self._auditor is None:
            return AgentResult.failed(self.name, "未配置审查器（auditor）")

        context = payload.get("context") or ""
        try:
            raw = self._auditor(text, context)
            issues = await raw if inspect.isawaitable(raw) else raw
        except Exception as exc:  # noqa: BLE001 - 审查失败要降级，不能拖垮整条链路
            logger.warning("审查失败：%s", str(exc)[:160])
            return AgentResult.failed(self.name, f"审查失败：{exc}")

        cleaned = [self._clean(item) for item in (issues or [])]
        cleaned = [item for item in cleaned if item]
        logger.info("审查完成 issues=%d", len(cleaned))
        return AgentResult.success(
            self.name,
            artifact={"issues": cleaned, "checked": len(cleaned)},
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            llm_calls=1,
        )

    @staticmethod
    def _clean(item: dict) -> dict | None:
        """字段收敛：非法风险级/判定一律回落安全值，缺依据降级为 unknown。"""
        if not isinstance(item, dict) or not item.get("clause"):
            return None
        verdict = item.get("verdict")
        risk = item.get("risk")
        basis = [str(b) for b in item.get("basis") or []]
        if verdict not in VERDICTS:
            verdict = "unknown"
        if verdict in {"violation", "risk"} and not basis:
            verdict = "unknown"
        if risk not in RISK_LEVELS:
            risk = "medium"
        return {
            "clause": str(item["clause"])[:200],
            "verdict": verdict,
            "risk": risk,
            "reason": str(item.get("reason") or "")[:400],
            "basis": basis,
            "fix": str(item.get("fix") or "")[:400],
        }
