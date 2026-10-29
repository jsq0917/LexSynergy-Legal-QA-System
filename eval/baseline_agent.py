"""单 Agent 基线：一次检索 + 一次生成，复刻 LawAnswer 的现链路。

**这不是稻草人**。它用的是与本系统完全相同的检索器、相同的 top-k、相同的证据文本，
唯一区别是把"取证 / 计算 / 生成 / 校验"四件事压给同一个模型、一次调用完成。
对照组因此只差**编排与校验**这一个变量 —— 这样得出的差值才能归因于架构，
而不是"基线检索更差"。

两处刻意保留的差别（必须在报告里说清，否则对比不成立）：
1. 基线**没有**子查询拆分：多跳问题只检索一次（这是它的架构限制，不是我们放水）；
2. 基线**没有**确定性测算引擎：金额由模型心算。
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from app.core.logger import get_logger
from app.evidence.ledger import EvidenceLedger
from app.generation.llm_client import DashScopeClient, get_llm_client
from app.retrieval.port import StatuteSearchPort
from config import Settings, get_settings

logger = get_logger("eval.baseline")

#: 金额行：`金额：72000.00 元` —— 要求模型给出可解析的单一数字，避免抽取靠猜
AMOUNT_LINE = re.compile(r"金额[：:]\s*([0-9][0-9,，]*(?:\.[0-9]+)?)\s*(万元|元)?")

SYSTEM = """你是资深中国劳动法律师，直接回答当事人的问题。

【硬性要求】
1. 每条结论后必须注明依据，格式为《法律全称》第X条（用全称，如"中华人民共和国劳动合同法"，不要写简称）；
2. 只能引用【法条资料】中出现的条文，不得编造条号或法名；
3. 【法条资料】未覆盖的内容，明确写"现有资料未涉及"，不要凭记忆补充；
4. 若问题涉及金额计算，最后单独一行输出：金额：<数字> 元（只放一个数字，保留两位小数）；
   不要在这一行写公式或区间；
5. 正文不超过 500 字，不要加免责声明。"""

USER_TMPL = """【法条资料】
{evidence}

【用户问题】
{query}"""


@dataclass
class BaselineAnswer:
    text: str = ""
    sources: list[dict] = field(default_factory=list)
    llm_calls: int = 0
    elapsed_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    amount: Decimal | None = None
    error: str | None = None


def parse_amount(text: str) -> Decimal | None:
    """取最后一条 `金额：` 行的数字；没有就返回 None（不猜）。"""
    matches = list(AMOUNT_LINE.finditer(text or ""))
    if not matches:
        return None
    raw, unit = matches[-1].group(1), matches[-1].group(2) or "元"
    try:
        value = Decimal(raw.replace(",", "").replace("，", ""))
    except InvalidOperation:
        return None
    return value * (Decimal("10000") if unit == "万元" else Decimal("1"))


class BaselineSingleAgent:
    """单 Agent：`检索 → 拼 Prompt → 一次生成`。"""

    def __init__(
        self,
        search: StatuteSearchPort,
        client: DashScopeClient | None = None,
        settings: Settings | None = None,
        *,
        topk: int = 5,
        max_tokens: int = 900,
    ) -> None:
        self._s = settings or get_settings()
        self._search = search
        self._client = client or get_llm_client(self._s)
        self._topk = topk
        self._max_tokens = max_tokens

    async def answer(self, query: str) -> BaselineAnswer:
        started = time.perf_counter()
        try:
            evidence = await self._search.search_many([query], topk=self._topk)
        except Exception as exc:  # noqa: BLE001 - 检索失败如实记账，不吞掉
            return BaselineAnswer(error=f"检索失败：{exc}", elapsed_ms=_ms(started))

        ledger = EvidenceLedger()
        ledger.add(evidence)
        block = ledger.for_prompt(budget_chars=self._s.evidence_budget_chars)
        if not block.strip():
            return BaselineAnswer(
                text="暂未在现有资料中找到与该问题直接相关的法条依据。",
                sources=ledger.to_dicts(),
                elapsed_ms=_ms(started),
            )

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(evidence=block, query=query)},
        ]
        try:
            result = await asyncio.to_thread(
                self._client.chat, messages, max_tokens=self._max_tokens
            )
        except Exception as exc:  # noqa: BLE001 - 单 Agent 没有降级路径，失败即失败
            logger.warning("基线生成失败：%s", str(exc)[:160])
            return BaselineAnswer(
                sources=ledger.to_dicts(),
                llm_calls=1,
                elapsed_ms=_ms(started),
                error=f"生成失败：{str(exc)[:120]}",
            )

        return BaselineAnswer(
            text=result.text,
            sources=ledger.to_dicts(),
            llm_calls=1,
            elapsed_ms=_ms(started),
            tokens_in=result.usage.prompt_tokens,
            tokens_out=result.usage.completion_tokens,
            amount=parse_amount(result.text),
        )


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
