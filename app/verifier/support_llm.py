"""G4 语义支撑判定的 LLM 实现。

两个必须做的防御：
1. **只认输入里的句子**：模型可能把断言"复述"一遍，字符串对不上就没法用；
   这里用相似度回落到原始 claim，避免把模型新造的句子当成失败项。
2. **失败即跳过**：judge 挂了不能让整条链路报错，Verifier 会把它记入 `skipped_gates`。
"""

from __future__ import annotations

import asyncio

from app.core.logger import get_logger
from app.generation.llm_client import DashScopeClient, get_llm_client
from app.generation.prompts import support_messages
from config import Settings, get_settings

logger = get_logger("support")


def _similar(left: str, right: str, threshold: float = 0.6) -> bool:
    """粗糙但够用的相似度：字符二元组 Jaccard。用于把模型复述对齐回原句。"""
    a = {left[i : i + 2] for i in range(max(0, len(left) - 1))}
    b = {right[i : i + 2] for i in range(max(0, len(right) - 1))}
    if not a or not b:
        return left.strip() == right.strip()
    return len(a & b) / len(a | b) >= threshold


class LLMSupportChecker:
    def __init__(
        self,
        client: DashScopeClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._s = settings or get_settings()
        self._client = client or get_llm_client(self._s)

    async def __call__(self, claims: list[str], context: str) -> list[str]:
        if not claims:
            return []
        data, usage = await asyncio.to_thread(
            self._client.json_chat,
            support_messages(claims, context),
            model=self._s.fast_model,
            max_tokens=600,
        )
        flagged = data.get("unsupported") or []
        if not isinstance(flagged, list):
            flagged = []
        flagged = [str(item) for item in flagged if isinstance(item, str)]

        # 把模型给的句子对齐回原始 claim：找不到对应原文的丢弃（防模型新造）
        unsupported: list[str] = []
        for claim in claims:
            if any(_similar(claim, item) or claim in item or item in claim for item in flagged):
                unsupported.append(claim)
        logger.info(
            "G4 支撑判定 claims=%d unsupported=%d tokens=%d",
            len(claims),
            len(unsupported),
            usage.total,
        )
        return unsupported
