"""引文—条号自动对位：把「引文是对的、条号挂错了」的引用改挂到真正的条文上。

**为什么需要这一步**：模型在密集引用段落里最常见的错误不是编造条文，而是**错位一格** ——
把第四十七条的原文挂到第四十六条上。这类答案里每个字都是真的，G2 只能判它失败，
然后走补检索回路（多烧 2 次 LLM），而问题其实出在"标签"而不是"证据"。

修复是**确定性**的：引文若逐字命中账本里另一条证据，就说明标签错了，改标签即可。
零 LLM、零额外检索、可审计（每条修复都进 verdict 的 `repaired` 列表）。

三条保守约束，宁可少修也不修错：
1. 引文为空 → 不修（没有依据判断它属于谁）；
2. 原文里该引用字符串出现多次 → 不修（无法确定该改哪一处）；
3. 命中不止一条证据（歧义）→ 不修。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.logger import get_logger
from app.evidence.ledger import EvidenceLedger
from app.evidence.models import Evidence
from app.verifier.gates import extract_citations, verbatim_ok

logger = get_logger("verifier.repair")


@dataclass(frozen=True)
class Repair:
    old: str          # 原引用 key（`法#条号`）
    new: str          # 对位后的 key
    quote: str = ""   # 引文前 40 字，便于人工复核

    def to_dict(self) -> dict:
        return {"from": self.old, "to": self.new, "quote": self.quote}


def _owner(quote: str, ledger: EvidenceLedger) -> Evidence | None:
    """引文真正属于哪一条证据：唯一命中才返回，否则 None。"""
    hits = [item for item in ledger.all() if verbatim_ok(quote, item.quote)]
    return hits[0] if len(hits) == 1 else None


def g2_violations(text: str, ledger: EvidenceLedger) -> int:
    """数一数有多少条引文与它自己的条号对不上（G2 口径）。"""
    count = 0
    for cite in extract_citations(text or ""):
        if not cite.quote:
            continue
        ev = ledger.lookup(cite.law, cite.article_no)
        if ev is None or not verbatim_ok(cite.quote, ev.quote):
            count += 1
    return count


def remap_citations(draft: str, ledger: EvidenceLedger) -> tuple[str, list[Repair]]:
    """返回 (修好的草稿, 修复记录)。没有可修的就不动原文。

    **最后一道保险**：改完必须让"引文对不上条号"的数量严格下降，否则退回原文。
    自动改引用是危险操作（改错比不改更糟），所以宁可整体放弃这一次修复。
    """
    if not draft:
        return draft, []
    before = g2_violations(draft, ledger)

    text = draft
    repairs: list[Repair] = []
    for cite in extract_citations(draft):
        if not cite.quote:
            continue
        current = ledger.lookup(cite.law, cite.article_no)   # lookup 内部已做法名简称解析
        if current is not None and verbatim_ok(cite.quote, current.quote):
            continue                      # 本来就对，不动
        if text.count(cite.raw) != 1:
            continue                      # 出现多次，无法确定改哪一处
        target = _owner(cite.quote, ledger)
        if target is None:
            continue                      # 无命中或有歧义
        if current is not None and target.key == current.key:
            continue
        logger.info(
            "引文对位：%s -> %s（%s…）", cite.key, target.key, cite.quote[:24]
        )
        text = text.replace(cite.raw, f"《{target.law}》{target.article_no}", 1)
        repairs.append(Repair(old=cite.key, new=target.key, quote=cite.quote[:40]))

    if repairs and g2_violations(text, ledger) >= before:
        logger.warning("对位未减少冲突（%d -> %d），放弃修复", before, g2_violations(text, ledger))
        return draft, []
    return text, repairs
