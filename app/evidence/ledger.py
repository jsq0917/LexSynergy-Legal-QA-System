"""证据账本：一次任务内所有证据的唯一事实源。

三条纪律：
1. **只增不改** —— 任何 Agent 都不能删/改别人的证据，避免互相覆盖；
2. **编号即引用** —— 答案里的 E3 必须能在账本里查到，Verifier 靠它做 G2/G4；
3. **超预算丢低分** —— 由账本统一裁剪，不让各 Agent 自己截断（各截各的必然错位）。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from app.evidence.models import Evidence


class EvidenceLedger:
    def __init__(self) -> None:
        self._items: dict[str, Evidence] = {}
        self._order: list[str] = []
        self._seq = 0

    # ---------------- 写 ----------------
    def add(self, items: list[Evidence], *, agent: str = "") -> list[Evidence]:
        """去重（同 law#article_no 保留高分）+ 编号，返回本次**新增**的条目。"""
        added: list[Evidence] = []
        for item in items:
            key = item.key
            prev = self._items.get(key)
            if prev is not None:
                if item.score > prev.score:
                    self._items[key] = item.with_eid(prev.eid)
                continue
            self._seq += 1
            eid = f"E{self._seq}"
            entry = item.with_eid(eid)
            if agent and not entry.from_agent:
                entry = replace(entry, from_agent=agent)
            self._items[key] = entry
            self._order.append(key)
            added.append(entry)
        return added

    # ---------------- 读 ----------------
    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def all(self) -> list[Evidence]:
        return [self._items[k] for k in self._order]

    def canonical_law(self, name: str) -> str | None:
        """把《劳动合同法》这类简称解析成证据里的全称（唯一命中才算）。

        不解析会制造**假阳性 G1**：答案写"《劳动合同法》第四十一条"是对的中文习惯，
        但按字面拼 key 就找不到证据，于是被判"引用了不存在的条号"，触发一次白跑的补检索。
        """
        laws = {item.law for item in self._items.values()}
        if name in laws:
            return name
        hits = [law for law in laws if law.endswith(name) or name.endswith(law)]
        return hits[0] if len(hits) == 1 else None

    def lookup(self, law: str, article_no: str) -> Evidence | None:
        return self._items.get(f"{self.canonical_law(law) or law}#{article_no}")

    def by_eid(self, eid: str) -> Evidence | None:
        for item in self._items.values():
            if item.eid == eid:
                return item
        return None

    def lookup_loose(self, article_no: str) -> Evidence | None:
        """只给了条号没给法名时，若唯一命中则返回（多义则 None，逼上游补法名）。"""
        hits = [e for e in self._items.values() if e.article_no == article_no]
        return hits[0] if len(hits) == 1 else None

    # ---------------- 装配 ----------------
    def for_prompt(self, *, as_of: date | None = None, budget_chars: int = 6000) -> str:
        """按分数降序灌入预算，跳过 as_of 时已失效的条文。"""
        usable = [e for e in self.all() if e.effective_at(as_of)]
        usable.sort(key=lambda e: (-e.priority, -e.score))
        blocks: list[str] = []
        total = 0
        for idx, item in enumerate(usable, start=1):
            # 标题必须让模型能"照抄"：`《法》第X条` 独立成行、紧跟证据编号。
            # 早先把法名/章节/条号拼成一个长串（`法·第四章·第四十七条（E1）`），
            # 模型在密集引用时会错位一格（把第四十七条的原文挂到第四十六条上）——
            # 实测 c02 就是这么被 G2 抓到的。少一层拼接，少一类系统性错误。
            block = f"【资料{idx}】《{item.law}》{item.article_no}（{item.eid}）\n{item.quote.strip()}"
            if blocks and total + len(block) > budget_chars:
                continue
            if not blocks and len(block) > budget_chars:
                block = block[:budget_chars]
            blocks.append(block)
            total += len(block)
            if total >= budget_chars:
                break
        return "\n\n".join(blocks)

    def to_dicts(self) -> list[dict]:
        return [e.to_dict() for e in self.all()]
