"""证据模型：证据与论断结构性分离的载体。

检索方（StatuteAgent）只产出 Evidence；撰稿方（Synthesizer）只消费 Evidence。
两边都不越界，是这套系统防幻觉最硬的一层。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date


@dataclass(frozen=True)
class Evidence:
    law: str                          # 中华人民共和国劳动合同法
    article_no: str                   # 第四十七条
    quote: str                        # 条文原文（G2 逐字比对的基准）
    chunk_id: str = ""
    chapter_path: str = ""            # 第四章
    score: float = 0.0                # 融合/重排分
    from_agent: str = "statute"
    eid: str = ""                     # E1..En，由账本统一编号
    effective_from: date | None = None
    effective_to: date | None = None
    #: 特别法优于一般法：劳动合同法 20，劳动法 10，民法典 0
    priority: int = 0

    @property
    def key(self) -> str:
        """去重与引用定位的唯一键：跨法同条号不会混淆。"""
        return f"{self.law}#{self.article_no}"

    @property
    def title(self) -> str:
        parts = [f"《{self.law}》"]
        if self.chapter_path:
            parts.append(self.chapter_path)
        if self.article_no:
            parts.append(self.article_no)
        return "·".join(parts)

    def with_eid(self, eid: str) -> "Evidence":
        return replace(self, eid=eid)

    def effective_at(self, as_of: date | None) -> bool:
        """G3：as_of 当日该条是否有效。无日期信息视为长期有效。"""
        if as_of is None:
            return True
        if self.effective_from and as_of < self.effective_from:
            return False
        if self.effective_to and as_of > self.effective_to:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "eid": self.eid,
            "law": self.law,
            "article_no": self.article_no,
            "quote": self.quote,
            "chunk_id": self.chunk_id,
            "chapter_path": self.chapter_path,
            "score": self.score,
            "from_agent": self.from_agent,
            "priority": self.priority,
            "effective_from": self.effective_from.isoformat() if self.effective_from else None,
            "effective_to": self.effective_to.isoformat() if self.effective_to else None,
        }

    @staticmethod
    def from_dict(raw: dict) -> "Evidence":
        def _date(value) -> date | None:
            return date.fromisoformat(str(value)) if value else None

        return Evidence(
            law=raw["law"],
            article_no=raw["article_no"],
            quote=raw.get("quote", ""),
            chunk_id=raw.get("chunk_id", ""),
            chapter_path=raw.get("chapter_path", ""),
            score=float(raw.get("score") or 0.0),
            from_agent=raw.get("from_agent", "statute"),
            eid=raw.get("eid", ""),
            effective_from=_date(raw.get("effective_from")),
            effective_to=_date(raw.get("effective_to")),
            priority=int(raw.get("priority") or 0),
        )
