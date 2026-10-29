"""校验结果模型。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Citation:
    law: str
    article_no: str
    quote: str = ""      # 答案里引号内的原文摘录（可为空 = 只标了条号）
    raw: str = ""        # 原文片段，便于报错定位

    @property
    def key(self) -> str:
        return f"{self.law}#{self.article_no}"


@dataclass(frozen=True)
class GateFail:
    gate: str                 # G1_not_found / G2_misquote / G3_expired / G4_misgrounded / G5_numeric
    detail: str
    citation: Citation | None = None
    extra: str = ""

    def to_dict(self) -> dict:
        out = {"gate": self.gate, "detail": self.detail}
        if self.citation:
            out["citation"] = self.citation.key
        if self.extra:
            out["extra"] = self.extra
        return out


@dataclass(frozen=True)
class Verdict:
    passed: bool
    action: str                       # pass | refine_retrieval | downgrade
    checked: int
    fails: tuple[GateFail, ...] = field(default_factory=tuple)
    missing_hints: tuple[str, ...] = field(default_factory=tuple)
    skipped_gates: tuple[str, ...] = field(default_factory=tuple)

    @property
    def failed_gates(self) -> list[str]:
        return [f.gate for f in self.fails]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "action": self.action,
            "checked": self.checked,
            "fails": [f.to_dict() for f in self.fails],
            "missing_hints": list(self.missing_hints),
            "skipped_gates": list(self.skipped_gates),
        }
