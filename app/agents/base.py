"""Agent 基类与统一结果契约。

**业务逻辑写在 `run()` 里，协议适配写在 protocol/executor 里** —— 这条分界让
单测可以完全绕过 A2A（不启容器、不构造 RequestContext），也避免把业务绑在协议上。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.evidence.models import Evidence

STATUSES = ("ok", "no_data", "input_required", "failed", "timeout")


@dataclass
class AgentResult:
    agent: str
    status: str = "ok"
    artifact: dict = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    llm_calls: int = 0
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"非法状态：{self.status}")

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @classmethod
    def success(cls, agent: str, **kw: Any) -> "AgentResult":
        return cls(agent=agent, status="ok", **kw)

    @classmethod
    def no_data(cls, agent: str, message: str = "") -> "AgentResult":
        return cls(agent=agent, status="no_data", error=message or "无可用数据")

    @classmethod
    def input_required(cls, agent: str, slots: list[str]) -> "AgentResult":
        return cls(agent=agent, status="input_required", missing_slots=list(slots))

    @classmethod
    def failed(cls, agent: str, message: str) -> "AgentResult":
        return cls(agent=agent, status="failed", error=message)

    def to_dict(self) -> dict:
        """跨进程传输用的稳定契约（DataPart 的 payload）。"""
        return {
            "agent": self.agent,
            "status": self.status,
            "artifact": self.artifact,
            "evidence": [e.to_dict() for e in self.evidence],
            "missing_slots": list(self.missing_slots),
            "elapsed_ms": self.elapsed_ms,
            "llm_calls": self.llm_calls,
            "error": self.error,
            "schema": "agent-result/1",
        }

    @staticmethod
    def from_dict(raw: dict) -> "AgentResult":
        return AgentResult(
            agent=raw.get("agent", ""),
            status=raw.get("status", "ok"),
            artifact=raw.get("artifact") or {},
            evidence=[Evidence.from_dict(e) for e in raw.get("evidence") or []],
            missing_slots=list(raw.get("missing_slots") or []),
            elapsed_ms=int(raw.get("elapsed_ms") or 0),
            llm_calls=int(raw.get("llm_calls") or 0),
            error=raw.get("error"),
        )


class BaseAgent(ABC):
    name: str = "agent"
    description: str = ""
    #: 每个 skill 的 examples 会被路由 Prompt 当 few-shot 消费，写真实用户说法
    skills: tuple[dict, ...] = ()

    @abstractmethod
    async def run(self, payload: dict) -> AgentResult:
        """纯业务入口：payload 为 dict（来自 DataPart 或进程内直调）。"""

    def card(self, url: str):
        """构造 A2A AgentCard（懒导入，避免 core 层硬依赖 SDK）。"""
        from a2a.types import AgentCapabilities, AgentCard, AgentSkill

        return AgentCard(
            name=self.name,
            description=self.description,
            url=url,
            version="1.0.0",
            capabilities=AgentCapabilities(streaming=False),
            default_input_modes=["application/json", "text/plain"],
            default_output_modes=["application/json", "text/plain"],
            skills=[
                AgentSkill(
                    id=skill["id"],
                    name=skill["name"],
                    description=skill.get("description", ""),
                    tags=list(skill.get("tags", ())),
                    examples=list(skill.get("examples", ())),
                )
                for skill in self.skills
            ],
        )
