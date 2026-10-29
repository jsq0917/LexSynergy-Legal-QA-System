"""恒通过的校验器 —— **仅用于消融实验**，生产链路绝不装配它。

存在的理由：要回答"多 Agent 的收益来自编排还是来自校验"，就必须有一档
"只加编排、不加校验"的对照。缺了它，只能证明"多 Agent + 校验"整体更好，
说不清是哪一半起了作用 —— 那在简历上站不住脚。

它在 artifact 里显式写 `ablation: true`，任何下游看到这个标记都该知道
这条答案没有经过五道闸。
"""

from __future__ import annotations

from app.agents.base import AgentResult, BaseAgent


class NullVerifierAgent(BaseAgent):
    name = "verifier"
    description = "（消融用）恒通过的校验器：测量「只加编排、不加校验」这一档的效果"

    async def run(self, payload: dict) -> AgentResult:
        return AgentResult.success(
            self.name,
            artifact={
                "passed": True,
                "action": "pass",
                "checked": 0,
                "fails": [],
                "missing_hints": [],
                "skipped_gates": ["G1", "G2", "G3", "G4", "G5"],
                "ablation": True,
            },
        )
