"""金额测算专家：**不含 LLM**，只做确定性计算。

法律场景里"算错赔偿金"是事故而不是"偶尔不准"，所以这一层从设计上就不允许
语言模型参与数字。槽位缺失时返回 ``input_required``，由编排层转成追问。
"""

from __future__ import annotations

import time

from app.agents.base import AgentResult, BaseAgent
from app.calc.engine import compute_many
from app.calc.models import Facts
from app.core.errors import MissingSlots
from app.core.logger import get_logger

logger = get_logger("agent.calc")


class CalcAgent(BaseAgent):
    name = "calc"
    description = "金额测算专家：经济补偿、违法解除赔偿金、代通知金、加班费、二倍工资、年假折算、竞业限制、试用期上限。"
    skills = (
        {
            "id": "calc_compensation",
            "name": "劳动报酬与补偿测算",
            "description": "输入结构化事实（月工资、起止日期、事由），返回公式、代入过程与结果",
            "tags": ["legal", "calculation", "deterministic"],
            "examples": [
                "月薪 12000，2021-03-01 入职，2026-10-20 被裁员，经济补偿多少",
                "月薪 8000，每天加班 2 小时，加班费怎么算",
            ],
        },
    )

    async def run(self, payload: dict) -> AgentResult:
        started = time.perf_counter()
        kinds = list(payload.get("kinds") or [])
        if not kinds:
            return AgentResult.input_required(self.name, ["kinds"])

        facts = Facts.from_dict(payload.get("facts") or {})
        try:
            results = compute_many(kinds, facts)
        except MissingSlots as exc:
            logger.info("缺槽位 %s（kinds=%s）", exc.slots, kinds)
            return AgentResult.input_required(self.name, exc.slots)
        except KeyError as exc:
            return AgentResult.failed(self.name, str(exc))

        return AgentResult.success(
            self.name,
            artifact={"results": [r.to_dict() for r in results], "facts": payload.get("facts") or {}},
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            llm_calls=0,
        )
