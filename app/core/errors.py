"""异常与错误码：四层防线（工具 / Agent / 协议 / 编排）共用一套错误码。"""

from __future__ import annotations


class LexSynergyError(Exception):
    code = "internal_error"
    retriable = False

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message or self.__class__.__name__)
        self.message = message or self.__class__.__name__
        if code:
            self.code = code


class MissingSlots(LexSynergyError):
    """业务上缺必填槽位：交由编排层转成追问，**不是**错误。"""

    code = "input_required"
    retriable = False

    def __init__(self, slots: list[str], message: str = "") -> None:
        self.slots = list(slots)
        super().__init__(message or f"缺少必要信息：{'、'.join(self.slots)}")


class NoData(LexSynergyError):
    code = "no_data"


class AgentFailed(LexSynergyError):
    code = "agent_failed"
    retriable = True


class StepTimeout(AgentFailed):
    code = "step_timeout"
    retriable = True


class PlanError(LexSynergyError):
    code = "plan_error"
