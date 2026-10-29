"""Day 0 探路脚本：升级 A2A SDK 前先跑一遍，确认关键签名没变。

    python -m scripts.check_a2a_sdk

退出码 0 = 期望的 API 都在；非 0 = 需要按 ADR 0001 重新核对。
"""

from __future__ import annotations

import importlib.metadata as md
import inspect
import sys

EXPECTED_VERSION = "0.3.22"


def main() -> int:
    problems: list[str] = []

    version = md.version("a2a-sdk")
    print(f"a2a-sdk: {version}")
    if version != EXPECTED_VERSION:
        problems.append(f"版本变化：{version} != {EXPECTED_VERSION}（先看 ADR 0001）")

    try:
        from a2a.server.agent_execution import AgentExecutor, RequestContext
        from a2a.server.apps import A2AStarletteApplication
        from a2a.server.events import EventQueue
        from a2a.server.request_handlers import DefaultRequestHandler
        from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
        from a2a.types import AgentCapabilities, AgentCard, DataPart, Part, TaskState
        from a2a.utils import get_data_parts, get_message_text, new_agent_text_message
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 导入失败：{exc}")
        return 2

    print(f"AgentExecutor.execute: {inspect.signature(AgentExecutor.execute)}")
    print(f"TaskUpdater.__init__:  {inspect.signature(TaskUpdater.__init__)}")
    print(f"TaskState 取值:        {[s.value for s in TaskState]}")
    print(f"AgentCard 必填:        {[k for k, v in AgentCard.model_fields.items() if v.is_required()]}")
    print(f"Part 字段:             {list(Part.model_fields)}")
    print(f"DataPart 字段:         {list(DataPart.model_fields)}")

    # 关键假设：execute 必须是 async（否则回退到"自研 async 层"）
    if not inspect.iscoroutinefunction(AgentExecutor.execute):
        problems.append("AgentExecutor.execute 不再是 async —— 整个异步方案需要重估")

    # 关键假设：DataPart 能承载 dict（结构化产物契约依赖它）
    part = Part(root=DataPart(data={"k": 1, "nested": {"x": [1, 2]}}))
    if not isinstance(part.root, DataPart) or part.root.data != {"k": 1, "nested": {"x": [1, 2]}}:
        problems.append("DataPart 无法原样承载嵌套 dict")

    # 关键假设：TaskState 可用字符串值构造
    try:
        assert TaskState("input-required") == TaskState.input_required
    except Exception:  # noqa: BLE001
        problems.append('TaskState("input-required") 构造方式失效')

    print(f"A2ACardResolver 默认路径: /.well-known/agent-card.json（见 A2ACardResolver.__init__）")

    if problems:
        print("\n".join(f"[WARN] {p}" for p in problems))
        return 1
    print("\n[OK] 关键 API 与 ADR 0001 一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
