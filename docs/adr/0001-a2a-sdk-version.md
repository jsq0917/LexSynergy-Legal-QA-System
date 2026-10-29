# ADR 0001 · A2A SDK 版本选型

- 状态：已接受（2026-10-28）
- 决策：锁定 `a2a-sdk[http-server]==0.3.22`，业务逻辑写在 `AgentExecutor` 之外
- 影响面：`app/protocol/*`、`app/agents/base.py`、`pyproject.toml`

## 背景

参考架构（SmartVoyage）用 `python-a2a 0.5.4`，遇到三个结构性代价：
1. `A2AServer.handle_task` 是**同步**方法（Flask 底座），而编排层是 async FastAPI，只能用
   `run_in_executor` + `asyncio.run` 绕事件循环冲突；
2. 每请求新建 event loop → MCP 连接无法复用（每次多 200–800ms）；
3. 协议层不支持结构化产物，导致 artifacts 里 String 与 dict 混用。

初版设计因此打算"自研 200 行 async 兼容层"。Day 0 实测后推翻了这个结论。

## 实测结果

| 版本 | 核心差异 | 结论 |
|---|---|---|
| `a2a-sdk==1.3.0` | 类型层改为 **protobuf**（`Part` 是 `a2a_pb2.Part`）；服务端重构为 `a2a.server.routes`（`fastapi_routes` / `jsonrpc_routes` / `rest_routes`），**已无 `A2AStarletteApplication`** | 过新，公开文档与示例仍以 0.3.x 为主，不适合 22 天工期的项目 |
| `a2a-sdk==0.3.22` | pydantic 类型；`A2AStarletteApplication` + `DefaultRequestHandler` + `InMemoryTaskStore` + `TaskUpdater`；`AgentExecutor.execute` 本身是 **async** | **采用** |

关键事实（已用 `.venv` 实跑确认）：

```python
from a2a.server.agent_execution import AgentExecutor
class AgentExecutor:
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None: ...
```

也就是说，"同步框架导致事件循环冲突"只对 `python-a2a` 成立，官方 SDK 从根上没有这个问题。
自研 200 行 = 重复实现一个还在演进的协议，收益为负。

其它实测细节（写代码时踩过）：

- Agent Card 路径：`/.well-known/agent-card.json`（`A2ACardResolver` 默认值），不是早期文档里的 `agent.json`；
- `get_data_parts()` 返回的是 **dict 列表**（已解包），不是 `Part` 对象；
- `TaskUpdater` 构造需 `(event_queue, task_id, context_id)`，方法有 `start_work` / `add_artifact` / `update_status` / `complete` / `failed`；
- `TaskState` 用值构造最稳：`TaskState("input-required")`；
- `AgentCard` 必填字段：`name / description / url / version / capabilities / skills / default_input_modes / default_output_modes`。

## 决策

用官方 SDK，但把**业务逻辑放在 executor 之外**：

```python
# 业务：纯 async，可单测，不依赖协议
class StatuteAgent(BaseAgent):
    async def run(self, payload: dict) -> AgentResult: ...

# 协议：约 40 行通用适配器，四个 Agent 复用同一个
class A2AExecutor(AgentExecutor):
    async def execute(self, ctx, queue):
        updater = TaskUpdater(queue, ctx.task_id, ctx.context_id)
        await updater.start_work()
        result = await self._agent.run(payload_of(ctx))
        await updater.add_artifact(parts=[Part(root=DataPart(data=result.to_dict()))])
        ...
```

这带来三个直接收益：

1. **单测绕过协议**：`await StatuteAgent(...).run(payload)`，不启容器、不造 `RequestContext`；
2. **结构化产物有协议支持**：`DataPart` 承载 `AgentResult`（带 `schema: agent-result/1`），
   彻底消灭"artifacts 有时是 str 有时是 dict"；
3. **协议可替换**：换 MCP / REST / gRPC 只改 `app/protocol/`。

## 回退方案

若 0.3.22 后续出现阻塞性 bug：回退到自研兼容层，成本约 200 行，`app/agents/*` 与
`app/orchestrator/*` 零改动（因为它们只依赖 `BaseAgent.run()`）。

## 复现

```bash
python -m scripts.check_a2a_sdk
```
该脚本会打印版本、关键类签名、Card 路径与 `TaskState` 取值，用于升级前快速回归。
