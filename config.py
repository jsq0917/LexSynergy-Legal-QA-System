"""全局配置：唯一配置入口，全部可由环境变量覆盖。

刻意不引入 pydantic-settings：core 层保持零外部依赖，单测可在裸环境下跑。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """极简 .env 加载：**不覆盖**已存在的环境变量。

    优先级：进程环境变量 > .env > 代码默认值。
    这样 CI/容器里用 `export LLM_API_KEY=...` 就能压过本地文件，
    也不会因为误提交 .env 而把生产 key 覆盖掉。仓库里 .env 已被 .gitignore 排除。
    """
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Budget:
    """单次请求的硬预算。三道闸：步数 / 回路 / 全局 deadline。"""

    max_steps: int = 8
    max_loop: int = 1
    deadline_s: float = 25.0
    step_timeout_s: float = 20.0


@dataclass(frozen=True)
class Settings:
    # --- 服务 ---
    api_port: int = 8201
    agent_transport: str = "inproc"  # inproc | a2a

    # --- Agent 端口（A2A）---
    statute_port: int = 5101
    calc_port: int = 5102
    compliance_port: int = 5103
    verifier_port: int = 5104

    # --- 编排 ---
    #: auto=按风险升级（默认）| fast=强制快速通道 | full=强制全编排
    routing_policy: str = "auto"
    #: **请求级**硬超时（秒）。scheduler 的 deadline 只罩住 DAG，
    #: 撰稿 / G4 / 回路都在它之外，所以必须另设一道总闸。
    request_deadline_s: float = 45.0

    # --- 检索 ---
    #: dense（bge-m3 稠密，R@5 最高）| hybrid（稠密+BM25 加权 RRF，R@8 最高）| bm25（零依赖基线）
    retrieval_backend: str = "dense"
    retrieval_topk: int = 5
    retrieval_pool_factor: int = 4
    evidence_budget_chars: int = 6000

    # --- LLM（阿里云百炼 DashScope，OpenAI 兼容模式）---
    llm_provider: str = "dashscope"
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "deepseek-v4.1-flash"          # 文本生成统一用它
    llm_model_fast: str = ""                        # 留空 = 复用 llm_model
    llm_api_key: str = ""
    llm_timeout_s: float = 35.0                     # 单次调用上限：一次问答不该等 60s
    llm_max_retries: int = 2
    #: 超时是否重试。默认关闭——重试把 35s 变成 70s，用户早走了；限流/5xx 仍然重试
    llm_retry_on_timeout: bool = False
    #: deepseek-v4.1-flash 默认走思考模式：不关掉的话小 max_tokens 会被 reasoning 吃光、
    #: content 返回空。实测 enable_thinking=false 最省（2 tokens 出结果）。
    llm_enable_thinking: bool = False
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1600

    # --- Embedding ---
    #: local=bge-m3（已建索引）| dashscope=qwen3.7-text-embedding-flash
    embedding_backend: str = "local"
    embedding_model_path: str = ""
    embedding_dashscope_model: str = "qwen3.7-text-embedding-flash"

    # --- 校验 ---
    verify_support_check: bool = True  # 关闭则只跑 4 道确定性闸
    verify_numeric_tolerance: float = 0.01

    budget: Budget = field(default_factory=Budget)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_port=_env_int("API_PORT", 8201),
            agent_transport=_env("AGENT_TRANSPORT", "inproc"),
            statute_port=_env_int("STATUTE_PORT", 5101),
            calc_port=_env_int("CALC_PORT", 5102),
            compliance_port=_env_int("COMPLIANCE_PORT", 5103),
            verifier_port=_env_int("VERIFIER_PORT", 5104),
            routing_policy=_env("ROUTING_POLICY", "auto"),
            request_deadline_s=_env_float("REQUEST_DEADLINE_S", 45.0),
            retrieval_backend=_env("RETRIEVAL_BACKEND", "dense"),
            retrieval_topk=_env_int("RETRIEVAL_TOPK", 5),
            retrieval_pool_factor=_env_int("RETRIEVAL_POOL_FACTOR", 4),
            evidence_budget_chars=_env_int("EVIDENCE_BUDGET_CHARS", 6000),
            llm_provider=_env("LLM_PROVIDER", "dashscope"),
            llm_model=_env("LLM_MODEL", "deepseek-v4.1-flash"),
            llm_model_fast=_env("LLM_MODEL_FAST", ""),
            llm_base_url=_env(
                "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            llm_api_key=_env("LLM_API_KEY", ""),
            llm_timeout_s=_env_float("LLM_TIMEOUT_S", 35.0),
            llm_max_retries=_env_int("LLM_MAX_RETRIES", 2),
            llm_retry_on_timeout=_env_bool("LLM_RETRY_ON_TIMEOUT", False),
            llm_enable_thinking=_env_bool("LLM_ENABLE_THINKING", False),
            llm_temperature=_env_float("LLM_TEMPERATURE", 0.1),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", 1600),
            embedding_backend=_env("EMBEDDING_BACKEND", "local"),
            embedding_model_path=_env("EMBEDDING_MODEL_PATH", ""),
            embedding_dashscope_model=_env(
                "EMBEDDING_DASHSCOPE_MODEL", "qwen3.7-text-embedding-flash"
            ),
            verify_support_check=_env_bool("VERIFY_SUPPORT_CHECK", True),
            verify_numeric_tolerance=_env_float("VERIFY_NUMERIC_TOLERANCE", 0.01),
            budget=Budget(
                max_steps=_env_int("MAX_STEPS", 8),
                max_loop=_env_int("MAX_LOOP", 1),
                deadline_s=_env_float("DEADLINE_S", 25.0),
                step_timeout_s=_env_float("STEP_TIMEOUT_S", 20.0),
            ),
        )

    @property
    def has_llm(self) -> bool:
        return bool(self.llm_api_key and self.llm_base_url and self.llm_model)

    @property
    def fast_model(self) -> str:
        return self.llm_model_fast or self.llm_model

    def agent_url(self, name: str) -> str:
        port = {
            "statute": self.statute_port,
            "calc": self.calc_port,
            "compliance": self.compliance_port,
            "verifier": self.verifier_port,
        }[name]
        return f"http://127.0.0.1:{port}"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reset_settings() -> None:
    """测试用：清掉单例。"""
    global _settings
    _settings = None
