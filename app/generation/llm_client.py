"""阿里云百炼（DashScope）OpenAI 兼容客户端。

选型与参数都是**实测**出来的，不是照文档抄的：

| 事实 | 实测结果 | 处理 |
|---|---|---|
| 端点 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `/chat/completions`、`/embeddings`、`/models` 均可用 |
| `deepseek-v4.1-flash` | 存在且可用（183 个模型中） | 文本生成统一用它 |
| **思考模式** | 默认开启：`max_tokens=16` 时 16 个 token **全进 reasoning，content 为空**（finish_reason=length） | 每个请求显式带 `enable_thinking: false`；关掉后同样问题只花 **2 个 token** |
| JSON 模式 | `response_format={"type":"json_object"}` 可用，但实测出现过键名被改写（`intent` → `int#`） | **必须做 schema 校验 + 失败回落**，不能假设 JSON 一定合法 |
| Embedding | `qwen3.7-text-embedding` 免费额度已耗尽（400）；`qwen3.7-text-embedding-flash` 可用、1024 维 | 云端 embedding 用 flash 版；本项目默认仍用本地 bge-m3 |
| Rerank | `/reranks` 返回 404（兼容模式未暴露该路径） | 重排继续用本地 bge-reranker |

错误分类驱动重试策略：401/403 不重试（配置问题），429/5xx/超时可重试（指数退避）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from app.core.logger import get_logger
from config import Settings, get_settings

logger = get_logger("llm")


# ---------------------------------------------------------------- 错误
class LLMError(RuntimeError):
    retriable = False


class LLMAuthError(LLMError):
    """401/403：key 无效、无权访问该模型、workspace 不匹配。重试无意义。"""


class LLMQuotaError(LLMError):
    """额度耗尽（百炼返回 400 + 'Free quota exhausted'）。需要充值或换模型。"""


class LLMBadRequest(LLMError):
    """400：参数非法（模型名错、消息格式错、max_tokens 越界）。"""


class LLMRateLimit(LLMError):
    retriable = True


class LLMServerError(LLMError):
    retriable = True


class LLMTimeout(LLMError):
    retriable = True


class LLMResponseError(LLMError):
    """返回结构不符合预期（空 content、JSON 解析失败）。可重试一次。"""

    retriable = True


# ---------------------------------------------------------------- 数据结构
@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def merge(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens


@dataclass
class ChatResult:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""
    raw: dict = field(default_factory=dict)


# ---------------------------------------------------------------- 客户端
class DashScopeClient:
    """无状态客户端：每次请求新建 httpx 连接，避免跨线程复用问题。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self._s = settings or get_settings()

    # ---------------- 内部 ----------------
    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._s.llm_api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self,
        messages: list[dict],
        *,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
        enable_thinking: bool | None,
    ) -> dict:
        thinking = self._s.llm_enable_thinking if enable_thinking is None else enable_thinking
        payload: dict[str, Any] = {
            "model": model or self._s.llm_model,
            "messages": messages,
            "temperature": self._s.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self._s.llm_max_tokens,
            # 关键：不显式关闭，deepseek-v4.1-flash 会把 max_tokens 全花在 reasoning 上
            "enable_thinking": thinking,
        }
        # 注意：stream_options 只能和 stream=true 一起发，非流式带上会被 400 拒绝
        # （{"'stream' and 'stream_options' must be set together"}）
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _raise_for(self, status: int, body: str) -> None:
        text = (body or "")[:300]
        if status in (401, 403):
            raise LLMAuthError(f"鉴权失败（{status}）：{text}")
        if status == 429:
            raise LLMRateLimit(f"限流（429）：{text}")
        if status >= 500:
            raise LLMServerError(f"上游错误（{status}）：{text}")
        if "quota" in text.lower() or "额度" in text:
            raise LLMQuotaError(f"额度不足：{text}")
        raise LLMBadRequest(f"请求被拒（{status}）：{text}")

    @staticmethod
    def _parse_usage(data: dict) -> Usage:
        u = data.get("usage") or {}
        details = u.get("completion_tokens_details") or {}
        return Usage(
            prompt_tokens=int(u.get("prompt_tokens") or 0),
            completion_tokens=int(u.get("completion_tokens") or 0),
            reasoning_tokens=int(details.get("reasoning_tokens") or 0),
        )

    # ---------------- 非流式 ----------------
    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        enable_thinking: bool | None = None,
    ) -> ChatResult:
        import httpx

        if not self._s.has_llm:
            raise LLMAuthError("未配置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL")

        payload = self._payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            enable_thinking=enable_thinking,
        )
        url = f"{self._s.llm_base_url.rstrip('/')}/chat/completions"
        last_error: Exception | None = None

        for attempt in range(self._s.llm_max_retries + 1):
            started = time.perf_counter()
            try:
                with httpx.Client(timeout=self._s.llm_timeout_s) as client:
                    resp = client.post(url, headers=self._headers, json=payload)
                if resp.status_code != 200:
                    self._raise_for(resp.status_code, resp.text)
                data = resp.json()
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                text = message.get("content") or ""
                usage = self._parse_usage(data)
                if not text.strip():
                    raise LLMResponseError(
                        f"空返回 finish={choice.get('finish_reason')} "
                        f"reasoning_tokens={usage.reasoning_tokens}"
                    )
                logger.info(
                    "llm ok model=%s %dms in=%d out=%d reasoning=%d",
                    data.get("model", payload["model"]),
                    int((time.perf_counter() - started) * 1000),
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.reasoning_tokens,
                )
                return ChatResult(
                    text=text,
                    model=data.get("model", payload["model"]),
                    usage=usage,
                    finish_reason=choice.get("finish_reason", ""),
                    raw=data,
                )
            except (LLMAuthError, LLMQuotaError, LLMBadRequest):
                raise
            except Exception as exc:  # noqa: BLE001 - 网络/限流/结构异常统一走重试判定
                last_error = _normalize_exc(exc)
                # **超时不重试**：一次 60s 超时再重试，等于把 60s 变成 120s，
                # 而用户早就走了。实测那一次 71.5s 的尾延迟就是这么来的。
                # 超时该做的是尽快降级（模板 / 部分结果），不是再赌一次。
                if isinstance(last_error, LLMTimeout) and not self._s.llm_retry_on_timeout:
                    logger.warning("llm 超时，不重试，直接降级：%s", str(last_error)[:80])
                    raise last_error
                if attempt >= self._s.llm_max_retries:
                    break
                backoff = 0.8 * (2**attempt)
                logger.warning(
                    "llm 第 %d 次失败（%s），%.1fs 后重试",
                    attempt + 1,
                    type(last_error).__name__,
                    backoff,
                )
                time.sleep(backoff)
        raise last_error or LLMError("未知错误")

    # ---------------- 流式 ----------------
    def chat_stream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[tuple[str, Usage | None]]:
        """逐段产出 (文本增量, 最终 usage)。"""

        import httpx

        if not self._s.has_llm:
            raise LLMAuthError("未配置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL")

        payload = self._payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=False,
            enable_thinking=None,
        )
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        url = f"{self._s.llm_base_url.rstrip('/')}/chat/completions"

        with httpx.Client(timeout=self._s.llm_timeout_s) as client:
            with client.stream("POST", url, headers=self._headers, json=payload) as resp:
                if resp.status_code != 200:
                    body = "".join(resp.iter_text())
                    self._raise_for(resp.status_code, body)
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    usage = self._parse_usage(data) if data.get("usage") else None
                    for choice in data.get("choices") or []:
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            yield delta, None
                    if usage and usage.total:
                        yield "", usage

    # ---------------- JSON ----------------
    def json_chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> tuple[dict, Usage]:
        """JSON 模式 + 清洗 + 解析。**不保证 schema 正确**，调用方必须自己校验字段。"""
        result = self.chat(
            messages, model=model, max_tokens=max_tokens, json_mode=True, enable_thinking=False
        )
        text = _strip_fences(result.text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"JSON 解析失败：{exc}；原文={text[:160]}") from exc
        if not isinstance(data, dict):
            raise LLMResponseError(f"期望 JSON 对象，实际 {type(data).__name__}")
        return data, result.usage


def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        if t.endswith("```"):
            t = t[:-3]
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        return t[start : end + 1]
    return t.strip()


def _normalize_exc(exc: Exception) -> LLMError:
    name = type(exc).__name__
    if "Timeout" in name:
        return LLMTimeout(f"超时：{exc}")
    return LLMError(f"{name}: {exc}")


_client: DashScopeClient | None = None


def get_llm_client(settings: Settings | None = None) -> DashScopeClient:
    global _client
    if _client is None:
        _client = DashScopeClient(settings)
    return _client
