"""模型连通性自检：换 Key / 换模型 / 换端点后先跑它。

    python -m scripts.check_llm
    python -m scripts.check_llm --embedding    # 额外测云端 embedding

逐项验证：Key → /models → 非流式 chat → JSON 模式 →（可选）embedding。
每项都打印耗时与 token 用量，失败时给出明确原因（鉴权 / 额度 / 参数）。
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import sys
import time

from app.core.errors import LexSynergyError
from app.generation.llm_client import (
    DashScopeClient,
    LLMAuthError,
    LLMError,
    LLMQuotaError,
    LLMResponseError,
)
from config import get_settings

OK = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def _mask(key: str) -> str:
    return f"{key[:8]}...{key[-4:]}" if len(key) > 14 else "（未配置或过短）"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts.check_llm")
    parser.add_argument("--embedding", action="store_true", help="额外测云端 embedding")
    args = parser.parse_args(argv)

    s = get_settings()
    print(f"httpx {md.version('httpx')}")
    print(f"base_url : {s.llm_base_url}")
    print(f"model    : {s.llm_model}（fast={s.fast_model}）")
    print(f"api_key  : {_mask(s.llm_api_key)}")
    print(f"thinking : {s.llm_enable_thinking}（False = 关闭思考模式）")
    print("-" * 62)

    if not s.has_llm:
        print(f"{FAIL} 未配置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL")
        return 2

    client = DashScopeClient(s)
    problems: list[str] = []

    # 1) /models
    try:
        import httpx

        t0 = time.time()
        r = httpx.get(
            f"{s.llm_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {s.llm_api_key}"},
            timeout=20,
        )
        if r.status_code == 200:
            ids = [m["id"] for m in r.json().get("data", [])]
            hit = s.llm_model in ids
            print(f"{OK} /models  {len(ids)} 个模型  {int((time.time()-t0)*1000)}ms  "
                  f"{'含' if hit else '不含'} {s.llm_model}")
            if not hit:
                near = [i for i in ids if s.llm_model.split("-")[0] in i][:5]
                print(f"       近似名：{near}")
                problems.append("目标模型不在 /models 列表里（可能是别名或未开通）")
        else:
            print(f"{FAIL} /models HTTP {r.status_code} {r.text[:120]}")
            problems.append("/models 不可达")
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} /models 异常 {type(exc).__name__}: {str(exc)[:120]}")
        problems.append("/models 异常")

    # 2) 非流式 chat
    try:
        t0 = time.time()
        res = client.chat(
            [{"role": "user", "content": "只回复两个字：可用"}], max_tokens=16
        )
        ms = int((time.time() - t0) * 1000)
        print(f"{OK} chat    {ms}ms  content={res.text.strip()[:12]!r}  "
              f"in={res.usage.prompt_tokens} out={res.usage.completion_tokens} "
              f"reasoning={res.usage.reasoning_tokens}")
        if res.usage.reasoning_tokens > 0:
            print(f"{WARN} 仍有 reasoning token（enable_thinking 可能没生效）")
            problems.append("思考模式未关闭")
    except LLMQuotaError as exc:
        print(f"{FAIL} chat    额度不足：{str(exc)[:160]}")
        problems.append("额度不足")
    except LLMAuthError as exc:
        print(f"{FAIL} chat    鉴权失败：{str(exc)[:160]}")
        problems.append("鉴权失败")
    except LLMError as exc:
        print(f"{FAIL} chat    {type(exc).__name__}: {str(exc)[:160]}")
        problems.append(f"chat 失败：{type(exc).__name__}")
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} chat    异常 {type(exc).__name__}: {str(exc)[:160]}")
        problems.append("chat 异常")

    # 3) JSON 模式（只验证"能拿到 dict"，不验证 schema —— 实测会改键名）
    try:
        data, usage = client.json_chat(
            [
                {"role": "system", "content": '只输出 JSON，字段名必须是 intent。'},
                {"role": "user", "content": "输出 {\"intent\": \"labor_law\"}"},
            ],
            max_tokens=64,
        )
        print(f"{OK} json    keys={list(data)}  tokens={usage.total}")
        if "intent" not in data:
            print(f"{WARN} JSON 键名被改写（实测存在）→ 调用方必须做字段校验")
    except LLMResponseError as exc:
        print(f"{WARN} json    JSON 解析失败（会降级到规则）：{str(exc)[:120]}")
    except Exception as exc:  # noqa: BLE001
        print(f"{WARN} json    未通过：{type(exc).__name__} {str(exc)[:100]}")

    # 4) embedding（可选）
    if args.embedding:
        try:
            import httpx

            t0 = time.time()
            r = httpx.post(
                f"{s.llm_base_url.rstrip('/')}/embeddings",
                headers={"Authorization": f"Bearer {s.llm_api_key}"},
                json={"model": s.embedding_dashscope_model, "input": ["试用期最长多久"]},
                timeout=30,
            )
            if r.status_code == 200:
                dim = len(r.json()["data"][0]["embedding"])
                print(f"{OK} embed   {dim} 维  {int((time.time()-t0)*1000)}ms  "
                      f"model={s.embedding_dashscope_model}")
            else:
                print(f"{WARN} embed   HTTP {r.status_code} {r.text[:120]}")
        except Exception as exc:  # noqa: BLE001
            print(f"{WARN} embed   异常 {type(exc).__name__}: {str(exc)[:100]}")

    print("-" * 62)
    if problems:
        print(f"{FAIL} 存在 {len(problems)} 个问题：")
        for item in problems:
            print(f"  - {item}")
        return 1
    print(f"{OK} 全部通过。当前 LLM 三件（Triage / 撰稿 / G4）可正常工作。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except LexSynergyError as exc:  # pragma: no cover
        print(f"{FAIL} {exc}")
        sys.exit(2)
