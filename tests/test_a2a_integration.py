"""A2A 集成用例：需要真实启动的服务，默认跳过。

    # 终端 A：python -m app.runtime agent calc --port 5102
    # 终端 B：
    A2A_CALC_URL=http://127.0.0.1:5102 pytest -m integration
"""

from __future__ import annotations

import os

import pytest

from app.protocol.invoker import A2AHttpInvoker

pytestmark = pytest.mark.integration

URL = os.getenv("A2A_CALC_URL", "")


@pytest.mark.asyncio
@pytest.mark.skipif(not URL, reason="未设置 A2A_CALC_URL，跳过真实 A2A 集成测试")
async def test_real_a2a_roundtrip():
    invoker = A2AHttpInvoker({"calc": URL}, timeout_s=20)
    result = await invoker.call(
        "calc",
        {
            "kinds": ["severance"],
            "facts": {"monthly_wage": 12000, "start_date": "2021-03-01", "end_date": "2026-10-20"},
        },
    )
    assert result.ok
    assert result.artifact["results"][0]["value"] == "72000.00"


@pytest.mark.asyncio
@pytest.mark.skipif(not URL, reason="未设置 A2A_CALC_URL，跳过真实 A2A 集成测试")
async def test_real_a2a_reports_missing_slots():
    invoker = A2AHttpInvoker({"calc": URL}, timeout_s=20)
    result = await invoker.call("calc", {"kinds": ["severance"], "facts": {}})
    assert result.status == "input_required"
    assert set(result.missing_slots) == {"monthly_wage", "start_date", "end_date"}
