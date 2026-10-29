"""阶段 9：支撑率评测（独立 judge）。零网络：判官换成假客户端。

一直盯四件事：
1. Prompt 必须**确定性且带版本号** —— 否则同一批答案的漂移无从归因；
2. temperature 必须钉死 0，判官模型必须是与生成模型**不同族**的那个；
3. stray（判官列回纯引文/过渡句）不能算成答案的错，但也不能让它把分拉高；
4. **判官全挂时 0.0000 漂移不是通过** —— 覆盖率不够要判"无法判定"。
"""

from __future__ import annotations

import json

import pytest

from eval import faithfulness as F


class _Usage:
    completion_tokens = 7


class _Result:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage = _Usage()


class FakeJudge:
    """按题循环返回预设 JSON；同时记录调用参数。"""

    def __init__(self, payloads: list[dict] | None = None, *, raises: Exception | None = None):
        self.payloads = payloads or []
        self.raises = raises
        self.calls: list[dict] = []
        self._i = 0

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self.raises is not None:
            raise self.raises
        data = self.payloads[self._i % len(self.payloads)] if self.payloads else {"unsupported": []}
        self._i += 1
        return _Result(json.dumps(data, ensure_ascii=False))


def _case(config: str, sid: str, text: str, sources: list[dict] | None = None) -> dict:
    return {
        "config": config,
        "sample_id": sid,
        "question": "被辞退能拿多少补偿？",
        "text": text,
        "sources": sources
        or [
            {
                "law": "中华人民共和国劳动合同法",
                "article_no": "第四十七条",
                "quote": "经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。",
            }
        ],
    }


ANSWER = (
    "你被违法辞退，可以主张赔偿金。"
    "用人单位违反本法规定解除劳动合同的，应当按经济补偿标准的二倍支付赔偿金。"
    "金额以你提供的事实为准。"
    "建议尽快申请劳动仲裁，避免错过时效。"
)


# ---------------------------------------------------------------- 1. Prompt
def test_prompt_is_deterministic_and_versioned():
    a = F.judge_messages("问题", ANSWER, _case("A", "q1", ANSWER)["sources"])
    b = F.judge_messages("问题", ANSWER, _case("A", "q1", ANSWER)["sources"])
    assert a == b, "同一输入必须产生逐字相同的 messages"
    assert F.PROMPT_VERSION in a[0]["content"]
    body = a[1]["content"]
    assert "【用户问题】" in body and "第四十七条" in body and "【回答】" in body


# ---------------------------------------------------------------- 2. 调用参数
@pytest.mark.asyncio
async def test_judge_pins_temperature_model_and_thinking():
    judge = FakeJudge([{"unsupported": []}])
    rows, failures, tokens = await F.judge_all(
        [_case("A_baseline", "q1", ANSWER)], judge, model="qwen3.8-27b", concurrency=2
    )
    assert failures == 0 and tokens == 7
    call = judge.calls[0]
    assert call["temperature"] == 0.0
    assert call["enable_thinking"] is False
    assert call["json_mode"] is True
    assert call["model"] == "qwen3.8-27b"
    assert rows[0]["score"] == 1.0


# ---------------------------------------------------------------- 3. 评分
def test_support_score_matches_truncated_claim_and_drops_stray():
    claims = ["经济补偿按工作年限计算", "你应当支付赔偿金", "可以申请劳动仲裁"]
    # 判官截短了第一条、把纯引文列了回来 → 只能计 1 条命中，stray 1 条
    score, stray = F.support_score(claims, ["经济补偿按工作年限", "“经济补偿按…支付”"])
    assert stray == 1
    assert score == round(1 - 1 / 3, 4)


def test_support_score_without_claims_is_nan():
    score, stray = F.support_score([], ["任何东西"])
    assert score != score and stray == 0


def test_parse_unsupported_tolerates_noise_and_bad_shapes():
    assert F.parse_unsupported('```json\n{"unsupported": ["A"]}\n```') == ["A"]
    assert F.parse_unsupported('前缀 {"unsupported": ["B"]} 后缀') == ["B"]
    assert F.parse_unsupported('{"unsupported": "不是列表"}') == []
    assert F.parse_unsupported("完全不是 JSON") == []
    assert F.parse_unsupported('{"other": []}') == []


# ---------------------------------------------------------------- 4. 漂移
def test_report_three_states():
    _, status = F.report(
        {"A": [0.90, 0.905, 0.902]}, {"judge_model": "m", "repeats": 3, "cases": 3, "failures": 0}
    )
    assert status == "ok"

    _, status = F.report(
        {"A": [0.90, 0.905, 0.93]}, {"judge_model": "m", "repeats": 3, "cases": 3, "failures": 0}
    )
    assert status == "fail"

    # 全挂：每次都是 NaN，漂移算出来是 0.0000，但绝不能判通过
    nan = float("nan")
    _, status = F.report(
        {"A": [nan, nan, nan]}, {"judge_model": "m", "repeats": 3, "cases": 3, "failures": 3}
    )
    assert status == "insufficient"


@pytest.mark.asyncio
async def test_judge_failure_marks_nan_not_zero():
    judge = FakeJudge(raises=RuntimeError("quota exhausted"))
    rows, failures, _ = await F.judge_all(
        [_case("A_baseline", "q1", ANSWER)], judge, model="m", concurrency=1
    )
    assert failures == 1
    assert rows[0]["score"] != rows[0]["score"]
    assert rows[0]["error"] == "RuntimeError"


@pytest.mark.asyncio
async def test_answer_without_claims_is_not_called():
    judge = FakeJudge([{"unsupported": []}])
    rows, failures, _ = await F.judge_all(
        [_case("A_baseline", "q1", "【结论】短。")], judge, model="m", concurrency=1
    )
    assert judge.calls == [], "没有断言就不该浪费一次调用"
    assert rows[0]["error"] == "no_claims" and failures == 1
