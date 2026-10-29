"""运行时装配：把 Agent / 检索 / 校验拼成一条可跑的流水线。

这里的语料是**演示语料**（10 条常用条文，用于把链路跑通与做种子缺陷评测）；
正式语料在 P0 从 LawAnswer 导入（3 部法律 + 司法解释，1,465 条）。
"""

from __future__ import annotations

import argparse
import asyncio
import time
import json
import sys
from datetime import date
from decimal import Decimal

from app.agents.base import BaseAgent
from app.agents.calc_agent import CalcAgent
from app.core.logger import get_logger
from app.agents.compliance import ComplianceAgent
from app.agents.statute import StatuteAgent
from app.agents.verifier_agent import VerifierAgent
from app.evidence.models import Evidence
from app.orchestrator.pipeline import Pipeline
from app.orchestrator.rules import triage as rule_triage
from app.retrieval.port import InMemoryStatuteSearch
from app.verifier.service import Verifier
from config import get_settings, reset_settings

logger = get_logger("runtime")

LCL = "中华人民共和国劳动合同法"
LABOR = "中华人民共和国劳动法"
ANNUAL = "职工带薪年休假条例"

#: 演示语料：条文原文（供 G2 逐字比对）
_RAW_CORPUS: list[tuple[str, str, str, str, int]] = [
    (
        LCL,
        "第十九条",
        "第二章 劳动合同的订立",
        "劳动合同期限三个月以上不满一年的，试用期不得超过一个月；劳动合同期限一年以上不满三年的，"
        "试用期不得超过二个月；三年以上固定期限和无固定期限的劳动合同，试用期不得超过六个月。"
        "同一用人单位与同一劳动者只能约定一次试用期。以完成一定工作任务为期限的劳动合同或者"
        "劳动合同期限不满三个月的，不得约定试用期。试用期包含在劳动合同期限内。",
        20,
    ),
    (
        LCL,
        "第四十条",
        "第四章 劳动合同的解除和终止",
        "有下列情形之一的，用人单位提前三十日以书面形式通知劳动者本人或者额外支付劳动者一个月工资后，"
        "可以解除劳动合同：（一）劳动者患病或者非因工负伤，在规定的医疗期满后不能从事原工作，"
        "也不能从事由用人单位另行安排的工作的；（二）劳动者不能胜任工作，经过培训或者调整工作岗位，"
        "仍不能胜任工作的；（三）劳动合同订立时所依据的客观情况发生重大变化，致使劳动合同无法履行，"
        "经用人单位与劳动者协商，未能就变更劳动合同内容达成协议的。",
        20,
    ),
    (
        LCL,
        "第四十六条",
        "第四章 劳动合同的解除和终止",
        "有下列情形之一的，用人单位应当向劳动者支付经济补偿：（一）劳动者依照本法第三十八条规定"
        "解除劳动合同的；（二）用人单位依照本法第三十六条规定向劳动者提出解除劳动合同并与劳动者"
        "协商一致解除劳动合同的；（三）用人单位依照本法第四十条规定解除劳动合同的；"
        "（四）用人单位依照本法第四十一条第一款规定解除劳动合同的。",
        20,
    ),
    (
        LCL,
        "第四十七条",
        "第四章 劳动合同的解除和终止",
        "经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。"
        "六个月以上不满一年的，按一年计算；不满六个月的，向劳动者支付半个月工资的经济补偿。"
        "劳动者月工资高于用人单位所在直辖市、设区的市级人民政府公布的本地区上年度职工月平均工资三倍的，"
        "向其支付经济补偿的标准按职工月平均工资三倍的数额支付，向其支付经济补偿的年限最高不超过十二年。"
        "本条所称月工资是指劳动者在劳动合同解除或者终止前十二个月的平均工资。",
        20,
    ),
    (
        LCL,
        "第八十二条",
        "第七章 法律责任",
        "用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，"
        "应当向劳动者每月支付二倍的工资。",
        20,
    ),
    (
        LCL,
        "第八十七条",
        "第七章 法律责任",
        "用人单位违反本法规定解除或者终止劳动合同的，应当依照本法第四十七条规定的经济补偿标准的"
        "二倍向劳动者支付赔偿金。",
        20,
    ),
    (
        LABOR,
        "第四十四条",
        "第四章 工作时间和休息休假",
        "有下列情形之一的，用人单位应当按照下列标准支付高于劳动者正常工作时间工资的工资报酬："
        "（一）安排劳动者延长工作时间的，支付不低于工资的百分之一百五十的工资报酬；"
        "（二）休息日安排劳动者工作又不能安排补休的，支付不低于工资的百分之二百的工资报酬；"
        "（三）法定休假日安排劳动者工作的，支付不低于工资的百分之三百的工资报酬。",
        10,
    ),
    (
        ANNUAL,
        "第五条",
        "第三章",
        "单位确因工作需要不能安排职工休年休假的，经职工本人同意，可以不安排职工休年休假。"
        "对职工应休未休的年休假天数，单位应当按照该职工日工资收入的300%支付年休假工资报酬。",
        5,
    ),
    (
        LCL,
        "第二十三条",
        "第二章 劳动合同的订立",
        "对负有保密义务的劳动者，用人单位可以在劳动合同或者保密协议中与劳动者约定竞业限制条款，"
        "并约定在解除或者终止劳动合同后，在竞业限制期限内按月给予劳动者经济补偿。",
        20,
    ),
    (
        LCL,
        "第三十九条",
        "第四章 劳动合同的解除和终止",
        "劳动者有下列情形之一的，用人单位可以解除劳动合同：（一）在试用期间被证明不符合录用条件的；"
        "（二）严重违反用人单位的规章制度的；（三）严重失职，营私舞弊，给用人单位造成重大损害的。",
        20,
    ),
]


def demo_corpus() -> list[Evidence]:
    return [
        Evidence(
            law=law,
            article_no=article,
            chapter_path=chapter,
            quote=text,
            chunk_id=f"{law}#{article}",
            score=0.0,
            priority=priority,
        )
        for law, article, chapter, text, priority in _RAW_CORPUS
    ]


def rule_based_auditor(text: str, context: str) -> list[dict]:
    """离线审查器：几条硬规则，用于跑通 ComplianceAgent 的契约。

    生产替换成 LLM 审计器（Prompt 要求逐项给 basis），接口不变。
    """
    issues: list[dict] = []
    basis = [line for line in context.splitlines() if line.startswith("【资料")]

    def _basis_of(article: str) -> list[str]:
        for line in basis:
            if article in line:
                return [f"{LCL}#{article}"]
        return []

    if "自愿放弃社保" in text or "放弃缴纳社会保险" in text:
        issues.append(
            {
                "clause": "自愿放弃社保",
                "verdict": "violation",
                "risk": "high",
                "reason": "社会保险为法定强制义务，劳动者放弃的约定无效",
                "basis": _basis_of("第十九条") or ["中华人民共和国社会保险法#第五十八条"],
                "fix": "删除该条款，依法为员工缴纳社会保险",
            }
        )
    if "试用期" in text:
        import re

        match = re.search(r"试用期[^\d]{0,6}(\d{1,2})\s*个?月", text)
        months = int(match.group(1)) if match else None
        if months and months > 2:
            issues.append(
                {
                    "clause": f"试用期 {months} 个月",
                    "verdict": "risk",
                    "risk": "high" if months > 6 else "medium",
                    "reason": "试用期长度须与合同期限匹配，最长不得超过六个月",
                    "basis": _basis_of("第十九条"),
                    "fix": "明确劳动合同期限，并按期限压缩试用期至法定上限以内",
                }
            )
    if "押金" in text or "保证金" in text:
        issues.append(
            {
                "clause": "收取押金/保证金",
                "verdict": "violation",
                "risk": "high",
                "reason": "用人单位不得要求劳动者提供担保或以其他名义收取财物",
                "basis": ["中华人民共和国劳动合同法#第九条"],
                "fix": "立即停止收取并退还已收款项",
            }
        )
    if "不缴社保" in text or "不缴纳社会保险" in text:
        issues.append(
            {
                "clause": "不缴纳社会保险",
                "verdict": "violation",
                "risk": "high",
                "reason": "缴纳社会保险是用人单位的法定义务",
                "basis": ["中华人民共和国社会保险法#第五十八条"],
                "fix": "自用工之日起三十日内办理社保登记",
            }
        )
    return issues


def build_search(settings) -> object:
    """三级降级：dense → bm25 → 10 条演示语料。

    这个链条保证：新克隆的仓库不跑 `build_corpus` 也能 `ask` 出结果；
    构建了语料但没建向量时退回 BM25；两者都在时用稠密（R@5 最高）。
    """
    from app.retrieval.corpus import Bm25StatuteSearch, Corpus

    backend = (settings.retrieval_backend or "dense").lower()
    corpus = None
    try:
        corpus = Corpus.load()
        if corpus.article_count < 1000:
            corpus = None
    except FileNotFoundError:
        logger.warning("未找到语料索引，回退演示语料（先跑 scripts.build_corpus）")

    if corpus is None:
        return InMemoryStatuteSearch(demo_corpus())

    if backend in {"dense", "hybrid"}:
        try:
            from app.retrieval.dense import DenseStatuteSearch, HybridStatuteSearch

            if backend == "hybrid":
                return HybridStatuteSearch(corpus)
            return DenseStatuteSearch(corpus)
        except FileNotFoundError:
            logger.warning("未找到 dense.npy，回退 BM25（先跑 scripts.build_embeddings）")
    return Bm25StatuteSearch(corpus)


def build_pipeline(*, use_llm: bool | None = None, verify: bool = True) -> Pipeline:
    """装配流水线。有 key 且未显式关闭时启用 LLM 三件（Triage / 撰稿 / G4）。

    三件都是**可切换**的：
    * Triage：LLM 失败逐字段回落规则；
    * 撰稿器：LLM 失败退回模板；
    * G4：judge 失败标为 skipped，不影响其余四道闸。
    任一处坏掉都不会让整条链路断。
    """
    settings = get_settings()
    llm_on = settings.has_llm if use_llm is None else (use_llm and settings.has_llm)

    triager = rule_triage
    synthesizer = None          # None → Pipeline 用模板撰稿器
    support = None
    if llm_on:
        from app.generation.synthesizer import LLMSynthesizer
        from app.orchestrator.triage_llm import LLMTriager
        from app.verifier.support_llm import LLMSupportChecker

        triager = LLMTriager(settings=settings)
        synthesizer = LLMSynthesizer(settings=settings)
        if settings.verify_support_check:
            support = LLMSupportChecker(settings=settings)
        logger.info("LLM 已启用：%s（G4=%s）", settings.llm_model, "on" if support else "off")

    if verify:
        verifier_agent: BaseAgent = VerifierAgent(
            Verifier(
                support_checker=support,
                tolerance=Decimal(str(settings.verify_numeric_tolerance)),
            )
        )
    else:
        from app.agents.null_verifier import NullVerifierAgent

        verifier_agent = NullVerifierAgent()
        logger.warning("校验器已关闭（消融模式）：本次答案**未经过五道闸**")

    agents = {
        "statute": StatuteAgent(build_search(settings), topk=settings.retrieval_topk),
        "calc": CalcAgent(),
        "compliance": ComplianceAgent(auditor=rule_based_auditor),
        "verifier": verifier_agent,
    }
    return Pipeline(
        agents, triager=triager, synthesizer=synthesizer, settings=settings
    )


# ----------------------------------------------------------------------
def _warmup_search(pipeline: Pipeline) -> None:
    """CLI 也因为同样的理由需要预热：bge-m3 冷启动 6~20s，
    这笔时间会落在检索步的 20s 超时里，让首个问题答成"检索超时/没资料"。
    API 侧在 lifespan 里预热，CLI 侧只能在这里同步做一次。
    """
    agent = getattr(pipeline, "_agents", {}).get("statute")
    if agent is None:
        return
    started = time.perf_counter()
    try:
        asyncio.run(agent.run({"sub_query": "试用期最长可以约定多久"}))
        logger.info("检索预热完成 %.1fs", time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 - 预热失败不影响后续请求
        logger.warning("检索预热失败（忽略）：%s", str(exc)[:120])


def _cmd_ask(
    query: str,
    facts: dict | None,
    as_of: str | None,
    backend: str | None = None,
    use_llm: bool | None = None,
    policy: str | None = None,
) -> int:
    if policy:
        import os

        os.environ["ROUTING_POLICY"] = policy
        reset_settings()
    if backend:
        import os

        os.environ["RETRIEVAL_BACKEND"] = backend
        reset_settings()
    pipeline = build_pipeline(use_llm=use_llm)
    _warmup_search(pipeline)
    answer = asyncio.run(
        pipeline.run(query, facts=facts, as_of=date.fromisoformat(as_of) if as_of else None)
    )
    print(answer.text)
    print("\n--- 执行摘要 ---")
    mode = answer.mode or "full"
    print(f"mode={mode} route={answer.route} loops={answer.loops} llm_calls={answer.llm_calls} "
          f"elapsed={answer.elapsed_ms}ms sources={len(answer.sources)} "
          f"backend={get_settings().retrieval_backend}")
    if answer.verdict:
        print(f"verify: passed={answer.verdict.get('passed')} action={answer.verdict.get('action')} "
              f"fails={[f.get('gate') for f in answer.verdict.get('fails') or []]}")
    for step in answer.steps:
        print(f"  - {step['step_id']:>3} {step['agent']:<10} {step['status']:<14} "
              f"{step['elapsed_ms']:>4}ms 证据={step['evidence']}")
    return 0


def _cmd_cards() -> int:
    settings = get_settings()
    pipeline = build_pipeline()
    for name, agent in pipeline._agents.items():  # noqa: SLF001 - 调试命令
        card = agent.card(settings.agent_url(name))
        print(json.dumps(card.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


def _cmd_agent(name: str, port: int) -> int:
    from app.protocol.server import serve

    pipeline = build_pipeline()
    agent = pipeline._agents.get(name)  # noqa: SLF001
    if agent is None:
        print(f"未知 agent：{name}，可选 {list(pipeline._agents)}", file=sys.stderr)
        return 2
    serve(agent, port=port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.runtime", description="LexSynergy 运行时")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="跑一个问题")
    ask.add_argument("query")
    ask.add_argument("--facts", default="", help="JSON 事实，例如 '{\"monthly_wage\":12000}'")
    ask.add_argument("--as-of", default=None, help="事发日期 YYYY-MM-DD（用于 G3 时效校验）")
    ask.add_argument(
        "--backend", default=None, choices=("dense", "hybrid", "bm25"), help="检索后端（默认 dense）"
    )
    ask.add_argument("--no-llm", action="store_true", help="强制用规则 Triage + 模板撰稿（不调模型）")
    ask.add_argument(
        "--policy", default=None, choices=("auto", "fast", "full"), help="路由策略（默认取配置）"
    )

    sub.add_parser("cards", help="打印所有 AgentCard")

    agent = sub.add_parser("agent", help="以 A2A 服务方式启动单个 Agent")
    agent.add_argument("name")
    agent.add_argument("--port", type=int, required=True)

    args = parser.parse_args(argv)
    if args.cmd == "ask":
        facts = json.loads(args.facts) if args.facts else None
        return _cmd_ask(
            args.query, facts, args.as_of, args.backend, False if args.no_llm else None, args.policy
        )
    if args.cmd == "cards":
        return _cmd_cards()
    return _cmd_agent(args.name, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
