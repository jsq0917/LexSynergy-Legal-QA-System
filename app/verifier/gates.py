"""五道闸的纯函数实现：除 G4 外全部零 LLM、零 IO、可单测。

G1 条号存在性   —— 编造引用 / 断链引用
G2 引文逐字一致 —— 篡改引文（允许用"……"显式省略，但每段必须原样出现）
G3 as_of 时效   —— 引用已废止/尚未生效的版本
G4 语义支撑     —— 引对了条文却支撑不了结论（Mode 2，最危险的一类）
G5 数字一致     —— 金额与测算结果不符
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from app.verifier.models import Citation

CN_NUM = "一二三四五六七八九十百千零〇两"

#: 法名与条号之间允许夹章节路径（`《法》·第二章·第四十七条`），但不能跨句跨引号，
#: 否则会把后文的条号误挂到前文的法名上。
CITATION_RE = re.compile(rf"《([^》]{{2,40}})》(?:[^《》\n。；]{{0,24}}?)第([{CN_NUM}]{{1,8}})条")
LOOSE_ARTICLE_RE = re.compile(rf"第([{CN_NUM}]{{1,8}})条")

AMOUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(万元|元)")

#: 归一化时抹掉的字符（空白 + 中英标点 + 引号括号）。只抹标点不动文字，
#: 因此"逐字比对"仍然是逐字，只是不让排版差异造成误报。
_STRIP_CHARS = set(
    " \t\r\n\u3000"
    "，。；：、,.!?！？;:\"'“”‘’（）()《》【】[]<>…—-－·~`|"
)
_ELLIPSIS = ("……", "...", "…")


def normalize(text: str) -> str:
    return "".join(ch for ch in (text or "") if ch not in _STRIP_CHARS)


def normalize_amount_unit(unit: str) -> Decimal:
    return Decimal("10000") if unit == "万元" else Decimal("1")


def extract_citations(text: str) -> list[Citation]:
    """抽取《法名》第X条；带引号的原文摘录一并取出（用于 G2）。"""
    out: list[Citation] = []
    for match in CITATION_RE.finditer(text or ""):
        law, article = match.group(1).strip(), f"第{match.group(2)}条"
        quote = _quote_near(text, match.start(), match.end())
        out.append(Citation(law=law, article_no=article, quote=quote, raw=match.group(0)))
    return out


_QUOTE_PAIRS = (("“", "”"), ("「", "」"), ('"', '"'))
_OPEN_TO_CLOSE = dict(_QUOTE_PAIRS)
_CLOSE_TO_OPEN = {close: open_q for open_q, close in _QUOTE_PAIRS}

#: 引文与条号之间允许出现的"排版"内容：标点/括号/空白，以及本系统的证据编号 `（E1）`。
#: 用正则而不是字符集，是因为 `（E1）：` 里夹着字母数字，字符集会把 E 当成正文而断掉。
_FWD_GAP = re.compile(r"^[\s（(【\[:：，,。、]*?(?:E\d{1,3})?[\s）)】\]：:，,。]*?[“「\"]")
_BWD_GAP = re.compile(r"[”」\"][\s）)】\]：:，,。]*?(?:E\d{1,3})?[\s（(【\[:：，,。]*?$")


def _quote_forward(text: str, end: int, window: int = 24) -> str:
    m = _FWD_GAP.match(text[end : end + window])
    if not m:
        return ""
    pos = end + m.end() - 1
    open_q = text[pos]
    close = text.find(_OPEN_TO_CLOSE[open_q], pos + 1)
    return text[pos + 1 : close] if close != -1 else ""


def _quote_backward(text: str, start: int, window: int = 24) -> str:
    lo = max(0, start - window)
    m = _BWD_GAP.search(text[lo:start])
    if not m:
        return ""
    pos = lo + m.start()
    open_q = _CLOSE_TO_OPEN[text[pos]]
    j = text.rfind(open_q, 0, pos)
    return text[j + 1 : pos] if j != -1 else ""


def _quote_near(text: str, start: int, end: int) -> str:
    """只取**语法上绑在条号上**的引文：`《法》第X条：“引文”` 或 `“引文”（《法》第X条）`。

    为什么必须这么严：早先版本在 ±(120/40) 的窗口里找"最近的引号"，结果在密集引用
    段落里抓到了**邻居条文**的引文，G2 把正确引用判成篡改（真实问答里 q01 一次误报 8 条）。

    判据是"中间只剩排版字符"：括号、冒号、标点、空白，以及本系统的证据编号 `（E1）`。
    只要中间夹了正文，就不是这条的引文 —— 于是"抓错邻居"在结构上不可能发生。
    宁可少校验（无引文 = 不判逐字），也不要错判。
    """
    return _quote_forward(text, end) or _quote_backward(text, start)


def verbatim_ok(quote: str, evidence_quote: str) -> bool:
    """引文是否与法条原文逐字一致。

    * 无引文 → 不判（True），G2 只对"摘录了原文"的引用生效；
    * 使用了省略号 → 每段必须按顺序原样出现在原文里；
    * 否则 → 归一化后必须是原文的**连续片段**。

    为什么不要求"完全相等"：法律写作里截取片段是常态（"经济补偿按劳动者在本单位
    工作的年限"），把它判成篡改只会制造误报。真正的篡改是**增删改任何一个字** ——
    那样就不再是连续片段，一定搜不到。所以"连续片段"既放过合法截取，又不放过篡改。
    """
    if not quote:
        return True
    q, e = normalize(quote), normalize(evidence_quote)
    if not q:
        return True
    if any(mark in quote for mark in _ELLIPSIS):
        segments = [normalize(s) for s in re.split(r"……|\.\.\.|…", quote)]
        segments = [s for s in segments if s]
        if not segments:
            return False
        cursor = 0
        for seg in segments:
            pos = e.find(seg, cursor)
            if pos == -1:
                return False
            cursor = pos + len(seg)
        return True
    return q in e


def extract_amounts(text: str) -> list[tuple[str, Decimal]]:
    """抽取"数字 + 元/万元"，统一折算为元。"""
    out: list[tuple[str, Decimal]] = []
    for match in AMOUNT_RE.finditer(text or ""):
        try:
            value = Decimal(match.group(1)) * normalize_amount_unit(match.group(2))
        except InvalidOperation:  # pragma: no cover - 正则已限定数字
            continue
        out.append((match.group(0), value))
    return out


def numeric_mismatches(
    text: str,
    allowed: list[Decimal],
    *,
    tolerance: Decimal = Decimal("0.01"),
) -> list[tuple[str, Decimal]]:
    """答案里出现的金额必须能在 allowed 里找到对应值，否则算失配。"""
    if not allowed:
        return []
    bad: list[tuple[str, Decimal]] = []
    for raw, value in extract_amounts(text):
        if not any(abs(value - a) <= tolerance for a in allowed):
            bad.append((raw, value))
    return bad


#: 这些开头的片段不是"作者的论断"：标题、纯引文（来自证据，由 G2 负责）、列表符号
_SKIP_PREFIX = ("#", "【", "- ", "* ", "“", "「", '"', "> ")
#: 纯引用证据的片段不需要 G4 判（原文就在账本里，G2 已经校过逐字）
_QUOTED_HINT = ("“", "”")
#: 段首小标题（`**1. 结论**`、`1. 结论：`、`依据：`）：去掉它再判，否则"结论：您问的是……"
#: 这类元陈述会被当成无依据的法律断言（实测每次误报 1 条并触发补检索回路）
_LABEL_RE = re.compile(
    r"^\s*\**\s*(?:[\d一二三四五六七八九十]+\s*[.、)）]\s*)?\**\s*"
    r"(结论|依据|计算|提示|分析|说明|建议|答案|小贴士|答复|要点)\s*\**\s*(?:[：:]\s*)?"
)
#: 只是复述问题的元陈述，不构成法律断言
_META_RE = re.compile(r"^(您问的是|你问的是|你的问题是|您的问题是|本问题|该问题|问题涉及|综上|因此建议)")


def claims_of(text: str, *, min_len: int = 10, limit: int = 24) -> list[str]:
    """切断言句：只保留"作者自己的论断"。

    三类不判：
    * 标题/列表符号（`**2. 依据**`、`- 提示`）；
    * 以引号开头的纯引文 —— 那是从证据里逐字抄的，G2 负责，G4 再判会重复计错；
    * 过短片段（噪声大、判不准）。
    """
    parts = re.split(r"[。！？\n]+", text or "")
    out: list[str] = []
    for raw in parts:
        # 先剥 markdown 粗体再剥小标题：`**2. 依据** 同一用人单位…` 是有内容的断言，
        # 不能因为以 `**` 开头就整句丢掉（早先版本就是这么丢的）。
        piece = _LABEL_RE.sub("", raw.strip().strip("*").strip()).strip("*").strip()
        if len(piece) < min_len:
            continue
        if piece.startswith(_SKIP_PREFIX) or _META_RE.match(piece):
            continue
        if piece.count("“") >= 1 and piece.count("”") >= 1 and len(piece) < 220:
            # 大部分内容在引号里 → 视为引文片段，跳过
            quoted = "".join(re.findall(r"“([^”]*)”", piece))
            if len(quoted) >= len(piece) * 0.5:
                continue
        out.append(piece)
    return out[:limit]
