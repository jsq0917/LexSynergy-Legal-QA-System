"""文本归一化：法律 PDF/Markdown 的字符级脏数据处理。

三个必须处理的真实问题（都在本机语料上实测验证过）：

1. **康熙部首变体**：部分 PDF 抽取出的"第⼀条"里 `⼀` 是 U+2F00（KANGXI RADICAL ONE），
   不是汉字"一"。劳动法 Markdown 里有 **1122 个**这样的字符，不做映射只能识别出
   6/107 条 —— 这是整条入库链路的第一道生死关。
   处理：U+2F00–U+2FD5 批量 NFKC 映射，再手工补 NFKC 覆盖不到的 4 个变体。
2. **版式空格**："总     则"（PDF 为对齐插入的空格）、全角空格 U+3000。
3. **零宽字符** U+200B/200C/200D/FEFF：肉眼不可见，却会让正则和向量都失准。
"""

from __future__ import annotations

import re
import unicodedata

# 康熙部首（U+2F00 起）与 CJK 部首补充（U+2E80 起）中 NFKC 无法自动归一的变体
_KANGXI_TABLE: dict[int, str] = {
    cp: unicodedata.normalize("NFKC", chr(cp)) for cp in range(0x2F00, 0x2FD6)
}
# 实测这 4 个 NFKC 不会转换，必须手工映射
_KANGXI_TABLE.update({ord("⺠"): "民", ord("⻅"): "见", ord("⻓"): "长", ord("⻔"): "门"})

#: 零宽/格式控制字符
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u2060"), None)

#: 连续空白（含全角空格）
_SPACES_RE = re.compile(r"[\s\u3000]+")
#: 两个 CJK 字符之间的空格（PDF 对齐空格，需删除）
_CJK_GAP_RE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")
#: 全角数字/字母转半角
_FULLWIDTH_ALNUM = {cp: cp - 0xFEE0 for cp in list(range(0xFF10, 0xFF1A)) + list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B))}


def to_standard_chars(text: str) -> str:
    """字符级归一：康熙部首 → 标准汉字，全角字母数字 → 半角，去零宽字符。"""
    text = text.translate(_KANGXI_TABLE)
    text = text.translate(_ZERO_WIDTH)
    text = text.translate(_FULLWIDTH_ALNUM)
    return text


def remove_cjk_gap(text: str) -> str:
    """删除两个汉字之间的空格（PDF 对齐空格）。"""
    return _CJK_GAP_RE.sub("", text)


def normalize_line(line: str, *, keep_cjk_gap: bool = False) -> str:
    """单行归一。

    ``keep_cjk_gap=True`` 时保留汉字间的单个空格 —— 结构识别需要它：
    "第一章 总则" 里的空格是判断"这是标题而不是正文交叉引用"的关键信号
    （正文里的"第三十九条和第四十条"条号后没有空格）。
    """
    text = to_standard_chars(line)
    text = _SPACES_RE.sub(" ", text).strip()
    return text if keep_cjk_gap else remove_cjk_gap(text)


def normalize_text(text: str) -> str:
    """整篇归一（逐行处理，保留段落边界）。"""
    return "\n".join(normalize_line(line) for line in text.splitlines())


def has_kangxi(text: str) -> bool:
    return any(0x2F00 <= ord(ch) <= 0x2FD5 or 0x2E80 <= ord(ch) <= 0x2EFF for ch in text)


def count_kangxi(text: str) -> int:
    return sum(
        1 for ch in text if 0x2F00 <= ord(ch) <= 0x2FD5 or 0x2E80 <= ord(ch) <= 0x2EFF
    )


def smart_join(left: str, right: str) -> str:
    """合并 PDF 折行的两段文本。

    * 中文接中文：直接相连（PDF 折行不带空格）
    * 英文/数字接英文/数字：加空格，避免 "2008年" 变成 "2008 年" 之外还把单词粘连
    """
    if not left:
        return right
    if not right:
        return left
    if re.search(r"[\u4e00-\u9fff]$", left) and re.match(r"^[\u4e00-\u9fff]", right):
        return left + right
    if re.search(r"[A-Za-z0-9%)]$", left) and re.match(r"^[A-Za-z0-9(]", right):
        return f"{left} {right}"
    # 左侧是中文标点，或右侧以中文标点开头：直接相连，不加空格
    if re.search(r"[，。；：、！？）》”’]$", left) or re.match(r"^[，。；：、！？）》”’]", right):
        return left + right
    return f"{left} {right}"


def cn_to_int(cn: str) -> int:
    """中文数字转整数，支持到千位（民法典有"第一千二百六十条"）。

    只覆盖法条编号会用到的写法：零 一 二 … 十 百 千。
    """
    digits = "零一二三四五六七八九"
    units = {"十": 10, "百": 100, "千": 1000}
    total = 0
    num = 0
    for ch in cn:
        if ch == "零":
            continue
        if ch in digits:
            num = digits.index(ch)
        elif ch in units:
            total += (num or 1) * units[ch]
            num = 0
    return total + num
