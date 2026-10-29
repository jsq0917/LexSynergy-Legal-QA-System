r"""版式清洗：页眉页脚、页码、网页导航、目录点线、表格噪音。

法律文档的脏数据有很强的规律性，所以这里不做通用启发式，而是逐条命中：

| 噪音类型 | 真实样例 |
|---|---|
| 页眉 | `中华人民共和国劳动法_中华人民共和国人力资源和社会保障部` |
| 页码 | `1`、`12/34`、`第 3 页 共 5 页` |
| 网页导航 | `首页 \| 宪法 \| 人大机构 \| 栗战书委员长 \| 代表大会会议` |
| 来源信息 | `来源： 新华网  浏览字号： 大 中 小` |
| 日期行 | `8/23/2020`、`2020年06月02日 08:28:13` |
| 目录点线 | `第一章 总则 ......... 1` |
| 站点声明 | `加入收藏`、`网站无障碍`、`京ICP备...` |
| 表格分隔 | `\| ----- \| ------- \|` |

**页眉靠频次识别**：出现 ≥3 次、长度 ≤40 且不像正文（不以句号/分号结尾、
不是条/章/节头）的短行，判为页眉页脚。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from app.ingest.normalize import normalize_line

NOISE_KEYWORDS = (
    "加入收藏",
    "网站无障碍",
    "网站声明",
    "网站管理",
    "网站地图",
    "联系我们",
    "网站值班",
    "主办单位",
    "京ICP",
    "网站标识码",
    "版权所有",
    "浏览字号",
    "打印本页",
    "扫一扫",
    "分享到",
    "当前位置",
    "首页",
    "图片报道",
    "权威发布",
    "法律研究",
    "专题",
    "资料库",
    "新闻",
)

#: 页眉行常见形态：文件名带下划线、或 xxx部/委员会 结尾
_HEADER_HINT_RE = re.compile(r"(人力资源和社会保障部|全国人民代表大会|中国人大网|新华网)")

#: 同时支持 2020/8/23、2020-08-23、2020年6月2日 与 8/23/2020，可带精确到秒的时间
_DATE_RE = re.compile(
    r"^(?:\d{4}\s*[/\-.年]\s*\d{1,2}\s*[/\-.月]\s*\d{1,2}\s*日?"
    r"|\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4})"
    r"(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?$"
)
_URL_RE = re.compile(r"^https?://|^www\.")
_PAGE_RE = re.compile(r"^(\d+|\d+\s*/\s*\d+|第\s*\d+\s*页(\s*共\s*\d+\s*页)?)$")
_TOC_DOTS_RE = re.compile(r"(?:[.．·]{3,}|[…]{2,})\s*\d+\s*$")
_TABLE_SEP_RE = re.compile(r"^\|?[\s:|-]+\|?$")
_HEADING_RE = re.compile(r"^第[一二三四五六七八九十百零千]+[编章节条]")


@dataclass(frozen=True)
class CleanedLine:
    line_no: int
    text: str


def _looks_like_content(text: str) -> bool:
    """判断一行是否像正文，避免把重复出现的正文短句误删。"""
    if _HEADING_RE.match(text):
        return True
    return text.endswith(("。", "；", "：", "！", "？", "”"))


def is_page_number(text: str) -> bool:
    return bool(_PAGE_RE.match(text.strip()))


def is_noise(text: str) -> bool:
    if _URL_RE.match(text):
        return True
    if _DATE_RE.match(text):
        return True
    if _HEADER_HINT_RE.search(text) and len(text) <= 60:
        return True
    if any(kw in text for kw in NOISE_KEYWORDS) and len(text) <= 80:
        # "首页"这类词也可能出现在条文里，用长度 + 是否像正文双重限制
        return not _looks_like_content(text)
    # 网页导航行：一行里塞了多个站点栏目词，长度往往超过 80，靠"命中数"识别
    if sum(1 for kw in NOISE_KEYWORDS if kw in text) >= 2:
        return True
    return False


def is_toc_line(text: str) -> bool:
    return bool(_TOC_DOTS_RE.search(text))


def is_table_separator(text: str) -> bool:
    if "|" not in text:
        return False
    body = text.replace("|", "").replace(" ", "").replace(":", "")
    return bool(body) and set(body) <= {"-"}


def table_row_to_text(text: str) -> str:
    """把 markdown 表格行压成一句话（法律文档里表格多是目录/排版残留）。"""
    cells = [c.strip() for c in text.strip().strip("|").split("|")]
    return " ".join(c for c in cells if c)


def clean_lines(raw_lines: list[str]) -> list[CleanedLine]:
    """逐行清洗，返回保留下来的行（含原始行号，便于溯源）。"""
    normalized: list[tuple[int, str]] = []
    for idx, raw in enumerate(raw_lines, start=1):
        if raw.strip().startswith("|"):  # 表格块
            if is_table_separator(raw):
                continue
            text = normalize_line(table_row_to_text(raw), keep_cjk_gap=True)
        else:
            text = normalize_line(raw, keep_cjk_gap=True)
        if text:
            normalized.append((idx, text))

    # 页眉/页脚：短、重复 ≥3 次、不像正文
    counter = Counter(text for _, text in normalized if len(text) <= 40)
    repeated_headers = {
        text
        for text, count in counter.items()
        if count >= 3 and not _looks_like_content(text) and not is_page_number(text)
    }

    kept: list[CleanedLine] = []
    for line_no, text in normalized:
        if text in repeated_headers:
            continue
        if is_page_number(text) or is_noise(text) or is_toc_line(text):
            continue
        kept.append(CleanedLine(line_no=line_no, text=text))
    return kept
