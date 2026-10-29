"""结构识别：把清洗后的行还原成"编 → 章 → 节 → 条"。

三层防误判（每层都对应真实踩过的坑）：

1. **标题后必须有空白**：`^第X条\\s+` 才算条头。正文折行处的交叉引用
   （"第三十九条和第四十条…"）条号后没有空白，不会被误判成新条。
2. **条号单调递增**：民法典正文里"依照本法第一百五十三条"这类整行引用
   即使带了空格，只要条号小于当前进度就按正文处理。
3. **正文起点判定**：首个合法条号出现之前的内容一律视为目录/元数据丢弃。
   民法典目录区出现过一次"第一编～第七编"，正文区又出现一次，靠这条排除。

另：民法典是 **编→章→节→条** 四级（1260 条），另两部是 章→节→条 三级，
路径按实际层级拼接，缺级自动省略。
"""

from __future__ import annotations

import re

from app.ingest.clean import CleanedLine
from app.ingest.models import Article, ParsedDocument
from app.ingest.normalize import cn_to_int, remove_cjk_gap, smart_join

_CN_NUM = "[一二三四五六七八九十百零千]+"

PART_RE = re.compile(rf"^第({_CN_NUM})编[\s\u3000]+(.*)$")
CHAPTER_RE = re.compile(rf"^第({_CN_NUM})章[\s\u3000]+(.*)$")
SECTION_RE = re.compile(rf"^第({_CN_NUM})节[\s\u3000]+(.*)$")
ARTICLE_RE = re.compile(rf"^第({_CN_NUM})条[\s\u3000]+(.*)$")

#: 文件名里的冗余后缀，取法名前先剥掉
_FILENAME_NOISE = ("（全文）", "(全文)", "_中华人民共和国人力资源和社会保障部")


def law_name_from_filename(filename: str) -> str:
    """`中华人民共和国劳动法_中华人民共和国人力资源和社会保障部.md` → `中华人民共和国劳动法`。"""
    stem = filename.rsplit(".", 1)[0]
    stem = stem.split("_")[0]
    for noise in _FILENAME_NOISE:
        stem = stem.replace(noise, "")
    return stem.strip() or filename


def _flush(
    buffer: list[str],
    *,
    article_no: str,
    article_num: int,
    chapter_path: str,
    line_start: int,
    line_end: int,
    out: list[Article],
) -> None:
    if not article_no:
        return
    text = ""
    for raw in buffer:
        text = smart_join(text, remove_cjk_gap(raw))
    text = text.strip()
    if not text:
        return
    out.append(
        Article(
            article_no=article_no,
            article_num=article_num,
            chapter_path=chapter_path,
            content=text,
            line_start=line_start,
            line_end=line_end,
        )
    )


def parse_law(source_file: str, law_name: str, lines: list[CleanedLine]) -> ParsedDocument:
    """把清洗后的行解析成法条列表。"""
    articles: list[Article] = []
    part = chapter = section = ""
    buffer: list[str] = []
    article_no = ""
    article_num = 0
    current_num = 0
    line_start = 0
    line_end = 0
    in_body = False
    skipped = 0

    for item in lines:
        text = item.text
        line_no = item.line_no

        m = PART_RE.match(text)
        if m:
            if in_body:
                _flush(buffer, article_no=article_no, article_num=article_num,
                       chapter_path=_path(part, chapter, section), line_start=line_start,
                       line_end=line_end, out=articles)
                buffer, article_no = [], ""
            part, chapter, section = _label("编", m.group(1), m.group(2)), "", ""
            line_start = line_no
            continue

        m = CHAPTER_RE.match(text)
        if m:
            if in_body:
                _flush(buffer, article_no=article_no, article_num=article_num,
                       chapter_path=_path(part, chapter, section), line_start=line_start,
                       line_end=line_end, out=articles)
                buffer, article_no = [], ""
            chapter, section = _label("章", m.group(1), m.group(2)), ""
            line_start = line_no
            continue

        m = SECTION_RE.match(text)
        if m:
            if in_body:
                _flush(buffer, article_no=article_no, article_num=article_num,
                       chapter_path=_path(part, chapter, section), line_start=line_start,
                       line_end=line_end, out=articles)
                buffer, article_no = [], ""
            section = _label("节", m.group(1), m.group(2))
            line_start = line_no
            continue

        m = ARTICLE_RE.match(text)
        if m:
            num = cn_to_int(m.group(1))
            # 防误判②：条号回退 → 当成正文里的交叉引用
            if num < current_num:
                if in_body and article_no:
                    buffer.append(text)
                    line_end = line_no
                else:
                    skipped += 1
                continue
            if in_body:
                _flush(buffer, article_no=article_no, article_num=article_num,
                       chapter_path=_path(part, chapter, section), line_start=line_start,
                       line_end=line_end, out=articles)
            in_body = True
            current_num = num
            article_no = f"第{m.group(1)}条"
            article_num = num
            buffer = [m.group(2)] if m.group(2).strip() else []
            line_start = line_no
            line_end = line_no
            continue

        if in_body:
            if not article_no:      # 正文首条之前（理论上不会到这）
                continue
            buffer.append(text)
            line_end = line_no

    if in_body:
        _flush(buffer, article_no=article_no, article_num=article_num,
               chapter_path=_path(part, chapter, section), line_start=line_start,
               line_end=line_end, out=articles)

    full_text = "\n".join(item.text for item in lines)
    return ParsedDocument(
        source_file=source_file,
        law_name=law_name,
        text=full_text,
        articles=articles,
        skipped_articles=skipped,
    )


def _label(unit: str, number: str, title: str) -> str:
    """章节标签带上标题，给生成层提供更完整的上下文。

    "第四章 劳动合同的解除和终止" 比裸的 "第四章" 更能帮模型定位语义。
    """
    clean = remove_cjk_gap((title or "").strip())
    return f"第{number}{unit} {clean}".strip() if clean else f"第{number}{unit}"


def _path(part: str, chapter: str, section: str) -> str:
    return "·".join(p for p in (part, chapter, section) if p)


def validate_article_sequence(articles: list[Article]) -> list[int]:
    """返回缺失/跳号的条号列表，用于入库硬校验（数量对不上时直接报错而不是静默入库）。"""
    if not articles:
        return []
    nums = {a.article_num for a in articles}
    expected = set(range(1, max(nums) + 1))
    return sorted(expected - nums)
