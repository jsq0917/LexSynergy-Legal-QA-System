"""摄入管线：Markdown → 清洗 → 结构识别 → 父/子分块 → 可持久化的 JSONL。

与 LawAnswer 的区别：这里不碰 MySQL / Milvus（那是存储适配器的事），
只产出纯数据 + 一份校验报告，因此可以离线跑、可以单测、可以换任何存储。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.ingest.chunker import build_chunks, truncation_rate
from app.ingest.clean import clean_lines
from app.ingest.loader import law_name_of, load_lines
from app.ingest.models import Chunk, ParsedDocument
from app.ingest.normalize import count_kangxi
from app.ingest.structure import parse_law, validate_article_sequence

#: 入库硬校验：条数不对就是解析错了，必须报错而不是静默入库
EXPECTED_ARTICLES = {
    "中华人民共和国劳动合同法": 98,
    "中华人民共和国劳动法": 107,
    "中华人民共和国民法典": 1260,
}


@dataclass
class IngestReport:
    law_name: str
    source_file: str
    articles: int
    parents: int
    children: int
    chars: int
    kangxi_chars: int
    missing_article_nums: list[int] = field(default_factory=list)
    truncation_rate: float = 0.0
    expected: int | None = None

    @property
    def ok(self) -> bool:
        if self.expected is not None and self.articles != self.expected:
            return False
        return not self.missing_article_nums

    def to_dict(self) -> dict:
        return {
            "law_name": self.law_name,
            "source_file": self.source_file,
            "articles": self.articles,
            "expected": self.expected,
            "ok": self.ok,
            "parents": self.parents,
            "children": self.children,
            "chars": self.chars,
            "kangxi_chars": self.kangxi_chars,
            "missing_article_nums": self.missing_article_nums,
            "truncation_rate": round(self.truncation_rate, 4),
        }


def ingest(
    path: str | Path,
    *,
    law_name: str | None = None,
    doc_id: str | None = None,
    kb_id: str = "",
    chunk_size: int = 300,
    overlap: int = 50,
    expected: int | None = None,
) -> tuple[ParsedDocument, list[Chunk], IngestReport]:
    p = Path(path)
    raw_lines = load_lines(p)
    kangxi = count_kangxi(p.read_text(encoding="utf-8", errors="replace"))

    name = law_name or law_name_of(p)
    cleaned = clean_lines(raw_lines)
    doc = parse_law(source_file=p.name, law_name=name, lines=cleaned)
    chunks = build_chunks(
        doc,
        doc_id=doc_id or p.stem,
        kb_id=kb_id,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    report = IngestReport(
        law_name=doc.law_name,
        source_file=p.name,
        articles=doc.article_count,
        parents=sum(1 for c in chunks if c.chunk_type == "parent"),
        children=sum(1 for c in chunks if c.chunk_type == "child"),
        chars=sum(len(a.content) for a in doc.articles),
        kangxi_chars=kangxi,
        missing_article_nums=validate_article_sequence(doc.articles),
        truncation_rate=truncation_rate(chunks),
        expected=expected if expected is not None else EXPECTED_ARTICLES.get(doc.law_name),
    )
    return doc, chunks, report
