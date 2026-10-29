"""加载器：把语料文件读成"行列表"，交给 clean/structure 处理。

只支持 Markdown 与纯文本 —— 本项目复用 LawAnswer 已抽取好的 Markdown，
PDF 抽取属于 LawAnswer 的职责（那边有 loaders.py 处理扫描件与双栏版面）。
"""

from __future__ import annotations

from pathlib import Path

from app.ingest.models import LoadedDocument


def load_text(path: str | Path) -> LoadedDocument:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    return LoadedDocument(source_file=p.name, text=text, file_type="text")


def load_lines(path: str | Path) -> list[str]:
    doc = load_text(path)
    return doc.text.splitlines()


def law_name_of(path: str | Path) -> str:
    """文件名 → 法名（复用 structure.law_name_from_filename 的清洗规则）。"""
    from app.ingest.structure import law_name_from_filename

    return law_name_from_filename(Path(path).name)
