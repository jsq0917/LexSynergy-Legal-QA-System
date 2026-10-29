"""构建本地语料：Markdown → articles.jsonl + chunks.jsonl + manifest.json

    python -m scripts.build_corpus                    # 用 data/raw
    python -m scripts.build_corpus --src <dir> --out data/corpus

条数硬校验（98 / 107 / 1260）不通过时退出码非 0 —— 解析错了就必须炸，
不能让错的语料静默进入检索。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from app.ingest.pipeline import EXPECTED_ARTICLES, ingest
from app.ingest.structure import _FILENAME_NOISE  # noqa: PLC2701 - 复用同一套文件名清洗规则

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "data" / "raw"
DEFAULT_OUT = ROOT / "data" / "corpus"


def _law_key(path: Path) -> str:
    """文件名 → 期望条数表的键。"""
    stem = path.stem.split("_")[0]
    for noise in _FILENAME_NOISE:
        stem = stem.replace(noise, "")
    return stem


def build(src: Path, out: Path, *, chunk_size: int = 300, overlap: int = 50) -> int:
    files = sorted(src.glob("*.md"))
    if not files:
        print(f"[FAIL] {src} 下没有 .md 语料", file=sys.stderr)
        return 2

    out.mkdir(parents=True, exist_ok=True)
    articles_path = out / "articles.jsonl"
    chunks_path = out / "chunks.jsonl"
    reports: list[dict] = []
    failed = False

    with articles_path.open("w", encoding="utf-8") as fa, chunks_path.open(
        "w", encoding="utf-8"
    ) as fc:
        for path in files:
            expected = EXPECTED_ARTICLES.get(_law_key(path))
            doc, chunks, report = ingest(
                path,
                doc_id=_law_key(path),
                kb_id="kb_labor" if "劳动" in _law_key(path) else "kb_civil",
                chunk_size=chunk_size,
                overlap=overlap,
                expected=expected,
            )
            for article in doc.articles:
                fa.write(
                    json.dumps(
                        {
                            "law_name": doc.law_name,
                            "doc_id": _law_key(path),
                            "article_no": article.article_no,
                            "article_num": article.article_num,
                            "chapter_path": article.chapter_path,
                            "content": article.content,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            for chunk in chunks:
                fc.write(
                    json.dumps(
                        {
                            "chunk_id": chunk.chunk_id,
                            "parent_id": chunk.parent_id,
                            "doc_id": chunk.doc_id,
                            "kb_id": chunk.kb_id,
                            "chunk_type": chunk.chunk_type,
                            "article_no": chunk.article_no,
                            "chapter_path": chunk.chapter_path,
                            "law_name": doc.law_name,
                            "title": chunk.title,
                            "content": chunk.content,
                            "char_count": chunk.char_count,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            reports.append(report.to_dict())
            failed = failed or not report.ok

    manifest = {
        "source_dir": str(src),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "documents": reports,
        "total_articles": sum(r["articles"] for r in reports),
        "total_chunks": sum(r["parents"] + r["children"] for r in reports),
        "ok": not failed,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _print(reports, manifest)
    return 0 if not failed else 1


def _print(reports: list[dict], manifest: dict) -> None:
    print(f"{'法律':<24}{'条数':>8}{'期望':>6}{'父块':>6}{'子块':>7}{'字数':>9}{'康熙':>7}{'截断率':>8}")
    print("-" * 78)
    for r in reports:
        mark = "" if r["ok"] else "  ✗"
        print(
            f"{r['law_name']:<24}{r['articles']:>8}{r['expected'] or '-':>6}"
            f"{r['parents']:>6}{r['children']:>7}{r['chars']:>9}{r['kangxi_chars']:>7}"
            f"{r['truncation_rate']:>8.2%}{mark}"
        )
    print("-" * 78)
    print(
        f"合计 {manifest['total_articles']} 条 / {manifest['total_chunks']} 块    "
        f"校验 {'通过' if manifest['ok'] else '失败'}"
    )
    for r in reports:
        if r["missing_article_nums"]:
            print(f"  ! {r['law_name']} 缺号：{r['missing_article_nums'][:10]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts.build_corpus")
    parser.add_argument("--src", default=str(DEFAULT_SRC))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument(
        "--copy-src",
        action="store_true",
        help="把源文件复制到 data/raw（首次从 LawAnswer 迁移时用）",
    )
    args = parser.parse_args(argv)

    if args.copy_src:
        src = Path(args.src)
        dst = ROOT / "data" / "raw"
        dst.mkdir(parents=True, exist_ok=True)
        for path in src.glob("*.md"):
            shutil.copy2(path, dst / path.name)
        print(f"已复制 {len(list(dst.glob('*.md')))} 个语料文件到 {dst}")

    return build(
        Path(args.src), Path(args.out), chunk_size=args.chunk_size, overlap=args.overlap
    )


if __name__ == "__main__":
    raise SystemExit(main())
