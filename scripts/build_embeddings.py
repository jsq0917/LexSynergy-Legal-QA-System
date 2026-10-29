"""构建稠密向量索引：把语料子块编码成 dense.npy。

    python -m scripts.build_embeddings                 # 全量
    python -m scripts.build_embeddings --limit 50      # 冒烟

产物：data/corpus/dense.npy（n×1024，L2 归一化，行序 = chunks.jsonl 中子块的顺序）
      data/corpus/dense_meta.json（模型路径、维度、条数、chunks 指纹）

为什么用 npy 而不是 Milvus：1,470 条 × 1024 维只有 6MB，精确余弦搜索 <5ms，
零基础设施、可进 CI、结果完全可复现。上规模再换 Milvus（接口不变）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from app.retrieval.corpus import Corpus
from app.retrieval.embedder import BgeM3Embedder

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "corpus"


def chunks_fingerprint(corpus: Corpus) -> str:
    h = hashlib.sha1()
    for child in corpus.children:
        h.update(child["chunk_id"].encode("utf-8"))
        h.update(child["content"].encode("utf-8"))
    return h.hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts.build_embeddings")
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--limit", type=int, default=0, help="只编码前 N 条（冒烟用）")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args(argv)

    corpus = Corpus.load(args.out)
    children = corpus.children[: args.limit] if args.limit else corpus.children
    if not children:
        print("[FAIL] 语料为空，先跑 python -m scripts.build_corpus", file=sys.stderr)
        return 2

    embedder = BgeM3Embedder(batch_size=args.batch)
    if not embedder.available:
        print(f"[FAIL] 找不到 bge-m3 权重：{embedder._path}", file=sys.stderr)  # noqa: SLF001
        return 2

    texts = [f"{c['title']}\n{c['content']}" for c in children]
    started = time.time()
    print(f"编码 {len(texts)} 条（batch={args.batch}）...")
    vectors = embedder.encode(texts)
    elapsed = time.time() - started

    out = Path(args.out)
    np.save(out / "dense.npy", vectors)
    meta = {
        "model": str(embedder._path),  # noqa: SLF001
        "dim": int(vectors.shape[1]),
        "count": int(vectors.shape[0]),
        "normalized": True,
        "chunks_fingerprint": chunks_fingerprint(corpus),
        "elapsed_s": round(elapsed, 1),
        "per_item_ms": round(elapsed * 1000 / max(1, len(texts)), 1),
    }
    (out / "dense_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"完成：{vectors.shape}  维度 {meta['dim']}  "
        f"耗时 {elapsed:.1f}s（{meta['per_item_ms']}ms/条）→ {out / 'dense.npy'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
