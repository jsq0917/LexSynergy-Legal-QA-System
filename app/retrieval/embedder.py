"""bge-m3 稠密向量：本地加载、懒初始化、批量编码。

为什么不走 FlagEmbedding：它会把 peft/datasets 等一堆东西拖进来，而 bge-m3 的
稠密部分只需要 `AutoModel` 取 CLS + L2 归一化。少一层封装，少一类版本冲突。

模型路径默认指向 HF 缓存里已下载好的目录（LawAnswer 用过同一份权重），
可用 `EMBEDDING_MODEL_PATH` 覆盖。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from app.core.logger import get_logger

logger = get_logger("retrieval.embedder")

DEFAULT_MODEL_PATH = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-m3"
MAX_LEN = 256


class BgeM3Embedder:
    def __init__(self, model_path: str | Path | None = None, *, batch_size: int = 16) -> None:
        self._path = Path(model_path or os.getenv("EMBEDDING_MODEL_PATH") or DEFAULT_MODEL_PATH)
        self._batch = batch_size
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._path.exists()

    @property
    def loaded(self) -> bool:
        """模型是否已驻留。调用方可据此决定"第一次是否该给更宽的超时"。"""
        return self._model is not None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        with self._lock:                    # 双重检查：并发首次调用只加载一次
            if self._model is not None:
                return
            self._load()

    def _load(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        # 懒导入 + 调用方持锁：既要避免"导入 app 就拉起 2GB 模型"（内存紧张时直接段错误），
        # 又要避免两个线程同时首次 import 触发 transformers 懒加载竞态。
        import torch
        from transformers import AutoModel, AutoTokenizer

        logger.info("加载 bge-m3：%s", self._path)
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._path), local_files_only=True)
        self._model = AutoModel.from_pretrained(str(self._path), local_files_only=True).eval()
        torch.set_grad_enabled(False)
        self._torch = torch

    def encode(self, texts: list[str]) -> np.ndarray:
        """返回 (n, 1024) 的 L2 归一化向量。"""
        self._ensure()
        import torch

        out: list[np.ndarray] = []
        for start in range(0, len(texts), self._batch):
            chunk = [t or " " for t in texts[start : start + self._batch]]
            batch = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            with torch.no_grad():
                hidden = self._model(**batch).last_hidden_state[:, 0]
                vec = torch.nn.functional.normalize(hidden, dim=-1)
            out.append(vec.cpu().numpy().astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, 1024), dtype=np.float32)


def get_embedder() -> BgeM3Embedder:
    return BgeM3Embedder()
