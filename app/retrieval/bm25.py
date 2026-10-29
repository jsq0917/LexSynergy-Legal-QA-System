"""BM25 检索：纯 Python、零依赖、确定性。

为什么先做 BM25 而不是直接上向量：
1. 法律语料是**术语密集型**（"代通知金""二倍工资""竞业限制"），词项信号极强；
2. 它不依赖 embedding 服务，可以把"语料 + 检索 + 评测"整条链路跑在 CI 里；
3. 它是稠密检索的**基线**——没有基线，就无法回答"向量到底带来了多少收益"。

中文切分用二元组（bigram）：不需要词典，对"劳动合同"这类四字术语天然友好，
且能避免分词错误直接毁掉召回。
"""

from __future__ import annotations

import math
import re
from collections import Counter

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_ALNUM = re.compile(r"[a-z0-9]+")

STOP_CHARS = set("的了和与或及在是为对以按由等该其此这那")


def tokenize(text: str) -> list[str]:
    """中文二元组 + 英文/数字整词。"""
    text = (text or "").lower()
    tokens: list[str] = []
    for run in _CJK_RUN.findall(text):
        chars = [ch for ch in run if ch not in STOP_CHARS]
        if len(chars) == 1:
            tokens.append(chars[0])
            continue
        tokens.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
    tokens.extend(_ALNUM.findall(text))
    return tokens


class BM25:
    def __init__(self, docs: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._doc_tokens = [tokenize(doc) for doc in docs]
        self._doc_len = [len(t) for t in self._doc_tokens]
        self._avgdl = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0
        self._tf = [Counter(tokens) for tokens in self._doc_tokens]
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self._doc_tokens)
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def __len__(self) -> int:
        return len(self._doc_tokens)

    def search(self, query: str, *, topk: int = 5) -> list[tuple[int, float]]:
        terms = tokenize(query)
        if not terms or not self._tf:
            return []
        scores: list[tuple[int, float]] = []
        for idx, tf in enumerate(self._tf):
            s = 0.0
            dl = self._doc_len[idx] or 1
            for term in terms:
                freq = tf.get(term)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = freq + self._k1 * (1 - self._b + self._b * dl / (self._avgdl or 1))
                s += idf * freq * (self._k1 + 1) / denom
            if s > 0:
                scores.append((idx, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:topk]
