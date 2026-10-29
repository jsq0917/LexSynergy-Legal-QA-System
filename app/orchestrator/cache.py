"""答案缓存：键必须带角色与计划指纹。

LawAnswer 用 `sha1(问题)` 做键，在单链路里没问题；多 Agent 下同一个问题
劳动者问（检索+测算）和企业问（检索+审查）走的是不同 DAG，共用键会互相污染。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import date

from app.orchestrator.models import Plan


def normalize(text: str) -> str:
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def cache_key(
    query: str, role: str, plan: Plan, as_of: date | None, mode: str = ""
) -> str:
    """键 = 问题 + 角色 + 计划指纹 + 事发日 + 档位。

    档位必须进键：同一个单跳问题，快速通道（无 G4、无回路）与全编排
    （有五道闸裁决）的产物不同，共用键会让后一次读到前一次的裁决。
    """
    digest = hashlib.sha1(normalize(query).encode("utf-8")).hexdigest()[:16]
    stamp = as_of.strftime("%Y%m%d") if as_of else "na"
    return f"ans:{digest}:{role}:{plan.fingerprint}:{stamp}:{mode or 'full'}"


@dataclass
class CacheEntry:
    key: str
    text: str
    sources: list[dict]
    verdict: dict | None
    created: float


class AnswerCache:
    """进程内 LRU + TTL。生产换 Redis 只需换掉 get/put 两处实现。"""

    def __init__(self, *, ttl_s: float = 7 * 24 * 3600, max_items: int = 512) -> None:
        self._ttl = ttl_s
        self._max = max_items
        self._data: dict[str, CacheEntry] = {}

    def get(self, key: str) -> CacheEntry | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        if time.time() - entry.created > self._ttl:
            self._data.pop(key, None)
            return None
        return entry

    def put(self, key: str, *, text: str, sources: list[dict], verdict: dict | None) -> None:
        if len(self._data) >= self._max:
            oldest = min(self._data, key=lambda k: self._data[k].created)
            self._data.pop(oldest, None)
        self._data[key] = CacheEntry(
            key=key, text=text, sources=sources, verdict=verdict, created=time.time()
        )

    def __len__(self) -> int:
        return len(self._data)
