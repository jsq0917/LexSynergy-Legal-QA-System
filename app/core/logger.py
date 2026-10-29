"""统一日志：控制台 + 文件双写，业务代码只从这里取 logger。"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_CONFIGURED = False
_FMT = "%(asctime)s %(levelname)-5s [%(name)s] %(message)s"


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root = logging.getLogger("lexsynergy")
    root.setLevel(getattr(logging, level, logging.INFO))
    root.propagate = False

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
    root.addHandler(console)

    try:
        log_dir = Path(os.getenv("LOG_DIR", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FMT))
        root.addHandler(file_handler)
    except OSError:  # 只读文件系统/无权限：降级为仅控制台
        pass

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(f"lexsynergy.{name}")
