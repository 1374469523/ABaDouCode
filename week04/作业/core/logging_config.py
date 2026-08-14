"""日志配置：同时输出到控制台与滚动文件。"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from core.config import BASE_DIR

_LOG_PATH = BASE_DIR / "app.log"


def setup_logger(name: str = "intent") -> logging.Logger:
    logger = logging.getLogger(name)
    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = RotatingFileHandler(
        _LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


logger = setup_logger()
