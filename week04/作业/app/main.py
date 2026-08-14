"""FastAPI 服务入口：暴露四条技术路线的意图识别接口。"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine  # noqa: F401  触发引擎注册
from fastapi import FastAPI

from app.schemas import ClassifyItem, ClassifyRequest, ClassifyResponse
from core.logging_config import logger
from engine.registry import ClassifierRegistry

app = FastAPI(
    title="车载意图识别服务",
    description="四条技术路线：正则 / TFIDF+SVM / BERT / 大模型",
    version="1.0.0",
)


def _classify(engine_name: str, req: ClassifyRequest) -> ClassifyResponse:
    start = time.time()
    try:
        classifier = ClassifierRegistry.get(engine_name)
        texts = [req.text] if isinstance(req.text, str) else req.text
        preds = classifier.classify(texts)
        items = [
            ClassifyItem(text=t, label=p.label, score=round(p.score, 4))
            for t, p in zip(texts, preds)
        ]
        error = "ok"
    except Exception:
        items = []
        error = traceback.format_exc()
        logger.exception("引擎 %s 识别失败", engine_name)

    elapsed_ms = round((time.time() - start) * 1000, 2)
    return ClassifyResponse(
        request_id=req.request_id,
        engine=engine_name,
        items=items,
        elapsed_ms=elapsed_ms,
        error=error,
    )


@app.post("/v1/intent/regex", response_model=ClassifyResponse, tags=["正则"])
def classify_regex(req: ClassifyRequest):
    return _classify("regex", req)


@app.post("/v1/intent/tfidf", response_model=ClassifyResponse, tags=["TFIDF+SVM"])
def classify_tfidf(req: ClassifyRequest):
    return _classify("tfidf", req)


@app.post("/v1/intent/bert", response_model=ClassifyResponse, tags=["BERT"])
def classify_bert(req: ClassifyRequest):
    return _classify("bert", req)


@app.post("/v1/intent/llm", response_model=ClassifyResponse, tags=["大模型"])
def classify_llm(req: ClassifyRequest):
    return _classify("llm", req)


@app.get("/health", tags=["健康检查"])
def health():
    return {"status": "ok", "engines": ClassifierRegistry.names()}
