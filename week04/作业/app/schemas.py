"""请求/响应数据模型（Pydantic）。"""
from __future__ import annotations

from typing import Union

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    request_id: str = Field(default="", description="请求ID，方便排查")
    text: Union[str, list[str]] = Field(..., description="待识别文本，支持单条或批量")


class ClassifyItem(BaseModel):
    text: str = Field(..., description="原始文本")
    label: str = Field(..., description="意图类别")
    score: float = Field(..., description="置信度 0~1")


class ClassifyResponse(BaseModel):
    request_id: str = Field(default="", description="请求ID")
    engine: str = Field(..., description="识别引擎")
    items: list[ClassifyItem] = Field(..., description="识别结果列表")
    elapsed_ms: float = Field(..., description="耗时(毫秒)")
    error: str = Field(default="ok", description="异常信息")
