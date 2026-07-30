from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, List
from pydantic import BaseModel, Field


class FaqItem(BaseModel):
    question: str = Field(..., description="用户提问", min_length=1)
    answer: str = Field(..., description="回答内容", min_length=1)
    type: str = Field(..., description="FAQ 类型/分类")


class FaqDataset(BaseModel):
    items: List[FaqItem] = Field(default_factory=list, description="FAQ 列表")

    @classmethod
    def from_json(cls, path: str | Path) -> FaqDataset:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return cls(items=[FaqItem(**item) for item in raw])
        if isinstance(raw, dict):
            return cls.model_validate(raw)
        raise TypeError(f"不支持的 JSON 结构：{type(raw)}")

    def filter_by_type(self, type_name: str) -> list[FaqItem]:
        return [item for item in self.items if item.type == type_name]

    @property
    def types(self) -> list[str]:
        return sorted({item.type for item in self.items})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> FaqItem:
        return self.items[index]


class FaqSearchResult(BaseModel):
    item: FaqItem = Field(..., description="匹配的 FAQ 条目")
    score: float = Field(..., ge=0.0, le=1.0, description="相关性分数 (0-1)")
    match_type: Literal["exact", "fuzzy", "semantic"] = Field(
        default="fuzzy", description="匹配方式"
    )


_DEFAULT_FAQ_PATH = Path(__file__).resolve().parent.parent / "asserts" / "faq.json"


def load_default_faq() -> FaqDataset:
    """加载默认的 FAQ 数据集（asserts/faq.json）。"""
    return FaqDataset.from_json(_DEFAULT_FAQ_PATH)
