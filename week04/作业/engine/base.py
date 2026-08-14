"""策略模式：定义意图分类器的统一抽象接口与结果数据结构。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Prediction:
    """单条文本的意图识别结果。"""

    label: str
    score: float  # 0~1 置信度


class BaseIntentClassifier(ABC):
    """意图分类器的抽象策略接口。

    四条技术路线（正则 / TFIDF+SVM / BERT / 大模型）都实现该接口，
    对外暴露统一的 ``classify`` 行为，FastAPI 层无需关心具体实现。
    """

    # 引擎唯一标识，注册到工厂时使用
    name: str = ""

    @abstractmethod
    def classify(self, texts: list[str]) -> list[Prediction]:
        """对若干文本做意图分类，返回与输入等长的结果列表。"""
        raise NotImplementedError
