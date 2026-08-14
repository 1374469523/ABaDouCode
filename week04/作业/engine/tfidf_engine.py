"""TF-IDF + LinearSVM 引擎（策略实现）。

加载训练好的 ``(tfidf, svm, class_names)`` 权重，用 SVM 的 decision_function
经 softmax 归一化得到置信度。
"""
from __future__ import annotations

import jieba
import numpy as np
import pandas as pd
from joblib import load

from core.config import PATHS
from engine.base import BaseIntentClassifier, Prediction
from engine.registry import ClassifierRegistry


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@ClassifierRegistry.register("tfidf")
class TfidfSvmClassifier(BaseIntentClassifier):
    name = "tfidf"

    def __init__(self) -> None:
        self.tfidf, self.model, self.class_names = load(PATHS.tfidf_weights)
        self.stopwords = set(pd.read_csv(PATHS.stopwords, header=None)[0].values)

    def _preprocess(self, text: str) -> str:
        return " ".join(
            w for w in jieba.lcut(text) if w.strip() and w not in self.stopwords
        )

    def classify(self, texts: list[str]) -> list[Prediction]:
        cleaned = [self._preprocess(t) for t in texts]
        x = self.tfidf.transform(cleaned)
        decision = self.model.decision_function(x)
        probs = _softmax(decision)
        idx = probs.argmax(axis=1)
        return [
            Prediction(label=self.class_names[int(i)], score=float(probs[j][int(i)]))
            for j, i in enumerate(idx)
        ]
