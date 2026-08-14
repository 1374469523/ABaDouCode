"""BERT 微调引擎（策略实现）。

加载预训练 ``bert-base-chinese`` 与微调权重 ``bert.pt``，用 softmax 概率
作为置信度。
"""
from __future__ import annotations

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from core.config import PATHS
from engine.base import BaseIntentClassifier, Prediction
from engine.registry import ClassifierRegistry


@ClassifierRegistry.register("bert")
class BertClassifier(BaseIntentClassifier):
    name = "bert"

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(str(PATHS.bert_pretrained))
        checkpoint = torch.load(PATHS.bert_weights, map_location=self.device)
        self.class_names = checkpoint["class_names"]
        self.model = BertForSequenceClassification.from_pretrained(
            str(PATHS.bert_pretrained), num_labels=len(self.class_names)
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def classify(self, texts: list[str]) -> list[Prediction]:
        enc = self.tokenizer(
            texts, truncation=True, padding=True, max_length=64, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        idx = probs.argmax(axis=1)
        return [
            Prediction(label=self.class_names[int(i)], score=float(probs[j][int(i)]))
            for j, i in enumerate(idx)
        ]
