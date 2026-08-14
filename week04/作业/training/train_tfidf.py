"""训练 TF-IDF + LinearSVM 意图分类模型。

产出 ``weights/tfidf_svm.pkl``，内含 ``(tfidf, svm, class_names)``。

运行方式（在 week04/作业 目录下）：
    python training/train_tfidf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jieba
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from core.config import PATHS
from core.constants import CLASS_NAMES, LABEL2ID


def preprocess(text: str, stopwords: set[str]) -> str:
    return " ".join(w for w in jieba.lcut(text) if w.strip() and w not in stopwords)


def main() -> None:
    df = pd.read_csv(PATHS.dataset, sep="\t", header=None)
    df.columns = ["text", "label"]
    df = df[df["label"].isin(CLASS_NAMES)].reset_index(drop=True)

    stopwords = set(pd.read_csv(PATHS.stopwords, header=None)[0].values)
    df["text"] = df["text"].map(lambda t: preprocess(t, stopwords))
    y = df["label"].map(LABEL2ID)

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], y, test_size=0.2, stratify=y, random_state=42
    )

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
    x_train_vec = tfidf.fit_transform(x_train)
    x_test_vec = tfidf.transform(x_test)

    model = LinearSVC()
    model.fit(x_train_vec, y_train)

    acc = model.score(x_test_vec, y_test)
    print(f"TF-IDF + LinearSVM 测试集准确率: {acc:.4f}")
    print(classification_report(y_test, model.predict(x_test_vec), target_names=CLASS_NAMES))

    dump((tfidf, model, CLASS_NAMES), PATHS.tfidf_weights)
    print(f"模型已保存: {PATHS.tfidf_weights}")


if __name__ == "__main__":
    main()
