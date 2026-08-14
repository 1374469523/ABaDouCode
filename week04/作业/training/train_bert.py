"""训练 BERT 意图分类模型（Trainer 版）。

产出 ``weights/bert.pt``，内含 ``{state_dict, class_names}``。

运行方式（在 week04/作业 目录下）：
    python training/train_bert.py --sample 2000     # 默认 2000 条快速训练
    python training/train_bert.py --sample 0        # 全量数据训练
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

from core.config import PATHS
from core.constants import CLASS_NAMES, LABEL2ID, NUM_CLASSES


def load_data(sample: int | None) -> pd.DataFrame:
    df = pd.read_csv(PATHS.dataset, sep="\t", header=None)
    df.columns = ["text", "label"]
    df = df[df["label"].isin(CLASS_NAMES)].reset_index(drop=True)

    if sample is not None:
        per_class = max(1, sample // NUM_CLASSES)
        parts = [
            g.sample(n=min(len(g), per_class), random_state=42)
            for _, g in df.groupby("label")
        ]
        df = pd.concat(parts).reset_index(drop=True)

    df["label_id"] = df["label"].map(LABEL2ID)
    return df


def encode(tokenizer: BertTokenizer, texts: list[str]):
    return tokenizer(texts, truncation=True, padding=True, max_length=64)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds == labels).mean()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=2000, help="训练样本数，0 表示全量")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    sample = None if args.sample == 0 else args.sample
    df = load_data(sample)
    texts = df["text"].tolist()
    labels = df["label_id"].tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained(str(PATHS.bert_pretrained))
    model = BertForSequenceClassification.from_pretrained(
        str(PATHS.bert_pretrained), num_labels=NUM_CLASSES
    )

    train_ds = Dataset.from_dict({
        "input_ids": encode(tokenizer, x_train)["input_ids"],
        "attention_mask": encode(tokenizer, x_train)["attention_mask"],
        "labels": y_train,
    })
    test_ds = Dataset.from_dict({
        "input_ids": encode(tokenizer, x_test)["input_ids"],
        "attention_mask": encode(tokenizer, x_test)["attention_mask"],
        "labels": y_test,
    })

    training_args = TrainingArguments(
        output_dir=str(PATHS.weights_dir / "bert_checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=str(PATHS.weights_dir / "logs"),
        logging_steps=50,
        report_to="none",  # 禁用 tensorboard，规避其 gfile 无法处理中文路径的问题
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("BERT 评估指标:", metrics)

    best_ckpt = trainer.state.best_model_checkpoint or str(PATHS.weights_dir / "bert_checkpoints")
    best_model = BertForSequenceClassification.from_pretrained(best_ckpt)
    torch.save(
        {"state_dict": best_model.state_dict(), "class_names": CLASS_NAMES},
        PATHS.bert_weights,
    )
    print(f"模型已保存: {PATHS.bert_weights}")


if __name__ == "__main__":
    main()
