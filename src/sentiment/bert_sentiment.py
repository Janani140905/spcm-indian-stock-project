"""
BERT-based 3-class sentiment classifier for investor posts (paper Sec III.B, Fig 1).

Classes:  -1 = bearish, 0 = neutral, 1 = bullish  (mapped internally to 0,1,2 for the model)

Usage:
    python -m src.sentiment.bert_sentiment train   --labeled_csv data/labeled/annotated_posts.csv --output_dir models/bert_sentiment
    python -m src.sentiment.bert_sentiment predict --posts_csv data/raw/posts.csv --model_dir models/bert_sentiment --output_csv data/processed/posts_with_sentiment.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
)

LABEL2ID = {-1: 0, 0: 1, 1: 2}   # bearish, neutral, bullish
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MAX_LEN = 128


class PostsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, max_length=self.max_len, padding=False,
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def train(
    labeled_csv: str,
    output_dir: str,
    base_model: str = "ProsusAI/finbert",  # Swapped to English FinBERT for Indian/global text
    test_size: float = 0.3,
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 2e-5,
    seed: int = 42,
):
    df = pd.read_csv(labeled_csv)
    assert {"text", "label"}.issubset(df.columns), "labeled_csv needs columns: text,label"
    df = df.dropna(subset=["text", "label"])
    df["label_id"] = df["label"].astype(int).map(LABEL2ID)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label_id"]
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=3)

    train_ds = PostsDataset(train_df["text"], train_df["label_id"], tokenizer)
    test_ds = PostsDataset(test_df["text"], test_df["label_id"], tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        seed=seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=test_ds,
        data_collator=collator, compute_metrics=compute_metrics,
    )
    trainer.train()

    metrics = trainer.evaluate()
    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    print("Test accuracy:", accuracy_score(test_ds.labels, y_pred))
    print(classification_report(test_ds.labels, y_pred, target_names=["bearish", "neutral", "bullish"]))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}. Eval metrics: {metrics}")
    return metrics


@torch.no_grad()
def predict(posts_csv: str, model_dir: str, output_csv: str, batch_size: int = 64, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    df = pd.read_csv(posts_csv, parse_dates=["post_time"])
    texts = df["text"].fillna("").astype(str).tolist()

    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits
        pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(pred_ids.tolist())

    df["sentiment"] = [ID2LABEL[p] for p in all_preds]  # -1 / 0 / 1
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} sentiment-labeled posts to {output_csv}")
    return df


def _cli():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--labeled_csv", required=True)
    p_train.add_argument("--output_dir", required=True)
    p_train.add_argument("--base_model", default="ProsusAI/finbert")
    p_train.add_argument("--epochs", type=int, default=4)
    p_train.add_argument("--batch_size", type=int, default=16)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--posts_csv", required=True)
    p_pred.add_argument("--model_dir", required=True)
    p_pred.add_argument("--output_csv", required=True)

    args = parser.parse_args()
    if args.cmd == "train":
        train(args.labeled_csv, args.output_dir, base_model=args.base_model,
              epochs=args.epochs, batch_size=args.batch_size)
    elif args.cmd == "predict":
        predict(args.posts_csv, args.model_dir, args.output_csv)


if __name__ == "__main__":
    _cli()
