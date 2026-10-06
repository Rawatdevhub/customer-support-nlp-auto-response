from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def train_transformer(data_path: str, model_name: str):
    df = pd.read_csv(data_path).dropna(subset=["instruction", "intent"])
    labels = sorted(df["intent"].unique())
    label_to_id = {label: i for i, label in enumerate(labels)}
    df["label"] = df["intent"].map(label_to_id)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels), id2label={i: x for x, i in label_to_id.items()}, label2id=label_to_id)

    def tokenize(batch):
        return tokenizer(batch["instruction"], truncation=True, padding="max_length", max_length=128)

    train_ds = Dataset.from_pandas(train_df[["instruction", "label"]]).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df[["instruction", "label"]]).map(tokenize, batched=True)
    start = time.perf_counter()
    args = TrainingArguments(output_dir=str(MODEL_DIR / model_name.replace("/", "-")), evaluation_strategy="epoch", save_strategy="epoch", num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=32, load_best_model_at_end=True, report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds, tokenizer=tokenizer)
    trainer.train()
    training_seconds = time.perf_counter() - start
    metrics = trainer.evaluate()
    output_dir = MODEL_DIR / model_name.replace("/", "-")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    result = {"model": model_name, "training_seconds": training_seconds, **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}}
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / f"{model_name.replace('/', '-')}-metrics.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/Bitext_Customer_Support_Dataset.csv")
    parser.add_argument("--model", required=True, choices=["distilbert-base-uncased", "bert-base-uncased"])
    args = parser.parse_args()
    train_transformer(args.data, args.model)
