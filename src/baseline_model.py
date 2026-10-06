from __future__ import annotations

import argparse
from pathlib import Path
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "tfidf_logistic.joblib"


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"instruction", "intent", "response"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df.dropna(subset=["instruction", "intent", "response"]).copy()
    df["instruction"] = df["instruction"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    return df.drop_duplicates(subset=["instruction", "intent"]).reset_index(drop=True)


def train(data_path: str | Path) -> dict:
    df = load_data(data_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["intent"])
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    start = time.perf_counter()
    pipeline.fit(train_df["instruction"], train_df["intent"])
    elapsed = time.perf_counter() - start
    pred = pipeline.predict(test_df["instruction"])
    report = classification_report(test_df["intent"], pred, output_dict=True, zero_division=0)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipeline, "responses": df.groupby("intent")["response"].first().to_dict()}, MODEL_PATH)
    return {"rows": len(df), "classes": df["intent"].nunique(), "test_accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"], "training_seconds": elapsed, "confusion_matrix": confusion_matrix(test_df["intent"], pred).tolist()}


def predict(text: str) -> dict:
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    label = pipe.predict([text])[0]
    probabilities = pipe.predict_proba([text])[0]
    classes = pipe.classes_
    top = sorted(zip(classes, probabilities), key=lambda pair: pair[1], reverse=True)[:3]
    return {"intent": label, "confidence": float(max(probabilities)), "top_intents": [(name, float(score)) for name, score in top], "response": bundle["responses"].get(label, "A support agent will review your request shortly.")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/Bitext_Customer_Support_Dataset.csv")
    args = parser.parse_args()
    print(train(args.data))
