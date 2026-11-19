"""Shared evaluation: every model is scored the same way on the same test set (R3).

Saves two things per model:
  results/metrics/<name>.json      -> macro-F1 etc. (goes into the results table)
  results/predictions/<name>.csv   -> every test message with its prediction
                                      (we pick the failure examples for the slides from here)
"""
import json

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.config import LABELS, RESULTS_DIR


def evaluate(name, texts, y_true, y_pred, scores=None, extra=None):
    metrics = {
        "model": name,
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "f1_offensive": round(f1_score(y_true, y_pred, pos_label=1), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "confusion_matrix [[TN, FP], [FN, TP]]": confusion_matrix(y_true, y_pred).tolist(),
        "n_test": len(y_true),
        **(extra or {}),
    }

    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "metrics" / f"{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "text": texts,
        "true": [LABELS[y] for y in y_true],
        "pred": [LABELS[y] for y in y_pred],
        "score_offensive": scores if scores is not None else [None] * len(y_true),
    }).to_csv(RESULTS_DIR / "predictions" / f"{name}.csv", index=False)

    print(f"[{name}] macro-F1 = {metrics['macro_f1']:.4f} | accuracy = {metrics['accuracy']:.4f}")
    return metrics
