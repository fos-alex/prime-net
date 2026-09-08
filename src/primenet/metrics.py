"""Classification metrics shared by training, evaluation and baselines."""

from __future__ import annotations

import numpy as np


def prf(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    """Precision/recall per class for binary labels (1 = prime).

    y_score may be probabilities or already-thresholded 0/1 predictions.
    """
    pred = (np.asarray(y_score) >= threshold).astype(np.int8)
    y = np.asarray(y_true).astype(np.int8)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))

    def safe(a: int, b: int) -> float:
        return a / b if b else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision_prime": safe(tp, tp + fp),
        "recall_prime": safe(tp, tp + fn),
        "f1_prime": safe(2 * tp, 2 * tp + fp + fn),
        "precision_composite": safe(tn, tn + fn),
        "recall_composite": safe(tn, tn + fp),
        "accuracy": safe(tp + tn, tp + tn + fp + fn),
    }


def format_metrics(name: str, m: dict) -> str:
    return (
        f"{name:<18} prime P={m['precision_prime']:.4f} R={m['recall_prime']:.4f} "
        f"F1={m['f1_prime']:.4f} | comp R={m['recall_composite']:.4f} | "
        f"acc={m['accuracy']:.4f} | FP={m['fp']:,}"
    )
