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


def _sorted_by_score(y_true: np.ndarray, y_score: np.ndarray):
    """Sort descending by score; return labels, scores, cumulative tp, precision, recall,
    and a mask of tie-group end positions (honest threshold points)."""
    y = np.asarray(y_true).astype(np.int8)
    s = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    tp = np.cumsum(y == 1)
    precision = tp / (np.arange(len(y)) + 1)
    recall = tp / max(int((y == 1).sum()), 1)
    group_end = np.empty(len(y), dtype=bool)
    group_end[:-1] = s[:-1] != s[1:]
    group_end[-1] = True
    return s, tp, precision, recall, group_end


def precision_at_recall(y_true: np.ndarray, y_score: np.ndarray, r_target: float) -> float | None:
    """Max precision over achievable thresholds with recall >= r_target.

    Returns None when the score curve never reaches r_target. Ties are handled
    by evaluating each threshold at the end of its tie group.
    """
    _, _, precision, recall, group_end = _sorted_by_score(y_true, y_score)
    valid = (recall >= r_target) & group_end
    if not valid.any():
        return None
    return float(precision[valid].max())


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Precision summed over recall increments (step-function area under the PR curve).

    One threshold per tie group, evaluated at the group end: the group's whole
    recall increment is multiplied by the precision at that threshold.
    """
    s, _, precision, recall, group_end = _sorted_by_score(y_true, y_score)
    ends = np.flatnonzero(group_end)
    recall_end, precision_end = recall[ends], precision[ends]
    delta = np.diff(np.concatenate([[0.0], recall_end]))
    return float(np.sum(delta * precision_end))
