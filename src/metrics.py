"""Evaluation metrics, curve generation and decision-threshold optimization."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

LABEL_NAMES = ("ham", "spam")
THRESHOLD_GRID = tuple(round(float(t), 2) for t in np.linspace(0.01, 0.99, 99))


def percentile(values: Sequence[float] | Iterable[float], q: float) -> float | None:
    """Linear-interpolated percentile; ``None`` for empty input."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return None
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    return float(np.percentile(arr, q))


def apply_threshold(probabilities: Sequence[float] | np.ndarray | float, threshold: float) -> np.ndarray:
    """Return 1 (spam) where probability >= threshold, else 0 (ham)."""
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    probs = np.atleast_1d(np.asarray(probabilities, dtype=float))
    return (probs >= threshold).astype(int)


def _check_binary(y_true: np.ndarray) -> None:
    values = set(np.unique(y_true).tolist())
    if not values <= {0, 1}:
        raise ValueError(f"y_true must be binary 0/1, got {sorted(values)}")


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Compute the full metric set at a given decision threshold."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if y_true.size == 0:
        raise ValueError("cannot compute metrics on empty arrays")
    _check_binary(y_true)
    y_pred = apply_threshold(y_prob, threshold)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "n_samples": int(y_true.size),
        "n_spam": int(y_true.sum()),
        "n_ham": int((1 - y_true).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f0_5": float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if both_classes else None,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def classification_report_dict(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, object]:
    return classification_report(
        y_true, y_pred, labels=[0, 1], target_names=list(LABEL_NAMES), output_dict=True, zero_division=0
    )


def find_optimal_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    beta: float = 0.5,
    grid: Sequence[float] = THRESHOLD_GRID,
) -> tuple[float, float]:
    """Pick the threshold on ``grid`` maximizing F-beta.

    beta < 1 weights precision more than recall, reflecting that a false positive
    (legitimate mail sent to spam) is usually costlier than a missed spam.
    Ties are broken by choosing the threshold closest to 0.5.
    Returns (threshold, best F-beta).
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    _check_binary(y_true)
    best_t, best_score = 0.5, -1.0
    for t in grid:
        score = fbeta_score(y_true, apply_threshold(y_prob, t), beta=beta, zero_division=0)
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
            best_t, best_score = float(t), float(score)
    return best_t, best_score


def threshold_table(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, float]]:
    """Precision/recall/F1/FPR and error counts at each threshold."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    thresholds = thresholds if thresholds is not None else [round(t, 2) for t in np.linspace(0.05, 0.95, 19)]
    rows = []
    for t in thresholds:
        y_pred = apply_threshold(y_prob, t)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "f0_5": float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
                "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
                "false_positives": int(fp),
                "false_negatives": int(fn),
            }
        )
    return rows


def _downsample(arrays: list[np.ndarray], max_points: int) -> list[list[float]]:
    n = len(arrays[0])
    if n <= max_points:
        idx = np.arange(n)
    else:
        idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    return [np.asarray(a)[idx].astype(float).round(6).tolist() for a in arrays]


def roc_points(y_true, y_prob, max_points: int = 300) -> dict[str, list[float]]:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fpr_s, tpr_s = _downsample([fpr, tpr], max_points)
    return {"fpr": fpr_s, "tpr": tpr_s}


def pr_points(y_true, y_prob, max_points: int = 300) -> dict[str, list[float]]:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    p_s, r_s = _downsample([precision, recall], max_points)
    return {"precision": p_s, "recall": r_s}


def calibration_points(y_true, y_prob, n_bins: int = 10) -> dict[str, list[float]]:
    """Reliability-diagram data: mean predicted vs observed spam rate per probability bin."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    mean_pred, frac_pos, counts = [], [], []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            mean_pred.append(float(y_prob[mask].mean()))
            frac_pos.append(float(y_true[mask].mean()))
            counts.append(int(mask.sum()))
    return {"mean_predicted": mean_pred, "fraction_positive": frac_pos, "count": counts}
