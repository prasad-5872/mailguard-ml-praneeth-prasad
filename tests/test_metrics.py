import numpy as np
import pytest

from src.metrics import (
    calibration_points,
    classification_report_dict,
    compute_classification_metrics,
    find_optimal_threshold,
    percentile,
    pr_points,
    roc_points,
    threshold_table,
)

Y = [0, 0, 1, 1]
P = [0.1, 0.6, 0.4, 0.9]


def test_metrics_known_values():
    m = compute_classification_metrics(Y, P, 0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["specificity"] == pytest.approx(0.5)
    assert m["false_positive_rate"] == pytest.approx(0.5)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["brier"] == pytest.approx(np.mean((np.array(P) - np.array(Y)) ** 2))
    assert (m["n_samples"], m["n_spam"], m["n_ham"]) == (4, 2, 2)


def test_metrics_threshold_changes_confusion():
    m = compute_classification_metrics(Y, P, 0.35)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
    assert m["recall"] == 1.0


def test_metrics_single_class_has_no_auc():
    m = compute_classification_metrics([1, 1], [0.2, 0.9], 0.5)
    assert m["roc_auc"] is None and m["pr_auc"] is None
    assert m["recall"] == 0.5


@pytest.mark.parametrize(
    "y,p",
    [([0, 1], [0.5]), ([], []), ([0, 2], [0.1, 0.2])],
)
def test_metrics_input_errors(y, p):
    with pytest.raises(ValueError):
        compute_classification_metrics(y, p)


def test_classification_report_has_both_classes():
    rep = classification_report_dict([0, 1, 1], [0, 1, 0])
    assert rep["spam"]["recall"] == pytest.approx(0.5)
    assert rep["ham"]["precision"] == pytest.approx(0.5)


def test_percentile():
    assert percentile(range(1, 101), 95) == pytest.approx(95.05)
    assert percentile([], 95) is None
    assert percentile([7.0], 50) == 7.0
    with pytest.raises(ValueError):
        percentile([1, 2], 101)


def test_find_optimal_threshold_separable():
    y = [0, 0, 0, 1, 1, 1]
    p = [0.05, 0.2, 0.3, 0.7, 0.8, 0.95]
    t, score = find_optimal_threshold(y, p, beta=1.0)
    assert score == 1.0
    assert 0.3 < t <= 0.7
    assert t == 0.5  # tie-break: closest to 0.5


def test_find_optimal_threshold_precision_weighted():
    # a ham at 0.55 is a false positive at 0.5; F0.5 prefers raising the threshold past it
    y = [0, 0, 1, 1, 1, 1]
    p = [0.1, 0.55, 0.52, 0.8, 0.9, 0.95]
    t, _ = find_optimal_threshold(y, p, beta=0.5)
    assert t > 0.55


def test_find_optimal_threshold_rejects_bad_beta():
    with pytest.raises(ValueError):
        find_optimal_threshold([0, 1], [0.1, 0.9], beta=0)


def test_threshold_table_monotone_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.4 + rng.random(200) * 0.6, 0, 1)
    rows = threshold_table(y, p)
    recalls = [r["recall"] for r in rows]
    assert recalls == sorted(recalls, reverse=True)
    assert rows[0]["threshold"] == 0.05 and rows[-1]["threshold"] == 0.95


def test_curve_points_are_downsampled():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 5000)
    p = rng.random(5000)
    roc = roc_points(y, p, max_points=100)
    pr = pr_points(y, p, max_points=100)
    assert len(roc["fpr"]) <= 100 and len(roc["fpr"]) == len(roc["tpr"])
    assert len(pr["precision"]) <= 100
    assert roc["fpr"][0] == 0.0 and roc["tpr"][-1] == 1.0


def test_calibration_points():
    cal = calibration_points([0, 0, 1, 1], [0.05, 0.15, 0.85, 0.95], n_bins=10)
    assert cal["count"] == [1, 1, 1, 1]
    assert cal["fraction_positive"] == [0.0, 0.0, 1.0, 1.0]
