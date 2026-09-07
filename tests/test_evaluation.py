import numpy as np

from clinical_review.evaluation import capacity_threshold, model_metrics, threshold_metrics, utility_table


def test_capacity_threshold_selects_upper_tail():
    prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    threshold = capacity_threshold(prob, 0.2)
    assert threshold >= 0.8


def test_metrics_are_bounded():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    m = model_metrics(y, p)
    assert 0 <= m["auroc"] <= 1
    assert 0 <= m["average_precision"] <= 1
    assert 0 <= m["brier"] <= 1


def test_threshold_metrics_return_review_fraction():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    m = threshold_metrics(y, p, 0.5)
    assert m["review_fraction"] == 0.5
    assert m["recall"] == 1.0


def test_utility_table_has_requested_capacities():
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1])
    p = np.linspace(0.05, 0.95, 10)
    table = utility_table(y, p, capacities=(0.1, 0.2))
    assert table["capacity_fraction"].tolist() == [0.1, 0.2]
