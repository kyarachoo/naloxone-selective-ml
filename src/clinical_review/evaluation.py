from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        if not mask.any():
            continue
        ece += (mask.sum() / total) * abs(prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def threshold_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (np.asarray(prob) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "review_fraction": float(pred.mean()),
    }


def model_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y_true, prob)),
        "average_precision": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "ece_10bin": expected_calibration_error(y_true, prob),
    }


def capacity_threshold(prob: np.ndarray, capacity_fraction: float) -> float:
    prob = np.asarray(prob)
    if not 0 < capacity_fraction < 1:
        raise ValueError("capacity_fraction must be between 0 and 1")
    return float(np.quantile(prob, 1.0 - capacity_fraction))


def utility_table(
    y_true: np.ndarray,
    prob: np.ndarray,
    capacities: tuple[float, ...] = (0.05, 0.10, 0.20),
    tp_benefit: float = 4.0,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
) -> pd.DataFrame:
    rows = []
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    for capacity in capacities:
        threshold = capacity_threshold(prob, capacity)
        pred = prob >= threshold
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        utility = tp_benefit * tp - fp_cost * fp - fn_cost * fn
        rows.append(
            {
                "capacity_fraction": capacity,
                "threshold": threshold,
                "reviewed": int(pred.sum()),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "utility": float(utility),
            }
        )
    return pd.DataFrame(rows)


def subgroup_metrics(df: pd.DataFrame, y_col: str, prob_col: str, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_col in group_cols:
        for group_value, group in df.groupby(group_col, dropna=False):
            if len(group) < 25 or group[y_col].nunique() < 2:
                continue
            metrics = model_metrics(group[y_col].to_numpy(), group[prob_col].to_numpy())
            rows.append(
                {
                    "group_column": group_col,
                    "group_value": str(group_value),
                    "n": len(group),
                    "event_rate": float(group[y_col].mean()),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)
