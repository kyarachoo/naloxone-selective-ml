from __future__ import annotations

import numpy as np
import pandas as pd


def clinical_rule_score(df: pd.DataFrame) -> np.ndarray:
    """Simple transparent comparator, not a clinical guideline."""
    score = (
        3 * df["prior_overdose"].fillna(0).to_numpy()
        + 2 * df["oud_history"].fillna(0).to_numpy()
        + 1 * df["benzodiazepine_active"].fillna(0).to_numpy()
        + 1 * df["chronic_opioid_therapy"].fillna(0).to_numpy()
        + 1 * (df["discharge_opioid_mme"].fillna(0).to_numpy() >= 90)
        + 1 * (df["ed_visits_180d"].fillna(0).to_numpy() >= 2)
    )
    return score.astype(float)


def normalized_rule_probability(df: pd.DataFrame) -> np.ndarray:
    score = clinical_rule_score(df)
    return 1.0 / (1.0 + np.exp(-(score - 3.0)))
