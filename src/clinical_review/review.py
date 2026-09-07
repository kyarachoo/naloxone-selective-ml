from __future__ import annotations

import numpy as np
import pandas as pd


HIGH_RISK_DRIVER_RULES: list[tuple[str, str]] = [
    ("prior_overdose", "prior overdose"),
    ("oud_history", "OUD history"),
    ("benzodiazepine_active", "active benzodiazepine"),
    ("chronic_opioid_therapy", "chronic opioid therapy"),
]


def eligibility_for_review(df: pd.DataFrame) -> pd.Series:
    return (
        df["discharge_opioid_mme"].fillna(0).gt(0)
        & df["hospice"].fillna(0).eq(0)
        & df["naloxone_at_discharge"].fillna(0).eq(0)
    )


def critical_missing_mask(df: pd.DataFrame, critical_features: list[str]) -> pd.Series:
    return df[critical_features].isna().any(axis=1)


def case_drivers(row: pd.Series) -> str:
    drivers: list[str] = []
    for feature, label in HIGH_RISK_DRIVER_RULES:
        if row.get(feature, 0) == 1:
            drivers.append(label)
    if pd.notna(row.get("discharge_opioid_mme")) and row["discharge_opioid_mme"] >= 90:
        drivers.append("high discharge opioid MME")
    if row.get("ed_visits_180d", 0) >= 2:
        drivers.append("multiple recent ED visits")
    if row.get("comorbidity_index", 0) >= 4:
        drivers.append("higher comorbidity burden")
    return "; ".join(drivers) if drivers else "no prespecified high-risk driver triggered"


def assign_selective_status(
    df: pd.DataFrame,
    selected_prob_col: str,
    alternate_prob_col: str,
    threshold: float,
    critical_features: list[str],
    disagreement_threshold: float = 0.20,
    uncertainty_margin: float = 0.02,
) -> pd.DataFrame:
    out = df.copy()
    out["eligible_for_review"] = eligibility_for_review(out)
    out["model_disagreement"] = (out[selected_prob_col] - out[alternate_prob_col]).abs()
    out["critical_feature_missing"] = critical_missing_mask(out, critical_features)
    out["review_status"] = "RANKED"
    out["review_reason"] = "Eligible case ranked by predicted post-discharge risk"

    not_eligible = ~out["eligible_for_review"]
    out.loc[not_eligible, "review_status"] = "NOT_ELIGIBLE"
    out.loc[not_eligible, "review_reason"] = "Already received naloxone, no discharge opioid, or hospice exclusion"

    quality = out["eligible_for_review"] & out["p1_unresolved_quality_flag"].eq(1)
    out.loc[quality, "review_status"] = "DEFER_DATA_QUALITY"
    out.loc[quality, "review_reason"] = "Unresolved upstream data-quality flag from P1"

    missing = out["eligible_for_review"] & out["critical_feature_missing"] & ~quality
    out.loc[missing, "review_status"] = "DEFER_MISSING_CRITICAL"
    out.loc[missing, "review_reason"] = "Critical discharge-time feature is missing"

    disagreement = (
        out["eligible_for_review"]
        & out["model_disagreement"].gt(disagreement_threshold)
        & ~quality
        & ~missing
    )
    out.loc[disagreement, "review_status"] = "DEFER_MODEL_DISAGREEMENT"
    out.loc[disagreement, "review_reason"] = "Simple and nonlinear models disagree beyond tolerance"

    near_threshold = (
        out["eligible_for_review"]
        & (out[selected_prob_col] - threshold).abs().le(uncertainty_margin)
        & ~quality
        & ~missing
        & ~disagreement
    )
    out.loc[near_threshold, "review_status"] = "DEFER_THRESHOLD_UNCERTAINTY"
    out.loc[near_threshold, "review_reason"] = "Risk estimate is too close to capacity threshold for automatic ranking"

    out["priority_flag"] = (
        out["review_status"].eq("RANKED") & out[selected_prob_col].ge(threshold)
    ).astype(int)
    out["case_drivers"] = out.apply(case_drivers, axis=1)
    return out


def build_review_queue(df: pd.DataFrame, selected_prob_col: str) -> pd.DataFrame:
    ranked = df.loc[df["review_status"].eq("RANKED") & df["priority_flag"].eq(1)].copy()
    ranked = ranked.sort_values(selected_prob_col, ascending=False)
    deferred = df.loc[df["review_status"].str.startswith("DEFER")].copy()
    deferred = deferred.sort_values(selected_prob_col, ascending=False)
    ranked["queue_type"] = "PRIORITY_REVIEW"
    deferred["queue_type"] = "DEFERRED_DIRECT_REVIEW"
    return pd.concat([deferred, ranked], ignore_index=True)
