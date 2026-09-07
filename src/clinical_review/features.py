from __future__ import annotations

import pandas as pd

TARGET = "opioid_event_30d"


def model_features(registry: pd.DataFrame) -> list[str]:
    return registry.loc[registry["include_in_model"], "feature"].tolist()


def audit_feature_availability(registry: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    audit = registry.copy()
    audit["leakage_risk"] = (~audit["available_by_discharge"]) & (audit["feature"] != target)
    audit["model_feature_ok"] = audit["include_in_model"] & audit["available_by_discharge"]
    return audit


def assert_no_leakage(registry: pd.DataFrame, features: list[str]) -> None:
    lookup = registry.set_index("feature")
    unknown = [f for f in features if f not in lookup.index]
    if unknown:
        raise ValueError(f"Features missing from registry: {unknown}")

    unavailable = [f for f in features if not bool(lookup.loc[f, "available_by_discharge"])]
    if unavailable:
        raise ValueError(f"Post-discharge/leaky features supplied to model: {unavailable}")

    excluded = [f for f in features if not bool(lookup.loc[f, "include_in_model"])]
    if excluded:
        raise ValueError(f"Registry marks these features as excluded from modeling: {excluded}")


def feature_types(registry: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    lookup = registry.set_index("feature")
    numeric = [f for f in features if lookup.loc[f, "kind"] in {"numeric", "binary"}]
    categorical = [f for f in features if lookup.loc[f, "kind"] == "categorical"]
    return numeric, categorical


def temporal_split(
    df: pd.DataFrame,
    train_end: str,
    validation_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(df["discharge_date"])
    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)

    train = df.loc[dates <= train_end_ts].copy()
    validation = df.loc[(dates > train_end_ts) & (dates <= validation_end_ts)].copy()
    test = df.loc[dates > validation_end_ts].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Temporal split produced an empty partition")
    return train, validation, test
