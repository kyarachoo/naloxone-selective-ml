from pathlib import Path

import pandas as pd
import pytest

from clinical_review.features import assert_no_leakage, audit_feature_availability, model_features, temporal_split
from clinical_review.io import load_feature_registry


def registry():
    return load_feature_registry(Path(__file__).resolve().parents[1] / "config" / "feature_registry.csv")


def test_model_features_exclude_target_and_treatment():
    features = model_features(registry())
    assert "opioid_event_30d" not in features
    assert "naloxone_at_discharge" not in features
    assert "post_discharge_refill_count_30d" not in features


def test_feature_availability_audit_flags_post_discharge():
    audit = audit_feature_availability(registry()).set_index("feature")
    assert bool(audit.loc["post_discharge_refill_count_30d", "leakage_risk"])
    assert not bool(audit.loc["age", "leakage_risk"])


def test_assert_no_leakage_accepts_registry_model_features():
    reg = registry()
    assert_no_leakage(reg, model_features(reg))


def test_assert_no_leakage_rejects_post_discharge_feature():
    reg = registry()
    with pytest.raises(ValueError):
        assert_no_leakage(reg, ["age", "post_discharge_refill_count_30d"])


def test_temporal_split_is_ordered_and_nonempty():
    df = pd.DataFrame(
        {
            "discharge_date": pd.to_datetime(["2024-01-01", "2025-06-01", "2026-03-01"]),
            "x": [1, 2, 3],
        }
    )
    train, val, test = temporal_split(df, "2024-12-31", "2025-12-31")
    assert len(train) == len(val) == len(test) == 1
    assert train["x"].iloc[0] == 1
    assert val["x"].iloc[0] == 2
    assert test["x"].iloc[0] == 3
