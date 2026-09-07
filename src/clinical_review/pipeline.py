from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .baseline import normalized_rule_probability
from .evaluation import (
    capacity_threshold,
    model_metrics,
    subgroup_metrics,
    threshold_metrics,
    utility_table,
)
from .features import (
    TARGET,
    assert_no_leakage,
    audit_feature_availability,
    feature_types,
    model_features,
    temporal_split,
)
from .io import load_feature_registry, load_settings, project_root
from .models import build_models, logistic_coefficients
from .review import assign_selective_status, build_review_queue
from .synth import generate_synth_encounters


def _ensure_dirs(root: Path) -> None:
    for rel in ["outputs", "models", "reports", "data/synth"]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def _model_selection_score(metrics: dict[str, float]) -> float:
    # Balance discrimination and calibration; lower Brier/ECE are better.
    return metrics["average_precision"] + 0.35 * metrics["auroc"] - 0.75 * metrics["brier"] - 0.40 * metrics["ece_10bin"]


def run_pipeline(root: Path | None = None) -> dict[str, object]:
    root = root or project_root()
    _ensure_dirs(root)
    settings = load_settings(root / "config" / "settings.json")
    registry = load_feature_registry(root / "config" / "feature_registry.csv")

    data_path = root / "data" / "synth" / "encounters.csv"
    if not data_path.exists():
        generate_synth_encounters(seed=settings["random_seed"], output_path=data_path)
    df = pd.read_csv(data_path, parse_dates=["discharge_date", "opioid_event_30d_date"])

    audit = audit_feature_availability(registry)
    audit.to_csv(root / "outputs" / "feature_availability_audit.csv", index=False)

    features = model_features(registry)
    assert_no_leakage(registry, features)
    numeric, categorical = feature_types(registry, features)

    train, validation, test = temporal_split(df, settings["train_end"], settings["validation_end"])
    X_train, y_train = train[features], train[TARGET].astype(int)
    X_val, y_val = validation[features], validation[TARGET].astype(int)
    X_test, y_test = test[features], test[TARGET].astype(int)

    comparison_rows: list[dict[str, object]] = []
    fitted: dict[str, object] = {}
    val_probs: dict[str, np.ndarray] = {}
    test_probs: dict[str, np.ndarray] = {}

    rule_val = normalized_rule_probability(validation)
    rule_test = normalized_rule_probability(test)
    rule_val_metrics = model_metrics(y_val.to_numpy(), rule_val)
    rule_test_metrics = model_metrics(y_test.to_numpy(), rule_test)
    comparison_rows.append({"model": "clinical_rule_baseline", "split": "validation", **rule_val_metrics})
    comparison_rows.append({"model": "clinical_rule_baseline", "split": "test", **rule_test_metrics})

    for bundle in build_models(numeric, categorical, seed=settings["random_seed"]):
        bundle.pipeline.fit(X_train, y_train)
        val_prob = bundle.pipeline.predict_proba(X_val)[:, 1]
        test_prob = bundle.pipeline.predict_proba(X_test)[:, 1]
        fitted[bundle.name] = bundle.pipeline
        val_probs[bundle.name] = val_prob
        test_probs[bundle.name] = test_prob
        comparison_rows.append({"model": bundle.name, "split": "validation", **model_metrics(y_val, val_prob)})
        comparison_rows.append({"model": bundle.name, "split": "test", **model_metrics(y_test, test_prob)})

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(root / "outputs" / "model_comparison.csv", index=False)

    validation_models = comparison.loc[
        (comparison["split"] == "validation") & (comparison["model"] != "clinical_rule_baseline")
    ].copy()
    validation_models["selection_score"] = validation_models.apply(
        lambda r: _model_selection_score(r.to_dict()), axis=1
    )
    selected_name = validation_models.sort_values("selection_score", ascending=False).iloc[0]["model"]
    alternate_name = [name for name in fitted if name != selected_name][0]
    selected_model = fitted[selected_name]

    joblib.dump(selected_model, root / "models" / "selected_model.joblib")
    joblib.dump(fitted[alternate_name], root / "models" / "alternate_model.joblib")

    # Capacity threshold is chosen only on the validation set.
    capacity = float(settings["review_capacity_fraction"])
    threshold = capacity_threshold(val_probs[selected_name], capacity)

    test_scored = test.copy()
    test_scored["prob_selected"] = test_probs[selected_name]
    test_scored["prob_alternate"] = test_probs[alternate_name]
    test_scored["prob_rule_baseline"] = rule_test

    selective = assign_selective_status(
        test_scored,
        selected_prob_col="prob_selected",
        alternate_prob_col="prob_alternate",
        threshold=threshold,
        critical_features=settings["critical_features"],
        disagreement_threshold=float(settings["model_disagreement_threshold"]),
        uncertainty_margin=float(settings["threshold_uncertainty_margin"]),
    )
    selective.to_csv(root / "outputs" / "scored_test_encounters.csv", index=False)

    queue = build_review_queue(selective, "prob_selected")
    queue_cols = [
        "queue_type",
        "encounter_id",
        "patient_id",
        "discharge_date",
        "site_id",
        "procedure_group",
        "prob_selected",
        "prob_alternate",
        "model_disagreement",
        "review_status",
        "review_reason",
        "case_drivers",
        "p1_unresolved_quality_flag",
        "critical_feature_missing",
        "naloxone_at_discharge",
        TARGET,
    ]
    queue[queue_cols].to_csv(root / "outputs" / "review_queue.csv", index=False)

    selected_test_metrics = model_metrics(y_test, test_probs[selected_name])
    threshold_stats = threshold_metrics(y_test, test_probs[selected_name], threshold)
    utility_cfg = settings["utility"]
    utilities = utility_table(
        y_test,
        test_probs[selected_name],
        tp_benefit=float(utility_cfg["true_positive_benefit"]),
        fp_cost=float(utility_cfg["false_positive_cost"]),
        fn_cost=float(utility_cfg["false_negative_cost"]),
    )
    utilities.to_csv(root / "outputs" / "capacity_utility.csv", index=False)

    subgroup_frame = selective.copy()
    subgroup_frame["predicted_risk"] = subgroup_frame["prob_selected"]
    subgroup = subgroup_metrics(
        subgroup_frame,
        y_col=TARGET,
        prob_col="predicted_risk",
        group_cols=["sex", "site_id", "procedure_group", "oud_history"],
    )
    subgroup.to_csv(root / "outputs" / "subgroup_metrics.csv", index=False)

    # Error-review artifact at the fixed validation-selected threshold.
    test_pred = test_probs[selected_name] >= threshold
    error_review = selective.copy()
    error_review["error_type"] = np.select(
        [
            (test_pred == 1) & (y_test.to_numpy() == 0),
            (test_pred == 0) & (y_test.to_numpy() == 1),
        ],
        ["FALSE_POSITIVE", "FALSE_NEGATIVE"],
        default="CORRECT",
    )
    error_review.loc[error_review["error_type"] != "CORRECT"].sort_values(
        "prob_selected", ascending=False
    ).to_csv(root / "outputs" / "error_review.csv", index=False)

    # Transparent logistic coefficients regardless of which model is selected.
    logistic = fitted["logistic_regression"]
    logistic_coefficients(logistic).to_csv(root / "outputs" / "logistic_coefficients.csv", index=False)

    # Model-agnostic test-set permutation importance for the selected model.
    pi = permutation_importance(
        selected_model,
        X_test,
        y_test,
        n_repeats=4,
        random_state=settings["random_seed"],
        scoring="average_precision",
    )
    importance = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": pi.importances_mean,
            "importance_std": pi.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(root / "outputs" / "permutation_importance.csv", index=False)

    summary = {
        "selected_model": selected_name,
        "alternate_model": alternate_name,
        "validation_selected_threshold": threshold,
        "review_capacity_fraction": capacity,
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "test_event_rate": float(y_test.mean()),
        "test_metrics": selected_test_metrics,
        "threshold_metrics": threshold_stats,
        "review_status_counts": selective["review_status"].value_counts().to_dict(),
        "priority_review_cases": int((selective["priority_flag"] == 1).sum()),
        "deferred_cases": int(selective["review_status"].str.startswith("DEFER").sum()),
    }
    with (root / "outputs" / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report = f"""# P2 Run Summary

- Selected model: **{selected_name}**
- Alternate model: **{alternate_name}**
- Temporal split sizes: train {len(train):,}, validation {len(validation):,}, test {len(test):,}
- Test event rate: {y_test.mean():.3f}
- Validation-selected review threshold: {threshold:.4f}
- Test AUROC: {selected_test_metrics['auroc']:.3f}
- Test average precision: {selected_test_metrics['average_precision']:.3f}
- Test Brier score: {selected_test_metrics['brier']:.3f}
- Priority review cases: {summary['priority_review_cases']}
- Deferred cases: {summary['deferred_cases']}

The model predicts a **post-discharge opioid-related event**, not existing naloxone prescribing.
Naloxone-at-discharge is excluded from model features and is used only in explicit review eligibility logic.
Cases with unresolved upstream data quality, missing critical data, strong model disagreement, or near-threshold uncertainty are deferred rather than silently ranked.
"""
    (root / "reports" / "run_summary.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    summary = run_pipeline()
    print("P2 pipeline completed successfully.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
