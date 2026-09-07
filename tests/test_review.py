import pandas as pd

from clinical_review.review import assign_selective_status, build_review_queue


def sample():
    return pd.DataFrame(
        {
            "encounter_id": ["E1", "E2", "E3", "E4", "E5"],
            "patient_id": ["P1", "P2", "P3", "P4", "P5"],
            "discharge_date": pd.to_datetime(["2026-01-01"] * 5),
            "site_id": ["SITE_A"] * 5,
            "procedure_group": ["general"] * 5,
            "discharge_opioid_mme": [60, 60, 60, 60, 60],
            "prior_overdose": [1, 0, 0, 0, 0],
            "oud_history": [0, 0, 0, 0, 0],
            "benzodiazepine_active": [0, 0, 0, 0, 0],
            "chronic_opioid_therapy": [0, 0, 0, 0, 0],
            "ed_visits_180d": [0] * 5,
            "comorbidity_index": [1] * 5,
            "hospice": [0, 0, 0, 0, 0],
            "naloxone_at_discharge": [0, 0, 1, 0, 0],
            "p1_unresolved_quality_flag": [0, 1, 0, 0, 0],
            "prob_selected": [0.80, 0.70, 0.90, 0.51, 0.75],
            "prob_alternate": [0.78, 0.69, 0.89, 0.20, 0.74],
            "opioid_event_30d": [1, 0, 0, 1, 1],
        }
    )


def test_not_eligible_if_already_has_naloxone():
    out = assign_selective_status(sample(), "prob_selected", "prob_alternate", 0.6, ["discharge_opioid_mme", "prior_overdose", "oud_history", "benzodiazepine_active"])
    assert out.loc[out.encounter_id == "E3", "review_status"].iloc[0] == "NOT_ELIGIBLE"


def test_upstream_quality_flag_defers():
    out = assign_selective_status(sample(), "prob_selected", "prob_alternate", 0.6, ["discharge_opioid_mme", "prior_overdose", "oud_history", "benzodiazepine_active"])
    assert out.loc[out.encounter_id == "E2", "review_status"].iloc[0] == "DEFER_DATA_QUALITY"


def test_model_disagreement_defers():
    out = assign_selective_status(sample(), "prob_selected", "prob_alternate", 0.6, ["discharge_opioid_mme", "prior_overdose", "oud_history", "benzodiazepine_active"], disagreement_threshold=0.2)
    assert out.loc[out.encounter_id == "E4", "review_status"].iloc[0] == "DEFER_MODEL_DISAGREEMENT"


def test_ranked_high_risk_case_is_priority():
    out = assign_selective_status(sample(), "prob_selected", "prob_alternate", 0.6, ["discharge_opioid_mme", "prior_overdose", "oud_history", "benzodiazepine_active"])
    row = out.loc[out.encounter_id == "E1"].iloc[0]
    assert row["review_status"] == "RANKED"
    assert row["priority_flag"] == 1


def test_review_queue_contains_deferred_and_priority():
    out = assign_selective_status(sample(), "prob_selected", "prob_alternate", 0.6, ["discharge_opioid_mme", "prior_overdose", "oud_history", "benzodiazepine_active"])
    queue = build_review_queue(out, "prob_selected")
    assert {"PRIORITY_REVIEW", "DEFERRED_DIRECT_REVIEW"}.issubset(set(queue["queue_type"]))
