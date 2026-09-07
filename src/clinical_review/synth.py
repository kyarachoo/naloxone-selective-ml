from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import project_root


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synth_encounters(
    n: int = 8000,
    seed: int = 42,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Generate a privacy-safe longitudinal encounter table with a known DGP.

    The target is a post-discharge opioid-related adverse event. Naloxone-at-discharge
    is generated separately and is intentionally NOT the prediction target.
    """
    rng = np.random.default_rng(seed)

    start = np.datetime64("2023-01-01")
    end = np.datetime64("2026-08-31")
    day_span = int((end - start).astype(int))
    discharge_offsets = rng.integers(0, day_span + 1, size=n)
    discharge_dates = pd.to_datetime(start + discharge_offsets.astype("timedelta64[D]"))

    age = np.clip(rng.normal(52, 16, n).round(), 18, 90).astype(int)
    sex = rng.choice(["F", "M"], p=[0.55, 0.45], size=n)
    site_id = rng.choice(["SITE_A", "SITE_B", "SITE_C", "SITE_D"], p=[0.30, 0.27, 0.23, 0.20], size=n)
    procedure_group = rng.choice(
        ["general", "orthopedic", "urologic", "gynecologic", "vascular"],
        p=[0.28, 0.26, 0.15, 0.18, 0.13],
        size=n,
    )

    comorbidity_index = np.clip(rng.poisson(2.0, n), 0, 10)
    oud_history = rng.binomial(1, sigmoid(-3.4 + 0.14 * comorbidity_index), size=n)
    prior_overdose = rng.binomial(1, sigmoid(-4.2 + 1.55 * oud_history + 0.08 * comorbidity_index), size=n)
    chronic_opioid_therapy = rng.binomial(1, sigmoid(-2.2 + 1.15 * oud_history + 0.08 * comorbidity_index), size=n)
    benzodiazepine_active = rng.binomial(1, sigmoid(-2.4 + 0.55 * chronic_opioid_therapy + 0.04 * (age - 50)), size=n)

    ed_visits_180d = np.clip(
        rng.poisson(0.5 + 0.8 * oud_history + 0.55 * prior_overdose + 0.12 * comorbidity_index),
        0,
        12,
    )
    inpatient_days_365d = np.clip(
        rng.poisson(1.0 + 0.35 * comorbidity_index + 1.0 * prior_overdose),
        0,
        30,
    )

    procedure_mme = {
        "general": 45,
        "orthopedic": 70,
        "urologic": 35,
        "gynecologic": 40,
        "vascular": 55,
    }
    base_mme = np.array([procedure_mme[x] for x in procedure_group], dtype=float)
    discharge_opioid_mme = np.clip(
        rng.gamma(shape=2.0, scale=base_mme / 2.0) + 28 * chronic_opioid_therapy,
        0,
        300,
    ).round(1)
    pain_score_discharge = np.clip(
        rng.normal(4.0 + 0.6 * chronic_opioid_therapy + discharge_opioid_mme / 180.0, 1.7, n),
        0,
        10,
    ).round(1)

    site_effect = pd.Series(site_id).map({"SITE_A": 0.0, "SITE_B": 0.18, "SITE_C": -0.10, "SITE_D": 0.08}).to_numpy()
    nonlinear_high_mme = (discharge_opioid_mme > 90).astype(float)
    interaction = prior_overdose * benzodiazepine_active

    logit_event = (
        -4.0
        + 2.00 * prior_overdose
        + 1.50 * oud_history
        + 1.00 * benzodiazepine_active
        + 0.80 * chronic_opioid_therapy
        + 0.012 * discharge_opioid_mme
        + 0.22 * ed_visits_180d
        + 0.10 * comorbidity_index
        + 0.70 * nonlinear_high_mme
        + 0.80 * interaction
        + site_effect
    )
    event_prob = sigmoid(logit_event)
    opioid_event_30d = rng.binomial(1, event_prob)

    event_days = rng.integers(1, 31, size=n)
    event_date = pd.Series(pd.NaT, index=np.arange(n), dtype="datetime64[ns]")
    event_date.loc[opioid_event_30d == 1] = discharge_dates[opioid_event_30d == 1] + pd.to_timedelta(
        event_days[opioid_event_30d == 1], unit="D"
    )

    post_discharge_refill_count_30d = np.clip(
        rng.poisson(0.25 + 0.85 * opioid_event_30d + 0.45 * chronic_opioid_therapy),
        0,
        8,
    )

    site_naloxone = pd.Series(site_id).map({"SITE_A": 0.25, "SITE_B": -0.10, "SITE_C": 0.05, "SITE_D": 0.35}).to_numpy()
    naloxone_prob = sigmoid(
        -2.6
        + 1.30 * prior_overdose
        + 0.95 * oud_history
        + 0.55 * chronic_opioid_therapy
        + 0.008 * discharge_opioid_mme
        + 0.45 * benzodiazepine_active
        + site_naloxone
    )
    naloxone_at_discharge = rng.binomial(1, naloxone_prob)

    hospice = rng.binomial(1, 0.012, size=n)
    p1_unresolved_quality_flag = rng.binomial(1, 0.035, size=n)

    df = pd.DataFrame(
        {
            "encounter_id": [f"E{i:06d}" for i in range(1, n + 1)],
            "patient_id": [f"P{int(x):05d}" for x in rng.integers(1, 5001, size=n)],
            "discharge_date": discharge_dates,
            "age": age,
            "sex": sex,
            "site_id": site_id,
            "procedure_group": procedure_group,
            "comorbidity_index": comorbidity_index,
            "prior_overdose": prior_overdose,
            "oud_history": oud_history,
            "ed_visits_180d": ed_visits_180d,
            "inpatient_days_365d": inpatient_days_365d,
            "discharge_opioid_mme": discharge_opioid_mme,
            "benzodiazepine_active": benzodiazepine_active,
            "chronic_opioid_therapy": chronic_opioid_therapy,
            "pain_score_discharge": pain_score_discharge,
            "p1_unresolved_quality_flag": p1_unresolved_quality_flag,
            "naloxone_at_discharge": naloxone_at_discharge,
            "hospice": hospice,
            "opioid_event_30d": opioid_event_30d,
            "opioid_event_30d_date": event_date,
            "post_discharge_refill_count_30d": post_discharge_refill_count_30d,
            "true_event_probability": event_prob.round(6),
        }
    )

    # A small amount of realistic pre-discharge missingness.
    for col, frac in {
        "pain_score_discharge": 0.025,
        "inpatient_days_365d": 0.012,
        "discharge_opioid_mme": 0.008,
    }.items():
        idx = rng.choice(df.index, size=max(1, int(frac * n)), replace=False)
        df.loc[idx, col] = np.nan

    output_path = output_path or project_root() / "data" / "synth" / "encounters.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    frame = generate_synth_encounters()
    print(f"Wrote {len(frame):,} synth encounters")
    print(f"Event rate: {frame['opioid_event_30d'].mean():.3f}")
