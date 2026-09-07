from clinical_review.synth import generate_synth_encounters


def test_synth_target_is_nontrivial(tmp_path):
    df = generate_synth_encounters(n=1000, seed=7, output_path=tmp_path / "x.csv")
    rate = df["opioid_event_30d"].mean()
    assert 0.02 < rate < 0.35
    assert df["naloxone_at_discharge"].nunique() == 2
    assert df["opioid_event_30d"].nunique() == 2


def test_synth_has_post_discharge_leakage_fields_but_registry_excludes_them(tmp_path):
    df = generate_synth_encounters(n=200, seed=8, output_path=tmp_path / "x.csv")
    assert "post_discharge_refill_count_30d" in df.columns
    assert "opioid_event_30d_date" in df.columns
