# P2 Run Summary

- Selected model: **logistic_regression**
- Alternate model: **hist_gradient_boosting**
- Temporal split sizes: train 4,470, validation 2,116, test 1,414
- Test event rate: 0.124
- Validation-selected review threshold: 0.2744
- Test AUROC: 0.774
- Test average precision: 0.415
- Test Brier score: 0.090
- Priority review cases: 51
- Deferred cases: 96

The model predicts a **post-discharge opioid-related event**, not existing naloxone prescribing.
Naloxone-at-discharge is excluded from model features and is used only in explicit review eligibility logic.
Cases with unresolved upstream data quality, missing critical data, strong model disagreement, or near-threshold uncertainty are deferred rather than silently ranked.
