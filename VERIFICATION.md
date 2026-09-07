# Verification

This repository was verified as a complete working baseline before packaging.

## Test suite

Command:

```bash
pytest -q
```

Result:

```text
16 passed
```

## End-to-end pipeline

Command:

```bash
p2-review
```

Result: pipeline completed successfully and regenerated all derived outputs.

Key deterministic run results with the included seed/configuration:

- Train encounters: 4,470
- Validation encounters: 2,116
- Test encounters: 1,414
- Test outcome rate: 12.4%
- Validation-selected model: logistic regression
- Nonlinear comparator: histogram gradient boosting
- Test AUROC: 0.774
- Test average precision: 0.415
- Test Brier score: 0.090
- 10-bin expected calibration error: 0.021
- Validation-selected review threshold: 0.2744
- Test precision at that threshold: 0.504
- Test recall at that threshold: 0.383
- Test fraction above threshold: 9.4%
- Priority review cases after eligibility/deferral logic: 52
- Deferred cases: 92

The logistic model outperformed the nonlinear model on the validation selection criterion, so the pipeline intentionally retained the simpler model rather than selecting the more complex model by default.

## Model comparison on included data

```text
clinical rule baseline  test AUROC 0.731, AP 0.300
logistic regression     test AUROC 0.774, AP 0.415
hist gradient boosting  test AUROC 0.752, AP 0.351
```

## Selective-review behavior

The included test set routes cases into ranked, not-eligible, and explicit defer states. Deferral is triggered by unresolved upstream P1 quality flags, missing critical data, substantial model disagreement, or near-threshold uncertainty.

No real patient data are included. The dataset is privacy-safe synthetic/semi-synthetic data created solely for portfolio and learning use.
