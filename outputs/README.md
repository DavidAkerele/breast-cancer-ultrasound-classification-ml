# Stored evidence record

This directory contains the small, machine-readable outputs used by the current
dissertation narrative. The record is derived from the audited BrEaST and OASBUD
test partition and is retained so that a reviewer can inspect the values behind
the tables and figures without downloading the image datasets or checkpoints.

## Included files

- `metrics.json` - primary EfficientNet-B0 test metrics and bootstrap intervals.
- `model_benchmark.json` - same-split internal comparison with Custom CNN and ResNet-50.
- `extended_evaluation.json` - precision, recall, average precision, Brier score,
  calibration, and source-stratified diagnostics.
- `predictions.csv` - one row per held-out image with the stored malignant probability.
- `confusion_matrix.png`, `roc_curve.png`, `discrimination_calibration.png`, and
  `source_stratified_performance.png` - figures regenerated from the stored record.

The checkpoint paths in the JSON files are provenance labels. The binary
checkpoints and source images are intentionally excluded from GitHub. Re-run the
audit before treating a newly produced output as validated evidence, and do not
interpret these local-cohort results as clinical performance.
