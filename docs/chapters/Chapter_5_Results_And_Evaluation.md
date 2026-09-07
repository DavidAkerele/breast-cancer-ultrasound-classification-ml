# Chapter 5: Results and Evaluation Status

The current project supports an end-to-end evaluation run but not a valid clinical-performance claim. The configured combined loader has 402 training, 107 validation, and 120 test images. The supplied checkpoint evaluated at 78.33% accuracy, macro F1 0.7811, and ROC-AUC 0.8612 on the current test partition.

These values are provisional because the split audit finds OASBUD subject `30nh` in training and validation and BrEaST cases `case140` and `case151` in multiple partitions. The local BUSI files were renamed, so their patient-level independence cannot be verified. Do not compare architectures, cite confidence intervals, or make clinical inferences from this cohort.

The corrected evaluator writes `outputs/metrics.json`, `confusion_matrix.png`, and `roc_curve.png`. All final tables and figures must be derived from per-run saved predictions after the cohort is reconstructed at subject level.
