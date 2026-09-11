# Chapter 5: Results and Evaluation Status

The repaired validated loader uses BrEaST and OASBUD only, with 302 training, 63 validation, and 62 held-out test images. The fresh EfficientNet-B0 checkpoint achieved 67.74% accuracy, macro F1 0.6774, and ROC-AUC 0.7419 on that test partition.

The split audit passes for BrEaST and OASBUD with no subject crossing partitions. BUSI remains excluded because its local files were renamed and patient-level independence cannot be verified. The result is therefore a local-cohort research finding, not a clinical-performance estimate. Do not make clinical inferences or claim external generalisation.

The corrected evaluator writes `outputs/metrics.json`, `confusion_matrix.png`, and `roc_curve.png`. All final tables and figures must be derived from per-run saved predictions after the cohort is reconstructed at subject level.

The revised results chapter also includes four image-level outcomes selected by a fixed rule from `outputs/predictions.csv`: the lowest-probability correct benign case, the highest-probability correct malignant case, the highest-probability false positive, and the lowest-probability false negative. The plate uses the same ROI crop and CLAHE path as evaluation. It supports transparent error review without claiming that visual inspection reveals why the network made a decision.

See [`latex_university/figures/fig10_prediction_examples.png`](../../latex_university/figures/fig10_prediction_examples.png) for the generated plate.
