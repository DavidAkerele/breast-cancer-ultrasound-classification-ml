# Breast Ultrasound Classification Using Machine Learning

## Dissertation status

This Markdown document is a companion to the canonical LaTeX dissertation in `latex/`. It records the project’s evidence status accurately: the repository implements a research prototype, but its supplied local cohort has not passed subject-level split validation. It must not be described as a validated diagnostic system.

## Aim

The project investigates a reproducible workflow for binary classification of B-mode breast-ultrasound images. It compares convolutional backbones and makes preprocessing explicit: CLAHE, aspect-ratio-preserving padding, data augmentation, normalisation, and optional lesion-mask cropping.

## Implementation

The PyTorch pipeline supports EfficientNet-B0, ResNet-50, and a small custom CNN. Training, evaluation, command-line prediction, and FastAPI inference share the same preprocessing code. The evaluator writes `outputs/metrics.json`, `outputs/predictions.csv`, a confusion matrix, and a ROC curve; a table utility reads that metrics file rather than constructing results from fixed target values. `scripts/audit_data.py` audits local split identifiers without changing data, and training refuses a failed audit unless a provisional override is explicit.

## Current evidence

The combined loader currently contains 402 training, 107 validation, and 120 test images. The audit finds OASBUD subject `30nh` in training and validation, and BrEaST cases `case140` and `case151` in more than one partition. The renamed BUSI files do not retain source subject identifiers, so patient-level separation cannot be verified. A run of the bundled EfficientNet-B0 checkpoint on this unaudited test partition gave 78.33% accuracy, macro F1 0.7811, and ROC-AUC 0.8612. These are software-regression values only, not valid estimates of clinical generalisation.

## Required final experiment

1. Recreate a manifest that retains every original subject and view identifier.
2. Assign all images and annotations for a subject to one partition.
3. Freeze the test partition before model selection.
4. Train with saved seed/configuration/checkpoint metadata.
5. Save per-image predictions and calculate all tables, intervals, calibration, and figures from those files.
6. Update the thesis, DOCX, slides, notebook, README, and dashboard from the final artifacts.

## Scope and safety

The interface is for research and education. It does not provide a diagnosis, BI-RADS assessment, or clinical decision support. Confidence values and visualisations are model outputs, not clinical certainty.

See [RESULTS_STATUS.md](RESULTS_STATUS.md) for the audit record and `latex/chapters/chapter4.tex` for the dissertation’s formal results statement.
