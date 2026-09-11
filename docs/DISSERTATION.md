# Breast Ultrasound Classification Using Machine Learning

## Project overview

This project implements a reproducible research prototype for classifying breast ultrasound images as benign or malignant. It combines subject-level dataset auditing, shared preprocessing, three convolutional neural networks, evaluation outputs and a local web dashboard. The repaired BrEaST and OASBUD partitions pass the subject-level audit. Results describe this local cohort and do not establish clinical effectiveness.

## Aim

The project investigates a reproducible workflow for binary classification of B-mode breast-ultrasound images. It compares convolutional backbones and makes preprocessing explicit: CLAHE, aspect-ratio-preserving padding, data augmentation, normalisation, and optional lesion-mask cropping.

## Implementation

The PyTorch pipeline supports EfficientNet-B0, ResNet-50, and a small custom CNN. Training, evaluation, command-line prediction, and FastAPI inference share the same preprocessing code. The evaluator writes `outputs/metrics.json`, `outputs/predictions.csv`, a confusion matrix, and a ROC curve; a table utility reads that metrics file rather than constructing results from fixed target values. `scripts/audit_data.py` audits local split identifiers without changing data, and training refuses a failed audit unless a provisional override is explicit.

## Current evidence

The validated loader uses repaired subject-level BrEaST and OASBUD splits with 302 training, 63 validation, and 62 test images. The audit passes for those datasets with no cross-partition subjects. BUSI is excluded because its renamed files do not retain source identifiers. The current EfficientNet-B0 checkpoint achieved 67.74% accuracy, macro F1 0.6774, and ROC-AUC 0.7419 on the held-out test partition. These are local-cohort research results, not clinical validation.

## Limitations and future evaluation

The current comparison uses 62 held-out images and one training seed. Further evaluation should use a frozen subject-level manifest, repeated training seeds, controlled preprocessing ablations, validation-fitted calibration and an independent external cohort. BUSI remains outside the validated experiment because the local derivative lacks verifiable original case identity. The current loader groups normal images with the non-malignant class, a modelling simplification that requires review for any clinical study.

## Scope and safety

The interface is for research and education. It does not provide a diagnosis, BI-RADS assessment, or clinical decision support. Confidence values and visualisations are model outputs, not clinical certainty.

See [Evaluation results](RESULTS_STATUS.md) for the measured results and evidence limits.
