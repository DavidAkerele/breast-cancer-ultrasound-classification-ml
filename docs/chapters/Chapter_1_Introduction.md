# Chapter 1: Introduction

## 1.1 Background

Breast ultrasound is a non-ionising imaging modality used alongside clinical assessment and other imaging. Its appearance varies with equipment, acquisition settings, anatomy, and operator technique. These differences, together with speckle and limited public sample sizes, make automated image classification a useful but difficult research problem.

Convolutional neural networks can learn image representations directly from pixels, while transfer learning can reduce the amount of task-specific data required. Nevertheless, a polished model result is not automatically a valid generalisation estimate. Related images from the same subject must remain in one partition, and the complete route from preprocessing to reported tables must be reproducible.

This dissertation develops a binary breast-ultrasound classification prototype with shared preprocessing, three CNN backbones, auditable evaluation outputs, and a local web interface. It is research software, not a diagnostic or clinical decision-support system.

## 1.2 Research problem

The project addresses three connected issues:

1. **Pipeline consistency:** training, evaluation, command-line prediction, and API inference must apply the same preprocessing rules.
2. **Geometric preprocessing:** direct rectangular-to-square resize changes apparent aspect ratios; padding permits isotropic resize, but any predictive benefit must be measured.
3. **Evidence integrity:** subject identifiers, split manifests, seeds, configuration, checkpoints, and per-image predictions are necessary to support a defensible result.

The supplied OASBUD and BrEaST folders have now been rebuilt into subject-level splits with no cross-partition identifiers. The renamed BUSI files still do not preserve enough provenance to verify patient separation, so BUSI is excluded from validated training and evaluation. Current metrics are local-cohort research evidence only.

## 1.3 Research questions

- **RQ1:** Can one testable preprocessing and inference path support every executable project surface?
- **RQ2:** How should CLAHE and square padding be compared with simpler preprocessing under controlled partitions, seeds, and training budgets?
- **RQ3:** Which conclusions are justified by the present data audit, and what controls are required for a patient-independent evaluation?

## 1.4 Objectives and contributions

The implementation provides deterministic split and audit utilities, shared CLAHE and crop strategies, EfficientNet-B0, ResNet-50 and custom-CNN options, checkpoint metadata, per-image predictions, generated metrics and plots, regression tests, and a research-only FastAPI interface. The written and visual artifacts use the same current evidence status and withdraw unsupported historical benchmarks.

## 1.5 Scope

The configured task maps images to `benign` or `malignant`; images stored under `normal` are currently included in the non-malignant class. This is a modelling simplification. The project does not assign BI-RADS, recommend management, or establish clinical efficacy. No pseudo-labelling, preprocessing ablation, multi-seed comparison, external validation, or prospective reader study is claimed as completed.
