# Preprocessing and Noise Analysis Protocol

## Purpose

This document specifies how to evaluate image resizing, cropping, CLAHE, and synthetic noise using saved predictions and recorded configurations. No completed ablation is claimed. The only current measured checkpoint result is recorded in `outputs/metrics.json` and is limited to the repaired local cohort; it is not clinical validation.

## Preprocessing strategies

The implementation exposes three mutually exclusive geometric strategies:

1. `direct_resize`: resize the full rectangular frame to $224\times224$. This is simple, but potentially anisotropic.
2. `center_crop`: crop the largest centred square, then resize. The geometry is isotropic, but peripheral content can be discarded.
3. `roi_crop`: use a mask-derived crop when available, otherwise a documented central fallback. Add 20% context, reflection-pad to a square, then resize.

CLAHE is a separate boolean factor. It uses clip limit 2.0 and a $16\times16$ tile grid on LAB luminance.

## Required experiment matrix

Run each model under a predeclared, equal training budget for every planned preprocessing condition. At minimum:

| Factor | Levels |
|---|---|
| Architecture | EfficientNet-B0, ResNet-50, custom CNN |
| Geometry | direct resize, centre crop, ROI/square pad |
| CLAHE | disabled, enabled |
| Seed | predeclared repeated seeds |
| Perturbation | clean, speckle, Gaussian, impulse |

The data manifest, subject partition, train/validation/test membership, model-selection rule, and test set must remain fixed across comparisons.

## Synthetic perturbations

The API can apply seeded perturbations to an individual image for demonstration. A performance study must instead apply the exact same seeded perturbation grid to the frozen test set and store one row per image, model, condition, intensity, and seed. Synthetic perturbations approximate image changes; they are not equivalent to acquisition on another scanner or site.

## Evidence schema

Each run should save:

- repository revision and environment lock;
- dataset manifest and audit checksum;
- architecture, preprocessing, optimiser, seed, and epoch budget;
- selected checkpoint and model-selection criterion;
- per-image true label, predicted label, and class probabilities;
- exclusion and failure logs; and
- generated aggregate metrics with uncertainty intervals.

## Analysis rules

- Generate tables and figures from saved predictions only.
- Compare paired image-level outputs for matched conditions.
- Report central estimates with confidence intervals and the number of independent subjects.
- Separate validation used for selection from final test evaluation.
- Treat softmax confidence as a model score, not clinical certainty.
- Do not infer a pseudo-labelling threshold, robustness claim, or superiority ranking from interface demonstrations.

## Current status

The repaired folders contain subject-level OASBUD and BrEaST partitions with no cross-split subjects; BUSI identifiers remain insufficient for patient-level verification and is excluded from validated runs. The experiment matrix above is therefore a proposed analysis plan, not a completed ablation results section.
