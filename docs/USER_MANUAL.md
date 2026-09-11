# Research Prototype User Manual

## Purpose

The application is a local research demonstration for binary breast-ultrasound image classification. It is not a medical device and must not be used for diagnosis, triage, BI-RADS assignment, or treatment decisions.

## Start the application

```bash
venv/bin/python api.py
```

Open `http://localhost:8000`. The static interface is served by FastAPI; there is no separate Vite or React application in this repository.

## Reproducible evaluation

Run the audit before any training or reporting:

```bash
venv/bin/python scripts/audit_data.py --datasets breast oasbud --output outputs/data_audit.json
venv/bin/python evaluate.py --split test
```

The evaluator creates `outputs/metrics.json`, `predictions.csv`, `confusion_matrix.png`, and `roc_curve.png`. Do not enter hand-calculated or illustrative values into reports. The validated run uses the repaired BrEaST and OASBUD subject-level splits. The local BUSI derivative remains excluded because its provenance cannot support the required patient-level audit.

Training stops after a failed audit. `--allow-unaudited-data` is available only for a clearly labelled engineering run; it does not make the resulting checkpoint valid for reporting.

## Input handling

The API accepts PNG, JPEG, and TIFF uploads up to 20 MB. It applies the selected preprocessing path and may produce a model confidence and saliency visualisation. These outputs are technical aids only. They do not identify a lesion, establish morphology, or replace a clinician.

Predictions below the configured confidence threshold are surfaced as `UNCERTAIN` rather than forced into a benign/malignant label. This is a safety-oriented abstention rule, not a validated uncertainty estimate; calibrate it on a larger held-out cohort before relying on it.

## Code map

- `src/dataset.py`: shared image preparation and data loader.
- `src/train.py`: training/checkpoint loop.
- `src/evaluate.py`: metrics and artifact generation.
- `src/predict.py`: local inference using shared preprocessing.
- `src/api.py`: research-demo API and static-file hosting.
- `scripts/audit_data.py`: non-destructive split audit.
