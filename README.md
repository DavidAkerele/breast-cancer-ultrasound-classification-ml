# Breast Ultrasound Classification Dissertation

This repository contains the software and dissertation sources for an MSc research project on binary breast-ultrasound image classification. It is a research prototype, not a clinical device and must not be used for diagnosis or patient-management decisions.

## Current evidence status

The repository is being prepared for reproducible evaluation. The current data audit identifies cross-split subjects in OASBUD and BrEaST and cannot verify patient-level separation in the renamed BUSI copy. Consequently, the bundled checkpoint's latest test result (**78.33% accuracy; ROC-AUC 0.8612; 120 images**) is a **provisional engineering result**, not a valid patient-independent clinical performance estimate.

Do not cite earlier headline values from superseded project assets. Submission-facing code, documentation, notebook, website, Word document, and slides now use only the generated provisional record above; withdrawn visual assets are isolated under `docs/archive/` with an explicit warning.

## Repository layout

- `src/` — PyTorch data, model, train, evaluate, prediction, and FastAPI modules.
- `scripts/audit_data.py` — read-only split and subject-identifier audit.
- `scripts/data_split.py` — deterministic, copy-only subject-level splitter with a JSON manifest.
- `tests/` — fast regression checks for preprocessing and evidence boundaries.
- `data/` — local dataset copies; excluded from Git because they are large and may have licence restrictions.
- `outputs/` — local checkpoints and generated metrics/figures; excluded from Git.
- `latex/` — canonical dissertation source.
- `docs/` — supporting documentation and a results-status record.
- `web/` — static research-demo interface served by FastAPI.

## Submission artifacts

- `latex/main.pdf` — compiled 32-page canonical dissertation.
- `docs/presentation/Breast_Cancer_Ultrasound_Dissertation.docx` — editable Word companion.
- `docs/presentation/Breast_Cancer_Ultrasound_Dissertation_Presentation.pptx` — editable eight-slide viva deck with speaker notes.
- `Breast_Cancer_Ultrasound_Classification_Dissertation.ipynb` — executed and validated analysis notebook.

Rebuild the authored outputs after changing source material:

```bash
venv/bin/python scripts/build_notebook.py
venv/bin/python scripts/execute_notebook.py
venv/bin/python docs/presentation/build_docx.py
(cd latex && tectonic --keep-logs main.tex)
venv/bin/python docs/presentation/build_pptx.py  # requires Codex Desktop's bundled presentation runtime
```

## Reproducible workflow

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python scripts/audit_data.py --output outputs/data_audit.json
venv/bin/python evaluate.py --split test
venv/bin/python src/compute_project_tables.py
venv/bin/python -m unittest discover -s tests -v
venv/bin/python api.py
```

Training and evaluation share CLAHE, resize/crop strategy, normalisation, and paired-mask discovery (`_mask.png`, `_tumor.png`, or `_lesion_mask.png`). Evaluation writes `outputs/metrics.json`, `outputs/predictions.csv`, a confusion matrix, and an ROC curve. Tables must be generated from those files rather than manually entered.

Training refuses a failed data audit by default. For an explicitly provisional engineering run only, the override is:

```bash
venv/bin/python train.py --model efficientnet_b0 --epochs 25 --seed 42 --allow-unaudited-data
```

## Before submission

1. Rebuild the dataset at subject level, retaining all views and masks in one split per subject.
2. Preserve original BUSI identifiers in a manifest; the current renamed `sample_*` files cannot prove patient separation.
3. Retrain all models on the audited cohort with fixed seeds and save per-run configuration, checkpoint, and predictions.
4. Replace the provisional Chapter 4 result with generated artifacts from those audited runs.
5. Compile the LaTeX source and verify every thesis, notebook, DOCX, slide, and dashboard claim against the final metrics and predictions files.
