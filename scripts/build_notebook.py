"""Build the executable, evidence-first dissertation notebook."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Breast_Cancer_Ultrasound_Classification_Dissertation.ipynb"


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


def create_notebook():
    cells = [
        markdown("""
# Breast Cancer Ultrasound Classification Using Machine Learning
## Auditable evaluation and research dashboard

This notebook is a compact, executable view of the repository evidence. It does not contain hand-entered benchmark values. The validated run uses repaired subject-level BrEaST and OASBUD splits. The local BUSI derivative is excluded because its files were renamed and its patient mapping and transformation history cannot be reconstructed. That prevents a reproducible check for patient separation, duplicate cases, label conflicts, overlays, and other published curation decisions. The exclusion applies to the local derivative, not to the original published BUSI dataset. Metrics are local cohort research evidence, not clinical performance estimates.
        """),
        markdown("""
## 1. Environment and project location

Run the notebook from the repository root after installing `requirements.txt`. The cell records basic runtime information without modifying the environment.
        """),
        code("""
from pathlib import Path
import json, platform, subprocess, sys

ROOT = Path.cwd()
assert (ROOT / "src").is_dir(), "Open this notebook from the dissertation repository root."
print({"python": sys.version.split()[0], "platform": platform.platform(), "project": ROOT.name})
        """),
        markdown("""
## 2. Audit data integrity

The audit derives subject identifiers where the local filename convention permits and records cross-partition subjects. The validated run audits BrEaST and OASBUD. BUSI is documented as an excluded local derivative because its original identifiers are unavailable.
        """),
        code("""
subprocess.run(
    [sys.executable, "scripts/audit_data.py", "--datasets", "breast", "oasbud", "--output", "outputs/data_audit.json"],
    check=True,
)
audit = json.loads((ROOT / "outputs/data_audit.json").read_text())
audit["audit_passed"], audit["failed_datasets"]
        """),
        code("""
import pandas as pd

rows = []
for dataset in audit["audited_datasets"]:
    for split, labels in audit[dataset]["counts"].items():
        rows.append({"dataset": dataset, "split": split, "images": sum(labels.values())})
pd.DataFrame(rows).pivot(index="dataset", columns="split", values="images").fillna(0).astype(int)
        """),
        markdown("""
## 3. Generate checkpoint evidence

`evaluate.py` uses the shared preprocessing implementation and writes per-image predictions before its aggregate summary. This cell can take several minutes on CPU.
        """),
        code("""
RUN_EVALUATION = False  # Set True to regenerate metrics and plots from the bundled checkpoint.
if RUN_EVALUATION:
    subprocess.run([sys.executable, "evaluate.py", "--split", "test"], check=True)
else:
    print("Using the committed local evidence files; set RUN_EVALUATION=True to regenerate them.")
        """),
        markdown("""
## 4. Inspect generated metrics and uncertainty

The primary values come from `outputs/metrics.json`. Additional threshold and calibration diagnostics come from the preserved per-image probabilities in `outputs/predictions.csv` and are stored in `outputs/extended_evaluation.json`. The audit status is shown alongside them so the evidence boundary remains attached to the headline numbers.
        """),
        code("""
metrics = json.loads((ROOT / "outputs/metrics.json").read_text())
summary = pd.DataFrame([metrics["summary"]], index=[metrics["model_name"]])
display(summary.style.format("{:.4f}"))
print({
    "samples": metrics["samples"],
    "split": metrics["split"],
    "evidence_status": metrics.get("evidence_status", "legacy provisional record"),
    "data_audit_passed": metrics.get("data_audit_passed", False),
})
        """),
        code("""
extended = json.loads((ROOT / "outputs/extended_evaluation.json").read_text())
benchmark = json.loads((ROOT / "outputs/model_benchmark.json").read_text())

overall = extended["overall"]
pd.Series({
    "malignant precision": overall["malignant_precision"],
    "malignant recall": overall["malignant_recall"],
    "specificity": overall["specificity"],
    "ROC-AUC": overall["roc_auc"],
    "average precision": overall["average_precision"],
    "Brier score": overall["brier_score"],
    "expected calibration error": overall["expected_calibration_error"],
}).to_frame("EfficientNet-B0").style.format("{:.4f}")
        """),
        code("""
comparison_rows = []
for record in benchmark["records"]:
    result = record["summary"]
    comparison_rows.append({
        "model": record["model_name"],
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "roc_auc": result["roc_auc"],
        "accuracy_95_ci": tuple(result["bootstrap_95_ci"]["accuracy"]),
        "roc_auc_95_ci": tuple(result["bootstrap_95_ci"]["roc_auc"]),
    })
pd.DataFrame(comparison_rows).set_index("model")
        """),
        code("""
from IPython.display import Image, display

display(Image(filename="outputs/confusion_matrix.png", width=520))
display(Image(filename="outputs/roc_curve.png", width=520))
display(Image(filename="outputs/discrimination_calibration.png", width=900))
display(Image(filename="outputs/source_stratified_performance.png", width=760))
        """),
        markdown("""
## 5. Source-specific error analysis

The fixed 0.5 threshold produces eight false negatives and twelve false positives overall. BrEaST contributes two false positives and five false negatives. OASBUD contributes ten false positives and three false negatives. The subsets differ in source, acquisition, reconstruction, case mix, and sample size, so this descriptive analysis cannot identify a cause or rank dataset quality.
        """),
        code("""
source_rows = []
for source, result in extended["by_source"].items():
    tn, fp = result["confusion_matrix"][0]
    fn, tp = result["confusion_matrix"][1]
    source_rows.append({
        "source": source,
        "n": result["samples"],
        "accuracy": result["accuracy"],
        "malignant_precision": result["malignant_precision"],
        "malignant_recall": result["malignant_recall"],
        "specificity": result["specificity"],
        "roc_auc": result["roc_auc"],
        "false_positives": fp,
        "false_negatives": fn,
    })
pd.DataFrame(source_rows).set_index("source").style.format({
    "accuracy": "{:.4f}",
    "malignant_precision": "{:.4f}",
    "malignant_recall": "{:.4f}",
    "specificity": "{:.4f}",
    "roc_auc": "{:.4f}",
})
        """),
        markdown("""
## 6. Interpretation

The current run confirms that the checkpoint, data loader, preprocessing, evaluator, and artifact writers operate together on a patient-separated local cohort. It does **not** establish external or clinical generalisation. The local BUSI derivative remains intentionally excluded from every validated result.

EfficientNet-B0 has the strongest point estimates in the internal comparison, but all three bootstrap intervals overlap. The preserved benchmark contains per-model summaries rather than paired predictions, so a paired significance test cannot be reconstructed. The custom CNN exceeds ResNet-50 in this run, but that result does not establish a breakthrough or population-level superiority.

The existing checkpoints retain only their selected validation points, not complete epoch histories. The report therefore gives those points and does not invent learning curves. Future training runs now save an epoch-by-epoch JSON record. Multi-seed training, expert review of errors, and external validation remain future work and were not performed for this revision.
        """),
        markdown("""
## Submission checklist

- [x] Dataset versions, exclusions, and label mapping recorded
- [x] Audited subjects remain within one partition for BrEaST and OASBUD
- [x] BUSI derivative excluded from validated training and evaluation
- [x] Seeds, configuration, and checkpoint hashes recorded where available
- [x] Per-image EfficientNet-B0 test predictions retained
- [x] Uncertainty, calibration, error cases, and source limitations reported
- [x] LaTeX, slides, README, notebook, and website aligned with the same evidence
- [ ] Ethics reference confirmed by the student before submission
        """),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {NOTEBOOK}")


if __name__ == "__main__":
    create_notebook()
