"""Generate submission figures from stored audit and evaluation artefacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import find_mask_path, preprocess_image  # noqa: E402


OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "latex_university" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 160})


def load_json(name: str):
    return json.loads((OUTPUTS / name).read_text())


def save(fig, name: str, evidence_name: str | None = None):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight")
    if evidence_name:
        fig.savefig(OUTPUTS / evidence_name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def dataset_composition():
    audit = load_json("data_audit.json")
    datasets = audit["audited_datasets"]
    splits = ["train", "val", "test"]
    labels = ["benign", "malignant", "normal"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(datasets) * len(splits))
    width = 0.24
    for i, label in enumerate(labels):
        values = []
        for dataset in datasets:
            for split in splits:
                values.append(audit[dataset]["counts"].get(split, {}).get(label, 0))
        ax.bar(x + (i - 1) * width, values, width, label=label.capitalize())
    names = [f"{d}\n{s}" for d in datasets for s in splits]
    ax.set_xticks(x, names)
    ax.set_ylabel("Images")
    ax.set_title("Audited image composition by dataset and split")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig4_dataset_composition.png")


def model_benchmark():
    benchmark = load_json("model_benchmark.json")["records"]
    names = [r["model_name"].replace("_", " ").title() for r in benchmark]
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("roc_auc", "ROC-AUC")]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(names))
    width = 0.24
    for i, (key, label) in enumerate(metrics):
        ax.bar(x + (i - 1) * width, [r["summary"][key] for r in benchmark], width, label=label)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Internal architecture comparison on the audited test split")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig5_model_benchmark.png")


def uncertainty_intervals():
    benchmark = load_json("model_benchmark.json")["records"]
    names = [r["model_name"].replace("_", " ").title() for r in benchmark]
    point = np.array([r["summary"]["accuracy"] for r in benchmark])
    intervals = np.array([r["summary"]["bootstrap_95_ci"]["accuracy"] for r in benchmark])
    lower = point - intervals[:, 0]
    upper = intervals[:, 1] - point
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(names))
    ax.errorbar(point, y, xerr=[lower, upper], fmt="o", capsize=4, color="#155e75", ecolor="#155e75")
    ax.set_yticks(y, names)
    ax.set_xlim(0.35, 0.85)
    ax.set_xlabel("Accuracy with bootstrap 95% interval")
    ax.set_title("Uncertainty around the internal model comparison")
    ax.grid(axis="x", alpha=0.25)
    save(fig, "fig6_accuracy_intervals.png")


def pipeline_overview():
    fig, ax = plt.subplots(figsize=(10.5, 2.8))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 2.8)
    ax.axis("off")
    boxes = [
        (0.2, "Source\nimage"),
        (1.8, "Audit +\nsubject split"),
        (3.4, "Mask / ROI\ndiscovery"),
        (5.0, "CLAHE +\nsquare padding"),
        (6.8, "CNN\ncheckpoint"),
        (8.4, "Metrics +\npredictions"),
        (9.7, "Dashboard\nresearch output"),
    ]
    for x, label in boxes:
        patch = FancyBboxPatch((x, 1.0), 1.15, 0.8, boxstyle="round,pad=0.04,rounding_size=0.06",
                               linewidth=1.2, edgecolor="#155e75", facecolor="#e0f2fe")
        ax.add_patch(patch)
        ax.text(x + 0.575, 1.4, label, ha="center", va="center", fontsize=8)
    for (x1, _), (x2, _) in zip(boxes, boxes[1:]):
        ax.add_patch(FancyArrowPatch((x1 + 1.15, 1.4), (x2, 1.4), arrowstyle="-|>",
                                     mutation_scale=12, linewidth=1.1, color="#334155"))
    ax.text(5.25, 2.25, "Shared implementation path used by training, evaluation, CLI prediction, and API inference",
            ha="center", va="center", fontsize=10, fontweight="bold", color="#0f172a")
    save(fig, "fig7_pipeline_overview.png")


def architecture_overview():
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.8), sharey=True)
    specs = [
        ("EfficientNet-B0", ["Input", "MBConv blocks", "Squeeze /\nexcitation", "Dropout", "2-class output"], "#dbeafe"),
        ("ResNet-50", ["Input", "Residual\nstages", "Skip\nconnections", "Dropout", "2-class output"], "#fef3c7"),
        ("Custom CNN", ["Input", "4 convolutional\nstages", "Batch norm +\nReLU + pooling", "Adaptive pool +\ndropout", "2-class output"], "#dcfce7"),
    ]
    for ax, (title, layers, colour) in zip(axes, specs):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(layers) + 0.5)
        ax.axis("off")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        for i, layer in enumerate(layers):
            y = len(layers) - i - 0.1
            patch = FancyBboxPatch((0.13, y - 0.55), 0.74, 0.58, boxstyle="round,pad=0.03,rounding_size=0.05",
                                   linewidth=1.0, edgecolor="#475569", facecolor=colour)
            ax.add_patch(patch)
            ax.text(0.5, y - 0.26, layer, ha="center", va="center", fontsize=7.5)
            if i < len(layers) - 1:
                ax.add_patch(FancyArrowPatch((0.5, y - 0.55), (0.5, y - 0.95), arrowstyle="-|>",
                                             mutation_scale=9, linewidth=0.8, color="#64748b"))
    fig.suptitle("Implementation-level model pathways used in the study", fontsize=11, fontweight="bold")
    save(fig, "fig8_architecture_overview.png")


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def dataset_quality_examples():
    """Compare the local dataset copies without stretching their aspect ratios."""
    samples = [
        ("BrEaST", "Benign", ROOT / "data/breast/test/benign/case054.png"),
        ("OASBUD", "Benign", ROOT / "data/oasbud/test/benign/22rw_view2.png"),
        ("Local BUSI derivative", "Benign", ROOT / "data/busi/test/benign/sample_10.png"),
        ("BrEaST", "Malignant", ROOT / "data/breast/test/malignant/case117.png"),
        ("OASBUD", "Malignant", ROOT / "data/oasbud/test/malignant/186MR_view1.png"),
        ("Local BUSI derivative", "Malignant", ROOT / "data/busi/test/malignant/sample_10.png"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.2), facecolor="white")
    for ax, (dataset, label, path) in zip(axes.flat, samples):
        image = _read_rgb(path)
        height, width = image.shape[:2]
        ax.imshow(image)
        ax.set_title(f"{dataset} | {label}\n{width} x {height} pixels", fontsize=9, pad=6)
        ax.set_facecolor("#111827")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#94a3b8")
            spine.set_linewidth(0.8)
    fig.suptitle("Local dataset appearance before the common model transformation", fontsize=12,
                 fontweight="bold", color="#0f172a")
    save(fig, "fig9_dataset_quality_examples.png")


def _prediction_rows():
    with (OUTPUTS / "predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["malignant_probability"] = float(row["malignant_probability"])
    return rows


def _prediction_arrays(rows):
    labels = np.array([1 if row["true_label"] == "malignant" else 0 for row in rows], dtype=int)
    predictions = np.array([1 if row["predicted_label"] == "malignant" else 0 for row in rows], dtype=int)
    probabilities = np.array([row["malignant_probability"] for row in rows], dtype=float)
    return labels, predictions, probabilities


def _source_name(sample: str) -> str:
    if "data/breast/" in sample:
        return "BrEaST"
    if "data/oasbud/" in sample:
        return "OASBUD"
    return "Other"


def _binary_summary(labels, predictions, probabilities):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "samples": int(len(labels)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "accuracy": float((tp + tn) / len(labels)),
        "malignant_precision": float(tp / (tp + fp)) if tp + fp else None,
        "malignant_recall": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }


def extended_evaluation():
    """Derive additional test diagnostics from the preserved primary predictions."""
    rows = _prediction_rows()
    labels, predictions, probabilities = _prediction_arrays(rows)
    overall = _binary_summary(labels, predictions, probabilities)
    overall["brier_score"] = float(brier_score_loss(labels, probabilities))
    overall["expected_calibration_error"] = load_json("metrics.json")["summary"]["expected_calibration_error"]

    sources = {}
    for source in ("BrEaST", "OASBUD"):
        source_rows = [row for row in rows if _source_name(row["sample"]) == source]
        source_labels, source_predictions, source_probabilities = _prediction_arrays(source_rows)
        sources[source] = _binary_summary(source_labels, source_predictions, source_probabilities)

    record = {
        "source": "outputs/predictions.csv",
        "threshold": 0.5,
        "positive_class": "malignant",
        "overall": overall,
        "by_source": sources,
        "interpretation": (
            "Post-hoc diagnostics for the stored EfficientNet-B0 test predictions. "
            "No model was retrained and no threshold was selected from the test set."
        ),
    }
    (OUTPUTS / "extended_evaluation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def discrimination_and_calibration():
    """Plot precision-recall and reliability views for the primary checkpoint."""
    rows = _prediction_rows()
    labels, _, probabilities = _prediction_arrays(rows)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    average_precision = average_precision_score(labels, probabilities)
    observed, predicted = calibration_curve(labels, probabilities, n_bins=5, strategy="uniform")
    prevalence = float(labels.mean())

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), facecolor="white")
    axes[0].plot(recall, precision, color="#0f766e", linewidth=2.2,
                 label=f"EfficientNet-B0 (AP = {average_precision:.3f})")
    axes[0].axhline(prevalence, color="#94a3b8", linestyle="--", linewidth=1.2,
                    label=f"Prevalence = {prevalence:.3f}")
    axes[0].set(xlabel="Malignant recall", ylabel="Malignant precision", xlim=(0, 1), ylim=(0, 1.02),
                title="Precision-recall curve")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    axes[0].grid(alpha=0.22)

    axes[1].plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.2,
                 label="Perfect calibration")
    axes[1].plot(predicted, observed, marker="o", color="#b45309", linewidth=2.2,
                 label="Five-bin estimate")
    axes[1].set(xlabel="Mean predicted malignant probability", ylabel="Observed malignant fraction",
                xlim=(0, 1), ylim=(0, 1), title="Reliability diagram")
    axes[1].legend(frameon=False, loc="upper left", fontsize=8)
    axes[1].grid(alpha=0.22)
    fig.suptitle("Threshold and calibration diagnostics from 62 stored test predictions",
                 fontsize=12, fontweight="bold", color="#0f172a")
    save(fig, "fig12_discrimination_calibration.png", "discrimination_calibration.png")


def source_stratified_performance():
    """Show source-specific performance without presenting it as external validation."""
    record = extended_evaluation()
    sources = ["BrEaST", "OASBUD"]
    series = [
        ("accuracy", "Accuracy"),
        ("malignant_recall", "Malignant recall"),
        ("specificity", "Specificity"),
        ("roc_auc", "ROC-AUC"),
    ]
    x = np.arange(len(sources))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.2, 4.5), facecolor="white")
    colours = ["#0f766e", "#2563eb", "#b45309", "#7c3aed"]
    for index, ((key, label), colour) in enumerate(zip(series, colours)):
        values = [record["by_source"][source][key] for source in sources]
        ax.bar(x + (index - 1.5) * width, values, width, label=label, color=colour)
    ax.set_xticks(x, [f"{source}\n(n = {record['by_source'][source]['samples']})" for source in sources])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Source-stratified EfficientNet-B0 test performance")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.22)
    save(fig, "fig13_source_stratified_performance.png", "source_stratified_performance.png")


def web_interface_tour():
    """Compose browser-verified dashboard views into one report plate."""
    screen_dir = ROOT / "docs" / "images" / "dashboard"
    screens = [
        ("A  Single-image inspection", screen_dir / "01_single_image.png"),
        ("B  Seeded perturbation", screen_dir / "02_noise_experiment.png"),
        ("C  Labelled batch", screen_dir / "03_batch_evaluation.png"),
        ("D  Evidence record", screen_dir / "04_evidence_record.png"),
        ("E  Dataset explorer", screen_dir / "05_dataset_explorer.png"),
        ("F  System guide", screen_dir / "06_documentation.png"),
    ]
    missing = [str(path) for _, path in screens if not path.exists()]
    if missing:
        print(f"Skipped web interface tour; missing captures: {', '.join(missing)}")
        return
    loaded_screens = []
    for title, path in screens:
        image = _read_rgb(path)
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        loaded_screens.append((title, image))
    cv2.imwrite(
        str(FIGURES / "fig11_web_interface.png"),
        cv2.cvtColor(loaded_screens[0][1], cv2.COLOR_RGB2BGR),
    )
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 10.0), facecolor="white")
    for ax, (title, image) in zip(axes.flat, loaded_screens):
        ax.imshow(image)
        ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", pad=5, color="#0f172a")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
            spine.set_linewidth(0.8)
    fig.suptitle("Implemented OncoVision research dashboard", fontsize=13,
                 fontweight="bold", color="#0f172a")
    save(fig, "fig14_web_interface_tour.png")


def _select_prediction_examples():
    rows = _prediction_rows()
    groups = {
        "Correct benign": [r for r in rows if r["true_label"] == "benign" and r["predicted_label"] == "benign"],
        "Correct malignant": [r for r in rows if r["true_label"] == "malignant" and r["predicted_label"] == "malignant"],
        "False positive": [r for r in rows if r["true_label"] == "benign" and r["predicted_label"] == "malignant"],
        "False negative": [r for r in rows if r["true_label"] == "malignant" and r["predicted_label"] == "benign"],
    }
    selectors = {
        "Correct benign": lambda values: min(values, key=lambda r: r["malignant_probability"]),
        "Correct malignant": lambda values: max(values, key=lambda r: r["malignant_probability"]),
        "False positive": lambda values: max(values, key=lambda r: r["malignant_probability"]),
        "False negative": lambda values: min(values, key=lambda r: r["malignant_probability"]),
    }
    return [(name, selectors[name](values)) for name, values in groups.items()]


def prediction_examples():
    """Show deterministic examples from the stored per-image prediction table."""
    selected = _select_prediction_examples()
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.0), facecolor="white")
    for ax, (category, row) in zip(axes.flat, selected):
        path = ROOT / row["sample"]
        image = _read_rgb(path)
        mask_path = find_mask_path(str(path))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
        processed = preprocess_image(image, use_clahe=True, crop_strategy="roi_crop", mask_np=mask)
        probability = row["malignant_probability"]
        correct = row["true_label"] == row["predicted_label"]
        ax.imshow(processed)
        ax.set_title(
            f"{category}\nTruth: {row['true_label'].capitalize()} | "
            f"Prediction: {row['predicted_label'].capitalize()} | p(malignant) = {probability:.3f}",
            fontsize=9,
            pad=6,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#15803d" if correct else "#b91c1c")
            spine.set_linewidth(2.4)
    fig.suptitle("EfficientNet-B0 image-level outcomes on the audited test split", fontsize=12,
                 fontweight="bold", color="#0f172a")
    save(fig, "fig10_prediction_examples.png")


if __name__ == "__main__":
    dataset_composition()
    model_benchmark()
    uncertainty_intervals()
    pipeline_overview()
    architecture_overview()
    dataset_quality_examples()
    prediction_examples()
    discrimination_and_calibration()
    source_stratified_performance()
    web_interface_tour()
    print(f"Generated figures in {FIGURES}")
