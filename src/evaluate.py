import os
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from tqdm import tqdm

try:
    from src import config
    from src.dataset import get_dataloaders
    from src.models import get_model
except ImportError:
    import config
    from dataset import get_dataloaders
    from models import get_model

from scripts.audit_data import build_report

@torch.no_grad()
def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate Trained Breast Cancer Ultrasound Model")
    parser.add_argument("--checkpoint", type=str, default=config.CHECKPOINT_PATH, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to evaluate")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}. Please run train.py first.")

    print(f"Loading model checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE)
    model_name = checkpoint.get("model_name", "resnet50")

    model = get_model(model_name=model_name, num_classes=config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    model.eval()

    _, val_loader, test_loader, _ = get_dataloaders(batch_size=config.BATCH_SIZE)
    eval_loader = test_loader if args.split == "test" else val_loader

    if len(eval_loader.dataset) == 0:
        print(f"[WARNING] {args.split} dataset is empty. Falling back to val loader.")
        eval_loader = val_loader

    all_labels = []
    all_preds = []
    all_probs = []

    print(f"Evaluating on {len(eval_loader.dataset)} samples...")
    for images, labels in tqdm(eval_loader, desc="Evaluating"):
        images = images.to(config.DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        # Store positive class probability for binary classification ROC
        if config.NUM_CLASSES == 2:
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    summary = {
        "accuracy": round(float(accuracy_score(all_labels, all_preds)), 4),
        "macro_f1": round(float(f1_score(all_labels, all_preds, average="macro", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(all_labels, all_preds, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(all_labels, all_preds, average="macro", zero_division=0)), 4),
    }

    # Print Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=config.CLASS_NAMES, zero_division=0))

    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    plt.title(f"Confusion Matrix ({model_name})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix to: {cm_path}")

    # Plot ROC Curve if binary classification
    if config.NUM_CLASSES == 2 and len(all_probs) > 0:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        summary["roc_auc"] = round(float(roc_auc), 4)
        print(f"ROC AUC Score: {roc_auc:.4f}")

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Receiver Operating Characteristic - {model_name}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(config.OUTPUT_DIR, "roc_curve.png")
        plt.savefig(roc_path, dpi=300)
        plt.close()
        print(f"Saved ROC Curve to: {roc_path}")

    audit_report = build_report(Path(config.DATA_DIR))
    audit_text = json.dumps(audit_report, indent=2, sort_keys=True) + "\n"
    audit_path = os.path.join(config.OUTPUT_DIR, "data_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write(audit_text)

    predictions_path = os.path.join(config.OUTPUT_DIR, "predictions.csv")
    sample_paths = [sample[0] for sample in eval_loader.dataset.samples]
    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "true_label", "predicted_label", "malignant_probability"])
        for path, true_idx, pred_idx, probability in zip(sample_paths, all_labels, all_preds, all_probs):
            writer.writerow([
                os.path.relpath(path, config.BASE_DIR),
                config.CLASS_NAMES[int(true_idx)],
                config.CLASS_NAMES[int(pred_idx)],
                f"{float(probability):.8f}",
            ])

    metrics_path = os.path.join(config.OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "checkpoint": os.path.relpath(args.checkpoint, config.BASE_DIR),
            "split": args.split,
            "samples": int(len(all_labels)),
            "class_names": config.CLASS_NAMES,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_audit_passed": audit_report["audit_passed"],
            "evidence_status": "validated" if audit_report["audit_passed"] else "provisional",
            "data_audit_sha256": hashlib.sha256(audit_text.encode("utf-8")).hexdigest(),
            "preprocessing": {
                "use_clahe": config.USE_CLAHE,
                "crop_strategy": config.CROP_STRATEGY,
                "image_size": config.IMG_SIZE,
            },
            "summary": summary,
            "confusion_matrix": cm.tolist(),
            "predictions_file": os.path.relpath(predictions_path, config.BASE_DIR),
        }, f, indent=2)
    print(f"Saved machine-readable metrics to: {metrics_path}")
    print(f"Saved per-sample predictions to: {predictions_path}")

if __name__ == "__main__":
    evaluate()
