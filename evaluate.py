import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tqdm import tqdm

import config
from dataset import get_dataloaders
from models import get_model

@torch.no_grad()
def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate Trained Breast Cancer Mammogram Model")
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

if __name__ == "__main__":
    evaluate()
