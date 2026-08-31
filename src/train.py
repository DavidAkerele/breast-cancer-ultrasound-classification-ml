import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

try:
    from src import config
    from src.dataset import get_dataloaders
    from src.models import get_model
except ImportError:
    import config
    from dataset import get_dataloaders
    from models import get_model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Validation", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def main():
    parser = argparse.ArgumentParser(description="Train Breast Cancer Ultrasound Classifier")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0", "custom_cnn"],
                        help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--combine", action=argparse.BooleanOptionalAction, default=True,
                        help="Combine multiple datasets for training (use --no-combine for the active dataset only)")
    args = parser.parse_args()

    print(f"--- Starting Training Pipeline on Device: {config.DEVICE} ---")
    
    # Check if data exists
    if args.combine:
        breast_dir = os.path.join(config.DATA_DIR, "breast", "train")
        oasbud_dir = os.path.join(config.DATA_DIR, "oasbud", "train")
        if (not os.path.exists(breast_dir) or len(os.listdir(breast_dir)) == 0) and \
           (not os.path.exists(oasbud_dir) or len(os.listdir(oasbud_dir)) == 0):
            print("[WARNING] Combined training datasets (BrEaST / OASBUD) are empty or missing.")
    else:
        if not os.path.exists(config.TRAIN_DIR) or len(os.listdir(config.TRAIN_DIR)) == 0:
            print(f"[WARNING] Training directory {config.TRAIN_DIR} is empty or missing.")
            print("Running dummy data generator so you can verify the pipeline right away...")
            from scripts.create_dummy_data import create_dummy_dataset
            create_dummy_dataset()

    train_loader, val_loader, _, class_weights = get_dataloaders(batch_size=args.batch_size, combine=args.combine)
    print(f"Loaded {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples.")

    model = get_model(model_name=args.model, num_classes=config.NUM_CLASSES, pretrained=True)
    model = model.to(config.DEVICE)

    # Use class weights if dataset is imbalanced
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")

        # Checkpoint saving & Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'model_name': args.model,
                'class_names': config.CLASS_NAMES
            }
            torch.save(checkpoint_data, config.CHECKPOINT_PATH)
            if args.model in config.MODEL_CHECKPOINT_PATHS:
                torch.save(checkpoint_data, config.MODEL_CHECKPOINT_PATHS[args.model])
                print(f"[*] Best model saved to {config.CHECKPOINT_PATH} and {config.MODEL_CHECKPOINT_PATHS[args.model]}")
            else:
                print(f"[*] Best model saved to {config.CHECKPOINT_PATH}")
        else:
            patience_counter += 1
            print(f"[!] No improvement in val loss. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered to prevent overfitting.")
                break

    total_time = time.time() - start_time
    print(f"\n--- Training Finished in {total_time/60:.2f} minutes ---")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
