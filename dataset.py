import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

def apply_clahe(img_np):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to improve contrast of breast structures and lesion margins in ultrasound scans.
    """
    # If image is RGB/BGR, convert to LAB or process intensity channel
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID_SIZE)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return enhanced
    else:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID_SIZE)
        return clahe.apply(img_np)

class BreastUltrasoundDataset(Dataset):
    """
    Custom Dataset for Breast Cancer Ultrasound Scans.
    Supports directory structure where subfolders represent classes:
        data/train/benign/img1.png
        data/train/malignant/img2.png
    Or a list of (file_path, label_idx) tuples.
    """
    def __init__(self, root_dir=None, file_list=None, transform=None, use_clahe=True):
        self.transform = transform
        self.use_clahe = use_clahe
        self.samples = []

        if file_list is not None:
            self.samples = file_list
        elif root_dir is not None and os.path.exists(root_dir):
            for class_idx, class_name in enumerate(config.CLASS_NAMES):
                class_dir = os.path.join(root_dir, class_name)
                # Support common medical image extensions
                extensions = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp', '*.dcm')
                files = []
                
                if os.path.exists(class_dir):
                    for ext in extensions:
                        files.extend(glob.glob(os.path.join(class_dir, ext)))
                        
                # Map normal cases to benign (class_idx = 0)
                if class_name == "benign":
                    normal_dir = os.path.join(root_dir, "normal")
                    if os.path.exists(normal_dir):
                        for ext in extensions:
                            files.extend(glob.glob(os.path.join(normal_dir, ext)))
                            
                for filepath in sorted(files):
                    fname = os.path.basename(filepath)
                    # Filter out mask/tumor label files from training/validation scans
                    if not fname.lower().endswith(('_tumor.png', '_mask.png')):
                        self.samples.append((filepath, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]

        # Load image using OpenCV or PIL
        try:
            img_np = cv2.imread(filepath)
            if img_np is None:
                raise ValueError(f"Failed to read image at {filepath}")
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        except Exception as e:
            # Fallback to PIL
            img_pil = Image.open(filepath).convert("RGB")
            img_np = np.array(img_pil)

        # Apply CLAHE enhancement if enabled
        if self.use_clahe and config.USE_CLAHE:
            img_np = apply_clahe(img_np)

        # Convert numpy array back to PIL Image for torchvision transforms
        img = Image.fromarray(img_np)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

def get_transforms():
    """
    Returns standard training and validation data transformations.
    Ultrasound scans benefit from random flips and gentle rotations (invariance to orientation).
    Using Resize and CenterCrop ensures uniform square inputs across all datasets without distorting aspect ratios.
    """
    train_transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        # Standard ImageNet normalization values required for ResNet/EfficientNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def get_dataloaders(batch_size=config.BATCH_SIZE, num_workers=0, combine=True):
    """
    Creates DataLoaders for train, val, and test splits.
    Also computes class weights for loss balancing if dataset is imbalanced.
    """
    train_transform, val_transform = get_transforms()

    if combine:
        datasets_to_load = ["breast", "oasbud"]
        train_samples = []
        val_samples = []
        test_samples = []
        for ds_name in datasets_to_load:
            train_dir = os.path.join(config.DATA_DIR, ds_name, "train")
            val_dir = os.path.join(config.DATA_DIR, ds_name, "val")
            test_dir = os.path.join(config.DATA_DIR, ds_name, "test")
            
            if os.path.exists(train_dir):
                train_ds = BreastUltrasoundDataset(root_dir=train_dir, transform=train_transform, use_clahe=config.USE_CLAHE)
                train_samples.extend(train_ds.samples)
            if os.path.exists(val_dir):
                val_ds = BreastUltrasoundDataset(root_dir=val_dir, transform=val_transform, use_clahe=config.USE_CLAHE)
                val_samples.extend(val_ds.samples)
            if os.path.exists(test_dir):
                test_ds = BreastUltrasoundDataset(root_dir=test_dir, transform=val_transform, use_clahe=config.USE_CLAHE)
                test_samples.extend(test_ds.samples)
                
        train_dataset = BreastUltrasoundDataset(file_list=train_samples, transform=train_transform, use_clahe=config.USE_CLAHE)
        val_dataset = BreastUltrasoundDataset(file_list=val_samples, transform=val_transform, use_clahe=config.USE_CLAHE)
        test_dataset = BreastUltrasoundDataset(file_list=test_samples, transform=val_transform, use_clahe=config.USE_CLAHE)
        print(f"[Dataset] Combined loading: {len(train_samples)} training, {len(val_samples)} validation samples.")
    else:
        train_dataset = BreastUltrasoundDataset(root_dir=config.TRAIN_DIR, transform=train_transform, use_clahe=config.USE_CLAHE)
        val_dataset = BreastUltrasoundDataset(root_dir=config.VAL_DIR, transform=val_transform, use_clahe=config.USE_CLAHE)
        test_dataset = BreastUltrasoundDataset(root_dir=config.TEST_DIR, transform=val_transform, use_clahe=config.USE_CLAHE)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Compute class weights to handle imbalanced datasets (common in benign vs malignant datasets)
    class_counts = [0] * config.NUM_CLASSES
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    total_samples = sum(class_counts)
    if total_samples > 0 and all(c > 0 for c in class_counts):
        class_weights = [total_samples / (len(class_counts) * c) for c in class_counts]
        class_weights = torch.FloatTensor(class_weights).to(config.DEVICE)
    else:
        class_weights = None

    return train_loader, val_loader, test_loader, class_weights
