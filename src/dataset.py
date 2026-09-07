import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
try:
    from src import config
except ImportError:
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

def crop_and_pad_roi(img_np: np.ndarray, mask_np: np.ndarray = None, margin_ratio: float = getattr(config, 'ROI_MARGIN_RATIO', 0.20)) -> np.ndarray:
    """
    Crops around localized lesion ROI, applies safety boundary margin to preserve acoustic shadowing,
    pads the crop to a square aspect ratio to prevent geometric distortion, and resizes to target resolution.
    """
    h, w = img_np.shape[:2]
    
    if mask_np is not None and np.any(mask_np > 0):
        gray_mask = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY) if len(mask_np.shape) == 3 else mask_np
        y_indices, x_indices = np.where(gray_mask > 0)
        x1, x2 = np.min(x_indices), np.max(x_indices) + 1
        y1, y2 = np.min(y_indices), np.max(y_indices) + 1
    else:
        # For unmasked scans, restrict to the central 80% tissue region where masses reside
        h_start, h_end = int(h * 0.10), int(h * 0.90)
        w_start, w_end = int(w * 0.10), int(w * 0.90)
        x1, y1 = w_start, h_start
        x2, y2 = w_end, h_end

    bw, bh = max(5, x2 - x1), max(5, y2 - y1)
    
    # Add safety margin around ROI to preserve acoustic shadowing context
    margin_w = int(bw * margin_ratio)
    margin_h = int(bh * margin_ratio)
    
    x1 = max(0, x1 - margin_w)
    y1 = max(0, y1 - margin_h)
    x2 = min(w, x2 + margin_w)
    y2 = min(h, y2 + margin_h)
    
    # Pad crop to square aspect ratio (using reflection padding to avoid dark border artifacts)
    crop = img_np[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    max_dim = max(ch, cw)
    
    pad_top = (max_dim - ch) // 2
    pad_bottom = max_dim - ch - pad_top
    pad_left = (max_dim - cw) // 2
    pad_right = max_dim - cw - pad_left
    
    padded = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
    return cv2.resize(padded, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_CUBIC)


def find_mask_path(filepath: str):
    """Return a paired lesion mask from a supported dataset, if present."""
    stem, _ = os.path.splitext(filepath)
    for suffix in ("_mask.png", "_tumor.png", "_lesion_mask.png"):
        candidate = stem + suffix
        if os.path.exists(candidate):
            return candidate
    return None


def preprocess_image(img_np, use_clahe=True, crop_strategy=None, mask_np=None):
    """Shared preparation for training, evaluation, CLI, and API inference."""
    strategy = crop_strategy or config.CROP_STRATEGY
    if use_clahe and config.USE_CLAHE:
        img_np = apply_clahe(img_np)
    if strategy == "roi_crop":
        return crop_and_pad_roi(img_np, mask_np)
    if strategy == "direct_resize":
        return cv2.resize(img_np, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    if strategy == "center_crop":
        h, w = img_np.shape[:2]
        side = min(h, w)
        y, x = (h - side) // 2, (w - side) // 2
        return cv2.resize(img_np[y:y + side, x:x + side], (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    raise ValueError(f"Unsupported crop strategy: {strategy}")

class BreastUltrasoundDataset(Dataset):
    """
    Custom Dataset for Breast Cancer Ultrasound Scans.
    Supports directory structure where subfolders represent classes:
        data/train/benign/img1.png
        data/train/malignant/img2.png
    Or a list of (file_path, label_idx) tuples.
    """
    def __init__(self, root_dir=None, file_list=None, transform=None, use_clahe=True, crop_strategy=None):
        self.transform = transform
        self.use_clahe = use_clahe
        self.crop_strategy = crop_strategy if crop_strategy is not None else getattr(config, 'CROP_STRATEGY', 'roi_crop')
        self.samples = []

        if file_list is not None:
            self.samples = file_list
        elif root_dir is not None and os.path.exists(root_dir):
            for class_idx, class_name in enumerate(config.CLASS_NAMES):
                class_dir = os.path.join(root_dir, class_name)
                extensions = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp', '*.dcm')
                files = []
                
                if os.path.exists(class_dir):
                    for ext in extensions:
                        files.extend(glob.glob(os.path.join(class_dir, ext)))
                        
                if class_name == "benign":
                    normal_dir = os.path.join(root_dir, "normal")
                    if os.path.exists(normal_dir):
                        for ext in extensions:
                            files.extend(glob.glob(os.path.join(normal_dir, ext)))
                            
                for filepath in sorted(files):
                    fname = os.path.basename(filepath)
                    if not fname.lower().endswith(('_tumor.png', '_mask.png', '_lesion_mask.png')):
                        self.samples.append((filepath, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]

        try:
            img_np = cv2.imread(filepath)
            if img_np is None:
                raise ValueError(f"Failed to read image at {filepath}")
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        except Exception:
            img_pil = Image.open(filepath).convert("RGB")
            img_np = np.array(img_pil)

        mask_path = find_mask_path(filepath) if self.crop_strategy == "roi_crop" else None
        mask_np = cv2.imread(mask_path) if mask_path else None
        img_np = preprocess_image(img_np, self.use_clahe, self.crop_strategy, mask_np)

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
        # Do not vertically flip ultrasound: image depth has anatomical meaning.
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
        datasets_to_load = ["busi", "breast", "oasbud"]
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
