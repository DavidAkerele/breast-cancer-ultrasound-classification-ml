#!/usr/bin/env python3
"""
Utility to split the BrEaST ultrasound dataset into train/val/test sets.
It assumes the dataset is extracted under `data/train/BrEaST-Lesions_USG-images_and_masks`.
Each case consists of an image `caseXYZ.png` and its mask `caseXYZ_tumor.png`.
The script randomly assigns each case to train (70%), val (15%), or test (15%)
while moving both files to the corresponding directories preserving the structure.
"""
import os, random, shutil

random.seed(42)  # reproducible split

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_ROOT, "raw", "BrEaST-Lesions_USG-images_and_masks")
SRC_DIR = RAW_DIR

# Target split directories (will be created if missing)
SPLITS = {"train": os.path.join(DATA_ROOT, "train", "BrEaST-Lesions_USG-images_and_masks"),
          "val": os.path.join(DATA_ROOT, "val", "BrEaST-Lesions_USG-images_and_masks"),
          "test": os.path.join(DATA_ROOT, "test", "BrEaST-Lesions_USG-images_and_masks")}

for split_path in SPLITS.values():
    os.makedirs(split_path, exist_ok=True)

# Gather case identifiers (without extension and without _tumor)
files = [f for f in os.listdir(SRC_DIR) if f.endswith('.png')]
case_ids = set()
for f in files:
    if f.endswith('_tumor.png'):
        case_ids.add(f.replace('_tumor.png', ''))
    else:
        case_ids.add(f.replace('.png', ''))

case_ids = sorted(case_ids)

for case in case_ids:
    r = random.random()
    if r < 0.70:
        split = "train"
    elif r < 0.85:
        split = "val"
    else:
        split = "test"
    dst_dir = SPLITS[split]
    img_src = os.path.join(SRC_DIR, f"{case}.png")
    mask_src = os.path.join(SRC_DIR, f"{case}_tumor.png")
    for src in (img_src, mask_src):
        if os.path.exists(src):
            shutil.move(src, dst_dir)
        else:
            print(f"Warning: missing file {src}")

print("Splitting completed.")
print(f"Train: {len(os.listdir(SPLITS['train'])) // 2} cases")
print(f"Val: {len(os.listdir(SPLITS['val'])) // 2} cases")
print(f"Test: {len(os.listdir(SPLITS['test'])) // 2} cases")
