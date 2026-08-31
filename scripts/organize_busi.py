import os
import shutil
import zipfile
import random
import glob

# Configuration
ZIP_PATTERNS = ["*busi*.zip", "*ultrasound*.zip", "archive.zip"]
SOURCE_DIR_NAME = "Dataset_BUSI_with_GT"
TARGET_DIR = "data"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.15
TEST_RATIO = 0.05

def find_dataset_zip():
    for pattern in ZIP_PATTERNS:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None

def main():
    print("=== BUSI Dataset Extractor & Organizer ===")
    
    # 1. Look for zip file
    zip_path = find_dataset_zip()
    source_dir = SOURCE_DIR_NAME
    
    if zip_path:
        print(f"Found dataset archive: {zip_path}")
        print("Extracting archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Extraction complete.")
    
    # Verify if source directory exists
    if not os.path.exists(source_dir):
        # Look if it was unzipped inside another subfolder
        nested_searches = glob.glob(f"**/{SOURCE_DIR_NAME}", recursive=True)
        if nested_searches:
            source_dir = nested_searches[0]
        else:
            print(f"[ERROR] Could not find the dataset folder '{SOURCE_DIR_NAME}'.")
            print("Please make sure you have downloaded the BUSI dataset zip file from Kaggle:")
            print("https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset")
            print("Place the ZIP file in this folder and rerun this script.")
            return

    print(f"Sourcing images from: {source_dir}")
    
    # Clear existing data splits to avoid mixing
    for split in ["train", "val", "test"]:
        split_path = os.path.join(TARGET_DIR, split)
        if os.path.exists(split_path):
            print(f"Cleaning existing split folder: {split_path}")
            shutil.rmtree(split_path)
            
    classes = ["benign", "malignant"]
    
    for cls in classes:
        cls_source = os.path.join(source_dir, cls)
        if not os.path.exists(cls_source):
            print(f"[WARNING] Class source folder {cls_source} does not exist. Skipping.")
            continue
            
        # Get all non-mask PNG/JPG images
        all_files = [f for f in os.listdir(cls_source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # Filter out mask images
        images = [f for f in all_files if "_mask" not in f]
        
        # Shuffle for random split
        random.seed(42)
        random.shuffle(images)
        
        total = len(images)
        train_idx = int(total * TRAIN_RATIO)
        val_idx = train_idx + int(total * VAL_RATIO)
        
        splits = {
            "train": images[:train_idx],
            "val": images[train_idx:val_idx],
            "test": images[val_idx:]
        }
        
        print(f"\nProcessing class '{cls}' (Total scans: {total}):")
        for split, split_files in splits.items():
            dest_dir = os.path.join(TARGET_DIR, split, cls)
            os.makedirs(dest_dir, exist_ok=True)
            
            print(f"  - Copying {len(split_files)} images to {dest_dir}...")
            for idx, fname in enumerate(split_files):
                src_path = os.path.join(cls_source, fname)
                # Save as clean sample names to match training dataset loader format
                dest_path = os.path.join(dest_dir, f"sample_{idx}.png")
                shutil.copy(src_path, dest_path)
                
    print("\nDataset successfully organized and split!")
    print(f"Training path: {os.path.join(TARGET_DIR, 'train')}")
    print(f"Validation path: {os.path.join(TARGET_DIR, 'val')}")
    print(f"Testing path: {os.path.join(TARGET_DIR, 'test')}")

if __name__ == "__main__":
    main()
