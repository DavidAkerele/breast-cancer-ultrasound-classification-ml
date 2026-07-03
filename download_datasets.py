# download_datasets.py
"""Utility script to prepare datasets for the dissertation project.

This version:
* Uses the locally‑provided ``OASBUD.mat`` and ``BrEaST‑Lesions_USG-images_and_masks‑Dec-15-2023.zip`` files.
* Splits each dataset into ``train``, ``val`` and ``test`` folders using the same ratios as BUSI (80/15/5).
* Still retains the optional BUSI download step – it will be skipped if the download fails.
* Installs ``scipy`` on‑the‑fly if it is not already available (required for reading ``.mat`` files).
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import random
import shutil
from pathlib import Path
import numpy as np

# ---------- Ensure SciPy availability ----------
try:
    import scipy.io
except ImportError:
    print("[INFO] SciPy not found. Installing via pip…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    import scipy.io

# ---------- Configuration ----------
DATA_ROOT = Path(__file__).parent / "data"

# URLs for the downloadable datasets (BUSI may be skipped if unavailable)
BUSI_ZIP_URL = "https://scholar.cu.edu.eg/dataset/download/busi_dataset.zip"
OASBUD_ZIP_URL = "https://zenodo.org/record/545928/files/OASBUD.zip?download=1"

# Local files supplied by the user
LOCAL_OASBUD_MAT = Path(__file__).parent / "OASBUD.mat"
LOCAL_BREAST_ZIP = Path(__file__).parent / "BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip"

# ---------- Helper Functions ----------

def ensure_dir(path: Path) -> None:
    """Create *path* if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path) -> None:
    """Download *url* to *dest* (skips if the file already exists)."""
    if dest.exists():
        print(f"[INFO] File already exists: {dest}")
        return
    print(f"[INFO] Downloading {url} -> {dest}")
    try:
        with urllib.request.urlopen(url) as response, open(dest, "wb") as out_file:
            total = int(response.info().get("Content-Length", -1))
            downloaded = 0
            block_sz = 8192
            while True:
                buffer = response.read(block_sz)
                if not buffer:
                    break
                out_file.write(buffer)
                downloaded += len(buffer)
                if total > 0:
                    percent = downloaded * 100 // total
                    print(f"   {percent}% completed", end="\r")
        print("   Download complete.            ")
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract *zip_path* into *extract_to* (creates folder if needed)."""
    if not zip_path.is_file():
        raise RuntimeError(f"Zip file not found: {zip_path}")
    print(f"[INFO] Extracting {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("   Extraction complete.")


def run_organize_busi() -> None:
    """Execute the existing ``organize_busi.py`` script to split the BUSI dataset."""
    script_path = Path(__file__).parent / "organize_busi.py"
    if not script_path.is_file():
        raise RuntimeError("organize_busi.py not found in repository root.")
    print("[INFO] Running BUSI organizer to create train/val/test splits.")
    result = subprocess.run([sys.executable, str(script_path)], cwd=Path(__file__).parent, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("organize_busi.py failed.")


def split_mat_dataset(mat_path: Path, out_root: Path, ratios=(0.8, 0.15, 0.05)) -> None:
    """Load a .mat file, split its third‑dimensional image stack, and store each split as a separate .mat.
    The split files are named ``<original>_train.mat`` etc. and placed under ``out_root/<split>/``.
    """
    print(f"[INFO] Loading .mat file: {mat_path}")
    mat_data = scipy.io.loadmat(str(mat_path))
    # Find the first 3‑D ndarray that is not metadata
    image_key = None
    for k, v in mat_data.items():
        if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 3:
            image_key = k
            break
    if image_key is None:
        raise RuntimeError("No suitable 3‑D image array found in the .mat file.")
    images = mat_data[image_key]
    total = images.shape[2]
    indices = list(range(total))
    random.seed(42)
    random.shuffle(indices)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])
    splits = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:]
    }
    for split_name, idxs in splits.items():
        split_dir = out_root / split_name
        ensure_dir(split_dir)
        split_mat_path = split_dir / f"{mat_path.stem}_{split_name}.mat"
        split_images = images[:, :, idxs]
        split_mat = {k: v for k, v in mat_data.items() if not k.startswith("__")}
        split_mat[image_key] = split_images
        scipy.io.savemat(str(split_mat_path), split_mat)
        print(f"   Saved {split_name} split to {split_mat_path}")


def split_zip_dataset(zip_path: Path, out_root: Path, ratios=(0.8, 0.15, 0.05)) -> None:
    """Extract a zip containing image files and split them into train/val/test folders.
    Supports typical image extensions.
    """
    temp_dir = out_root / "_temp_extract"
    ensure_dir(temp_dir)
    extract_zip(zip_path, temp_dir)
    # Gather image paths recursively
    image_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp"):
        image_files.extend(list(temp_dir.rglob(ext)))
    total = len(image_files)
    random.seed(42)
    random.shuffle(image_files)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])
    splits = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:]
    }
    for split_name, files in splits.items():
        split_dir = out_root / split_name
        ensure_dir(split_dir)
        for src in files:
            dest = split_dir / src.name
            shutil.move(str(src), str(dest))
        print(f"   Moved {len(files)} files to {split_name} folder.")
    # Clean up temporary extraction directory
    shutil.rmtree(temp_dir)

# ---------- Main Workflow ----------

def main() -> None:
    ensure_dir(DATA_ROOT)

    # ---- BUSI (optional) ----
    try:
        download_file(BUSI_ZIP_URL, DATA_ROOT / "busi_dataset.zip")
        extract_zip(DATA_ROOT / "busi_dataset.zip", DATA_ROOT)
        run_organize_busi()
    except Exception as e:
        print(f"[WARN] BUSI step skipped: {e}")

    # ---- OASBUD ----
    if LOCAL_OASBUD_MAT.is_file():
        print(f"[INFO] Found local OASBUD.mat at {LOCAL_OASBUD_MAT}")
        split_mat_dataset(LOCAL_OASBUD_MAT, DATA_ROOT / "oasbud")
    else:
        print("[WARN] OASBUD.mat not found locally – falling back to automatic download.")
        download_file(OASBUD_ZIP_URL, DATA_ROOT / "OASBUD.zip")
        oasbud_dir = DATA_ROOT / "oasbud"
        ensure_dir(oasbud_dir)
        extract_zip(DATA_ROOT / "OASBUD.zip", oasbud_dir)

    # ---- BrEaST ----
    if LOCAL_BREAST_ZIP.is_file():
        print(f"[INFO] Found local BrEaST zip at {LOCAL_BREAST_ZIP}")
        split_zip_dataset(LOCAL_BREAST_ZIP, DATA_ROOT / "breast")
    else:
        print("[WARN] BrEaST zip not found locally – cannot process.")

    print("[SUCCESS] Dataset preparation complete.")

if __name__ == "__main__":
    main()
