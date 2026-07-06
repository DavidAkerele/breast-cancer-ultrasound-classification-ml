import os
import random
import scipy.io
import scipy.signal
import numpy as np
import cv2

mat_path = "data/OASBUD.mat"
out_base = os.path.join("data", "oasbud")

print(f"Loading {mat_path}...")
mat = scipy.io.loadmat(mat_path)
cases = mat['data'][0]
num_cases = len(cases)

# Create splits reproducible
random.seed(42)
case_indices = list(range(num_cases))
random.shuffle(case_indices)

train_end = int(num_cases * 0.70)
val_end = train_end + int(num_cases * 0.15)

splits = {
    "train": case_indices[:train_end],
    "val": case_indices[train_end:val_end],
    "test": case_indices[val_end:]
}

def reconstruct_bmode(rf):
    # Envelope detection along axial axis
    analytic_signal = scipy.signal.hilbert(rf, axis=0)
    envelope = np.abs(analytic_signal)
    # Log compression
    log_envelope = 20 * np.log10(envelope + 1e-5)
    # Normalize to 0-255
    min_val = np.min(log_envelope)
    max_val = np.max(log_envelope)
    if max_val > min_val:
        bmode = 255 * (log_envelope - min_val) / (max_val - min_val)
    else:
        bmode = np.zeros_like(log_envelope)
    return cv2.resize(bmode.astype(np.uint8), (512, 512))

def resize_mask(mask):
    # Normalize mask to binary (0 and 255)
    mask_bin = (mask > 0).astype(np.uint8) * 255
    return cv2.resize(mask_bin, (512, 512), interpolation=cv2.INTER_NEAREST)

print("Starting B-mode reconstruction and organizing...")

for split_name, idxs in splits.items():
    print(f"Processing split '{split_name}'...")
    for idx in idxs:
        case = cases[idx]
        case_id = str(case['id'][0]).strip()
        cls_val = int(case['class'][0, 0])
        cls_name = "malignant" if cls_val == 1 else "benign"
        
        target_dir = os.path.join(out_base, split_name, cls_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # Process view stage 1
        rf1 = case['rf1']
        roi1 = case['roi1']
        img1 = reconstruct_bmode(rf1)
        mask1 = resize_mask(roi1)
        
        cv2.imwrite(os.path.join(target_dir, f"{case_id}_view1.png"), img1)
        cv2.imwrite(os.path.join(target_dir, f"{case_id}_view1_tumor.png"), mask1)
        
        # Process view stage 2
        rf2 = case['rf2']
        roi2 = case['roi2']
        img2 = reconstruct_bmode(rf2)
        mask2 = resize_mask(roi2)
        
        cv2.imwrite(os.path.join(target_dir, f"{case_id}_view2.png"), img2)
        cv2.imwrite(os.path.join(target_dir, f"{case_id}_view2_tumor.png"), mask2)

print("OASBUD dataset organization complete.")
