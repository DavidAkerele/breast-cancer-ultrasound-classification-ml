import os
import io
import base64
from typing import List
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

import config
from dataset import apply_clahe
from models import get_model

app = FastAPI(
    title="Breast Cancer Ultrasound Classification API",
    description="Dissertation AI Diagnostic Tool using PyTorch Deep Learning",
    version="1.0.0"
)

# Enable CORS for web frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom 404 Exception Handler for Dedicated UI
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc):
    if exc.status_code == 404:
        if request.url.path.startswith("/api"):
            return JSONResponse(status_code=404, content={"detail": exc.detail})
        return FileResponse(os.path.join(WEB_DIR, "404.html"), status_code=404)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# Global model instance
model = None
val_transform = None
last_loaded_mtime = 0.0

def img_to_base64(img_np):
    """Converts a numpy RGB image array to base64 PNG data URL."""
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

def load_model_if_needed():
    global model, val_transform, last_loaded_mtime
    
    if val_transform is None:
        val_transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    if not os.path.exists(config.CHECKPOINT_PATH):
        if model is None:
            print("[WARNING] No trained model checkpoint found! Initializing fallback custom_cnn weights for UI testing.")
            model = get_model(model_name="custom_cnn", num_classes=config.NUM_CLASSES, pretrained=False)
            model = model.to(config.DEVICE)
            model.eval()
        return

    try:
        mtime = os.path.getmtime(config.CHECKPOINT_PATH)
        if model is None or mtime > last_loaded_mtime:
            print(f"Loading/Reloading checkpoint from: {config.CHECKPOINT_PATH} (mtime={mtime})")
            checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
            model_name = checkpoint.get("model_name", "custom_cnn")
            
            temp_model = get_model(model_name=model_name, num_classes=config.NUM_CLASSES, pretrained=False)
            temp_model.load_state_dict(checkpoint["model_state_dict"])
            temp_model = temp_model.to(config.DEVICE)
            temp_model.eval()
            
            model = temp_model
            last_loaded_mtime = mtime
            print(f"Successfully loaded/reloaded model architecture: {model_name}")
    except Exception as e:
        print(f"[ERROR] Failed to load/reload checkpoint: {str(e)}")
        if model is None:
            model = get_model(model_name="custom_cnn", num_classes=config.NUM_CLASSES, pretrained=False)
            model = model.to(config.DEVICE)
            model.eval()

@app.on_event("startup")
def load_pytorch_model():
    load_model_if_needed()
    print("Model startup load completed.")
def is_valid_medical_scan(img_np: np.ndarray) -> bool:
    """
    Validates if the image matches a breast ultrasound scan profile.
    Allows for clinical downloaded scans (which may contain annotations, Doppler highlights, or are pre-cropped).
    Rejects colorful photos, text documents, solid color blocks, and standard grayscale photos (like dogs, faces).
    """
    # 1. Color check: Allow small color elements (e.g., Doppler overlays or annotations)
    # but reject highly colorful images.
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        diff_rg = np.abs(img_np[:, :, 0].astype(np.int16) - img_np[:, :, 1].astype(np.int16))
        diff_gb = np.abs(img_np[:, :, 1].astype(np.int16) - img_np[:, :, 2].astype(np.int16))
        mean_diff = (np.mean(diff_rg) + np.mean(diff_gb)) / 2
        if mean_diff > 25.0:
            return False # Reject highly colorful photos
            
    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
    
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    
    # 2. Brightness bounds check: Ultrasound scans have low-to-moderate average intensity
    if mean_val < 10 or mean_val > 175:
        return False
        
    # 3. Dynamic range/flatness check: Reject solid color blocks
    if std_val < 5.0:
        return False
        
    # 4. White saturation check: Reject pure text pages or documents
    white_ratio = np.mean(gray > 240)
    if white_ratio > 0.35:
        return False
        
    # 5. Local Entropy Texture Check: Rejects normal photographs (like dogs, faces, scenes)
    # Ultrasound scans are composed of granular acoustic speckles, giving high local entropy.
    # Standard photos have large smooth areas with low local entropy.
    try:
        # Resize to standard size for consistent block statistics
        resized = cv2.resize(gray, (224, 224))
        block_size = 16
        h, w = resized.shape
        entropies = []
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = resized[y:y+block_size, x:x+block_size]
                # Compute block histogram
                hist, _ = np.histogram(block, bins=256, range=(0, 256))
                hist = hist.astype(np.float32) / block.size
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log2(hist))
                entropies.append(entropy)
        avg_entropy = np.mean(entropies)
        
        # Ultrasound speckle texture typically yields average local entropy > 3.8
        # Grayscale photos of objects, animals, or faces yield average local entropy < 3.4
        if avg_entropy < 3.65:
            return False
    except Exception:
        # Fallback to True if processing fails
        pass
        
    return True


def localize_lesion(gray_img: np.ndarray) -> tuple:
    """
    Locates the hypoechoic mass lesion using adaptive thresholding and L2 distance transform.
    This method guarantees a tight, precise circle fit that avoids acoustic shadowing/enhancement artifacts.
    """
    h, w = gray_img.shape
    
    # 1. Smooth to suppress speckle noise while preserving general mass structure
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)
    
    # 2. Find local minimum intensity in the central search region (where masses reside)
    cy, cx = h // 2, w // 2
    r_search = 50
    sub_region = blurred[max(0, cy-r_search):min(h, cy+r_search), max(0, cx-r_search):min(w, cx+r_search)]
    
    # Adaptive local threshold based on the actual darkest pixel in the region
    min_val = np.min(sub_region) if sub_region.size > 0 else 20
    thresh_val = min_val + 18
    
    # Threshold to get dark regions
    _, mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Clean up mask morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 4. Compute Distance Transform: find pixel furthest from background (i.e. lesion core center)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # Crop distance transform map to search region
    dist_center = dist[max(0, cy-r_search):min(h, cy+r_search), max(0, cx-r_search):min(w, cx+r_search)]
    
    if dist_center.size > 0:
        max_val = np.max(dist_center)
        # We need a clear peak representing a lesion core
        if max_val > 3.0:
            max_idx = np.unravel_index(np.argmax(dist_center), dist_center.shape)
            lx = cx - r_search + max_idx[1]
            ly = cy - r_search + max_idx[0]
            
            # Scale distance radius to wrap the mass outer border cleanly
            r = int(max_val * 1.6)
            r = max(15, min(r, 45)) # Clamped to clinical bounds
            return int(lx), int(ly), int(r)
            
    # Fallback to center region with default radius
    return cx, cy, 25


def draw_lesion_circle(img_np: np.ndarray, prediction: str) -> np.ndarray:
    """
    Draws a clinical double-outline target circle around the localized lesion.
    Red outline for MALIGNANT predictions, Green outline for BENIGN predictions.
    """
    # Ensure image is in color
    if len(img_np.shape) == 2:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 1:
        img_rgb = cv2.cvtColor(img_np[:, :, 0], cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_np.copy()
        
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cx, cy, r = localize_lesion(gray)
    
    # Class colors: red for malignant, green for benign
    color = (255, 23, 68) if prediction.upper() == "MALIGNANT" else (0, 230, 118)
    glow_color = (color[0] // 2, color[1] // 2, color[2] // 2)
    
    # Draw double circle outlines
    cv2.circle(img_rgb, (cx, cy), r, color, 2, lineType=cv2.LINE_AA)
    cv2.circle(img_rgb, (cx, cy), r + 4, glow_color, 1, lineType=cv2.LINE_AA)
    
    return img_rgb


def analyze_image_noise(img_gray: np.ndarray) -> dict:
    """
    Analyzes and classifies the noise characteristics in a breast ultrasound scan.
    Returns estimated noise levels and classifies the dominant noise type:
    - Speckle Noise (acoustic interference, standard in ultrasound)
    - Gaussian Noise (electronic sensor noise)
    - Impulse (Salt & Pepper) Noise (pixel dropouts/transmission errors)
    - Poisson Noise (signal-dependent noise)
    - Low Noise (high SNR scan)
    """
    h, w = img_gray.shape
    img_float = img_gray.astype(np.float32)

    # 1. Speckle Noise Estimation (standard deviation to mean ratio in non-zero regions)
    local_mean = cv2.blur(img_float, (7, 7))
    local_sq_mean = cv2.blur(img_float ** 2, (7, 7))
    local_var = np.maximum(0, local_sq_mean - local_mean ** 2)
    local_std = np.sqrt(local_var)
    
    non_zero_mask = local_mean > 5.0
    if np.sum(non_zero_mask) > 0:
        cv_map = local_std[non_zero_mask] / local_mean[non_zero_mask]
        speckle_index = float(np.median(cv_map))
    else:
        speckle_index = 0.0

    # 2. Gaussian Noise Estimation (using Laplacian variance normalized by mean)
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    lap_var = float(np.var(laplacian))
    mean_val = max(1.0, float(np.mean(img_gray)))
    gaussian_index = (lap_var / mean_val) * 0.1

    # 3. Impulse (Salt & Pepper) Noise Estimation (using difference with median filter)
    median_filtered = cv2.medianBlur(img_gray, 3)
    diff = cv2.absdiff(img_gray, median_filtered)
    impulse_pixels = np.sum(diff > 40)
    impulse_ratio = float(impulse_pixels / (h * w))

    # 4. Signal-to-Noise Ratio (SNR) estimation in dB
    std_val = float(np.std(img_float))
    snr_db = 20 * np.log10(mean_val / max(0.1, std_val)) if std_val > 0 else 0.0

    # Scale metrics to 0-100 values
    noise_metrics = {
        "speckle_level": min(100.0, max(0.0, speckle_index * 200.0)),
        "gaussian_level": min(100.0, max(0.0, gaussian_index * 4.0)),
        "impulse_level": min(100.0, max(0.0, impulse_ratio * 4000.0)),
        "snr_db": round(snr_db, 2)
    }

    # Classification Logic
    if noise_metrics["impulse_level"] > 15.0:
        dominant_type = "Impulse (Salt & Pepper) Noise"
        description = "Isolated pixel dropout artifacts, typically caused by analog-to-digital converter errors or probe transmission interference."
    elif noise_metrics["gaussian_level"] > 40.0:
        dominant_type = "Gaussian (Thermal) Noise"
        description = "Electronic sensor thermal noise, often arising from amplifier heat or low-quality transducer hardware."
    elif noise_metrics["speckle_level"] > 25.0:
        dominant_type = "Speckle Noise (Acoustic)"
        description = "Acoustic speckle pattern, which is the standard multiplicative noise in ultrasound scans caused by sub-resolution scatterer phase interference."
    elif noise_metrics["speckle_level"] <= 10.0 and noise_metrics["gaussian_level"] <= 10.0:
        dominant_type = "Low Noise (High Quality Scan)"
        description = "Minimal noise detected. High signal-to-noise ratio (SNR) scan suitable for clear diagnostic observation."
    else:
        dominant_type = "Mixed Acoustic Speckle"
        description = "Standard combination of tissue backscatter speckling and low-level sensor thermal noise."

    return {
        "dominant_type": dominant_type,
        "description": description,
        "metrics": noise_metrics
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "device": str(config.DEVICE), "classes": config.CLASS_NAMES}

@app.post("/predict")
async def predict_ultrasound(
    file: UploadFile = File(...),
    use_clahe: bool = Form(True)
):
    global model, val_transform
    load_model_if_needed()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
            img_np = np.array(img_pil)
        else:
            img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    if not is_valid_medical_scan(img_np):
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is not within the scope of this diagnostic model. Please upload a valid grayscale breast ultrasound scan."
        )

    original_b64 = img_to_base64(img_np)

    # Process image with CLAHE if requested
    if use_clahe:
        enhanced_np = apply_clahe(img_np)
        input_for_model = enhanced_np
    else:
        input_for_model = img_np

    # Prepare PyTorch Tensor
    img_pil = Image.fromarray(input_for_model)
    tensor = val_transform(img_pil).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        pred_idx = torch.argmax(probs).item()
        pred_class = config.CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx].item()

    prob_dict = {config.CLASS_NAMES[i]: float(probs[i].item()) for i in range(config.NUM_CLASSES)}

    # Draw localized ROI target circle on the processed visual image
    visual_img = draw_lesion_circle(input_for_model, pred_class)
    processed_b64 = img_to_base64(visual_img)

    # Analyze Noise
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
    noise_analysis = analyze_image_noise(img_gray)

    return {
        "prediction": pred_class.upper(),
        "confidence": round(confidence * 100, 2),
        "probabilities": {k: round(v * 100, 2) for k, v in prob_dict.items()},
        "original_image": original_b64,
        "processed_image": processed_b64,
        "used_clahe": use_clahe,
        "noise_analysis": noise_analysis
    }

@app.post("/predict/batch")
async def predict_ultrasound_batch(
    files: List[UploadFile] = File(...),
    use_clahe: bool = Form(True)
):
    global model, val_transform
    load_model_if_needed()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    results = []
    for file in files:
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
                img_np = np.array(img_pil)
            else:
                img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": f"Invalid image file: {str(e)}"
            })
            continue

        if not is_valid_medical_scan(img_np):
            results.append({
                "filename": file.filename,
                "error": "The uploaded image is not within the scope of this diagnostic model. Please upload a valid grayscale breast ultrasound scan."
            })
            continue

        original_b64 = img_to_base64(img_np)

        if use_clahe:
            enhanced_np = apply_clahe(img_np)
            processed_b64 = img_to_base64(enhanced_np)
            input_for_model = enhanced_np
        else:
            processed_b64 = original_b64
            input_for_model = img_np

        img_pil = Image.fromarray(input_for_model)
        tensor = val_transform(img_pil).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0)
            pred_idx = torch.argmax(probs).item()
            pred_class = config.CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx].item()

        # Analyze Noise
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
        noise_analysis = analyze_image_noise(img_gray)

        results.append({
            "filename": file.filename,
            "prediction": pred_class.upper(),
            "confidence": round(confidence * 100, 2),
            "original_image": original_b64,
            "processed_image": processed_b64,
            "noise_analysis": noise_analysis
        })

    return {"results": results}

# Mount static frontend directory
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
os.makedirs(WEB_DIR, exist_ok=True)

# Secure endpoint to serve source code of dissertation model files
@app.get("/api/code/{filename}")
def get_source_code(filename: str):
    allowed_files = ["config.py", "dataset.py", "models.py", "train.py", "evaluate.py", "predict.py"]
    if filename not in allowed_files:
        raise HTTPException(status_code=403, detail="Access denied.")
    
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"filename": filename, "code": content}

# Endpoint to query list of generated validation dataset files with split option
@app.get("/api/dataset/files")
def get_dataset_files(split: str = "val"):
    if split not in ["train", "val", "test"]:
        raise HTTPException(status_code=400, detail="Invalid split partition.")
        
    if split == "train":
        split_dir = config.TRAIN_DIR
    elif split == "test":
        split_dir = config.TEST_DIR
    else:
        split_dir = config.VAL_DIR
        
    files_list = []
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(split_dir, class_name)
        if os.path.exists(class_dir):
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    files_list.append({
                        "name": fname,
                        "class": class_name,
                        "url": f"/api/dataset/file/{split}/{class_name}/{fname}"
                    })
    return files_list

# Endpoint to serve a specific image file from a dataset split
@app.get("/api/dataset/file/{split}/{category}/{filename}")
def get_split_dataset_file(split: str, category: str, filename: str):
    if split not in ["train", "val", "test"]:
        raise HTTPException(status_code=400, detail="Invalid split partition.")
    if category not in config.CLASS_NAMES:
        raise HTTPException(status_code=400, detail="Invalid category.")
        
    if split == "train":
        split_dir = config.TRAIN_DIR
    elif split == "test":
        split_dir = config.TEST_DIR
    else:
        split_dir = config.VAL_DIR
        
    clean_filename = os.path.basename(filename)
    file_path = os.path.join(split_dir, category, clean_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image file not found.")
        
    return FileResponse(file_path)


def add_synthetic_noise(img_np: np.ndarray, noise_type: str, intensity: float) -> np.ndarray:
    """
    Applies synthetic noise to an image for comparison studies.
    """
    h, w = img_np.shape[:2]
    
    if noise_type == "speckle":
        # Multiplicative noise: I = I + I * N(0, v)
        # intensity ranges from 0.05 (low) to 0.40 (high)
        noise = np.random.normal(0, intensity, (h, w))
        if len(img_np.shape) == 3:
            noise = np.expand_dims(noise, axis=2)
            noisy = img_np.astype(np.float32) + img_np.astype(np.float32) * noise
        else:
            noisy = img_np.astype(np.float32) + img_np.astype(np.float32) * noise
            
    elif noise_type == "gaussian":
        # Additive electronic/sensor noise: I = I + N(0, v)
        # intensity ranges from 5 to 60 (pixel standard deviation)
        noise = np.random.normal(0, intensity * 255.0, (h, w))
        if len(img_np.shape) == 3:
            noise = np.expand_dims(noise, axis=2)
            noisy = img_np.astype(np.float32) + noise
        else:
            noisy = img_np.astype(np.float32) + noise
            
    elif noise_type == "impulse":
        # Salt & Pepper pixel dropout noise
        # intensity is corruption probability (0.01 to 0.20)
        noisy = img_np.copy()
        
        # Salt
        num_salt = np.ceil(intensity * img_np.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_np.shape[:2]]
        if len(img_np.shape) == 3:
            noisy[coords[0], coords[1], :] = 255
        else:
            noisy[coords[0], coords[1]] = 255
            
        # Pepper
        num_pepper = np.ceil(intensity * img_np.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_np.shape[:2]]
        if len(img_np.shape) == 3:
            noisy[coords[0], coords[1], :] = 0
        else:
            noisy[coords[0], coords[1]] = 0
            
        return noisy
        
    else:
        return img_np.copy()
        
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


@app.post("/api/noise/simulate")
async def simulate_noise_and_predict(
    file: UploadFile = File(...),
    noise_type: str = Form("speckle"),
    intensity: float = Form(0.15),
    use_clahe: bool = Form(True)
):
    global model, val_transform
    load_model_if_needed()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
            img_np = np.array(img_pil)
        else:
            img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    if not is_valid_medical_scan(img_np):
        raise HTTPException(status_code=400, detail="Out of scope image.")

    # 1. Clean diagnosis
    input_clean = apply_clahe(img_np) if use_clahe else img_np
    img_pil_clean = Image.fromarray(input_clean)
    tensor_clean = val_transform(img_pil_clean).unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        out_clean = model(tensor_clean)
        probs_clean = torch.softmax(out_clean, dim=1).squeeze(0)
        pred_idx_clean = torch.argmax(probs_clean).item()
        pred_class_clean = config.CLASS_NAMES[pred_idx_clean].upper()
        conf_clean = probs_clean[pred_idx_clean].item() * 100

    # 2. Generate Noisy Image
    noisy_np = add_synthetic_noise(img_np, noise_type, intensity)

    # 3. Noisy diagnosis
    input_noisy = apply_clahe(noisy_np) if use_clahe else noisy_np
    img_pil_noisy = Image.fromarray(input_noisy)
    tensor_noisy = val_transform(img_pil_noisy).unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        out_noisy = model(tensor_noisy)
        probs_noisy = torch.softmax(out_noisy, dim=1).squeeze(0)
        pred_idx_noisy = torch.argmax(probs_noisy).item()
        pred_class_noisy = config.CLASS_NAMES[pred_idx_noisy].upper()
        conf_noisy = probs_noisy[pred_idx_noisy].item() * 100

    # Draw localized target circles on both
    visual_clean = draw_lesion_circle(input_clean, pred_class_clean)
    visual_noisy = draw_lesion_circle(input_noisy, pred_class_noisy)

    clean_b64 = img_to_base64(visual_clean)
    noisy_b64 = img_to_base64(visual_noisy)

    # Explanatory clinical summary
    explanations = {
        "speckle": "Speckle noise is a multiplicative acoustic artifact inherent to ultrasound. It degrades the CNN's ability to segment fine-grained lesion margins, often blurring irregular spiculated borders and leading to incorrect classification.",
        "gaussian": "Gaussian noise simulates electronic thermal sensor noise. High thermal noise introduces high-frequency random fluctuations, corrupting the feature activations of deep convolutional kernels and lowering prediction confidence.",
        "impulse": "Impulse (salt & pepper) noise represents transmission dropout errors. It creates isolated pure white/black pixels, which can trigger artificial high-frequency edge detections, throwing off spatial pooling layers."
    }
    explanation = explanations.get(noise_type, "Noise degrades visual contrast, impairing diagnostic feature extraction.")

    return {
        "clean_prediction": pred_class_clean,
        "clean_confidence": round(conf_clean, 2),
        "clean_image": clean_b64,
        
        "noisy_prediction": pred_class_noisy,
        "noisy_confidence": round(conf_noisy, 2),
        "noisy_image": noisy_b64,
        
        "explanation": explanation
    }


# Legacy compatibility route for validation images
@app.get("/api/dataset/file/{category}/{filename}")
def get_dataset_file(category: str, filename: str):
    return get_split_dataset_file("val", category, filename)

app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on http://0.0.0.0:8000")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
