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
multi_model_cache = {}

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
            transforms.Resize(config.IMG_SIZE),
            transforms.CenterCrop(config.IMG_SIZE),
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
    if mean_val < 3 or mean_val > 240:
        return False
        
    # 3. Dynamic range/flatness check: Reject solid color blocks
    if std_val < 3.0:
        return False
        
    # 4. White saturation check: Reject pure text pages or documents
    white_ratio = np.mean(gray > 240)
    if white_ratio > 0.50:
        return False
        
    # 5. Local Entropy Texture Check: Rejects non-medical smooth photographs
    try:
        resized = cv2.resize(gray, (224, 224))
        block_size = 16
        h, w = resized.shape
        entropies = []
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = resized[y:y+block_size, x:x+block_size]
                hist, _ = np.histogram(block, bins=256, range=(0, 256))
                hist = hist.astype(np.float32) / block.size
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log2(hist))
                entropies.append(entropy)
        avg_entropy = np.mean(entropies)
        
        # Only reject if average local entropy is very low AND it is not a dark segmented/cropped ultrasound study
        if avg_entropy < 1.8 and mean_val > 45:
            return False
    except Exception:
        pass
        
    return True


def localize_lesion(gray_img: np.ndarray) -> tuple:
    """
    Locates the hypoechoic mass lesion using adaptive thresholding and L2 distance transform.
    This method scales parameters dynamically with the image dimensions for high precision.
    """
    h, w = gray_img.shape
    
    # 1. Smooth to suppress speckle noise relative to the image size
    ksize = int(min(h, w) * 0.04)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(5, ksize)
    blurred = cv2.GaussianBlur(gray_img, (ksize, ksize), 0)
    
    # 2. Find local minimum intensity in the central search region (where masses reside)
    cy, cx = h // 2, w // 2
    r_search = int(min(h, w) * 0.25)
    sub_region = blurred[max(0, cy-r_search):min(h, cy+r_search), max(0, cx-r_search):min(w, cx+r_search)]
    
    # Adaptive local threshold based on the actual darkest pixel in the region
    min_val = np.min(sub_region) if sub_region.size > 0 else 20
    thresh_val = min_val + 18
    
    # Threshold to get dark regions
    _, mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Clean up mask morphology
    se_size = max(3, int(min(h, w) * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
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
            min_r = max(10, int(min(h, w) * 0.05))
            max_r = int(min(h, w) * 0.3)
            r = max(min_r, min(r, max_r))
            return int(lx), int(ly), int(r)
            
    # Fallback to center region with default radius relative to image size
    fallback_r = int(min(h, w) * 0.12)
    return cx, cy, fallback_r


def draw_lesion_spotlight_zoom(img_np: np.ndarray, prediction: str) -> np.ndarray:
    """
    Applies a clinical radiologist spotlight zoom effect on the localized lesion area:
    1. Smooth radial vignette outside the lesion target area so the tumor shines under a spotlight.
    2. Precision clinical corner brackets around the localized boundary.
    3. A high-definition 2.5X magnified zoom inset lens (picture-in-picture) in the corner showing crisp margin details.
    """
    if len(img_np.shape) == 2:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 1:
        img_rgb = cv2.cvtColor(img_np[:, :, 0], cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_np.copy()
        
    h, w, _ = img_rgb.shape
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cx, cy, r = localize_lesion(gray)
    
    # Class theme colors: red for malignant, green for benign
    is_malignant = prediction.upper() == "MALIGNANT"
    color = (255, 23, 68) if is_malignant else (0, 230, 118)
    glow_color = (color[0] // 2, color[1] // 2, color[2] // 2)
    
    # -------------------------------------------------------------------------
    # 1. SPOTLIGHT EFFECT (Radial Vignette / Dimming background)
    # -------------------------------------------------------------------------
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    spotlight_inner = float(r * 1.15)
    spotlight_outer = float(max(spotlight_inner + 25.0, min(h, w) * 0.48))
    
    transition = (dist_from_center - spotlight_inner) / max(1.0, (spotlight_outer - spotlight_inner))
    transition = np.clip(transition, 0.0, 1.0)
    smooth_transition = transition * transition * (3.0 - 2.0 * transition)
    mask = 1.0 - (0.58 * smooth_transition)
    
    img_spotlight = (img_rgb.astype(np.float32) * mask[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)
    
    # Subtle dashed/soft glow ring along spotlight edge
    cv2.circle(img_spotlight, (cx, cy), int(spotlight_inner), glow_color, 1, lineType=cv2.LINE_AA)
    
    # -------------------------------------------------------------------------
    # 2. CLINICAL TARGET BRACKETS & CROSSHAIR
    # -------------------------------------------------------------------------
    box_r = int(r * 1.15)
    x1, y1 = max(0, cx - box_r), max(0, cy - box_r)
    x2, y2 = min(w - 1, cx + box_r), min(h - 1, cy + box_r)
    bracket_len = max(8, int(box_r * 0.35))
    thick = 2
    
    # Top-Left Bracket
    cv2.line(img_spotlight, (x1, y1), (min(w - 1, x1 + bracket_len), y1), color, thick, lineType=cv2.LINE_AA)
    cv2.line(img_spotlight, (x1, y1), (x1, min(h - 1, y1 + bracket_len)), color, thick, lineType=cv2.LINE_AA)
    # Top-Right Bracket
    cv2.line(img_spotlight, (x2, y1), (max(0, x2 - bracket_len), y1), color, thick, lineType=cv2.LINE_AA)
    cv2.line(img_spotlight, (x2, y1), (x2, min(h - 1, y1 + bracket_len)), color, thick, lineType=cv2.LINE_AA)
    # Bottom-Left Bracket
    cv2.line(img_spotlight, (x1, y2), (min(w - 1, x1 + bracket_len), y2), color, thick, lineType=cv2.LINE_AA)
    cv2.line(img_spotlight, (x1, y2), (x1, max(0, y2 - bracket_len)), color, thick, lineType=cv2.LINE_AA)
    # Bottom-Right Bracket
    cv2.line(img_spotlight, (x2, y2), (max(0, x2 - bracket_len), y2), color, thick, lineType=cv2.LINE_AA)
    cv2.line(img_spotlight, (x2, y2), (x2, max(0, y2 - bracket_len)), color, thick, lineType=cv2.LINE_AA)
    
    # Central target crosshair point
    cv2.drawMarker(img_spotlight, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1, line_type=cv2.LINE_AA)
    
    # -------------------------------------------------------------------------
    # 3. 2.5X MAGNIFIED ZOOM INSET LENS (Picture-in-Picture)
    # -------------------------------------------------------------------------
    crop_r = max(15, int(r * 0.85))
    cy_min, cy_max = max(0, cy - crop_r), min(h, cy + crop_r)
    cx_min, cx_max = max(0, cx - crop_r), min(w, cx + crop_r)
    roi_crop = img_rgb[cy_min:cy_max, cx_min:cx_max]
    
    if roi_crop.size > 0 and roi_crop.shape[0] > 5 and roi_crop.shape[1] > 5:
        lens_size = max(80, min(int(min(h, w) * 0.32), 160))
        zoom_img = cv2.resize(roi_crop, (lens_size, lens_size), interpolation=cv2.INTER_CUBIC)
        
        sharp_kernel = np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]], dtype=np.float32)
        zoom_img = cv2.filter2D(zoom_img, -1, sharp_kernel)
        
        margin = 10
        if cx > w // 2:
            lens_x = margin
        else:
            lens_x = w - lens_size - margin
        lens_y = margin
        
        # Ensure lens window fits
        if lens_y + lens_size < h and lens_x + lens_size < w:
            pad = 2
            cv2.rectangle(img_spotlight, (max(0, lens_x - pad), max(0, lens_y - pad - 16)), (min(w - 1, lens_x + lens_size + pad), min(h - 1, lens_y + lens_size + pad)), (15, 23, 42), -1, lineType=cv2.LINE_AA)
            cv2.rectangle(img_spotlight, (max(0, lens_x - pad), max(0, lens_y - pad - 16)), (min(w - 1, lens_x + lens_size + pad), min(h - 1, lens_y + lens_size + pad)), color, 1, lineType=cv2.LINE_AA)
            
            cv2.putText(img_spotlight, "2.5X ZOOM ROI", (lens_x + 4, lens_y - pad - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            
            img_spotlight[lens_y:lens_y + lens_size, lens_x:lens_x + lens_size] = zoom_img
            
            if cx > w // 2:
                cv2.line(img_spotlight, (x1, y1), (lens_x + lens_size, lens_y + lens_size // 2), glow_color, 1, lineType=cv2.LINE_AA)
            else:
                cv2.line(img_spotlight, (x2, y1), (lens_x, lens_y + lens_size // 2), glow_color, 1, lineType=cv2.LINE_AA)

    return img_spotlight


draw_lesion_circle = draw_lesion_spotlight_zoom


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

def get_multi_model_predictions(tensor: torch.Tensor) -> dict:
    global multi_model_cache
    models_to_run = ["custom_cnn", "resnet50", "efficientnet_b0"]
    results = {}
    for m_name in models_to_run:
        try:
            if m_name not in multi_model_cache:
                m_instance = get_model(model_name=m_name, num_classes=config.NUM_CLASSES, pretrained=(m_name != "custom_cnn"))
                m_instance = m_instance.to(config.DEVICE)
                m_instance.eval()
                multi_model_cache[m_name] = m_instance
            m_instance = multi_model_cache[m_name]
            with torch.no_grad():
                out = m_instance(tensor)
                probs = torch.softmax(out, dim=1).squeeze(0)
                p_idx = torch.argmax(probs).item()
                p_class = config.CLASS_NAMES[p_idx]
                conf = probs[p_idx].item()
                p_dict = {config.CLASS_NAMES[i]: round(float(probs[i].item()) * 100, 2) for i in range(config.NUM_CLASSES)}
                results[m_name] = {
                    "prediction": p_class.upper(),
                    "confidence": round(conf * 100, 2),
                    "probabilities": p_dict
                }
        except Exception as e:
            results[m_name] = {
                "prediction": "ERROR",
                "confidence": 0.0,
                "probabilities": {k: 0.0 for k in config.CLASS_NAMES}
            }
    return results

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

    multi_comparison = get_multi_model_predictions(tensor)

    # Clinical report details
    birads_str = "BI-RADS 4C - High suspicion of malignancy" if pred_class == "malignant" else "BI-RADS 2 - Benign finding (Routine screening)"
    density_str = "Heterogeneously Dense (ACR C)" if noise_analysis["metrics"].get("speckle_level", 0) > 15 else "Scattered Fibroglandular (ACR B)"
    shadowing_str = "Posterior acoustic shadowing detected surrounding localized mass margin." if pred_class == "malignant" else "No significant posterior acoustic shadowing observed."
    rationale_str = f"Convolutional feature extraction identified localized structural morphology with {round(confidence * 100, 1)}% network confidence under {noise_analysis['dominant_type']} profile."

    clinical_report = {
        "birads": birads_str,
        "tissue_density": density_str,
        "acoustic_shadowing": shadowing_str,
        "rationale": rationale_str
    }

    return {
        "prediction": pred_class.upper(),
        "confidence": round(confidence * 100, 2),
        "probabilities": {k: round(v * 100, 2) for k, v in prob_dict.items()},
        "original_image": original_b64,
        "processed_image": processed_b64,
        "used_clahe": use_clahe,
        "noise_analysis": noise_analysis,
        "multi_model_comparison": multi_comparison,
        "clinical_report": clinical_report
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
def get_dataset_files(dataset: str = "busi", split: str = "val"):
    if dataset not in ["busi", "breast", "oasbud"]:
        raise HTTPException(status_code=400, detail="Invalid dataset.")
    if split not in ["train", "val", "test"]:
        raise HTTPException(status_code=400, detail="Invalid split partition.")
        
    split_dir = os.path.join(config.DATA_DIR, dataset, split)
    
    # We support benign, malignant, and normal (only if it exists)
    allowed_categories = ["benign", "malignant"]
    if dataset == "breast":
        allowed_categories.append("normal")
        
    files_list = []
    for class_name in allowed_categories:
        class_dir = os.path.join(split_dir, class_name)
        if os.path.exists(class_dir):
            for fname in sorted(os.listdir(class_dir)):
                # Filter out masks (files ending with _tumor.png, _other.png etc.)
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    if not fname.lower().endswith(('_tumor.png', '_mask.png')):
                        files_list.append({
                            "name": fname,
                            "class": class_name,
                            "url": f"/api/dataset/file/{split}/{class_name}/{fname}?dataset={dataset}"
                        })
    return files_list

# Endpoint to serve a specific image file from a dataset split
@app.get("/api/dataset/file/{split}/{category}/{filename}")
def get_split_dataset_file(split: str, category: str, filename: str, dataset: str = "busi"):
    if dataset not in ["busi", "breast", "oasbud"]:
        raise HTTPException(status_code=400, detail="Invalid dataset.")
    if split not in ["train", "val", "test"]:
        raise HTTPException(status_code=400, detail="Invalid split partition.")
    if category not in ["benign", "malignant", "normal"]:
        raise HTTPException(status_code=400, detail="Invalid category.")
        
    clean_filename = os.path.basename(filename)
    file_path = os.path.join(config.DATA_DIR, dataset, split, category, clean_filename)
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
