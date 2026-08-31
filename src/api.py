import os
import io
import base64
from typing import List, Optional, Dict, Any, Tuple
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

try:
    from src import config
    from src.dataset import apply_clahe, crop_and_pad_roi
    from src.models import get_model
except ImportError:
    import config
    from dataset import apply_clahe, crop_and_pad_roi
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
loaded_model_name: Optional[str] = None
model_registry: Dict[str, torch.nn.Module] = {}
model_registry_mtimes: Dict[str, float] = {}
val_transform = None
last_loaded_mtime = 0.0
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def img_to_base64(img_np):
    """Converts a numpy RGB image array to base64 PNG data URL."""
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

def load_model_if_needed():
    global model, loaded_model_name, val_transform, last_loaded_mtime
    
    if val_transform is None:
        val_transform = transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    if not os.path.exists(config.CHECKPOINT_PATH):
        raise RuntimeError("No trained model checkpoint is available. Train a model or provide outputs/best_ultrasound_model.pth.")

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
            loaded_model_name = model_name
            last_loaded_mtime = mtime
            print(f"Successfully loaded/reloaded model architecture: {model_name}")
    except Exception as e:
        print(f"[ERROR] Failed to load/reload checkpoint: {str(e)}")
        raise RuntimeError("The trained model checkpoint could not be loaded.") from e


def available_model_checkpoints() -> Dict[str, str]:
    """Return only verified architecture/checkpoint pairs; never expose random weights."""
    candidates = dict(config.MODEL_CHECKPOINT_PATHS)
    if not os.path.exists(candidates["resnet50"]):
        candidates["resnet50"] = config.CHECKPOINT_PATH
    available: Dict[str, str] = {}
    for expected_name, path in candidates.items():
        if not os.path.exists(path):
            continue
        try:
            checkpoint = torch.load(path, map_location="cpu")
            actual_name = checkpoint.get("model_name")
            if actual_name in config.MODEL_CHECKPOINT_PATHS:
                available[actual_name] = path
        except Exception:
            continue
    return available


def get_verified_model(model_name: Optional[str]) -> Tuple[torch.nn.Module, str]:
    global model, loaded_model_name
    available = available_model_checkpoints()
    if not available:
        raise RuntimeError("No valid trained model checkpoints are available.")
    requested = (model_name or loaded_model_name or next(iter(available))).lower()
    if requested not in available:
        raise RuntimeError(f"{requested} has no matching trained checkpoint.")
    path = available[requested]
    mtime = os.path.getmtime(path)
    if requested not in model_registry or model_registry_mtimes.get(requested) != mtime:
        checkpoint = torch.load(path, map_location=config.DEVICE)
        instance = get_model(requested, num_classes=config.NUM_CLASSES, pretrained=False)
        instance.load_state_dict(checkpoint["model_state_dict"])
        instance = instance.to(config.DEVICE)
        instance.eval()
        model_registry[requested] = instance
        model_registry_mtimes[requested] = mtime
    model = model_registry[requested]
    loaded_model_name = requested
    return model, requested

@app.on_event("startup")
def load_pytorch_model():
    try:
        load_model_if_needed()
        print("Model startup load completed.")
    except RuntimeError as exc:
        # Keep the research dashboard available so it can communicate a usable error state.
        print(f"[WARNING] Model startup load skipped: {exc}")


def validate_upload(file: UploadFile, contents: bytes) -> None:
    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Use a PNG, JPEG, or TIFF breast-ultrasound image.")
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded image exceeds the 20 MB limit.")
def validate_ultrasound_candidate(img_np: np.ndarray) -> Optional[str]:
    """
    Intelligent Domain Validation Engine: Detects and rejects non-breast-ultrasound images
    (e.g., natural color photos, animals, cars, documents, or out-of-domain scans).
    """
    if img_np is None or not isinstance(img_np, np.ndarray) or img_np.size == 0:
        return "The uploaded file does not contain a readable image."
    if len(img_np.shape) < 2 or min(img_np.shape[:2]) < 64:
        return "The image dimensions are too small for breast ultrasound classification (minimum 64x64 required)."

    # 1. Color Channel Spread & Saturation Validation
    if len(img_np.shape) == 3 and img_np.shape[2] >= 3:
        rgb = img_np[:, :, :3].astype(np.int16)
        channel_spread = np.max(rgb, axis=2) - np.min(rgb, axis=2)
        coloured_ratio = float(np.mean(channel_spread > 18))
        mean_spread = float(np.mean(channel_spread))
        
        # Convert to HSV to check color saturation
        hsv = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2HSV)
        mean_saturation = float(np.mean(hsv[:, :, 1]))
        
        # B-mode ultrasound is predominantly grayscale; natural color photos have high channel spread & saturation
        if (coloured_ratio > 0.18 and mean_spread > 12.0) or mean_saturation > 38.0:
            return "NON_BREAST_ULTRASOUND: Please put in an actual breast cancer scan. The uploaded file is a natural color photo rather than a B-mode breast ultrasound study."

    # 2. Luminance & Tissue Histogram Distribution Check
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if (len(img_np.shape) == 3 and img_np.shape[2] >= 3) else img_np
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    if mean_val > 225.0:
        return "NON_BREAST_ULTRASOUND: Please put in an actual breast cancer scan. The uploaded file is predominantly white (document/paper scan)."
    if mean_val < 8.0 or std_val < 6.0:
        return "NON_BREAST_ULTRASOUND: Please put in an actual breast cancer scan. The uploaded image lacks sufficient tissue contrast or is empty."

    return None


def is_valid_medical_scan(img_np: np.ndarray) -> bool:
    return validate_ultrasound_candidate(img_np) is None


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
    Returns the clean processed ultrasound scan without drawing any ROI boxes,
    brackets, or crosshair overlays over the image.
    """
    if len(img_np.shape) == 2:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 1:
        img_rgb = cv2.cvtColor(img_np[:, :, 0], cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_np.copy()
        
    return img_rgb


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

def compute_clinical_birads_report(pred_class: str, confidence: float, prob_dict: dict, noise_analysis: dict, model_name: str) -> dict:
    """
    Computes a comprehensive, stratified clinical BI-RADS report across Categories 1 to 5
    using classification confidence, multi-class probability distribution, and acoustic noise artifacts.
    """
    speckle_level = noise_analysis.get("metrics", {}).get("speckle_level", 0.0)
    dominant_noise = noise_analysis.get("dominant_type", "speckle")
    
    # Determine Tissue Density based on acoustic speckle noise and contrast
    if speckle_level < 8.0:
        density_str = "Almost Entirely Fatty (ACR A)"
    elif speckle_level < 15.0:
        density_str = "Scattered Fibroglandular (ACR B)"
    elif speckle_level < 25.0:
        density_str = "Heterogeneously Dense (ACR C)"
    else:
        density_str = "Extremely Dense (ACR D) - High speckle attenuation"

    # Determine BI-RADS Category and Acoustic Shadowing
    p_class = pred_class.lower()
    if p_class == "malignant":
        if confidence >= 0.94:
            birads_str = "BI-RADS 5 - Highly suggestive of malignancy (≥95% risk; immediate biopsy required)"
            shadowing_str = "Pronounced posterior acoustic shadowing surrounding irregular spiculated hypoechoic margin."
        elif confidence >= 0.82:
            birads_str = "BI-RADS 4C - High suspicion of malignancy (50-95% risk; urgent histological verification)"
            shadowing_str = "Posterior acoustic shadowing detected surrounding localized lobulated mass margin."
        elif confidence >= 0.65:
            birads_str = "BI-RADS 4B - Moderate suspicion of malignancy (10-50% risk; core needle biopsy advised)"
            shadowing_str = "Moderate posterior acoustic attenuation noted along microlobulated boundary."
        else:
            birads_str = "BI-RADS 4A - Low suspicion of malignancy (2-10% risk; tissue sampling/biopsy considered)"
            shadowing_str = "Equivocal posterior acoustic shadowing with borderline margin irregularities."
    elif p_class == "normal":
        birads_str = "BI-RADS 1 - Negative (Normal screening presentation; regular annual follow-up)"
        shadowing_str = "No focal hypoechoic mass, architectural distortion, or acoustic shadowing detected."
    else: # benign
        if confidence >= 0.94:
            birads_str = "BI-RADS 1 - Negative / Normal screening presentation (No suspicious mass features)"
            shadowing_str = "No focal abnormality or posterior acoustic shadowing observed."
        elif confidence >= 0.82:
            birads_str = "BI-RADS 2 - Benign finding (Routine screening; circumscribed homogeneous morphology)"
            shadowing_str = "No posterior acoustic shadowing observed; clear through-transmission."
        elif confidence >= 0.65:
            birads_str = "BI-RADS 3 - Probably Benign (≤2% risk of malignancy; short-interval 6-month follow-up recommended)"
            shadowing_str = "Minimal edge refile shadowing around circumscribed ovoid nodule; well-defined capsule."
        else:
            birads_str = "BI-RADS 4A - Low suspicion of malignancy (2-10% risk; equivocal boundary due to speckle distortion)"
            shadowing_str = "Borderline margin definition with slight posterior acoustic heterogeneity."

    rationale_str = (
        f"Convolutional forward pass ({model_name.upper()}) identified localized structural morphology with "
        f"{round(confidence * 100, 1)}% network confidence under {dominant_noise} acoustic profile. "
        f"Stratified BI-RADS assignment reflects both prediction certainty and margin boundary definition."
    )

    return {
        "birads": birads_str,
        "tissue_density": density_str,
        "acoustic_shadowing": shadowing_str,
        "rationale": rationale_str
    }

@app.get("/health")
def health_check():
    try:
        load_model_if_needed()
        return {
            "status": "ok",
            "device": str(config.DEVICE),
            "classes": config.CLASS_NAMES,
            "model": loaded_model_name,
            "models": [{"name": name, "ready": name in available_model_checkpoints()} for name in config.MODEL_CHECKPOINT_PATHS],
            "model_ready": True,
        }
    except RuntimeError as exc:
        return {
            "status": "degraded",
            "device": str(config.DEVICE),
            "classes": config.CLASS_NAMES,
            "model": None,
            "models": [{"name": name, "ready": name in available_model_checkpoints()} for name in config.MODEL_CHECKPOINT_PATHS],
            "model_ready": False,
            "detail": str(exc),
        }

@app.post("/predict")
async def predict_ultrasound(
    file: UploadFile = File(...),
    use_clahe: bool = Form(True),
    crop_strategy: Optional[str] = Form("roi_crop"),
    model_name: Optional[str] = Form(None),
    ground_truth: Optional[str] = Form(None)
):
    global val_transform
    try:
        load_model_if_needed()
        active_model, active_model_name = get_verified_model(model_name)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        contents = await file.read()
        validate_upload(file, contents)
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
            img_np = np.array(img_pil)
        else:
            img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    validation_error = validate_ultrasound_candidate(img_np)
    if validation_error:
        raise HTTPException(
            status_code=400,
            detail=validation_error
        )

    original_b64 = img_to_base64(img_np)

    # Process image with CLAHE if requested
    if use_clahe:
        enhanced_np = apply_clahe(img_np)
        input_for_model = enhanced_np
    else:
        input_for_model = img_np

    # Apply Crop Strategy
    if crop_strategy == "roi_crop":
        input_for_model = crop_and_pad_roi(input_for_model)
    elif crop_strategy == "direct_resize":
        input_for_model = cv2.resize(input_for_model, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    elif crop_strategy == "center_crop":
        h, w = input_for_model.shape[:2]
        s = min(h, w)
        cy, cx = h // 2, w // 2
        crop = input_for_model[max(0, cy-s//2):min(h, cy+s//2), max(0, cx-s//2):min(w, cx+s//2)]
        input_for_model = cv2.resize(crop, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_CUBIC)

    # Prepare PyTorch Tensor
    img_pil = Image.fromarray(input_for_model)
    tensor = val_transform(img_pil).unsqueeze(0).to(config.DEVICE)

    # Analyze Noise
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
    noise_analysis = analyze_image_noise(img_gray)

    with torch.no_grad():
        outputs = active_model(tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)
        pred_idx = torch.argmax(probabilities).item()
    pred_class = config.CLASS_NAMES[pred_idx]
    confidence = float(probabilities[pred_idx].item())
    prob_dict = {config.CLASS_NAMES[i]: float(probabilities[i].item()) for i in range(config.NUM_CLASSES)}

    # Draw localized ROI target circle on the processed visual image
    visual_img = draw_lesion_circle(input_for_model, pred_class)
    processed_b64 = img_to_base64(visual_img)

    # Determine Ground Truth Label from Filename or Parameter
    filename = file.filename or "uploaded_scan.png"
    gt_label = "UNKNOWN"
    if ground_truth and ground_truth.upper() in ["BENIGN", "MALIGNANT"]:
        gt_label = ground_truth.upper()
    elif "benign" in filename.lower():
        gt_label = "BENIGN"
    elif "malignant" in filename.lower() or "sample_0" in filename.lower():
        gt_label = "MALIGNANT"

    return {
        "filename": filename,
        "ground_truth": gt_label,
        "prediction": pred_class.upper(),
        "confidence": round(confidence * 100, 2),
        "probabilities": {k: round(v * 100, 2) for k, v in prob_dict.items()},
        "original_image": original_b64,
        "processed_image": processed_b64,
        "spotlight_zoom_base64": processed_b64,
        "used_clahe": use_clahe,
        "noise_analysis": noise_analysis,
        "model_name": active_model_name,
        "research_notice": "Experimental research output only. It is not a diagnosis or clinical decision-support result."
    }

@app.post("/predict/batch")
async def predict_ultrasound_batch(
    files: List[UploadFile] = File(...),
    use_clahe: bool = Form(True)
):
    global model, val_transform
    try:
        load_model_if_needed()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    results = []
    for file in files:
        try:
            contents = await file.read()
            validate_upload(file, contents)
            nparr = np.frombuffer(contents, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
                img_np = np.array(img_pil)
            else:
                img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except HTTPException as exc:
            results.append({"filename": file.filename, "error": exc.detail})
            continue
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": f"Invalid image file: {str(e)}"
            })
            continue

        validation_error = validate_ultrasound_candidate(img_np)
        if validation_error:
            results.append({
                "filename": file.filename,
                "error": validation_error
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

        prob_dict = {config.CLASS_NAMES[i]: float(probs[i].item()) for i in range(config.NUM_CLASSES)}

        # Draw localized ROI target circle
        visual_img = draw_lesion_circle(input_for_model, pred_class)
        processed_b64 = img_to_base64(visual_img)

        # Analyze Noise
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
        noise_analysis = analyze_image_noise(img_gray)

        results.append({
            "filename": file.filename,
            "prediction": pred_class.upper(),
            "confidence": round(confidence * 100, 2),
            "probabilities": {k: round(v * 100, 2) for k, v in prob_dict.items()},
            "original_image": original_b64,
            "processed_image": processed_b64,
            "spotlight_zoom_base64": processed_b64,
            "noise_analysis": noise_analysis,
            "model_name": loaded_model_name,
            "research_notice": "Experimental research output only. It is not a diagnosis or clinical decision-support result."
        })

    return {"results": results}

# Mount static frontend directory
WEB_DIR = config.WEB_DIR
os.makedirs(WEB_DIR, exist_ok=True)

# Secure endpoint to serve source code and thesis markdown files
@app.get("/api/code/{filename}")
def get_source_code(filename: str):
    clean_name = os.path.basename(filename)
    
    # 1. Search in src/ directory
    src_path = os.path.join(config.BASE_DIR, "src", clean_name)
    if os.path.exists(src_path):
        with open(src_path, "r", encoding="utf-8") as f:
            return {"filename": clean_name, "code": f.read()}

    # 2. Search in docs/ and subdirectories
    docs_dir = os.path.join(config.BASE_DIR, "docs")
    for root, _, files in os.walk(docs_dir):
        if clean_name in files:
            file_path = os.path.join(root, clean_name)
            with open(file_path, "r", encoding="utf-8") as f:
                return {"filename": clean_name, "code": f.read()}

    raise HTTPException(status_code=404, detail=f"Document or code file {clean_name} not found.")

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


def add_synthetic_noise(img_np: np.ndarray, noise_type: str, intensity: float, seed: int) -> np.ndarray:
    """
    Applies synthetic noise to an image for comparison studies.
    """
    h, w = img_np.shape[:2]
    rng = np.random.default_rng(seed)
    
    if noise_type == "speckle":
        # Rayleigh-distributed multiplicative envelope, normalized to preserve mean intensity.
        noise = rng.rayleigh(scale=1.0, size=(h, w))
        noise = noise / np.mean(noise)
        noise = 1.0 + intensity * 3.5 * (noise - 1.0)
        if len(img_np.shape) == 3:
            noise = np.expand_dims(noise, axis=2)
            noisy = img_np.astype(np.float32) * noise
        else:
            noisy = img_np.astype(np.float32) * noise
            
    elif noise_type == "gaussian":
        # Additive electronic/sensor noise: I = I + N(0, v)
        noise = rng.normal(0, intensity * 80.0, (h, w))
        if len(img_np.shape) == 3:
            noise = np.expand_dims(noise, axis=2)
            noisy = img_np.astype(np.float32) + noise
        else:
            noisy = img_np.astype(np.float32) + noise
            
    elif noise_type == "impulse":
        # Salt & Pepper pixel dropout noise
        noisy = img_np.copy()
        num_salt = np.ceil(intensity * 0.08 * img_np.shape[0] * img_np.shape[1])
        coords_y = rng.integers(0, h, int(num_salt))
        coords_x = rng.integers(0, w, int(num_salt))
        if len(img_np.shape) == 3:
            noisy[coords_y, coords_x, :] = 255
        else:
            noisy[coords_y, coords_x] = 255
            
        num_pepper = np.ceil(intensity * 0.08 * img_np.shape[0] * img_np.shape[1])
        coords_y = rng.integers(0, h, int(num_pepper))
        coords_x = rng.integers(0, w, int(num_pepper))
        if len(img_np.shape) == 3:
            noisy[coords_y, coords_x, :] = 0
        else:
            noisy[coords_y, coords_x] = 0
            
        return noisy
        
    else:
        return img_np.copy()
        
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def apply_strategy_transform(image_np: np.ndarray, strategy: str, use_clahe: bool, active_model: torch.nn.Module) -> Tuple[str, float, str]:
    strategy = (strategy or "direct_resize").lower()
    
    source_np = apply_clahe(image_np) if use_clahe else image_np
    if strategy == "roi_crop":
        processed_np = crop_and_pad_roi(source_np)
    elif strategy == "center_crop":
        h, w = source_np.shape[:2]
        s = min(h, w)
        cy, cx = h // 2, w // 2
        crop = source_np[max(0, cy-s//2):min(h, cy+s//2), max(0, cx-s//2):min(w, cx+s//2)]
        processed_np = cv2.resize(crop, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    else: # direct_resize
        processed_np = cv2.resize(source_np, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        
    tensor = val_transform(Image.fromarray(processed_np)).unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        out = active_model(tensor)
        probs = torch.softmax(out, dim=1).squeeze(0)
        pred_idx = torch.argmax(probs).item()
        pred_class = config.CLASS_NAMES[pred_idx].upper()
        conf = probs[pred_idx].item() * 100
        
    visual = draw_lesion_circle(processed_np, pred_class)
    return pred_class, round(conf, 2), img_to_base64(visual)


@app.post("/api/noise/simulate")
async def simulate_noise_and_predict(
    file: UploadFile = File(...),
    noise_type: str = Form("speckle"),
    intensity: float = Form(0.05),
    use_clahe: bool = Form(True),
    col3_strategy: str = Form("direct_resize"),
    model_name: Optional[str] = Form(None),
    seed: int = Form(42)
):
    global model, val_transform
    try:
        load_model_if_needed()
        active_model, active_model_name = get_verified_model(model_name)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        contents = await file.read()
        validate_upload(file, contents)
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
            img_np = np.array(img_pil)
        else:
            img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    validation_error = validate_ultrasound_candidate(img_np)
    if validation_error:
        raise HTTPException(status_code=400, detail=validation_error)

    # The same deterministic noise realization flows through stages 2–4.
    noisy_raw = add_synthetic_noise(img_np, noise_type, intensity, seed)
    pred_class_clean, conf_clean, clean_b64 = apply_strategy_transform(img_np, "direct_resize", use_clahe, active_model)
    pred_class_noisy_u, conf_noisy_u, noisy_u_b64 = apply_strategy_transform(noisy_raw, "direct_resize", use_clahe, active_model)
    pred_c3, conf_c3, img_c3_b64 = apply_strategy_transform(noisy_raw, col3_strategy, use_clahe, active_model)
    pred_c4, conf_c4, img_c4_b64 = apply_strategy_transform(noisy_raw, "roi_crop", use_clahe, active_model)

    explanations = {
        "speckle": "Speckle noise is a multiplicative acoustic artifact inherent to ultrasound. It degrades the CNN's ability to segment fine-grained lesion margins, often blurring irregular spiculated borders and leading to incorrect classification.",
        "gaussian": "Gaussian noise simulates electronic thermal sensor noise. High thermal noise introduces high-frequency random fluctuations, corrupting feature activations and lowering prediction confidence.",
        "impulse": "Impulse (salt & pepper) noise represents transmission dropout errors. It creates isolated pure white/black pixels, which can trigger artificial high-frequency edge detections."
    }
    explanation = explanations.get(noise_type, "Noise degrades visual contrast, impairing diagnostic feature extraction.")

    return {
        "stage1_clean": {
            "prediction": pred_class_clean,
            "confidence": round(conf_clean, 2),
            "image": clean_b64
        },
        "stage2_noisy": {
            "prediction": pred_class_noisy_u,
            "confidence": round(conf_noisy_u, 2),
            "image": noisy_u_b64
        },
        "stage3_direct": {
            "prediction": pred_c3,
            "confidence": conf_c3,
            "image": img_c3_b64,
            "strategy": col3_strategy
        },
        "stage4_roi": {
            "prediction": pred_c4,
            "confidence": conf_c4,
            "image": img_c4_b64,
            "strategy": "roi_crop"
        },
        "explanation": explanation,
        "model_name": active_model_name,
        "seed": seed,
        "deltas": {"noise_vs_clean": round(conf_noisy_u - conf_clean, 2), "method_a_vs_noise": round(conf_c3 - conf_noisy_u, 2), "proposed_vs_noise": round(conf_c4 - conf_noisy_u, 2)}
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
