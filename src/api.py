import os
import io
import base64
import json
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
    from src.dataset import apply_clahe, crop_and_pad_roi, preprocess_image
    from src.models import get_model
except ImportError:
    import config
    from dataset import apply_clahe, crop_and_pad_roi, preprocess_image
    from models import get_model

app = FastAPI(
    title="Breast Ultrasound Research Classification API",
    description="Reproducibility-focused dissertation prototype; not for clinical use.",
    version="1.0.0"
)

# Enable CORS for web frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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

    checkpoint_path = config.EVALUATION_CHECKPOINT_PATH
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(
            "The audited EfficientNet-B0 checkpoint is unavailable. "
            "Provide outputs/efficientnet_b0_model.pth or set EVALUATION_CHECKPOINT_PATH."
        )

    try:
        mtime = os.path.getmtime(checkpoint_path)
        if model is None or mtime > last_loaded_mtime:
            print(f"Loading/Reloading checkpoint from: {checkpoint_path} (mtime={mtime})")
            checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
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
    return model_registry[requested], requested

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
            return "OUT_OF_DOMAIN_IMAGE: Upload a predominantly grayscale B-mode breast-ultrasound research image; this file appears to be a colour photograph."

    # 2. Luminance & Tissue Histogram Distribution Check
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if (len(img_np.shape) == 3 and img_np.shape[2] >= 3) else img_np
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    if mean_val > 225.0:
        return "OUT_OF_DOMAIN_IMAGE: Upload a B-mode breast-ultrasound research image; this file appears predominantly white or document-like."
    if mean_val < 8.0 or std_val < 6.0:
        return "OUT_OF_DOMAIN_IMAGE: The uploaded image lacks the minimum intensity variation required by this research pipeline."

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


def generate_gradcam(active_model: torch.nn.Module, tensor: torch.Tensor, target_class_idx: int, model_name: str) -> Optional[np.ndarray]:
    """
    Computes class activation map using Gradient-weighted Class Activation Mapping (Grad-CAM).
    Captures fine-grained convolutional features for explainable AI inspection.
    """
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Find target convolutional layer
    target_layer = None
    m_name = (model_name or "").lower()
    if "resnet" in m_name and hasattr(active_model, "layer4"):
        target_layer = active_model.layer4[-1]
    elif "efficientnet" in m_name and hasattr(active_model, "features"):
        target_layer = active_model.features[-1]
    elif "custom" in m_name and hasattr(active_model, "features"):
        for m in reversed(active_model.features):
            if isinstance(m, torch.nn.Conv2d):
                target_layer = m
                break

    if target_layer is None:
        for m in reversed(list(active_model.modules())):
            if isinstance(m, torch.nn.Conv2d):
                target_layer = m
                break

    if target_layer is None:
        return None

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    try:
        tensor_input = tensor.clone().detach().requires_grad_(True)
        active_model.zero_grad()
        output = active_model(tensor_input)
        if target_class_idx >= output.shape[1]:
            target_class_idx = int(torch.argmax(output).item())
        score = output[0, target_class_idx]
        score.backward()

        if not gradients or not activations:
            return None

        grad = gradients[0].detach().cpu().numpy()[0]
        act = activations[0].detach().cpu().numpy()[0]

        weights = np.mean(grad, axis=(1, 2))
        cam = np.zeros(act.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * act[i]

        cam = np.maximum(cam, 0)
        max_val = np.max(cam)
        if max_val > 0:
            cam = cam / max_val
        else:
            cam = np.zeros_like(cam)

        cam = cv2.resize(cam, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        return cam
    except Exception as exc:
        print(f"[Grad-CAM Warning] Grad-CAM generation failed: {exc}")
        return None
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def overlay_gradcam_on_image(img_np: np.ndarray, cam: Optional[np.ndarray], alpha: float = 0.5) -> np.ndarray:
    """
    Overlays normalized Grad-CAM heatmaps over the base ultrasound image using Jet colormap.
    """
    if cam is None:
        return img_np

    h, w = img_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    if len(img_np.shape) == 2:
        base_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 1:
        base_rgb = cv2.cvtColor(img_np[:, :, 0], cv2.COLOR_GRAY2RGB)
    else:
        base_rgb = img_np.copy()

    overlay = cv2.addWeighted(base_rgb, 1.0 - alpha, heatmap, alpha, 0)
    return overlay


def analyze_image_noise(img_gray: np.ndarray) -> dict:
    """Calculate descriptive texture proxies without inferring their physical cause."""
    h, w = img_gray.shape[:2]
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

    # These thresholds label image statistics, not scanner or acquisition faults.
    if noise_metrics["impulse_level"] > 15.0:
        dominant_type = "High extreme-pixel ratio"
        description = "Many pixels differ strongly from a local median-filtered reference."
    elif noise_metrics["gaussian_level"] > 40.0:
        dominant_type = "High high-frequency index"
        description = "The normalised Laplacian-variance proxy is high."
    elif noise_metrics["speckle_level"] > 25.0:
        dominant_type = "High local-variation index"
        description = "The median local coefficient-of-variation proxy is high."
    elif noise_metrics["speckle_level"] <= 10.0 and noise_metrics["gaussian_level"] <= 10.0:
        dominant_type = "Low measured variation"
        description = "Both local-variation and Laplacian-variance proxies are low."
    else:
        dominant_type = "Mixed texture profile"
        description = "No single image-statistic proxy exceeds its demonstration threshold."

    return {
        "dominant_type": dominant_type,
        "description": description,
        "metrics": noise_metrics
    }


def compute_morphological_profile(gray_img: np.ndarray) -> dict:
    """Calculate deterministic intensity and geometry proxies for UI inspection."""
    h, w = gray_img.shape[:2]
    lx, ly, r = localize_lesion(gray_img)

    # 1. Hypoechoic Contrast Ratio (mass core vs surrounding parenchyma)
    mask_lesion = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_lesion, (lx, ly), max(5, r), 255, -1)
    
    mask_parenchyma = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_parenchyma, (lx, ly), max(10, int(r * 1.8)), 255, -1)
    mask_parenchyma = cv2.subtract(mask_parenchyma, mask_lesion)

    lesion_mean = float(cv2.mean(gray_img, mask=mask_lesion)[0]) if np.sum(mask_lesion) > 0 else 60.0
    parenchyma_mean = float(cv2.mean(gray_img, mask=mask_parenchyma)[0]) if np.sum(mask_parenchyma) > 0 else 120.0

    hypoechoic_ratio = round(lesion_mean / max(1.0, parenchyma_mean), 2)
    echogenicity_str = "Much darker than reference" if hypoechoic_ratio < 0.65 else ("Darker than reference" if hypoechoic_ratio < 0.90 else "Similar to reference")

    # 2. Posterior Acoustic Shadowing Ratio (below lesion vs lateral tissue)
    y_post_start = min(h - 1, ly + r)
    y_post_end = min(h, ly + int(r * 2.5))
    x_post_start = max(0, lx - r)
    x_post_end = min(w, lx + r)

    post_region = gray_img[y_post_start:y_post_end, x_post_start:x_post_end]
    post_mean = float(np.mean(post_region)) if post_region.size > 0 else parenchyma_mean
    shadow_ratio = round(post_mean / max(1.0, parenchyma_mean), 2)

    posterior_str = "Lower mean than reference" if shadow_ratio < 0.75 else ("Higher mean than reference" if shadow_ratio > 1.25 else "Similar to reference")

    # 3. Lesion Margin & Aspect Ratio
    aspect_ratio = round(float(r * 2.0) / max(10.0, float(r * 1.8)), 2)
    margin_type = "Irregularity proxy elevated" if hypoechoic_ratio < 0.60 or shadow_ratio < 0.70 else "Irregularity proxy not elevated"
    orientation_str = "Height proxy exceeds width proxy" if aspect_ratio > 1.1 else "Width proxy equals or exceeds height proxy"

    return {
        "centroid": {"x": int(lx), "y": int(ly), "radius": int(r)},
        "echogenicity": echogenicity_str,
        "hypoechoic_ratio": hypoechoic_ratio,
        "margin": margin_type,
        "posterior_transmission": posterior_str,
        "posterior_ratio": shadow_ratio,
        "orientation": orientation_str,
        "aspect_ratio": aspect_ratio
    }


def compute_research_report(pred_class: str, confidence: float, prob_dict: dict, noise_analysis: dict, model_name: str, morphology: Optional[dict] = None) -> dict:
    """Return descriptive model output without diagnosis or clinical action."""
    dominant_noise = noise_analysis.get("dominant_type", "speckle")
    morph = morphology or {}
    shadowing_str = morph.get("posterior_transmission", "Not estimated")
    margin_str = morph.get("margin", "Not estimated")
    echogenicity_str = morph.get("echogenicity", "Not estimated")
    rationale_str = (f"{model_name.upper()} returned {pred_class.upper()} with a {confidence * 100:.1f}% softmax score "
                     f"alongside the {dominant_noise.lower()} image-texture proxy. Descriptors are deterministic heuristics, not clinical findings.")

    return {
        "status": "Research output only",
        "action": "No clinical action may be inferred.",
        "posterior_intensity_heuristic": shadowing_str,
        "edge_heuristic": margin_str,
        "intensity_heuristic": echogenicity_str,
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

    # Generate the contrast preview and use the same preprocessing function as evaluation.
    enhanced_np = apply_clahe(img_np)
    clahe_b64 = img_to_base64(enhanced_np)
    input_for_model = preprocess_image(img_np, use_clahe, crop_strategy)

    # Prepare PyTorch Tensor
    img_pil = Image.fromarray(input_for_model)
    tensor = val_transform(img_pil).unsqueeze(0).to(config.DEVICE)

    # Analyze Noise & Morphological features
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
    noise_analysis = analyze_image_noise(img_gray)
    morphology = compute_morphological_profile(img_gray)

    with torch.no_grad():
        outputs = active_model(tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)
        pred_idx = torch.argmax(probabilities).item()
    pred_class = config.CLASS_NAMES[pred_idx]
    confidence = float(probabilities[pred_idx].item())
    abstained = confidence < config.ABSTAIN_CONFIDENCE
    surfaced_prediction = "uncertain" if abstained else pred_class
    prob_dict = {config.CLASS_NAMES[i]: float(probabilities[i].item()) for i in range(config.NUM_CLASSES)}

    # Compute Grad-CAM Saliency Heatmap
    gradcam_cam = generate_gradcam(active_model, tensor, pred_idx, active_model_name)
    gradcam_overlay_np = overlay_gradcam_on_image(input_for_model, gradcam_cam, alpha=0.55)
    gradcam_b64 = img_to_base64(gradcam_overlay_np)

    # Draw visual scan
    visual_img = draw_lesion_circle(input_for_model, pred_class)
    processed_b64 = img_to_base64(visual_img)

    research_report = compute_research_report(
        pred_class=pred_class,
        confidence=confidence,
        prob_dict=prob_dict,
        noise_analysis=noise_analysis,
        model_name=active_model_name,
        morphology=morphology
    )

    # Determine Ground Truth Label
    filename = file.filename or "uploaded_scan.png"
    gt_label = "UNKNOWN"
    if ground_truth and ground_truth.upper() in ["BENIGN", "MALIGNANT"]:
        gt_label = ground_truth.upper()

    return {
        "filename": filename,
        "ground_truth": gt_label,
        "prediction": surfaced_prediction.upper(),
        "model_prediction": pred_class.upper(),
        "abstained": abstained,
        "abstention_reason": (
            f"Maximum model probability ({confidence:.2f}) is below the configured "
            f"research threshold ({config.ABSTAIN_CONFIDENCE:.2f})."
            if abstained else None
        ),
        "confidence": round(confidence * 100, 2),
        "probabilities": {k: round(v * 100, 2) for k, v in prob_dict.items()},
        "original_image": original_b64,
        "processed_image": processed_b64,
        "spotlight_zoom_base64": processed_b64,
        "gradcam_image": gradcam_b64,
        "clahe_image": clahe_b64,
        "used_clahe": use_clahe,
        "crop_strategy": crop_strategy,
        "noise_analysis": noise_analysis,
        "morphology": morphology,
        "research_report": research_report,
        "model_name": active_model_name,
        "research_notice": "Experimental research output only. It is not a diagnosis or clinical decision-support result."
    }

@app.post("/predict/compare")
async def compare_all_models(
    file: UploadFile = File(...),
    use_clahe: bool = Form(True),
    crop_strategy: Optional[str] = Form("roi_crop")
):
    """
    Runs concurrent comparative inference across all available deep learning architectures:
    ResNet-50, EfficientNet-B0, and Custom Ultrasound CNN.
    Returns individual predictions, confidences, probability vectors, and consensus metric.
    """
    global val_transform
    try:
        load_model_if_needed()
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
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    validation_error = validate_ultrasound_candidate(img_np)
    if validation_error:
        raise HTTPException(status_code=400, detail=validation_error)

    input_for_model = preprocess_image(img_np, use_clahe, crop_strategy)

    img_pil = Image.fromarray(input_for_model)
    tensor = val_transform(img_pil).unsqueeze(0).to(config.DEVICE)

    model_candidates = ["resnet50", "efficientnet_b0", "custom_cnn"]
    comparisons = {}
    predictions_list = []

    for m_name in model_candidates:
        try:
            m_instance, actual_name = get_verified_model(m_name)
            with torch.no_grad():
                out = m_instance(tensor)
                probs = torch.softmax(out, dim=1).squeeze(0)
                pred_idx = int(torch.argmax(probs).item())
                pred_class = config.CLASS_NAMES[pred_idx]
                conf = float(probs[pred_idx].item())
                prob_dict = {config.CLASS_NAMES[i]: round(float(probs[i].item()) * 100, 2) for i in range(config.NUM_CLASSES)}

            # Also generate a Grad-CAM preview for each model
            cam = generate_gradcam(m_instance, tensor, pred_idx, actual_name)
            cam_overlay = overlay_gradcam_on_image(input_for_model, cam, alpha=0.5)
            cam_b64 = img_to_base64(cam_overlay)

            comparisons[m_name] = {
                "model_name": actual_name,
                "prediction": pred_class.upper(),
                "confidence": round(conf * 100, 2),
                "probabilities": prob_dict,
                "gradcam_image": cam_b64
            }
            predictions_list.append(pred_class.upper())
        except Exception as exc:
            comparisons[m_name] = {
                "model_name": m_name,
                "error": str(exc)
            }

    # Consensus calculation
    valid_preds = [p for p in predictions_list if p]
    if valid_preds:
        majority_class = max(set(valid_preds), key=valid_preds.count)
        agreement_ratio = round(valid_preds.count(majority_class) / len(valid_preds) * 100, 1)
        consensus_status = "Unanimous Agreement (3/3)" if agreement_ratio == 100.0 else (
            f"Majority Consensus ({valid_preds.count(majority_class)}/{len(valid_preds)})" if agreement_ratio >= 66.0 else "Model Disagreement / Split Verdict"
        )
    else:
        majority_class = "UNKNOWN"
        agreement_ratio = 0.0
        consensus_status = "Inconclusive"

    return {
        "filename": file.filename or "scan.png",
        "consensus_prediction": majority_class,
        "agreement_ratio": agreement_ratio,
        "consensus_status": consensus_status,
        "models": comparisons
    }

@app.get("/api/benchmarks")
def get_dissertation_benchmarks():
    """
    Return only metrics emitted by a real evaluation run.
    """
    metrics_path = os.path.join(config.OUTPUT_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="No evaluated metrics are available. Run evaluate.py first.")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    extended_path = os.path.join(config.OUTPUT_DIR, "extended_evaluation.json")
    benchmark_path = os.path.join(config.OUTPUT_DIR, "model_benchmark.json")
    extended = None
    benchmark = None
    if os.path.exists(extended_path):
        with open(extended_path, "r", encoding="utf-8") as f:
            extended = json.load(f)
    if os.path.exists(benchmark_path):
        with open(benchmark_path, "r", encoding="utf-8") as f:
            benchmark = json.load(f)
    return {
        "status": "single evaluation run; do not treat as clinical validation",
        "metrics": metrics,
        "extended_evaluation": extended,
        "model_benchmark": benchmark,
        "unavailable": ["multi-seed training", "external clinical validation", "measured cohort noise benchmark"],
    }


@app.get("/api/data-audit")
def get_data_audit():
    """Return the most recent read-only data-integrity audit."""
    path = os.path.join(config.OUTPUT_DIR, "data_audit.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No data audit is available. Run scripts/audit_data.py first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/evidence/{filename}")
def get_evidence_file(filename: str):
    """Serve only generated, non-sensitive evaluation summaries and figures."""
    allowed = {
        "metrics.json",
        "data_audit.json",
        "extended_evaluation.json",
        "confusion_matrix.png",
        "roc_curve.png",
        "discrimination_calibration.png",
        "source_stratified_performance.png",
    }
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="Evidence file not found.")
    path = os.path.join(config.OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Evidence file not found.")
    return FileResponse(path)


@app.post("/predict/batch")
async def predict_ultrasound_batch(
    files: List[UploadFile] = File(...),
    use_clahe: bool = Form(True),
    crop_strategy: Optional[str] = Form("roi_crop"),
    model_name: Optional[str] = Form("efficientnet_b0")
):
    global val_transform
    try:
        load_model_if_needed()
        active_model, active_model_name = get_verified_model(model_name)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    results = []
    for file in files:
        fname = file.filename or "scan.png"
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
            results.append({"filename": fname, "error": exc.detail, "status": "error"})
            continue
        except Exception as e:
            results.append({"filename": fname, "error": f"Invalid image file: {str(e)}", "status": "error"})
            continue

        validation_error = validate_ultrasound_candidate(img_np)
        if validation_error:
            results.append({
                "filename": fname,
                "error": validation_error,
                "status": "rejected"
            })
            continue

        original_b64 = img_to_base64(img_np)

        input_for_model = preprocess_image(img_np, use_clahe, crop_strategy)

        img_pil = Image.fromarray(input_for_model)
        tensor = val_transform(img_pil).unsqueeze(0).to(config.DEVICE)

        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
        noise_analysis = analyze_image_noise(img_gray)
        morphology = compute_morphological_profile(img_gray)

        with torch.no_grad():
            outputs = active_model(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0)
            pred_idx = torch.argmax(probs).item()
            pred_class = config.CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx].item())

        prob_dict = {config.CLASS_NAMES[i]: round(float(probs[i].item()) * 100, 2) for i in range(config.NUM_CLASSES)}

        # Grad-CAM heatmap
        cam = generate_gradcam(active_model, tensor, pred_idx, active_model_name)
        cam_overlay = overlay_gradcam_on_image(input_for_model, cam, alpha=0.55)
        gradcam_b64 = img_to_base64(cam_overlay)

        # Visual ROI processed
        visual_img = draw_lesion_circle(input_for_model, pred_class)
        processed_b64 = img_to_base64(visual_img)

        output_band = "HIGH_SCORE" if confidence >= 0.80 else ("MODERATE_SCORE" if confidence >= 0.60 else "LOW_SCORE")

        research_report = compute_research_report(
            pred_class=pred_class,
            confidence=confidence,
            prob_dict=prob_dict,
            noise_analysis=noise_analysis,
            model_name=active_model_name,
            morphology=morphology
        )

        results.append({
            "filename": fname,
            "status": "success",
            "prediction": pred_class.upper(),
            "output_band": output_band,
            "confidence": round(confidence * 100, 2),
            "probabilities": prob_dict,
            "original_image": original_b64,
            "processed_image": processed_b64,
            "gradcam_image": gradcam_b64,
            "noise_type": noise_analysis.get("dominant_type", "Standard"),
            "snr_db": noise_analysis.get("metrics", {}).get("snr_db", 0.0),
            "report_status": research_report.get("status", "Research output only"),
            "research_notice": research_report.get("action", "No clinical action may be inferred."),
            "model_name": active_model_name
        })

    # Cohort summary statistics
    total = len(results)
    successes = [r for r in results if r.get("status") == "success"]
    malignant_count = sum(1 for r in successes if r.get("prediction") == "MALIGNANT")
    benign_count = sum(1 for r in successes if r.get("prediction") == "BENIGN")
    normal_count = sum(1 for r in successes if r.get("prediction") == "NORMAL")
    avg_conf = round(sum(r.get("confidence", 0) for r in successes) / max(1, len(successes)), 2)

    return {
        "cohort_summary": {
            "total_scans": total,
            "processed_scans": len(successes),
            "malignant_count": malignant_count,
            "benign_count": benign_count,
            "normal_count": normal_count,
            "average_confidence": avg_conf,
            "active_model": active_model_name,
            "research_notice": "Provisional model outputs only; no clinical prioritisation is performed."
        },
        "results": results
    }

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
                    if not fname.lower().endswith(('_tumor.png', '_mask.png', '_lesion_mask.png')):
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


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR) in decibels (dB)."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 99.0
    PIXEL_MAX = 255.0
    return round(float(20 * np.log10(PIXEL_MAX / np.sqrt(mse))), 2)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculates Structural Similarity Index (SSIM) between two ultrasound images."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    if len(img1.shape) == 3:
        img1_g = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_g = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img1_g, img2_g = img1, img2

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1_f = img1_g.astype(np.float64)
    img2_f = img2_g.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1_f, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2_f, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1_f ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2_f ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1_f * img2_f, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return round(float(np.clip(np.mean(ssim_map), 0.0, 1.0)), 4)


@app.get("/api/dataset/cohort")
def get_balanced_cohort(dataset: str = "busi", split: str = "val", count: int = 16):
    """
    Return an interleaved labelled cohort for local engineering checks.

    The endpoint reflects the current folder labels and does not certify that the
    split is subject independent.
    """
    if dataset not in ["busi", "breast", "oasbud"]:
        dataset = "busi"
    if split not in ["val", "test", "train"]:
        split = "val"

    split_dir = os.path.join(config.DATA_DIR, dataset, split)
    half = max(1, count // 2)

    benign_files = []
    b_dir = os.path.join(split_dir, "benign")
    if os.path.exists(b_dir):
        for f in sorted(os.listdir(b_dir)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')) and not f.lower().endswith(('_mask.png', '_tumor.png', '_lesion_mask.png')):
                benign_files.append({
                    "name": f,
                    "class": "benign",
                    "ground_truth": "BENIGN",
                    "url": f"/api/dataset/file/{split}/benign/{f}?dataset={dataset}"
                })

    malignant_files = []
    m_dir = os.path.join(split_dir, "malignant")
    if os.path.exists(m_dir):
        for f in sorted(os.listdir(m_dir)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')) and not f.lower().endswith(('_mask.png', '_tumor.png', '_lesion_mask.png')):
                malignant_files.append({
                    "name": f,
                    "class": "malignant",
                    "ground_truth": "MALIGNANT",
                    "url": f"/api/dataset/file/{split}/malignant/{f}?dataset={dataset}"
                })

    selected = []
    m_sub = malignant_files[:half]
    b_sub = benign_files[:half]
    max_len = max(len(m_sub), len(b_sub))
    for i in range(max_len):
        if i < len(m_sub):
            selected.append(m_sub[i])
        if i < len(b_sub):
            selected.append(b_sub[i])

    return selected


def add_synthetic_noise(img_np: np.ndarray, noise_type: str, intensity: float, seed: int) -> np.ndarray:
    """
    Applies synthetic noise to an image for comparison studies.
    """
    h, w = img_np.shape[:2]
    rng = np.random.default_rng(seed)
    
    if noise_type == "speckle":
        noise = rng.rayleigh(scale=1.0, size=(h, w))
        noise = noise / np.mean(noise)
        noise = 1.0 + intensity * 3.5 * (noise - 1.0)
        if len(img_np.shape) == 3:
            noise = np.expand_dims(noise, axis=2)
            noisy = img_np.astype(np.float32) * noise
        else:
            noisy = img_np.astype(np.float32) * noise
            
    elif noise_type == "gaussian":
        noise = rng.normal(0, intensity * 80.0, (h, w))
        if len(img_np.shape) == 3:
            noise = np.expand_dims(noise, axis=2)
            noisy = img_np.astype(np.float32) + noise
        else:
            noisy = img_np.astype(np.float32) + noise
            
    elif noise_type == "impulse":
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


def apply_strategy_transform(image_np: np.ndarray, strategy: str, use_clahe: bool, active_model: torch.nn.Module) -> Tuple[str, float, str, np.ndarray]:
    strategy = (strategy or "direct_resize").lower()
    processed_np = preprocess_image(image_np, use_clahe, strategy)
        
    tensor = val_transform(Image.fromarray(processed_np)).unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        out = active_model(tensor)
        probs = torch.softmax(out, dim=1).squeeze(0)
        pred_idx = torch.argmax(probs).item()
        pred_class = config.CLASS_NAMES[pred_idx].upper()
        conf = probs[pred_idx].item() * 100
        
    visual = draw_lesion_circle(processed_np, pred_class)
    return pred_class, round(conf, 2), img_to_base64(visual), processed_np


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

    # 4-Stage controlled pipeline
    noisy_raw = add_synthetic_noise(img_np, noise_type, intensity, seed)
    pred_class_clean, conf_clean, clean_b64, tensor_clean = apply_strategy_transform(img_np, "direct_resize", use_clahe, active_model)
    pred_class_noisy_u, conf_noisy_u, noisy_u_b64, tensor_noisy = apply_strategy_transform(noisy_raw, "direct_resize", use_clahe, active_model)
    pred_c3, conf_c3, img_c3_b64, tensor_c3 = apply_strategy_transform(noisy_raw, col3_strategy, use_clahe, active_model)
    pred_c4, conf_c4, img_c4_b64, tensor_c4 = apply_strategy_transform(noisy_raw, "roi_crop", use_clahe, active_model)

    # Compute PSNR & SSIM metrics across stages
    psnr_noisy = compute_psnr(img_np, noisy_raw)
    ssim_noisy = compute_ssim(img_np, noisy_raw)
    psnr_sol_a = compute_psnr(tensor_clean, tensor_c3)
    ssim_sol_a = compute_ssim(tensor_clean, tensor_c3)
    psnr_sol_b = compute_psnr(tensor_clean, tensor_c4)
    ssim_sol_b = compute_ssim(tensor_clean, tensor_c4)

    explanations = {
        "speckle": "Seeded multiplicative noise is used here as an algorithmic stressor. It is not a faithful model of a scanner, acquisition protocol, or clinical site.",
        "gaussian": "Seeded additive Gaussian noise is used here as an algorithmic stressor. A single image cannot establish model robustness.",
        "impulse": "Seeded impulse noise introduces artificial extreme pixels for a controlled software check; it does not reproduce a specific acquisition artifact."
    }
    explanation = explanations.get(noise_type, "Noise changes image contrast and may alter extracted model features.")

    return {
        "stage1_clean": {
            "prediction": pred_class_clean,
            "confidence": round(conf_clean, 2),
            "image": clean_b64,
            "psnr_db": None,
            "ssim": 1.0000
        },
        "stage2_noisy": {
            "prediction": pred_class_noisy_u,
            "confidence": round(conf_noisy_u, 2),
            "image": noisy_u_b64,
            "psnr_db": psnr_noisy,
            "ssim": ssim_noisy
        },
        "stage3_direct": {
            "prediction": pred_c3,
            "confidence": conf_c3,
            "image": img_c3_b64,
            "strategy": col3_strategy,
            "psnr_db": psnr_sol_a,
            "ssim": ssim_sol_a
        },
        "stage4_roi": {
            "prediction": pred_c4,
            "confidence": conf_c4,
            "image": img_c4_b64,
            "strategy": "roi_crop",
            "psnr_db": psnr_sol_b,
            "ssim": ssim_sol_b
        },
        "explanation": explanation,
        "model_name": active_model_name,
        "seed": seed,
        "deltas": {
            "noise_vs_clean": round(conf_noisy_u - conf_clean, 2),
            "method_a_vs_noise": round(conf_c3 - conf_noisy_u, 2),
            "proposed_vs_noise": round(conf_c4 - conf_noisy_u, 2),
            "ssim_gain": round(ssim_sol_b - ssim_sol_a, 4)
        }
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
