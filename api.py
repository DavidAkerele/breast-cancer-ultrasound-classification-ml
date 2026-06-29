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
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import config
from dataset import apply_clahe
from models import get_model

app = FastAPI(
    title="Breast Cancer Mammogram Classification API",
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

# Global model instance
model = None
val_transform = None

def img_to_base64(img_np):
    """Converts a numpy RGB image array to base64 PNG data URL."""
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

@app.on_event("startup")
def load_pytorch_model():
    global model, val_transform
    print(f"Loading checkpoint from: {config.CHECKPOINT_PATH}")
    
    if os.path.exists(config.CHECKPOINT_PATH):
        checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
        model_name = checkpoint.get("model_name", "custom_cnn")
        print(f"Loaded architecture: {model_name}")
        model = get_model(model_name=model_name, num_classes=config.NUM_CLASSES, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("[WARNING] No trained model checkpoint found! Initializing fallback custom_cnn weights for UI testing.")
        model = get_model(model_name="custom_cnn", num_classes=config.NUM_CLASSES, pretrained=False)

    model = model.to(config.DEVICE)
    model.eval()

    val_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("Model initialized and ready for inference.")
def is_valid_medical_scan(img_np: np.ndarray) -> bool:
    """
    Heuristic to check if the uploaded image matches a breast cancer scan profile (mammogram/ultrasound).
    Medical scans are primarily grayscale, meaning the variance between R, G, B channels is very low.
    Also, they typically have a high proportion of dark/background pixels.
    """
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        diff_rg = np.abs(img_np[:, :, 0].astype(np.int16) - img_np[:, :, 1].astype(np.int16))
        diff_gb = np.abs(img_np[:, :, 1].astype(np.int16) - img_np[:, :, 2].astype(np.int16))
        mean_diff = (np.mean(diff_rg) + np.mean(diff_gb)) / 2
        if mean_diff > 10.0:
            return False
            
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
    dark_pixels_ratio = np.mean(gray < 30)
    if dark_pixels_ratio < 0.05:
        return False
        
    return True

@app.get("/health")
def health_check():
    return {"status": "ok", "device": str(config.DEVICE), "classes": config.CLASS_NAMES}

@app.post("/predict")
async def predict_mammogram(
    file: UploadFile = File(...),
    use_clahe: bool = Form(True)
):
    global model, val_transform
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
            detail="The uploaded image is not within the scope of this diagnostic model. Please upload a valid grayscale mammography or ultrasound scan."
        )

    original_b64 = img_to_base64(img_np)

    # Process image with CLAHE if requested
    if use_clahe:
        enhanced_np = apply_clahe(img_np)
        processed_b64 = img_to_base64(enhanced_np)
        input_for_model = enhanced_np
    else:
        processed_b64 = original_b64
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

    return {
        "prediction": pred_class.upper(),
        "confidence": round(confidence * 100, 2),
        "probabilities": {k: round(v * 100, 2) for k, v in prob_dict.items()},
        "original_image": original_b64,
        "processed_image": processed_b64,
        "used_clahe": use_clahe
    }

@app.post("/predict/batch")
async def predict_mammograms_batch(
    files: List[UploadFile] = File(...),
    use_clahe: bool = Form(True)
):
    global model, val_transform
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
                "error": "The uploaded image is not within the scope of this diagnostic model. Please upload a valid grayscale mammography or ultrasound scan."
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

        results.append({
            "filename": file.filename,
            "prediction": pred_class.upper(),
            "confidence": round(confidence * 100, 2),
            "original_image": original_b64,
            "processed_image": processed_b64
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

# Endpoint to query list of generated validation dataset files
@app.get("/api/dataset/files")
def get_dataset_files():
    files_list = []
    # Scan benign and malignant validation folders
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(config.VAL_DIR, class_name)
        if os.path.exists(class_dir):
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    files_list.append({
                        "name": fname,
                        "class": class_name,
                        "url": f"/api/dataset/file/{class_name}/{fname}"
                    })
    return files_list

# Endpoint to serve a specific image file from validation database
@app.get("/api/dataset/file/{category}/{filename}")
def get_dataset_file(category: str, filename: str):
    if category not in config.CLASS_NAMES:
        raise HTTPException(status_code=400, detail="Invalid category.")
    
    # Clean file name to prevent directory traversal
    clean_filename = os.path.basename(filename)
    file_path = os.path.join(config.VAL_DIR, category, clean_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image file not found.")
        
    return FileResponse(file_path)

app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on http://0.0.0.0:8000")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
