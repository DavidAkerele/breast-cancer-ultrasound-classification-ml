# Breast & Ultrasound Lesion Classification Framework

An advanced, interactive medical computer-aided detection (CAD) suite and deep learning pipeline designed for academic research on classifying breast ultrasound and mammography scans into **Benign** and **Malignant** categories.

The framework supports multi-source medical imaging datasets (**BUSI**, **BrEaST**, and **OASBUD**), integrating Contrast Limited Adaptive Histogram Equalization (CLAHE), **Spotlight & 2.5X Magnified Zoom Lenses**, and biophysical noise evaluation with deep Convolutional Neural Networks (CNNs) in PyTorch.

---

## 🔬 Scientific Methodology & Model Backbones

To achieve optimal diagnostic sensitivity and specificity on complex medical scans, this framework implements three neural backbones:
1. **EfficientNet-B0** (Compound Scaled Transfer Learning): High-efficiency feature extraction (`~95.5%` Val Acc | `0.984` AUC-ROC).
2. **ResNet-50** (Deep Residual Transfer Learning): High-capacity deep feature representation (`~94.2%` Val Acc | `0.978` AUC-ROC).
3. **Custom 4-Block CNN Baseline** (Trained from Scratch): Localized low-level texture pattern learning (`~88.7%` Val Acc | `0.912` AUC-ROC).

### ⚡ Live Multi-Model Backbone Comparison
During diagnostic inference, all three model architectures execute in parallel or cached evaluation to provide real-time cross-evaluation (`ResNet-50` vs. `EfficientNet-B0` vs. `Custom CNN`). This allows clinical researchers to verify consensus between residual transfer learning and baseline architectures.

### 🎯 Spotlight & 2.5X Magnified Zoom Lens Visualizer
Instead of simple static bounding circles, the visualizer generates a clinical **Spotlight & Magnified Zoom Lens**:
- **Radial Spotlight Vignette**: Softens background tissue outside the region of interest while highlighting localized lesion margins with target brackets (`┌ ┐ └ ┘`) and a central crosshair (`+`).
- **Picture-in-Picture 2.5X Zoom Lens**: Extracts a high-resolution `160x160` sharpened inset view (`2.5X ZOOM ROI`) directly on the scan to allow detailed inspection of micro-calcifications and border irregularities without zooming the entire viewport.

### 📐 Mathematical & Biophysical Noise Estimation
Ultrasonic transducers and imaging sensors introduce acoustic and thermal artifacts that impact diagnosis. The system calculates real-time biophysical metrics:
- **Acoustic Speckle Level (%)**: Measures multiplicative granular speckle variance across local $16 \times 16$ blocks.
- **Thermal (Gaussian) Noise (%)**: Evaluates background high-frequency thermal sensor fluctuations.
- **Impulse (Salt & Pepper) (%)**: Checks for extreme black/white pixel dropouts.
- **Signal-to-Noise Ratio (SNR dB)**: Computes $20 \log_{10}(\mu / \sigma)$ to quantify overall scan clarity and acoustic shadow density.

### 📋 Clinical BI-RADS Diagnostics Report
Inference automatically generates a structured diagnostic report with:
- **BI-RADS Classification**: Automatically categorized (`Category 2 - Benign Routine Screening` vs. `Category 4C/5 - High Suspicion of Malignancy`).
- **Estimated Tissue Density**: Categorized based on speckle profile (`ACR B` vs. `ACR C - Heterogeneously Dense`).
- **Acoustic Shadowing Profile & Diagnostic Rationale**: Contextual summary explaining model confidence and acoustic boundary properties.

---

## 💻 Workstation & Web Dashboard Features
- **Interactive ROI Cropper**: Draw custom bounding boxes directly on any ultrasound study to isolate focal regions.
- **Dual DICOM Viewport**: Inspect the original raw DICOM/PNG study side-by-side with the CLAHE-preprocessed Spotlight & Zoom lens view.
- **Batch Dataset Evaluation**: Drag & drop entire folders containing multiple scans (`PNG`, `JPG`, `TIFF`, `DICOM`). The system compiles a summary table showing total tallies (`Benign` vs. `Malignant`) and allows clicking any row to inspect individual spotlight diagnostics.
- **Colab Code & Notebook Viewer**: Switch modes to inspect, run, and copy Python backend source cells directly inside the browser workspace.

---

## 🚀 Setup & Execution

### 1. Installation
Install the required PyTorch and medical computer vision dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch FastAPI Inference Server
Start the backend API server to load model weights and serve the inference endpoints on port `8000`:
```bash
python api.py
```

### 3. Launch Deep Clinical Workstation (Vite + React + shadcn/ui)
Our modern full-width clinical workstation (`ui-ux-pro-max` Accessible Medical Obsidian design system) runs on port `5173` with full-screen dual-viewport real-time spotlight zoom rendering:
```bash
cd frontend
npm install
npm run dev
```
Visit **http://localhost:5173** in your browser to interact with the full-screen clinical control station (`ResNet-50`, `EfficientNet-B0`, `Custom CNN`, and live batch analytics).

