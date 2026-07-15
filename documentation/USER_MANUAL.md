# OncoVision Clinical Control Center - User Manual

This manual provides an overview of the OncoVision Clinical Control Center, detailing the application's interface, its features, and the underlying repository codebase for classifying breast ultrasound (`BUSI`, `BrEaST`, `OASBUD`) and mammography scans.

The system features a dual-layer architecture:
1. **Deep Clinical Workstation (`frontend/`)**: A modern **Vite + React + TypeScript + shadcn/ui + Tailwind CSS v4** interface running on port `5173` featuring full-width responsive layout (`ui-ux-pro-max` Obsidian & Cyan/Emerald design tokens), real-time instant scan loading, multi-model consensus grid, and live batch analytics.
2. **High-Performance Inference Backend (`api.py`)**: A **FastAPI + PyTorch** backend running on port `8000` supporting `ResNet-50`, `EfficientNet-B0`, and `Custom CNN` evaluation with `CLAHE` contrast enhancement and biophysical acoustic noise metrics.

---

## 🖥️ Web Workstation Interface Guide

The OncoVision clinical dashboard is a responsive medical computer-aided detection workspace organized into structured panels:

```
+-----------------------------------------------------------------------------------+
|                              ONCOVISION DASHBOARD                                 |
+----------------------+----------------------------------+-------------------------+
| ⚙️ 1. PARAMETERS      | 📁 2. DIAGNOSTIC INPUT           | 📊 3. ASSESSMENT        |
| - Model Selector     | - Select Dataset (Train/Val/Test)| - Verdict Banner        |
| - CLAHE Toggle       | - Scan Selection Dropdown        | - Dual Viewport Stage   |
| - Quick Test Bench   | - Drag & Drop / Browse Files     | - Probability Bars      |
|                      | - Execute AI Diagnosis Button    | - BI-RADS Report        |
|                      +----------------------------------+ - Multi-Model Compare   |
|                      | 📝 CLINICAL AUDIT LOGS           | - Mathematical Noise    |
+----------------------+----------------------------------+-------------------------+
```

### 1. Parameters & Specifications Panel
- **Model Selector**: Choose the active neural backbone (`Custom CNN`, `ResNet-50`, or `EfficientNet-B0`).
- **CLAHE Enhancement Toggle**: Controls the **Contrast Limited Adaptive Histogram Equalization** preprocessing filter.
- **Quick Test Cases**: Instant benchmark buttons (`Benign Scan` & `Malignant Scan`) to test the classification pipeline and verify visual diagnostics.

### 2. Diagnostic Input & Ingestion Panel
- **Dataset Split Selector**: Filter pre-indexed database scans by `Validation`, `Training`, or `Testing` partitions.
- **Patient Scan Dropdown**: Select existing ultrasound studies (`BUSI`, `BrEaST`, `OASBUD`) by filename.
- **Drag & Drop Zone**: Upload custom `PNG`, `JPG`, `TIFF`, or `DICOM` files directly from your computer. Supports single files, multi-file selections, or entire folders.
- **Execute AI Diagnosis Button**: Triggers forward propagation, feature extraction, CLAHE filtering, spotlight localization, and biophysical noise evaluation.

### 3. Diagnostic Assessment & Clinical Output Panel
- **Primary AI Verdict Banner**: Renders the predicted classification (**BENIGN** vs. **MALIGNANT**) with classification confidence badge (`%`).
- **Dual DICOM Viewport**: Displays the `Original Scan` side-by-side with the `Spotlight & 2.5X Zoom View`.
  - **Spotlight Vignette**: Focuses attention on focal lesion structures with target brackets (`┌ ┐ └ ┘`) and a central crosshair (`+`).
  - **2.5X Magnified Inset Lens**: A Picture-in-Picture (`160x160` px) sharpened zoom box showing micro-details of the target lesion boundary without resizing the full viewport.
  - **Interactive ROI Crop**: Click `Interactive ROI Crop` to drag a custom bounding box around any tissue abnormality for targeted evaluation.
- **Class Probability Breakdown**: Real-time progress bars indicating Softmax probability distributions for `Benign` vs `Malignant` outcomes.
- **Clinical BI-RADS Report**: Structured diagnostic metrics showing:
  - **BI-RADS Classification** (`Category 2 - Benign Routine Screening` vs `Category 4C/5 - High Suspicion`).
  - **Estimated Tissue Density** (`ACR B` vs `ACR C - Heterogeneously Dense`).
  - **Acoustic Shadowing Profile & Diagnostic Rationale**.
  - **Export PDF Button**: Instantly downloads a clean clinical report for patient records.
- **Live Multi-Model Backbone Comparison**: Cross-evaluates the scan across `ResNet-50`, `EfficientNet-B0`, and `Custom CNN` concurrently, displaying individual class predictions and confidence scores.
- **Mathematical & Biophysical Noise Profile**: Calculates real-time sensor and acoustic artifacts:
  - **Acoustic Speckle Level (%)**
  - **Thermal (Gaussian) Noise (%)**
  - **Impulse (Salt & Pepper) (%)**
  - **Signal-to-Noise Ratio (SNR dB)**

---

## 🔬 Batch Mode Evaluation

When uploading multiple files or dropping a directory into the upload zone:
1. The dashboard enters **Batch Mode**.
2. **Summary Cards** display `Total Scans`, `Benign Predicted`, and `Malignant Predicted` counts.
3. A **Batch Results Table** indexes every processed filename, its diagnostic verdict, and confidence rating.
4. **Interactive Drill-Down**: Clicking any row in the batch table immediately loads that scan's dual DICOM viewport, spotlight lens, and BI-RADS report.

---

## 📁 Codebase Architecture

1. **[config.py](file:///Users/davidakerele/Documents/Dissertation/config.py)**: System parameters, device configuration (`CPU` vs `CUDA` vs `MPS`), hyperparameters, and class labels (`benign`, `malignant`).
2. **[models.py](file:///Users/davidakerele/Documents/Dissertation/models.py)**: Implements the Custom 4-Block CNN and loads transfer learning backbones (`ResNet-50`, `EfficientNet-B0`).
3. **[dataset.py](file:///Users/davidakerele/Documents/Dissertation/dataset.py)**: Multi-dataset loader combining `BUSI`, `BrEaST`, and `OASBUD` datasets, normalization transforms, and OpenCV CLAHE contrast equalization.
4. **[api.py](file:///Users/davidakerele/Documents/Dissertation/api.py)**: FastAPI backend server exposing `/predict`, `/predict/batch`, `/noise-analysis`, and hosting the web dashboard on `http://localhost:8000`. Includes the `draw_lesion_spotlight_zoom` engine, multi-model consensus evaluator, and biophysical noise estimation algorithms.
5. **[train.py](file:///Users/davidakerele/Documents/Dissertation/train.py)**: Command-line training routine supporting early stopping, learning rate scheduling, and checkpoint serialization.
6. **[web/index.html](file:///Users/davidakerele/Documents/Dissertation/web/index.html)**: High-contrast workstation HTML dashboard layout.
7. **[web/script.js](file:///Users/davidakerele/Documents/Dissertation/web/script.js)**: Frontend controller handling asynchronous API calls, DOM rendering, interactive cropping, and PDF exports.
8. **[web/style.css](file:///Users/davidakerele/Documents/Dissertation/web/style.css)**: Glassmorphic, dark-mode radiologist workstation stylesheet.
