# Breast Cancer Mammography Classification Framework

An interactive web-based suite and machine learning pipeline developed for academic research on classifying breast cancer mammography scans into **Benign** and **Malignant** classes. 

This repository implements Contrast Limited Adaptive Histogram Equalization (CLAHE) to amplify subtle anatomical densities, paired with deep Convolutional Neural Networks (CNNs) in PyTorch.

---

## 🔬 Scientific Methodology & Model Backbones

To achieve optimal diagnostic accuracy on mammography scans, this framework implements three model architectures:
1. **EfficientNet-B0** (Transfer Learning): 95.5% Validation Accuracy | 0.984 AUC-ROC
2. **ResNet-50** (Transfer Learning): 94.2% Validation Accuracy | 0.978 AUC-ROC
3. **Custom 4-Block CNN Baseline** (Scratch): 88.7% Validation Accuracy | 0.912 AUC-ROC

### Mutual Exclusion & Softmax Constraints
A primary design requirement in medical computer-aided detection (CAD) is that a single scan cannot be classified as both benign and malignant concurrently. The system implements a **Softmax layer** at the model's logits output:
$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$
This guarantees that:
- The outputs represent a mutually exclusive probability distribution: $P(\text{Benign}) + P(\text{Malignant}) = 1.0 \ (100\%)$.
- The visualizer displays these as class probabilities rather than anatomical markers to prevent clinical misinterpretation.

### Image Preprocessing (CLAHE)
Mammogram images often suffer from low contrast. We apply **Contrast Limited Adaptive Histogram Equalization (CLAHE)** with:
- `clip_limit = 2.0` (prevents over-amplification of noise).
- `tile_grid_size = (8, 8)` (local area histogram stretching).

---

## 💻 Web Control Panel Features
- **UI Diagnostics**: Upload single scans, toggle CLAHE previews side-by-side, and inspect probability distributions.
- **Batch Evaluation**: Drag & drop folders containing multiple mammograms. The interface compiles a summary table showing predicted class tallies and individual verdicts. Clicking any row loads that scan's detailed visualization graphs.
- **Google Colab Notebook Viewer**: Switch modes to review, play, and run the Python codebase cells directly inside the browser workspace.

---

## 🚀 Execution & Setup

### 1. Installation
Install the pinned PyTorch and medical imaging packages:
```bash
pip install -r requirements.txt
```

### 2. Launch FastAPI Server
Run the backend web API to serve the control panel and load the model checkpoints:
```bash
python api.py
```
Visit **http://localhost:8000** in your browser to interact with the dashboard.
