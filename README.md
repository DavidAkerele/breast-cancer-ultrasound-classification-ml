# Automated Breast Ultrasound Cancer Classification Framework

**Master of Science (MSc) Dissertation Research Project**  
**Author:** Akerele David Damilola  
**Institution:** Department of Computing and Mathematics | Faculty of Science and Engineering  
**Academic Year:** 2026  

---

## 🔬 Project Overview

This repository houses the complete, end-to-end research framework for the MSc Dissertation: **"Breast Cancer Ultrasound Classification Using Machine Learning"**. The project investigates the quantifiable effects of acoustic degradation (multiplicative Rayleigh speckle noise, thermal sensor noise, and impulse transmission artifacts) and geometric spatial preprocessing on the diagnostic stability and latent feature representations of deep convolutional neural networks.

The framework integrates multi-center benchmark clinical imaging repositories (**BUSI**, **OASBUD**, and **BrEaST**), a novel **Proposed ROI Reflection Square Padding Algorithm** that eliminates the $+38.5\%$ aspect ratio distortion error ($\mathcal{AR} = 0.0\%$), localized **Contrast Limited Adaptive Histogram Equalization (CLAHE)** boosting Contrast-to-Noise Ratios by $+210\%$, and deep transfer learning backbones (**EfficientNet-B0**, **ResNet-50**, and a **Custom 4-Block CNN** baseline) in PyTorch.

---

## 📁 Repository Software Architecture

```
Dissertation/
├── api.py                                       # Root Launcher: FastAPI Clinical Backend
├── train.py                                     # Root Launcher: Multi-Model Training Pipeline
├── evaluate.py                                  # Root Launcher: Noise Resilience & Metric Evaluation
├── predict.py                                   # Root Launcher: Single & Batch Inference Engine
├── requirements.txt                             # Python Dependencies Specification
├── README.md                                    # Project Master Documentation
├── Breast_Cancer_Ultrasound_Classification_Dissertation.ipynb  # Interactive Scientific Research Notebook
│
├── src/                                         # Core Modular Python Package
│   ├── __init__.py                              # Package Initializer
│   ├── config.py                                # Hyperparameters, Preprocessing & Path Config
│   ├── dataset.py                               # PyTorch Dataset, CLAHE & ROI Reflection Padding
│   ├── models.py                                # Neural Architectures (EfficientNet, ResNet, Custom CNN)
│   ├── train.py                                 # Training, Validation & Checkpoint Management
│   ├── evaluate.py                              # Metric Computation, Confusion Matrices & ROC Plots
│   ├── predict.py                               # Diagnostic Inference Functions
│   ├── compute_project_tables.py                # Scientific Metric & Ablation Table Generator
│   └── api.py                                   # FastAPI REST API & Clinical Endpoints
│
├── latex/                                       # Master's Thesis LaTeX Source Files
│   ├── main.tex                                 # Master LaTeX Root Document
│   ├── abstract.tex                             # Comprehensive Dissertation Abstract
│   ├── declaration.tex                          # Academic Integrity Declaration
│   ├── acknowledgements.tex                     # Acknowledgements
│   ├── abbreviations.tex                        # Clinical & Technical Terminology List
│   ├── references.bib                           # 30+ Peer-Reviewed Literature Bibliography
│   ├── figures/                                 # Publication Figures (Fig 1 to Fig 8)
│   └── chapters/                                # Full Dissertation Chapters
│       ├── chapter1.tex                         # Chapter 1: Introduction
│       ├── chapter2.tex                         # Chapter 2: Literature Review
│       ├── chapter3.tex                         # Chapter 3: Methodology & Experimental Design
│       ├── chapter4.tex                         # Chapter 4: Results & Comparative Analysis
│       ├── chapter5.tex                         # Chapter 5: Discussion & Critical Reflection
│       └── chapter6.tex                         # Chapter 6: Conclusion & Future Work
│
├── docs/                                        # Comprehensive Academic Documentation
│   ├── DISSERTATION.md                          # Full MSc Dissertation in Markdown
│   ├── USER_MANUAL.md                           # System & User Manual
│   ├── OVERLEAF_INTEGRATION_GUIDE.md            # Overleaf Git Sync Guide
│   ├── chapters/                                # Individual Markdown Chapters (Ch 1 to 6)
│   │   ├── Chapter_1_Introduction.md
│   │   ├── Chapter_2_Literature_Review.md
│   │   ├── Chapter_3_Methodology.md
│   │   ├── Chapter_4_Implementation_And_Training.md
│   │   ├── Chapter_5_Results_And_Evaluation.md
│   │   └── Chapter_6_Discussion_And_Conclusion.md
│   └── analysis/                                # Deep Theoretical & Empirical Reports
│       ├── IMAGE_RESIZING_AND_CROPPING_NOISE_ANALYSIS.md
│       └── TABLE_INTERPRETATION_AND_METHODOLOGY_GUIDE.md
│
├── scripts/                                     # Data Curation & Preprocessing Tools
│   ├── download_datasets.py                     # Multi-Center Dataset Downloader
│   ├── organize_busi.py                         # BUSI Dataset Preprocessor
│   ├── organize_oasbud.py                       # OASBUD Dataset Formatter
│   ├── data_split.py                            # Stratified Train/Val/Test Splitter
│   └── create_dummy_data.py                     # Synthetic Test Data Generator
│
├── web/                                         # Interactive Clinical Diagnostic Dashboard
│   ├── index.html                               # Dashboard User Interface
│   ├── style.css                                # Dark Mode Clinical Styling
│   └── script.js                                # Frontend Controller & Real-Time Charting
│
├── data/                                        # Multi-Center Ultrasound Repositories
│   ├── busi/                                    # Formatted BUSI Dataset
│   ├── oasbud/                                  # Formatted OASBUD Dataset
│   └── BrEaST/                                  # Formatted BrEaST Dataset
│
└── outputs/                                     # Model Weights & Benchmark Artifacts
    ├── best_ultrasound_model.pth
    ├── efficientnet_b0_model.pth
    ├── resnet50_model.pth
    ├── custom_cnn_model.pth
    ├── confusion_matrix.png
    └── roc_curve.png
```

---

## ⚡ Quick Start & Execution Guide

### 1. Environment Installation
Install the necessary PyTorch and scientific computing dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch Interactive Clinical Dashboard
Start the FastAPI server (serves the clinical web interface on `http://localhost:8000`):
```bash
python api.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser to access the real-time inference workstation, 4-stage noise comparative laboratory, and BI-RADS clinical report generator.

### 3. Model Training & Validation
To train or fine-tune an architecture (`efficientnet_b0`, `resnet50`, or `custom_cnn`):
```bash
python train.py --model efficientnet_b0 --epochs 25
```

### 4. Noise Sensitivity & Diagnostic Evaluation
To evaluate trained checkpoints on held-out test partitions and generate confusion matrices and ROC curves:
```bash
python evaluate.py --split test
```

### 5. Single Image Inference
To execute diagnostic inference on a specific sonogram:
```bash
python predict.py --image data/busi/val/malignant/sample.png
```

### 6. Reproduce Scientific Tables
To recompute all quantitative macro/micro metrics and pseudo-labeling ablation tables:
```bash
python src/compute_project_tables.py
```

---

## 📊 Core Scientific Benchmark Results

### Primary Diagnostic Performance (Clean Held-Out Validation)
| Model Backbone | Accuracy (%) | Sensitivity (%) | Specificity (%) | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0 (Proposed)** | **98.2%** | **97.5%** | **98.7%** | **0.981** | **0.991** |
| **ResNet-50 (Transfer)** | 96.5% | 96.2% | 96.8% | 0.965 | 0.985 |
| **Custom 4-Block CNN** | 89.4% | 85.0% | 93.3% | 0.889 | 0.924 |

### Noise Resilience Benchmark (Rayleigh Speckle Scatter $\sigma$)
| Preprocessing Strategy | Clean ($\sigma=0.0$) | $\sigma=0.01$ | $\sigma=0.05$ | $\sigma=0.10$ | $\sigma=0.15$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Proposed: ROI Crop + Square Pad** | **98.2%** | **97.9%** | **96.8%** | **94.5%** | **92.1%** |
| Strategy A: Direct Anamorphic Resize | 88.4% | 85.2% | 79.1% | 68.4% | 52.3% |
| Strategy B: Default Center Crop | 81.2% | 78.4% | 74.2% | 62.1% | 48.7% |

---

## 📚 Master's Thesis Documentation Links
- 📄 **Full Dissertation (Markdown):** [docs/DISSERTATION.md](file:///Users/davidakerele/Documents/Dissertation/docs/DISSERTATION.md)
- 📝 **LaTeX Thesis Directory:** [latex/](file:///Users/davidakerele/Documents/Dissertation/latex/main.tex)
- 📓 **Jupyter Research Notebook:** [Breast_Cancer_Ultrasound_Classification_Dissertation.ipynb](file:///Users/davidakerele/Documents/Dissertation/Breast_Cancer_Ultrasound_Classification_Dissertation.ipynb)
- 📖 **User & System Manual:** [docs/USER_MANUAL.md](file:///Users/davidakerele/Documents/Dissertation/docs/USER_MANUAL.md)
- 🔬 **Biophysical Noise & Aspect Ratio Report:** [docs/analysis/IMAGE_RESIZING_AND_CROPPING_NOISE_ANALYSIS.md](file:///Users/davidakerele/Documents/Dissertation/docs/analysis/IMAGE_RESIZING_AND_CROPPING_NOISE_ANALYSIS.md)
- 📊 **Table Interpretation & Methodology Guide:** [docs/analysis/TABLE_INTERPRETATION_AND_METHODOLOGY_GUIDE.md](file:///Users/davidakerele/Documents/Dissertation/docs/analysis/TABLE_INTERPRETATION_AND_METHODOLOGY_GUIDE.md)
