# Chapter 5: Results and Evaluation

## 5.1 Multi-Model Diagnostic Performance Matrix
The models were evaluated on the held-out validation and test partitions across multi-center datasets (**BUSI**, **OASBUD**, **BrEaST**):

### Table 1: Primary Diagnostic Metrics across Model Backbones
| Architecture | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0 (Proposed)** | **98.2%** | **97.5%** | **98.7%** | **98.5%** | **0.981** | **0.991** |
| **ResNet-50 (Transfer)** | 96.5% | 96.2% | 96.8% | 96.8% | 0.965 | 0.985 |
| **Custom 4-Block CNN** | 89.4% | 85.0% | 93.3% | 89.8% | 0.889 | 0.924 |

### Table 7: Multi-Model Architecture Macro & Micro Metric Performance Matrix
| Model Architecture | Epoch | Macro F1 | Normal F1 | Benign F1 | Malignant F1 | Micro F1 | Micro AUC | Macro AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0 (Proposed)** | 25 | **0.9821** | **0.9773** | **0.9872** | **0.9818** | **0.9831** | **0.9865** | **0.9885** |
| **ResNet-50 (Transfer)** | 25 | 0.9531 | 0.9213 | 0.9936 | 0.9444 | 0.9605 | 0.9625 | 0.9611 |
| **Custom Ultrasound CNN** | 20 | 0.8466 | 0.8000 | 0.8774 | 0.8624 | 0.8531 | 0.9106 | 0.9079 |
| **0.95 Pseudo-Label Ensemble** | 62 | **0.9871** | **0.9767** | **0.9936** | **0.9910** | **0.9887** | **0.9895** | **0.9891** |

## 5.2 Noise Resilience Stress-Testing
Under increasing Rayleigh speckle noise ($\sigma \in [0.01, 0.15]$), the Proposed ROI Crop + Square Pad pipeline maintains high accuracy, while direct resizing degrades rapidly:

| Preprocessing Strategy | Clean ($\sigma=0.0$) | $\sigma=0.01$ | $\sigma=0.05$ | $\sigma=0.10$ | $\sigma=0.15$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Proposed: ROI Crop + Square Pad** | **98.2%** | **97.9%** | **96.8%** | **94.5%** | **92.1%** |
| Strategy A: Direct Anamorphic Resize | 88.4% | 85.2% | 79.1% | 68.4% | 52.3% |
| Strategy B: Default Center Crop | 81.2% | 78.4% | 74.2% | 62.1% | 48.7% |

## 5.3 Semi-Supervised Pseudo-Labeling Confidence Threshold Ablation

### Table 8: Semi-Supervised Pseudo-Labeling Confidence Threshold Ablation Matrix
| Method / Strategy | Confidence Threshold (τ) | Added | Train Size | Epoch | Macro F1 | Malignant F1 | Micro F1 | Macro AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Supervised)** | N/A | 0 | 828 | 25 | 0.9606 | 0.9541 | 0.9605 | 0.9730 |
| **Pseudo-Label** | 0.80 | 520 | 1,348 | 32 | 0.9661 | 0.9818 | 0.9661 | 0.9783 |
| **Pseudo-Label** | 0.90 | 415 | 1,243 | 46 | 0.9702 | 0.9818 | 0.9718 | 0.9770 |
| **Pseudo-Label (Proposed)** | **0.95** | **350** | **1,178** | **62** | **0.9871** | **0.9910** | **0.9887** | **0.9891** |
| **Pseudo-Label** | 0.97 | 280 | 1,108 | 44 | 0.9837 | 0.9815 | 0.9831 | 0.9859 |
| **Pseudo-Label** | 0.99 | 185 | 1,013 | 45 | 0.9749 | 0.9725 | 0.9774 | 0.9802 |

## 5.4 Discussion of Findings
1. **Clinical Safety in Malignant Tumor Detection:** EfficientNet-B0 produces only 2 false negatives out of 80 malignant cases ($97.5\%$ sensitivity), outperforming the Custom CNN baseline (12 false negatives).
2. **Aspect Ratio Preservation:** Proposed ROI padding eliminates the $+38.5\%$ distortion error ($\mathcal{AR} = 0.0\%$), preserving the diagnostic utility of the $H/W$ biomarker.
3. **Biophysical Enhancement:** CLAHE yields a $+210\%$ boost in Contrast-to-Noise Ratio ($3.48$ vs. $1.12$).
4. **Pseudo-Labeling Sweet Spot:** Filtering candidates at $\tau = 0.95$ delivers optimal performance by balancing confirmation bias and data starvation.
