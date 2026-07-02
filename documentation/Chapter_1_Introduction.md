# Chapter 1: Introduction

## 1.1 Research Context and Background
Breast cancer represents a major global health challenge, accounting for the highest incidence of oncological diagnoses among women globally. The World Health Organization (WHO) estimates that millions of new cases are diagnosed annually, with early detection remaining the single most critical factor in improving survival rates. Mammography screening, which uses low-dose X-rays to visualize internal tissue structures, is the primary radiological tool for identifying early signs of breast abnormalities before they become clinically palpable.

However, the efficacy of mammography screening is highly dependent on the quality of the image and the expertise of the interpreting radiologist. Glandular breast structures and malignant masses exhibit similar attenuation properties under X-rays, appearing as white regions. This overlap causes dense breast tissue to mask malignancies, resulting in high rates of false negatives. Conversely, normal glandular tissue overlap can simulate the appearance of masses, leading to false positives, patient anxiety, and unnecessary follow-up biopsies.

## 1.2 Problem Statement
The diagnostic accuracy of screening mammograms faces two primary challenges:
1. **Low Visual Contrast**: Masses and architectural distortions are frequently obscured by dense fibroglandular tissue, especially in younger patients.
2. **Inter-Observer Variability**: Visual interpretation is subjective. Cognitive fatigue and differences in clinical experience can lead to diagnostic errors.

To address these limitations, computer-aided detection (CAD) systems have emerged. Early CAD systems suffered from high false-positive rates due to reliance on hand-crafted features. Deep Convolutional Neural Networks (CNNs) offer a promising alternative by learning feature representations directly from the raw pixel data.

## 1.3 Aims and Objectives
The aim of this dissertation is to build and evaluate a deep learning-based CAD framework for binary mammography classification (Benign vs. Malignant). The specific objectives are:
- Develop a preprocessing module utilizing **Contrast Limited Adaptive Histogram Equalization (CLAHE)** to improve image contrast.
- Implement and train three distinct model architectures: **ResNet-50**, **EfficientNet-B0**, and a **Custom CNN** baseline.
- Establish a clinically valid prediction system where outputs are strictly constrained to a mutually exclusive softmax probability distribution, preventing physically impossible dual diagnoses.
- Construct an interactive diagnostic control center to display neural classifications, comparisons, and audit logs.
