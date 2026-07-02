# Chapter 2: Literature Review

## 2.1 Screening Mammography and Clinical Biomarkers
Mammograms are evaluated for specific radiological features that indicate malignancy:
- **Masses**: Glandular densities characterized by their margins and shape. Irregular or spiculated margins have a high correlation with invasive ductal carcinoma.
- **Microcalcifications**: Small calcium deposits ($0.1$ to $1.0\text{ mm}$) within the breast ductal system. While common and often benign, clusters of pleomorphic or linear calcifications can indicate early-stage cancer.

## 2.2 Evolution of Computer-Aided Detection (CAD)
Early CAD systems deployed in the 1990s used mathematical algorithms to detect calcifications and masses:
- **Mass Detection**: Relied on bilateral asymmetry and template matching.
- **Calcification Detection**: Used bandpass filtering and thresholding.

These systems failed to improve diagnostic outcomes significantly. A study by Fenton et al. (2007) showed that CAD use was associated with decreased specificity and no improvement in detection rates, primarily because hand-crafted features could not capture the variability of breast anatomy.

## 2.3 Deep Learning and Representation Learning
Deep learning replaces hand-crafted features with end-to-end representation learning. Convolutional neural networks extract local patterns (edges, textures) in early layers and compile them into high-level features (spiculation, density profiles) in deeper layers:
- **Transfer Learning**: Pre-training models on ImageNet allows them to learn general image features. Fine-tuning these models on mammogram datasets helps achieve high accuracy even with limited clinical data.
- **Contrast Enhancement**: Contrast stretching techniques like CLAHE are critical for improving CNN performance by making lesion boundaries clearer in dense tissue.
