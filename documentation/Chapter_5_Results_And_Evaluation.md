# Chapter 5: Results and Evaluation

## 5.1 Quantitative Performance Metrics
The models were evaluated on the validation split. Metrics include Accuracy, F1-Score, Precision, Recall, and Area Under the ROC Curve (AUC):

| Model Architecture | Accuracy | F1-Score | Precision | Recall (Sensitivity) | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | **95.5%** | **0.954** | **0.958** | **0.950** | **0.984** |
| **ResNet-50** | 94.2% | 0.941 | 0.945 | 0.938 | 0.978 |
| **Custom CNN** | 88.7% | 0.885 | 0.890 | 0.880 | 0.912 |

## 5.2 Discussion of Model Comparison
- **EfficientNet-B0**: Achieved the highest performance across all metrics. Its inverted residual blocks and attention-based squeeze-and-excitation layers allowed it to capture localized tissue densities and microcalcification patterns while minimizing the parameter footprint.
- **ResNet-50**: Showed comparable sensitivity but required significantly more parameters, making it more prone to overfitting on our mammography sample size.
- **Custom CNN Baseline**: Provided a solid baseline, proving that standard convolutional layers can classify mammograms reasonably well, though it lacked the deep spatial abstraction of the pre-trained backbones.
