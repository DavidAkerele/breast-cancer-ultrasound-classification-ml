# Chapter 3: Methodology

## 3.1 Preprocessing: CLAHE Optimization
Contrast Limited Adaptive Histogram Equalization (CLAHE) addresses the limitations of standard Global Histogram Equalization (GHE) by operating on localized regions (tiles) and clipping contrast amplification:
1. **Tiling**: The image is split into a grid of $8 \times 8$ contextual tiles.
2. **Clipping**: For each tile, the histogram is computed and clipped at a threshold (clip limit = $2.0$). Bins exceeding this limit are redistributed uniformly.
3. **Equalization**: The cumulative distribution function (CDF) is mapped for local pixel values:
   $$s = T(r) = \int_{0}^{r} p_r(w) dw$$
4. **Interpolation**: Bilinear interpolation is applied to smooth boundaries and prevent artifact edges.

## 3.2 Deep Learning Architectures

### 3.2.1 Custom 4-Block CNN
Our baseline CNN comprises:
- **Four Convolutional Blocks**: Each contains two $3 \times 3$ Conv layers, Batch Normalization, ReLU activation, and a $2 \times 2$ Max Pooling layer. Filters double (32, 64, 128, 256) to capture hierarchical features.
- **Fully-Connected Head**: Adaptive Average Pooling maps feature dimensions to $1 \times 1$. We apply Flatten, a 128-unit dense layer with Dropout ($p=0.5$), and a linear mapping to the classification logits.

### 3.2.2 ResNet-50
ResNet-50 uses bottleneck blocks with residual connections to enable training of deeper networks. The residual block computes:
$$y = F(x, \{W_i\}) + W_s x$$
where $F$ represents the residual mapping and $W_s$ is a linear projection matching dimensions.

### 3.2.3 EfficientNet-B0
EfficientNet uses compound scaling to optimize network depth, width, and input resolution. It employs Mobile Inverted Bottleneck Convolutions (MBConv) with Squeeze-and-Excitation attention to focus on relevant mammographic regions.

## 3.3 Classification Constraints (Softmax Exclusivity)
A clinical diagnosis must be mutually exclusive. A mammogram scan is classified as either benign or malignant. We enforce this constraint using a Softmax activation on logits $z$:
$$P(y = i | x) = \frac{e^{z_i}}{\sum_{j=1}^{2} e^{z_j}}$$
This ensures $P(\text{Benign}) + P(\text{Malignant}) = 1.0$, preventing physically impossible overlapping diagnoses.
