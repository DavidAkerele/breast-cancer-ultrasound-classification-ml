# Chapter 3: Methodology

## 3.1 Preprocessing: CLAHE Optimization
Contrast Limited Adaptive Histogram Equalization (CLAHE) [Zuiderveld, 1994] addresses the limitations of standard Global Histogram Equalization (GHE) by operating on localized regions (tiles) and clipping contrast amplification:
1. **Tiling:** The luminance plane is split into a grid of $16 \times 16$ contextual tiles.
2. **Clipping:** For each tile, the histogram is computed and clipped at a threshold ($\beta = 2.0$). Surplus bins are redistributed uniformly across gray levels.
3. **Equalization:** The cumulative distribution function (CDF) is mapped for local pixel values:
   $$P(k) = \frac{1}{M \cdot N} \sum_{j=0}^k h_{\text{clipped}}(j)$$
4. **Interpolation:** Bilinear interpolation is applied across tile boundaries to eliminate artificial boundary edges.

## 3.2 Proposed ROI Reflection Square Padding Formulation
Given an input scan $I$ and lesion mask $M$, bounding box extrema $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$ are extracted. A 20% margin ratio ($\alpha = 0.20$) is added:
$$y_1 = \max(0, y_{\min} - \alpha \cdot h_{box}), \quad y_2 = \min(H, y_{\max} + \alpha \cdot h_{box})$$
$$x_1 = \max(0, x_{\min} - \alpha \cdot w_{box}), \quad x_2 = \min(W, x_{\max} + \alpha \cdot w_{box})$$

To convert the cropped matrix $I_{\text{crop}} \in \mathbb{R}^{h_c \times w_c}$ into a square tensor without aspect ratio distortion, the maximum dimension $d_{\max} = \max(h_c, w_c)$ is computed. Symmetric reflection boundary padding is applied:
$$pad_{\text{top}} = \lfloor (d_{\max} - h_c)/2 \rfloor, \quad pad_{\text{bottom}} = d_{\max} - h_c - pad_{\text{top}}$$
$$pad_{\text{left}} = \lfloor (d_{\max} - w_c)/2 \rfloor, \quad pad_{\text{right}} = d_{\max} - w_c - pad_{\text{left}}$$

Boundary pixels are mirrored using reflection padding (`BORDER_REFLECT_101`), preventing sharp edge gradient spikes. The padded square tensor is resized to $224 \times 224$ using bicubic interpolation ($s_x = s_y \implies \mathcal{AR} = 0.0\%$).

## 3.3 Deep Learning Architectures
1. **EfficientNet-B0 (5.3M parameters):** Employs compound scaling with Mobile Inverted Bottleneck (MBConv) blocks and Squeeze-and-Excitation (SE) channel attention [Tan & Le, 2019, Hu et al., 2018].
2. **ResNet-50 (25.6M parameters):** 50-layer deep residual network utilizing bottleneck residual units with identity shortcut mappings [He et al., 2016].
3. **Custom 4-Block CNN Baseline (1.2M parameters):** 4 Conv blocks ($32 \rightarrow 64 \rightarrow 128 \rightarrow 256$ filters), Batch Normalization, ReLU, $2 \times 2$ Max Pooling, Adaptive Average Pooling, and Dropout ($p=0.5$).

## 3.4 Classification Constraints (Softmax Exclusivity)
A clinical diagnosis must be mutually exclusive. An ultrasound scan is classified as either benign or malignant. We enforce this constraint using a Softmax activation on logits $\mathbf{z}$:
$$P(Y = c \mid \mathbf{x}) = \frac{\exp(z_c)}{\sum_{j=1}^K \exp(z_j)}, \quad \text{such that } \sum_{c=1}^K P(Y = c \mid \mathbf{x}) \equiv 1.000$$

## 3.5 Synthetic Acoustic Noise Simulation Engine
- **Rayleigh Speckle Noise:** $I_{\text{speckle}}(x, y) = I(x, y) + I(x, y) \cdot \eta_m(x, y)$, where $\eta_m \sim \mathcal{N}(0, \sigma^2)$ for $\sigma \in [0.01, 0.15]$.
- **Gaussian Thermal Noise:** $I_{\text{gaussian}}(x, y) = I(x, y) + \eta_a(x, y)$, where $\eta_a \sim \mathcal{N}(0, \sigma_g^2)$.
- **Impulse Noise:** $P(I=0) = p/2$, $P(I=255) = p/2$ for ADC transmission dropout simulation.
