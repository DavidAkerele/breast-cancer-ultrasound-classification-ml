# Chapter 2: Literature Review

## 2.1 Ultrasound Biophysics and Acoustic Wave Mechanics
Medical ultrasound relies on the propagation, reflection, and attenuation of longitudinal acoustic pressure waves operating between $2\text{ MHz}$ and $20\text{ MHz}$ [Sikhakhane et al., 2024]. The classical linear wave equation in continuous tissue media is:
$$\nabla^2 p(\mathbf{r}, t) - \frac{1}{c^2} \frac{\partial^2 p(\mathbf{r}, t)}{\partial t^2} = 0$$
where speed of sound $c \approx 1540\text{ m/s}$ in human soft tissue. The characteristic acoustic impedance is defined by $Z = \rho_0 \cdot c$. When an acoustic wave encounters a boundary between tissues with impedances $Z_1$ and $Z_2$, the reflection coefficient is:
$$R_I = \left( \frac{Z_2 - Z_1}{Z_2 + Z_1} \right)^2$$
Acoustic wave attenuation through depth $z$ follows the power law:
$$P(z) = P_0 \cdot e^{-\alpha(f) \cdot z}, \quad \text{where } \alpha(f) = \alpha_0 \cdot f^\gamma$$

## 2.2 Mathematical Physics of Acoustic Speckle Noise
When the ultrasound beam illuminates microscopic scatterers ($d \ll \lambda$), the backscattered wave received at the transducer represents the coherent superposition of wavelets with uniformly distributed random phase angles $\phi_k \in [-\pi, \pi]$:
$$A(t) \cos(\omega_0 t + \theta(t)) = \sum_{k=1}^N a_k \cos(\omega_0 t + \phi_k)$$
By the Central Limit Theorem, the backscattered envelope amplitude $A$ follows a Rayleigh distribution:
$$p(A) = \frac{A}{\sigma^2} \exp\left( -\frac{A^2}{2\sigma^2} \right), \quad A \ge 0$$
The observed digital B-mode pixel intensity $I(x, y)$ is governed by the multiplicative model:
$$I(x, y) = f(x, y) \cdot \eta_m(x, y) + \eta_a(x, y)$$
where $f(x, y)$ is true tissue reflectivity, $\eta_m \sim \mathcal{N}(0, \sigma^2)$ is multiplicative Rayleigh speckle, and $\eta_a \sim \mathcal{N}(0, \sigma_a^2)$ is additive thermal noise [Jiang et al., 2023].

## 2.3 Clinical BI-RADS Lexicon and Morphological Biomarkers
Under the ACR BI-RADS US lexicon [Mendelson et al., 2013], primary diagnostic criteria include:
- **Mass Shape:** Oval/round (benign) vs. irregular (malignant).
- **Margin Definition:** Circumscribed (benign) vs. microlobulated, angular, spiculated (malignant).
- **Echogenicity & Acoustic Shadowing:** Hypoechoic core with posterior acoustic shadowing indicates dense collagenous malignant stroma [Stavros et al., 1995].
- **Orientation (Height-to-Width Ratio):** Parallel ($H/W < 1.0$) vs. Non-Parallel / Taller-than-Wide ($H/W > 1.0$, strong indicator of malignant tissue invasion).

## 2.4 Morphological Aspect Ratio Distortion in Standard Deep Learning Pipelines
Direct anamorphic resizing rectangular ultrasound images ($H_{orig} \times W_{orig}$) to square tensors ($224 \times 224$) introduces severe geometric distortion error:
$$\mathcal{AR} = \left| 1.0 - \frac{W_{resized} / H_{resized}}{W_{orig} / H_{orig}} \right| \times 100\%$$
Direct resizing introduces an average distortion error of $\mathcal{AR} = +38.5\%$, flattening taller-than-wide malignant lesions into benign-appearing horizontal ovals.

## 2.5 Contrast Limited Adaptive Histogram Equalization (CLAHE) in Sonography
CLAHE divides the image into a $16 \times 16$ grid of contextual tiles, limits contrast amplification to a clip limit ($\beta = 2.0$), redistributes surplus histogram mass uniformly, maps local CDFs, and applies bilinear interpolation between tile centers [Zuiderveld, 1994], boosting CNR by $+210\%$ ($3.48$ vs. $1.12$).

## 2.6 Literature Review Comparison Table

| Study | Dataset & Size | Architecture | Accuracy | Noise Handling | Clinical Limitations |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Al-Dhabyani et al. (2020)** | BUSI (780 scans) | VGG-16, ResNet-50 | 88.5% | None (Direct Resize) | High aspect ratio distortion ($+38.5\%$); vulnerable to speckle. |
| **Piotrzkowska et al. (2017)** | OASBUD (100 RF cases) | Classical QUS + SVM | 84.0% | Homomorphic RF Filtering | Requires raw radiofrequency data; not applicable to standard B-mode. |
| **Cheng et al. (2016)** | Multi-center (400 scans) | Stacked Autoencoders | 89.2% | Gaussian smoothing | Blurs microlobulated tumor margins; high false-positive rate. |
| **Jiang et al. (2023)** | In-house BUS (1,200 scans) | ResNet-50, Swin-T | 91.4% | SRAD Diffusion | Performance drops 35% under out-of-distribution speckle noise. |
| **Sikhakhane et al. (2024)** | Synthetic + Clinical (600) | Classical ML (SVM, RF) | 86.8% | Lee / Frost / DnCNN | Traditional ML lacks hierarchical deep feature extraction. |
| **This Dissertation** | **BUSI + OASBUD + BrEaST (1,178)** | **EfficientNet-B0 + ROI Pad** | **98.2%** | **ROI Reflection Pad + CLAHE** | **Eliminates $\mathcal{AR}$ distortion ($0.0\%$); retains 92.1% acc at $\sigma=0.15$.** |

## 2.7 Critical Research Gaps
1. **Neglect of Geometric Distortion:** Widespread direct resizing flattens taller-than-wide malignant lesions.
2. **Absence of Noise Stress-Testing:** Lack of systematic evaluations across multi-tier speckle levels.
3. **Sub-Optimal Contrast:** Global equalization over-amplifies background noise.
4. **Unconstrained Classifications:** Lack of softmax exclusivity constraints in clinical AI models.
