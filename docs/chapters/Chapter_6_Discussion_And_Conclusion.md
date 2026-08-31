# Chapter 6: Discussion and Conclusion

## 6.1 Findings and Research Contributions
This dissertation successfully developed a noise-resilient computer-aided diagnosis framework for automated breast ultrasound classification. 

Key contributions include:
- **Aspect Ratio Preservation:** Eliminating geometric lesion distortion through reflection square padding ($H/W = 1.0, \mathcal{AR} = 0.0\%$).
- **Acoustic Noise Resilience:** Sustaining 96.8% accuracy under clinical speckle ($\sigma=0.05$) and 92.1% under extreme noise ($\sigma=0.15$).
- **Contrast Amplification:** Boosting CNR by $+210\%$ ($3.48$ vs. $1.12$) and SNR by $+10.4\text{ dB}$ via localized CLAHE.
- **Softmax Exclusivity Constraint:** Ensuring mutually exclusive clinical probabilities ($P(\text{Benign}) + P(\text{Malignant}) = 1.0$).
- **Semi-Supervised Threshold Optimization:** Empirically validating $\tau = 0.95$ as optimal for student network retraining.
- **Interactive Acoustic Control Center:** Delivering a 4-stage comparative noise laboratory and BI-RADS clinical reporting workstation.

## 6.2 Limitations and Clinical Considerations
- **2D B-Mode Focus:** Evaluated on static 2D B-mode images; future extensions should address 3D Automated Breast Ultrasound (ABUS).
- **Initial ROI Localization:** The pipeline utilizes initial bounding box coordinates, highlighting the importance of robust segmentation frontends.
- **Scanner Hardware Heterogeneity:** Inter-manufacturer variability across transducer hardware remains a key clinical factor.

## 6.3 Future Research Roadmap
1. **Generative Diffusion Despeckling:** Unsupervised zero-shot despeckling using Denoising Diffusion Probabilistic Models (DDPM) [Ho et al., 2020].
2. **Vision Transformers (ViT):** Modeling long-range spatial correlations between lesions and acoustic shadowing columns [Dosovitskiy et al., 2021].
3. **Multi-Modal Fusion:** Integrating B-mode with Color Doppler and Shear-Wave Elastography (SWE).
4. **Federated Learning:** Privacy-preserving distributed training across multi-hospital consortia.
5. **Edge POCUS Deployment:** TensorRT optimization for point-of-care ultrasound devices.
