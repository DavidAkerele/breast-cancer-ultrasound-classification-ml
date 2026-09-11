# Chapter 2: Background and Literature Review

## 2.1 Ultrasound image formation and speckle

B-mode ultrasound displays processed echo information produced by reflection, scattering, attenuation, beamforming, compression, and device-specific post-processing. A simplified observation model is:

$$I(x,y)=f(x,y)\eta_m(x,y)+\eta_a(x,y),$$

where $f$ is an underlying reflectivity field, $\eta_m$ is a multiplicative component, and $\eta_a$ is an additive component. This abstraction motivates robustness tests but does not fully reproduce clinical acquisition.

Radiologists use descriptors such as shape, margin, orientation, echogenicity, and posterior acoustic behaviour within the BI-RADS ultrasound lexicon. These concepts explain why geometry and contrast can matter; they do not authorise this software to infer a BI-RADS category.

## 2.2 Geometric preprocessing

For an image of height $H$ and width $W$ resized to $H'$ by $W'$, $s_y=H'/H$ and $s_x=W'/W$. Direct resize is anisotropic when $s_x\neq s_y$. Padding a crop to a square before resizing applies one final scale factor. This preserves crop aspect ratio by construction, but cropping and reflected boundaries can create their own biases. Predictive effects require a matched ablation.

## 2.3 CLAHE

Contrast Limited Adaptive Histogram Equalisation applies local histogram mappings and clips histogram mass before redistribution. The project applies OpenCV CLAHE to the LAB luminance channel with clip limit 2.0 and a $16\times16$ tile grid. This is a reproducible configuration, not proof of improved classification.

## 2.4 CNN architectures

The project compares two transfer-learning backbones with a compact custom model:

- **ResNet-50** uses residual identity paths to support deep optimisation.
- **EfficientNet-B0** uses mobile inverted bottlenecks, squeeze-and-excitation, and coordinated scaling.
- **Custom CNN** uses four convolutional stages, batch normalisation, pooling, adaptive average pooling, and dropout.

Softmax produces a mutually exclusive probability vector but does not guarantee calibration. Grad-CAM produces a coarse gradient-based localisation map but does not prove causal or clinically relevant reasoning.

## 2.5 Reproducibility gap

Public datasets such as BUSI and OASBUD make experimentation possible, but subject grouping and provenance determine whether an evaluation is interpretable. A complete evidence record includes source identifiers, grouping rules, checksums, exclusions, split manifest, preprocessing, model configuration, seeds, checkpoints, and per-image predictions.

The research gap addressed here is therefore practical integration: connecting a working classifier and user interface to an auditable evidence chain, while preventing illustrative interface values from being mistaken for measured findings.

## 2.6 Public breast-ultrasound datasets

The revised dissertation reviews five datasets discussed by Bristow and Yap: OASBUD, RODTOOK, UDIAT, BUSI, and BrEaST. They differ in country, scanner source, image formation, annotation, ground truth, and file organisation. Those differences affect both training and the meaning of a test result.

OASBUD and BrEaST were selected for the validated experiment because their local copies retain subject identifiers, paired lesion masks, and labels linked to biopsy or follow-up evidence. Their contrasting appearance also gives the local experiment some acquisition diversity. The study still has a narrow geographic base because both sources were collected in Poland.

RODTOOK and UDIAT were not in the supplied project data. Bringing them into the present experiment would require a fresh acquisition, licence review, curation pass, and frozen subject-level manifest. They are sensible external cohorts for later work.

BUSI was present only as a renamed 224 by 224 derivative. The generated `sample_*` identifiers cannot be traced back to the original cases, and the visible images show repeated banding. Published correspondence also reports duplicates and inconsistent cases in the public release. The local derivative is therefore illustrated in the report but excluded from validated training and evaluation.

The dataset plate is stored at [`latex_university/figures/fig9_dataset_quality_examples.png`](../../latex_university/figures/fig9_dataset_quality_examples.png). It shows benign and malignant examples from the three local folders while making the BUSI derivative status explicit.
