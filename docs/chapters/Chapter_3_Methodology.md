# Chapter 3: Methodology

## 3.1 Evidence design

The project distinguishes **engineering evidence**, which confirms that the software path executes consistently, from **generalisation evidence**, which estimates performance on independent subjects. The current local cohort supports only the first level because it fails the split-integrity audit.

## 3.2 Data audit

The validated loader uses BrEaST and OASBUD only and finds 302 training, 63 validation, and 62 test images after subject-level rebuilding. The audit reports no cross-partition identifiers in either dataset. BUSI patient separation cannot be reconstructed from its renamed files, so it is excluded by default; training still stops if any configured dataset fails its audit unless `--allow-unaudited-data` is supplied for an explicitly provisional run.

The subject-level splitter copies grouped files into deterministic class-stratified partitions and saves a JSON manifest. It never moves the source dataset.

## 3.3 Shared preprocessing

All executable surfaces call the same function. The sequence is RGB decoding, optional CLAHE, one selected crop/resize strategy, tensor conversion, and ImageNet normalisation.

Supported masks end with `_mask.png`, `_tumor.png`, or `_lesion_mask.png`. A valid mask defines a bounding box expanded by a 20% context margin. Without a mask, the central 80% is used as a deterministic fallback. For a crop of height $h_c$ and width $w_c$, the square side is $d=\max(h_c,w_c)$; symmetric `BORDER_REFLECT_101` padding is added and the result is resized to $224\times224$.

Training augmentation uses horizontal flips and rotations up to 15 degrees. Vertical flips are excluded because image depth has acquisition meaning.

## 3.4 Models and optimisation

EfficientNet-B0 and ResNet-50 use ImageNet initialisation during training; the custom CNN is trained from scratch. The default optimiser is AdamW with learning rate $10^{-4}$ and weight decay $10^{-4}$, cosine annealing, class-weighted cross-entropy, batch size 16, and early stopping. Python, NumPy, PyTorch, and CUDA are seeded; each checkpoint stores its run configuration and audit status.

## 3.5 Evaluation artifacts

`evaluate.py` writes `predictions.csv`, `metrics.json`, `confusion_matrix.png`, and `roc_curve.png`. The JSON record includes audit status and checksum, generation time, preprocessing configuration, sample count, summary metrics, and confusion matrix. Tables are generated from these outputs rather than entered manually.

## 3.6 Interface boundary

The FastAPI interface exposes research inference, model comparison where compatible checkpoints exist, batch evaluation, and seeded synthetic-noise demonstrations. It does not infer ground truth from filenames or provide diagnosis, BI-RADS, clinical priority, or management advice.
