# Results and reproducibility status

**Status: provisional; not suitable for clinical-performance claims.**

The currently bundled EfficientNet-B0 checkpoint was evaluated on 120 images using the repository's shared preprocessing path on 7 September 2026. It achieved 78.33% accuracy, macro F1 0.7811, and ROC-AUC 0.8612. The machine-readable record is stored in `outputs/metrics.json`, with per-image probabilities in `outputs/predictions.csv`.

The accompanying split audit reports subject leakage (`30nh` in OASBUD; `case140` and `case151` in BrEaST) and cannot verify the renamed BUSI image copy at patient level. The result must therefore be used only to confirm that the software runs end to end. It is not evidence of generalisation.

Historical figures, pseudo-labelling tables, and fixed noise curves were generated from static or simulated values and must not be reported as measured findings. They are isolated under `docs/archive/legacy_illustrative_figures/`. The canonical submission workflow is: audit a subject-level cohort, train with saved configuration and seed, evaluate once on an untouched test partition, and generate each table and figure from the saved predictions.
