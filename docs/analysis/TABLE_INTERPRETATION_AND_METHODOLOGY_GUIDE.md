# Table Interpretation and Methodology Guide

Only tables generated from saved held-out predictions may appear in the final dissertation. `src/compute_project_tables.py` reads `outputs/metrics.json`, which is created by `evaluate.py`; it no longer simulates labels, scores, or pseudo-labelling experiments.

The current metrics file represents an unaudited engineering run and must be labelled provisional. Before submitting a performance table, verify that the data audit contains no cross-split subjects and that the manifest retains original source identifiers. Report sample count, split definition, model/checkpoint, preprocessing, random seed, class-wise metrics, and uncertainty intervals.

Pseudo-labelling, noise robustness, and preprocessing ablations require separate stored experiments. They must not be inferred from fixed values or dashboard displays.
