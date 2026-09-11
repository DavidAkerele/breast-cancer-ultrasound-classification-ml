# Breast Cancer Ultrasound Classification Using Machine Learning

**Dissertation title:** *Breast Cancer Ultrasound Classification Using Machine Learning*

This folder is the university-template version of the dissertation. It preserves the archive's `mmuthesis.cls`, `command.tex`, `mmu_logo.pdf`, root-level chapter layout, appendices, and `report.tex` entry point while using the corrected evidence-aware dissertation content. Appendices 3 to 5 contain the updated Terms of Reference, the EthOS approval evidence, and the product repository link.

Compile locally from this folder with:

```bash
tectonic --keep-logs report.tex
```

The validated local experiment is documented throughout the chapters: BrEaST and OASBUD use repaired subject-level partitions, while the processed local BUSI derivative is excluded because its patient grouping and case-level curation cannot be reconstructed. The EfficientNet-B0 result is local-cohort evidence rather than clinical validation.

The report includes the confusion matrix, ROC curve, precision-recall curve, reliability diagram, source-stratified results, audited dataset composition, internal architecture comparison, confidence intervals, local dataset appearance, image-level prediction examples, and a six-screen dashboard tour. Regenerate those figures from the repository root with:

```bash
venv/bin/python scripts/generate_report_figures.py
```

The personal explanation and presentation briefing is in [`../docs/PERSONAL_PROJECT_SUMMARY_AND_PRESENTATION_GUIDE.md`](../docs/PERSONAL_PROJECT_SUMMARY_AND_PRESENTATION_GUIDE.md).

## Uploading to Overleaf

Upload the complete contents of this folder, including the `figures/` and `appendix_assets/` directories. In Overleaf, the files should appear as `report.tex`, `figures/fig2_confusion_matrix.png`, and `appendix_assets/Akerele_David_25908322_Terms_of_Reference.pdf`. Set `report.tex` as the main document and choose pdfLaTeX or XeLaTeX with BibTeX-compatible bibliography processing. Uploading only the `.tex` files will produce missing-file errors.

## Ethics submission gate

`appendix_assets/EthOS_approval_email.pdf` reproduces the favourable-opinion email issued on 18 June 2026. It gives EthOS reference number **92016**, which is entered in the declaration and front matter. `appendix_assets/Application.pdf` is retained after it as the supporting application record. The report and ToR use the same registered project title shown in the email.
