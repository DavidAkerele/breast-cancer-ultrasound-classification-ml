# Final dissertation overhaul record

Date completed: 11 September 2026

## Scope followed

This revision changed the report, evaluation presentation, documentation, notebook, web interface, and packaging. It did not retrain any model, reorganise BUSI, restore BUSI provenance, or change the validated BrEaST and OASBUD data.

## Main report

The MMU-standard report is titled *Breast Cancer Ultrasound Classification Using Machine Learning*. The six chapters contain 12,947 raw LaTeX words before front matter, references, and appendices. After the ethics evidence was inserted, the compiled document has 100 A4 pages.

The report now includes:

- a public dataset review and a clear explanation for selecting BrEaST and OASBUD;
- an explicit exclusion of the local BUSI derivative because patient mapping and curation history cannot be reconstructed;
- a targeted literature review of ten relevant works by Professor Moi Hoon Yap and collaborators;
- the confusion matrix, ROC curve, precision-recall curve, reliability diagram, bootstrap intervals, model comparison, source-specific evaluation, and correct and incorrect image examples;
- a six-screen tour of the research dashboard;
- expanded appendices covering the audit, reproducibility, API contract, tests, artifacts, and evidence limitations;
- consistent author details, student ID 25908322, project title, and supervisor information.

The source contains no em dash characters or placeholder citation markers. The old standalone `TermRef.tex` was also rewritten so that it describes the current binary BrEaST and OASBUD study rather than the earlier BUSI multiclass proposal.

## Evaluation record

The existing EfficientNet-B0 checkpoint was evaluated on 62 images. Its accuracy is 0.6774, macro F1 is 0.6774, and ROC-AUC is 0.7419. The preserved predictions also give average precision 0.6803, malignant precision 0.6364, malignant recall 0.7241, Brier score 0.2144, and five-bin expected calibration error 0.0652.

EfficientNet-B0 has the strongest point estimates in the internal three-model comparison. The bootstrap intervals overlap, and paired model predictions are unavailable, so the report does not claim statistical superiority. Complete learning histories are also unavailable for the existing checkpoints. The report gives the selected validation points and does not fabricate learning curves. Future training now records an epoch-by-epoch JSON history.

## Website, notebook, and presentation

The research dashboard contains six documented pages for single-image inspection, seeded noise experiments, batch evaluation, evidence, local dataset inspection, and project documentation. Its documentation page distinguishes appropriate research use from unsupported clinical use.

The executed notebook now loads the audit, primary metrics, extended evaluation, model benchmark, source-specific errors, and generated figures. Evaluation remains optional and disabled by default. No training cell was added or run.

The 12-slide PowerPoint uses the same data and evidence boundary as the report. It contains editable metric charts and tables, source-specific errors, calibration, and six full dashboard screenshots. Speaker notes carry the interpretation limits.

## Verification completed

- Main report and standalone Terms of Reference compile successfully.
- The final Overleaf package compiles as a self-contained source folder.
- No missing figures, unresolved citations, undefined references, overfull boxes, or TeX errors appear in the report log.
- The final PDF contains no blank pages and no repeated text pages.
- All 12 presentation slides passed package and layout validation and were visually inspected after rendering.
- The notebook executed and passed `nbformat` validation.
- All six repository regression tests pass.
- The upload ZIP contains 35 required files and no generated LaTeX auxiliary files.

## Post-overhaul ethics and declaration update

The candidate supplied the EthOS favourable-opinion email on 11 September 2026. The email is reproduced in `latex_university/appendix_assets/EthOS_approval_email.pdf` and embedded as the first item in Appendix D. It records reference number **92016**, a favourable ethical opinion dated 18 June 2026, and an approval period tied to the application end date of 11 September 2026.

The report front matter and declaration now use reference number 92016. The declaration is populated with the candidate name, `Akerele David Damilola`, the supplied signature image, and the date `6 September 2026`. The rebuilt MMU report is 100 A4 pages and was checked after the change.

The clean package is `submission/Akerele_David_25908322_Submission_Package.zip`; it contains the final dissertation, presentation, ToR, approval evidence, product link, self-contained Overleaf source, complete project source, and SHA-256 checksums.

## Student action still required

The dissertation and ToR now use the registered EthOS title exactly. The supervisor's Moodle confirmation should be retained as the course record; no assistant-supplied signature or date should replace it.
