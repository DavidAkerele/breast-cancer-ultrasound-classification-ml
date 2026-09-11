# Supervisor Revision Checklist

This record maps the supervisor's comments to the completed dissertation changes.

## Phase 1: Evidence audit

- Reviewed Joe Bristow and Moi Hoon Yap's supplied MIUA 2026 manuscript.
- Checked its dataset summary and reference list against the current dissertation.
- Audited the local BrEaST, OASBUD, and BUSI-derived folders and the stored prediction file.

## Phase 2: Dataset literature revision

- Added Section 2.7, covering OASBUD, RODTOOK, UDIAT, BUSI, and BrEaST.
- Added a comparison table with curated counts, country, evidence, and curation considerations.
- Explained why the validated experiment uses OASBUD and BrEaST.
- Added key dataset, benchmark, detection, classification, observer, and systematic-review references.
- Verified ten relevant Moi Hoon Yap works against Google Scholar and an independent bibliographic source, with the forthcoming 2026 manuscript labelled separately.

## Phase 3: Scientific illustrations

- Added a six-panel comparison of local dataset appearance.
- Added four image-level EfficientNet-B0 outcomes selected deterministically from `outputs/predictions.csv`.
- Captured the implemented web interface with its model controls, evidence banner, source image, and Grad-CAM panel.
- Added a precision-recall curve, reliability diagram, source-stratified performance chart, and a six-screen web interface tour.

## Phase 4: Dissertation integration

- Added captions, labels, and in-text cross-references for every new figure.
- Expanded the Chapter 4 error analysis without assigning clinical causes from visual inspection.
- Added class-level precision and recall, average precision, Brier score, source-level error counts, confidence intervals, and an explicit statistical-comparison limit.
- Reported the selected checkpoint epochs and explained why a full historical learning curve cannot be reconstructed without retraining.
- Added figure provenance to Appendix A and synchronised the supporting Markdown documentation.
- Retained cautious, research-only wording throughout the additions.

## Phase 5: Submission verification

- Rebuilt the MMU LaTeX report with all images and references present.
- Checked the complete page sequence and found no repeated text pages.
- Checked that there are no unresolved citations, missing references, missing figures, or overfull boxes.
- Confirmed that the main dissertation body is within the requested 10,000 to 15,000 word range.
- Ran the automated project tests and created a clean Overleaf upload archive.

## Deliberate exclusions

- BUSI was not restored, reorganised, or added to the validated cohort. The report explains why the processed local copy cannot support repeatable case-level curation or patient separation.
- No model was retrained. All added evaluation figures come from the preserved test predictions.

Added the supplied favourable-opinion email to Appendix D, entered EthOS reference number 92016 in the front matter and declaration, and populated the declaration with the candidate's name, signature image, and date.

The dissertation and ToR now use the registered EthOS title exactly. The signed declaration and Moodle supervisor confirmation should not be backdated or altered by the assistant.
