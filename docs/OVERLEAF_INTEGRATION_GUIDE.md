# Overleaf Synchronization & Integration Guide

This guide explains how to connect your local codebase directory **`latex/`** to your **Overleaf** project.

---

## Method 1: Direct Overleaf Git Sync (Recommended: Works on All Overleaf Accounts)

Every Overleaf project has a dedicated Git URL. You can link this repository directly to Overleaf so that pushing local changes updates your Overleaf project instantly.

### Step 1: Get Your Overleaf Git URL
1. Open your project on [Overleaf](https://www.overleaf.com).
2. Click **Menu** in the top-left corner.
3. Under **Synchronise**, click **Git**.
4. Copy the Git URL (e.g., `https://git.overleaf.com/123456789abcdef`).

### Step 2: Add Remote and Sync
Open your terminal in the dissertation workspace root:

```bash
# Add Overleaf as a remote
git remote add overleaf https://git.overleaf.com/YOUR_PROJECT_ID

# Fetch and merge Overleaf template changes
git fetch overleaf
git pull overleaf main --allow-unrelated-histories

# Push local LaTeX chapters, figures, and bibtex to Overleaf
git push overleaf main
```

---

## Method 2: 1-Click ZIP Export (Drag & Drop to Overleaf)

If you prefer uploading directly without Git:

```bash
# Create a clean ZIP archive of the latex/ directory
zip -r overleaf_thesis.zip latex/
```

1. Go to Overleaf $\rightarrow$ **New Project** $\rightarrow$ **Upload Project**.
2. Drag and drop **`overleaf_thesis.zip`**.
3. Overleaf will automatically unpack and compile `main.tex` with all 6 chapters, high-res figures, and BibTeX citations!

---

## Directory Map (`latex/`)

```text
latex/
├── main.tex                  # Master thesis file with MMU template & packages
├── abstract.tex              # Evidence-aware abstract and local-cohort result
├── declaration.tex           # MMU ethics declaration page
├── acknowledgements.tex      # Acknowledgements page
├── abbreviations.tex         # Abbreviations table
├── references.bib            # Complete BibTeX bibliography database
├── chapters/
│   ├── chapter1.tex          # Introduction, Problem Statement, Objectives, Contributions
│   ├── chapter2.tex          # Background, Ultrasound Physics, Speckle Noise, Aspect Ratio Error
│   ├── chapter3.tex          # Methodology, Multi-Center Datasets, Proposed ROI Padding
│   ├── chapter4.tex          # Experimental Results, Multi-Model Matrix, ROC Curves
│   ├── chapter5.tex          # Discussion & Critical Reflection (Clinical, Legal, Ethical)
│   └── chapter6.tex          # Conclusion & Future Work
└── figures/                  # High-resolution PNG figures (fig1 through fig8)
    ├── fig1_4stage_matrix.png
    ├── fig2_confusion_matrix.png
    ├── fig3_roc_curve.png
    ├── fig4_noise_resilience.png
    ├── fig5_snr_cnr.png
    ├── fig6_aspect_ratio_error.png
    ├── fig7_certainty_boxplot.png
    └── fig8_loss_convergence.png
```
