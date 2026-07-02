# OncoVision Clinical Control Center - User Manual

This manual provides an overview of the OncoVision Clinical Control Center, detailing the application's interface, its features, and the underlying repository codebase.

---

## 🖥️ Web Interface Guide

The OncoVision clinical dashboard is a single-screen responsive workspace split into three primary panels:

```
+-----------------------------------------------------------------------------+
|                            ONCOVISION DASHBOARD                             |
+------------------+----------------------------------+-----------------------+
| ⚙️ PARAMETERS     | 📁 1. DIAGNOSTIC INPUT            | 📊 2. ASSESSMENT      |
| - Model Selector | - Select Existing Study Dropdown | - Verdict Banner      |
| - CLAHE Toggle   | - Drag & Drop / Browse Files     | - Side-by-Side Imgs   |
|                  | - Execute Button                 | - Probability Bars    |
| - Quick Start    |                                  | - Batch Results Grid  |
| - Model Specs    +----------------------------------+-----------------------+
|                  | 📝 CLINICAL AUDIT LOGS (Console output feed)              |
+------------------+----------------------------------------------------------+
```

### 1. Left Sidebar: Parameters & Specifications
- **Model Selector**: Dropdown menu to choose the neural backbone for classification.
  - `ResNet-50`: High-depth residual network.
  - `EfficientNet-B0`: Compound-scaled, parameter-optimized network.
  - `Custom CNN`: Lightweight baseline convolutional model built from scratch.
- **CLAHE Enhancement Toggle**: Turns the **Contrast Limited Adaptive Histogram Equalization** preprocessing filter ON or OFF.
- **Dissertation Models Card**: Displays key validation accuracies and Area Under the ROC Curve (AUC) statistics for reference.
- **Quick Start Guide**: A short step-by-step walkthrough explaining the system workflow.

### 2. Center Panel: Diagnostic Input & Logs
- **Validation Study Selector**: A compact dropdown menu that indexes the pre-generated database files. Selecting a file loads it immediately.
- **Drag & Drop Upload Zone**: Allows importing scans from your computer. You can drop a single image, multiple images, or a folder containing a dataset.
- **Execute Diagnosis Button**: A theme-matched, compact execution button that triggers neural evaluation.
- **Clinical Audit Logs**: A log terminal tracking pipeline activity, connection health, hardware configurations, and diagnostic summaries.

### 3. Right Panel: Diagnostic Assessment (Results)
- **Single Mode**:
  - **Verdict Banner**: Displays the predicted class (**BENIGN** or **MALIGNANT**) with classification confidence.
  - **Visual Comparison**: Renders the original mammogram side-by-side with the CLAHE-preprocessed scan.
  - **Class Probabilities**: Sliding metrics showing the model's Softmax output distribution (representing classification certainty).
- **Batch Mode**:
  - Activated when multiple files or folders are uploaded.
  - **Summary Metrics**: Shows Total Scans, Benign Predicted, and Malignant Predicted.
  - **Results Table**: Lists file names, verdicts, and confidence ratings.
  - **Interactive Selection**: Clicking any row in the table loads that scan's metrics and image comparisons in the visualizer.

---

## 🔬 Google Colab / Notebook Mode

Click the **Notebook Mode** switch in the header to view the code repository.

- **Jupyter Cells**: Python source files are segmented into code blocks.
- **Run Cells**: Click the play icon on the left of any block to simulate execution (prompt changes to `In [*]` and prints logs).
- **Run All**: Executes all notebook blocks sequentially.
- **Copy Source**: Copies the raw source code of the selected file to the clipboard.

---

## 📁 Codebase Architecture

Here is a summary of the repository files:

1. **[config.py](file:///Users/davidakerele/Documents/Dissertation/config.py)**: Stores hyperparameters, input dimension limits ($224 \times 224$), classes, and system parameters (e.g. CPU vs GPU selection).
2. **[models.py](file:///Users/davidakerele/Documents/Dissertation/models.py)**: Implements the Custom 4-Block CNN and imports the ResNet-50 and EfficientNet-B0 models.
3. **[dataset.py](file:///Users/davidakerele/Documents/Dissertation/dataset.py)**: Handles image loading, normalization transforms, weighted random sampling, and the OpenCV CLAHE preprocessing filter.
4. **[train.py](file:///Users/davidakerele/Documents/Dissertation/train.py)**: Manages model training, early stopping validation, and saving model weights.
5. **[evaluate.py](file:///Users/davidakerele/Documents/Dissertation/evaluate.py)**: Computes performance metrics (F1-score, sensitivity, specificity) and generates confusion matrices.
6. **[predict.py](file:///Users/davidakerele/Documents/Dissertation/predict.py)**: Handles inference via CLI arguments.
7. **[api.py](file:///Users/davidakerele/Documents/Dissertation/api.py)**: Serves the FastAPI server, exposes prediction endpoints, and hosts the visual dashboard at `http://localhost:8000`.
