# Chapter 4: Implementation and Training Controls

## 4.1 Repository structure

The executable code is separated into configuration, dataset, model, training, evaluation, prediction, and API modules. Thin root-level entry points keep commands stable. Generated evidence is written to `outputs/`, while the dissertation source, supporting documents, notebook, slides, and web interface remain version-controlled.

## 4.2 Training guard and checkpoint metadata

Before training, the pipeline regenerates `outputs/data_audit.json`. A failed audit stops the run unless the user explicitly supplies `--allow-unaudited-data`; this override does not convert the result into valid evidence. The best validation-loss checkpoint stores the model and optimiser states plus the model name, epoch, validation metrics, class names, seed, batch size, learning rate, epoch budget, dataset-combination flag, crop strategy, CLAHE flag, creation time, and audit status.

## 4.3 Evaluation implementation

Evaluation reconstructs the selected model without downloading pretrained weights and loads the checkpoint state. The test loader contains no stochastic augmentation. Per-image outputs are saved before aggregate metrics are calculated so every reported number can be regenerated.

The current EfficientNet-B0 checkpoint was retrained after rebuilding BrEaST and OASBUD subject-level splits. Its evaluation on the 62-image held-out test partition is reported as local-cohort research evidence in Chapter 5; BUSI is excluded because its source identifiers are unavailable.

## 4.4 Verification

The test suite checks output dimensions for every crop strategy, discovery of supported mask suffixes, explicit failure of the current data audit, and the API's no-clinical-action response. Syntax compilation, notebook validation, LaTeX compilation, browser checks, and render inspection of Word and PowerPoint outputs form separate release gates.
