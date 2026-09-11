# Personal project summary and presentation guide

## 1. The project in one paragraph

This project builds and evaluates a research prototype for classifying breast ultrasound images as benign or malignant. It combines a shared image-preprocessing pipeline, three convolutional neural networks, a dataset-integrity audit, reproducible evaluation outputs, and a local web interface. The important part of the project is the link between those pieces: the same preprocessing and model-loading path is used for training, evaluation, command-line prediction, and the dashboard. The work is deliberately presented as a research baseline. It does not claim to diagnose patients or to be ready for clinical deployment.

## 2. What problem the project addresses

Breast ultrasound is useful because it is non-ionising and can show soft-tissue structure, but the appearance of an image changes with the scanner, acquisition settings, operator technique, probe position, gain, and image composition. Speckle and other acquisition effects make the task difficult. A machine-learning model can also appear to perform well if related images from the same subject are placed in different partitions. That is why the project treats subject separation and data provenance as part of the modelling problem.

## 3. What was implemented

The project includes:

- CLAHE applied to the luminance channel as an explicit, configurable preprocessing step.
- Mask-aware region-of-interest cropping when a recognised mask is available.
- A deterministic central fallback crop when a mask is missing. This is documented as a fallback, not as lesion localisation.
- Square padding before resizing so that the final geometric transformation is isotropic.
- Direct resize and centre-crop strategies for future controlled comparisons.
- Random horizontal flips and rotations for training only.
- EfficientNet-B0 and ResNet-50 transfer-learning models.
- A compact custom CNN trained from scratch.
- Class-weighted cross-entropy, AdamW, cosine scheduling, early stopping, and recorded random seeds.
- A read-only audit that checks subject identifiers and cross-partition leakage.
- Per-image predictions, confusion matrices, ROC curves, calibration error, and bootstrap intervals.
- A FastAPI research service and a local dashboard with confidence, uncertainty, Grad-CAM, batch inference, and controlled perturbation demonstrations.

## 4. Dataset and evidence boundary

The validated local experiment uses repaired BrEaST and OASBUD subject-level partitions:

- 302 training images
- 63 validation images
- 62 test images
- 427 images in total

The audit found no cross-partition subjects in those two datasets. BUSI is not included in the validated headline result because the local files were renamed and converted to square derivatives. That breaks the link needed to repeat published duplicate and label checks and to prove patient-level separation. BUSI is not being restored for this dissertation. The exclusion applies to the local derivative, not to the value of the original public dataset.

The binary task uses benign and malignant output classes. Images in a `normal` directory are mapped to the non-malignant side of the current loader. Explain this as an implementation decision that needs review before a definitive clinical study, because normal and benign are not the same clinical category.

## 5. Main measured findings

The fresh EfficientNet-B0 checkpoint was evaluated on the audited 62-image test partition. The measured results were:

- Accuracy: 67.74%
- Macro F1: 0.6774
- Macro precision: 0.6803
- Macro recall / balanced accuracy: 0.6803
- ROC-AUC: 0.7419
- Average precision: 0.6803
- Malignant precision: 0.6364
- Malignant recall: 0.7241
- Brier score: 0.2144
- Expected calibration error: 0.0652
- Bootstrap 95% accuracy interval: 0.5645 to 0.7903
- Bootstrap 95% macro-F1 interval: 0.5589 to 0.7889
- Bootstrap 95% ROC-AUC interval: 0.6217 to 0.8616

The confusion matrix was:

|                     | Predicted benign | Predicted malignant |
|---------------------|-----------------:|--------------------:|
| Actual benign       | 21               | 12                  |
| Actual malignant    | 8                | 21                  |

The internal same-split comparison was:

| Model           | Accuracy | Macro F1 | Macro recall | ROC-AUC |
|----------------|---------:|---------:|-------------:|--------:|
| EfficientNet-B0 | 0.6774 | 0.6774 | 0.6803 | 0.7419 |
| Custom CNN      | 0.6452 | 0.6452 | 0.6479 | 0.6708 |
| ResNet-50       | 0.5645 | 0.5374 | 0.5512 | 0.6364 |

The correct interpretation is that EfficientNet-B0 was the strongest of the three models in this local run. The intervals are wide and overlap, so this does not establish general superiority. It is an internal baseline comparison on 62 test images, not a published-model benchmark or clinical validation.

### How to interpret the custom CNN versus ResNet-50

Yes, the custom CNN is better than ResNet-50 in the current run:

- Custom CNN accuracy: 64.52%; ResNet-50 accuracy: 56.45%.
- Custom CNN macro F1: 0.6452; ResNet-50 macro F1: 0.5374.
- Custom CNN ROC-AUC: 0.6708; ResNet-50 ROC-AUC: 0.6364.

That is a genuine finding from your experiment, but it is not a breakthrough. The comparison uses one small 62-image test partition and one training seed. The accuracy intervals overlap: 0.5323-0.7581 for the custom CNN and 0.4355-0.6935 for ResNet-50. A deeper model can lose on a small dataset because it may be harder to optimise or more prone to fitting incidental patterns. The current evidence supports the sentence: “The custom CNN outperformed ResNet-50 on this audited local test run.” It does not support: “The custom CNN is generally better than ResNet-50.”

In the presentation, use this as a thoughtful result. Say that the experiment challenged the assumption that a larger pretrained backbone must perform best, then explain that multi-seed and external-site testing are needed to determine whether the difference persists. That answer sounds scientifically mature and gives you a clear next experiment.

## 6. What you should say the project proves

The project demonstrates that:

1. A single preprocessing and inference contract can be shared across training, evaluation, command-line prediction, and the web interface.
2. Subject-level audit checks can expose whether a local folder structure supports a patient-independent estimate.
3. The repository can produce traceable predictions, metrics, plots, and checkpoint metadata.
4. The current audited cohort supports a reproducible local baseline for further experimentation.

## 7. What you must not claim

Avoid saying that the model diagnoses breast cancer, is clinically accurate, is ready for hospital use, or is state of the art. Do not describe the 67.74% result as sensitivity or specificity. Do not present the local architecture comparison as a comparison against external published systems. Do not include the withdrawn historical values or simulated robustness curves as measured findings. If someone asks whether the model works on BUSI, explain that the processed local copy is excluded because its original case identity and curation trail cannot be reconstructed.

## 8. Recommended presentation structure

For a 10 to 12 minute presentation, use about 9 or 10 slides.

### Slide 1: Title and research question

State the title and one sentence describing the task: “Can a reproducible deep-learning pipeline classify breast ultrasound images while making its data and evaluation limitations explicit?”

### Slide 2: Clinical and technical motivation

Explain why ultrasound is useful and why it is difficult for machine learning. Mention speckle, acquisition variability, and the risk of subject leakage. Keep this practical rather than turning the slide into a long literature review.

### Slide 3: Data and audit

Show the new dataset-composition figure. State the 427-image validated cohort, the 302/63/62 split, and the fact that BUSI is excluded because identifiers cannot be verified. This is one of the strongest slides because it shows that you checked the evidence before reporting a score.

### Slide 4: Pipeline

Draw or show the sequence: source image, audit, crop and mask discovery, CLAHE option, square padding, CNN, evaluator, and dashboard. Emphasise that the same preprocessing function is reused at every inference route.

### Slide 5: Models

Introduce EfficientNet-B0, ResNet-50, and the custom CNN. Explain why three models were used: two transfer-learning backbones and a compact baseline. Do not spend too long describing every layer.

### Slide 6: Evaluation design

Explain the subject-level split rule, the frozen-test principle, the saved per-image predictions, and the metrics. Mention that the present experiment is local-cohort evidence and that external validation remains future work.

### Slide 7: Main results

Show the notebook-derived confusion matrix and ROC curve. State the EfficientNet-B0 result: 67.74% accuracy, 0.6774 macro F1, and 0.7419 ROC-AUC on 62 test images. Immediately add the limitation: the intervals are wide and the result is not clinical validation.

Then show the precision-recall and reliability figure. Explain that average precision is 0.6803 and that the reliability diagram is a descriptive check, not a fitted calibration model. The source plot is useful for discussion: BrEaST and OASBUD have different error profiles, so the pooled score hides source effects. Do not claim that one dataset is better.

### Slide 8: Model comparison and uncertainty

Show the model benchmark and accuracy-interval figures. Explain that EfficientNet-B0 ranked highest in this run, but the overlapping intervals mean that more subjects and multiple seeds are needed before making a strong ranking claim.

### Slide 9: Dashboard and safety boundary

Show the dashboard. Explain the uncertainty display, Grad-CAM, batch results, and perturbation controls. State that the interface is research-only and does not assign BI-RADS, provide a diagnosis, or recommend treatment.

### Slide 10: Contribution, limitations, and next step

Finish with three points: the pipeline is reproducible, the evidence boundary is explicit, and the next experiment requires a frozen manifest, multi-seed training, validation-fitted calibration, and external evaluation. BUSI remains outside the validated cohort.

## 9. Suggested opening script

“My project investigates breast ultrasound classification using deep learning. The main focus is not only the neural network score; it is whether the score can be traced to a valid subject-level split, a shared preprocessing path, and saved evidence. I built a research prototype using BrEaST and OASBUD, compared three CNN approaches, and connected the evaluator to a local dashboard. The current EfficientNet-B0 result is 67.74% accuracy on an audited 62-image test set. I present that as a reproducible local baseline, not as a clinical diagnostic system.”

## 10. Suggested closing script

“The project shows that a functioning dashboard and a high-looking number are not enough to establish clinical performance. The strongest contribution is the evidence-aware pipeline: the data audit, shared preprocessing, recorded predictions, and explicit separation between engineering evidence and clinical claims. The next step is a locked, multi-seed, subject-independent evaluation with restored dataset provenance and an external test cohort.”

## 11. Questions you are likely to receive

### Why was BUSI excluded?

The local BUSI files had generated names and processed square images rather than a traceable copy of the original release. Without original case identity, the project cannot repeat published duplicate and label-conflict checks or verify that one patient occurs in only one partition. Including those files would enlarge the test set without strengthening its validity.

### Why is there no learning curve?

The completed checkpoints store the selected epoch, validation loss, and validation accuracy, but not every epoch. A full curve cannot be reconstructed from one saved point. The code now saves epoch histories for future runs, while the dissertation reports the available checkpoint record rather than drawing a synthetic curve.

### Why use CLAHE?

CLAHE is a documented local contrast enhancement option. It may help expose local structure, but the project does not claim that it improves classification until a controlled ablation is run. The important point is that the setting is recorded and shared across the pipeline.

### Why is square padding preferable to direct resizing?

Square padding preserves the aspect ratio before the final resize, so horizontal and vertical geometry are scaled by the same factor. It may still introduce boundary texture and it may not improve performance. Its value must be tested empirically.

### Why is accuracy only 67.74%?

The result comes from a small, difficult, heterogeneous local cohort. It is also a transparent measured result rather than a historical or simulated number. The project prioritises traceability and correct interpretation over reporting an unsupported higher score.

### Why compare these models?

EfficientNet-B0 and ResNet-50 provide transfer-learning baselines with different architectures, while the custom CNN shows what a smaller from-scratch model can do under the same evaluator. The comparison is internal and exploratory.

### Can the model distinguish ultrasound from unrelated images?

Not reliably. It was trained on breast-ultrasound image folders and has no broad out-of-distribution detector. An unrelated image may still receive a benign or malignant probability. The interface should therefore be treated as research-only, and a future system would need explicit input validation and out-of-distribution testing.

### What would make this clinically useful?

Restored provenance, a much larger subject-level cohort, external site validation, calibration, clinically meaningful labels, reader studies, prospective evaluation, privacy governance, and the appropriate regulatory pathway. A larger backbone alone would not solve those requirements.

## 12. Presentation habits

- Lead with the research question, not the codebase.
- Explain the evidence boundary before showing the headline score.
- Use the confusion matrix to discuss the two error directions.
- Say “local audited test partition” instead of “clinical test set.”
- When uncertain, describe what the current experiment can support and what would need to be run next.
- Keep one slide available for the dashboard, but spend most of the time on data validity and evaluation design.
- Finish with the contribution and the next experiment, not with a claim of deployment readiness.
