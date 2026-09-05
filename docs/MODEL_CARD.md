# EggNet v0.2 model card

Purpose: egg-level scientific investigation of sex prediction from transmission images and HSI. No real-domain weights or accuracy claims are distributed.

## Models

- hsi: calibrated ROI median, IQR and SNV spectra (3×B), training-only per-band/channel normalization, Conv1d/GroupNorm/SiLU encoder, pooled 64-dimensional embedding.
- rgb: ResNet18, optional torchvision ImageNet initialization, 512-dimensional embedding. RGB/green/CLAHE variants. ROI crop, aspect-preserving resize, zero padding, fixed ImageNet normalization.
- fusion: embeddings concatenated, LayerNorm → Linear(64) → SiLU → Dropout(0.25) → binary logit. This is an engineering baseline, not a claimed novel biological architecture.
- Freezing the image encoder also freezes its BatchNorm running statistics.

Training uses BCE, AdamW, gradient clipping, training-only flips and validation-loss early stopping. A temperature is fitted on a separate calibration split. Positive class F, research threshold 0.5. Invalid input is reported separately.

## Evaluation

Traditional benchmark: nested grouped CV, inner parameter selection and explicit grouped sigmoid calibration, common valid cohort, final-test assets excluded. Logistic regression, PLS-DA, SVM and Extra Trees.

Neural benchmark: outer batch CV with separate fitting/early-stopping/calibration groups inside each outer training fold. Architecture and learning rate are fixed per run. Repeated seeds do not create additional independent subjects.

Fixed-prediction batch-bootstrap intervals do not refit models or correct architecture selection. Classical permutation testing shuffles within batches and reruns the nested procedure. Deep-model permutation testing and multiplicity adjustment are not implemented.

## Scope

One observation per egg; RGB uint8 and user-provided masks; one device/protocol contract per config; age bounds enforced when configured. Calibration assumes linear response and matched references. HSI uses ROI summaries rather than spatial-spectral patches. RGB has no automated vessel annotation or Grad-CAM yet. HSI arrays load in memory during preprocessing; large data need a cache/memmap adapter.

No real eggs were used in v0.2 validation. Neural forward/backward, training/reload, grouped CV, missing-input handling and test isolation were tested on synthetic data. Optional ImageNet weights and CUDA execution require separate environment verification. No device-control behavior is included.
