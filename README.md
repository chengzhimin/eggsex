# eggsex

**EggSex-HSI v0.1.0 — research baseline for paired transmission images and hyperspectral data.**

> This repository is an engineering/research baseline. It does **not** contain a validated production checkpoint or real-domain performance claim.

## What is included

- ENVI `.hdr` adapter and a strict NPZ data contract.
- Dark/reference correction, ROI statistics, wavelength-contract checks and QC.
- RGB green-channel + CLAHE + lightweight texture statistics.
- Three comparable modes: `hsi`, `rgb`, `fusion`.
- StandardScaler + RBF-SVM baseline with validation selection and separate probability calibration.
- Egg-level uniqueness and batch-level split-leakage checks.
- Accuracy/AUC/Brier/coverage reports and an explicit `REVIEW` abstention state.
- Unit tests, synthetic end-to-end demo and GitHub Actions CI.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
eggsex demo --out runs/synthetic
```

The demo creates synthetic data with an intentionally injected separable signal. **Its metrics have no scientific or production meaning.**

## Real data

Copy `configs/example.json` to an instrument-specific configuration, then provide a manifest following `docs/DATA_CONTRACT.md`.

```bash
python -m eggsex.cli train --manifest data/manifest.csv --config configs/instrument.json --mode hsi --out runs/hsi
python -m eggsex.cli train --manifest data/manifest.csv --config configs/instrument.json --mode rgb --out runs/rgb
python -m eggsex.cli train --manifest data/manifest.csv --config configs/instrument.json --mode fusion --out runs/fusion
python -m eggsex.cli predict --model runs/hsi/model.joblib --manifest data/inference.csv --out runs/predictions.csv
```

Do not use test results to choose the final mode, hyperparameters or threshold. Use validation data or a separate development set, and keep a final blind batch for evaluation.

## Repository layout

```text
configs/        instrument configuration template
docs/           data contract, model card and roadmap
src/eggsex/     preprocessing, ENVI adapter, model and CLI
tests/          unit/integration smoke tests
.github/        continuous integration
```

## Data and model policy

Do not commit raw data, ground-truth labels, checkpoints or `runs/`. The `.gitignore` is intentionally conservative. Joblib artifacts are executable pickle-based objects; load only trusted artifacts.

Current v0.1 does not include camera SDK integration, automatic segmentation, deep CNN/ViT training, Grad-CAM, ONNX/TensorRT or PLC control. Those are later engineering phases, not hidden claims of completion.

## Research roadmap

See [docs/研发路线.md](docs/研发路线.md) for the path from reproducible baseline to multi-modal model, independent-batch validation and production deployment.

## License

MIT. Third-party papers, repositories and model weights are not bundled into this repository.
