# AutoVisionLab

Industrial vision experimentation platform. The current implementation focuses on image classification and provides structured experiments, background auto-search, and cross-model comparison.

## Overview

AutoVisionLab uses `run` as the experiment container, `experiment` as the unit of training history, `task` as the background orchestration unit, and structured `proposal`, `result`, and `reflection` objects to connect AI search, training, and review.

Current implementation includes:

- Image-classification training loop
- Structured `model_recipe`, `train_hyp`, and `dataset_recipe`
- Whitelisted parameter spaces and search-policy validation
- Background `Auto Train`
- Background `Compare Models`
- SQLite metadata storage and local artifact persistence

## Current Support

- Frontend: `React`
- Backend: `FastAPI`
- Training stack: `PyTorch`
- Database: `SQLite`
- Supported models: `MobileNetV2`, `MobileNetV3 Small`, `GoogLeNet`, `ResNet18`
- Current task type: `classification`

## Data and Artifacts

Classification data uses:

```text
data/
  raw/
    <dataset_name>/
  classification/
    <dataset_name>/
      train.txt
      val.txt
      test.txt
```

Training artifacts are written locally:

- `artifacts/runs/<run_id>.log`
- `artifacts/checkpoints/<experiment_id>.pt`

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd frontend/web
npm install
./scripts/run_backend.sh
./scripts/run_frontend.sh
```

Default endpoints:

- Backend: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:5173`

## Documentation

- [docs/overview.md](docs/overview.md)
- [docs/api.md](docs/api.md)
- [docs/schemas/model_recipe.md](docs/schemas/model_recipe.md)
- [docs/schemas/train_hyp.md](docs/schemas/train_hyp.md)
- [docs/schemas/dataset_recipe.md](docs/schemas/dataset_recipe.md)
