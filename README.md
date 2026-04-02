# AutoVisionLab

Industrial vision model optimization still involves a lot of manual work: tuning parameters, running experiments, checking results, and iterating again. The real cost is often not the problem itself, but the repetition, scattered comparisons, and constant tool switching.

AutoVisionLab focuses on that repetitive part of the workflow. It brings AI into the training and experimentation loop so results are collected, structured, and fed into a shared analysis flow. The system compares past runs, summarizes trends, and suggests the next direction to explore.

Engineers still define the goals, constraints, and acceptance criteria, while AI handles the repetitive but necessary analysis and iteration work. The result is a workflow that is easier to trace, compare, and build on over time.

English | [中文](README.zh.md)

![React](https://img.shields.io/badge/Frontend-React-61DAFB?logo=react&logoColor=white)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/Training-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?logo=sqlite&logoColor=white)

![Task page](docs/screenshots/task.png)

![Search mode](docs/screenshots/search.png)

![Compare mode](docs/screenshots/compare.png)

## Data Preparation

Classification datasets are organized in two layers:

- `data/raw/<dataset_name>/` stores images grouped by class name
- `data/classification/<dataset_name>/` stores split manifests generated from `raw/`

The trainer reads split manifests such as `train.txt`, `val.txt`, and optional `test.txt`.

Example with the bundled `NEU` dataset:

1. Download `NEU-CLS` from the official page: [NEU surface defect database](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/)
2. Extract `NEU-CLS` under `data/raw/NEU-CLS/`
3. Use `data/prepare_neucls_split.py` to generate `train.txt`, `val.txt`, and `test.txt`

   ```bash
   python3 data/prepare_neucls_split.py --source-root data/raw/NEU-CLS --dataset-name NEU --val-ratio 0.2 --test-ratio 0.1 --seed 42 --force
   ```

Demo Mode is available for quick local testing. When enabled, it uses a smaller deterministic subset if the dataset is larger than the demo limit.

## Quick Start

1. Create and activate a Python virtual environment.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```

2. Install the backend dependencies.

   ```bash
   pip install -r requirements.txt
   ```

3. Install the frontend dependencies.

   ```bash
   cd frontend/web
   npm install
   ```

4. Configure the environment.

   ```bash
   cp .env.example .env
   ```

   Fill in the API key, model, and base URL in `.env` before starting the backend.

5. Start the backend and frontend.

   ```bash
   ./scripts/run_backend.sh
   ./scripts/run_frontend.sh
   ```

Default endpoints:

- Backend: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:5173`

## Documentation

- [docs/overview.md](docs/overview.md)
- [docs/artifacts.md](docs/artifacts.md)
- [docs/llm.md](docs/llm.md)
- [docs/api.md](docs/api.md)
- [docs/schemas/model_recipe.md](docs/schemas/model_recipe.md)
- [docs/schemas/train_hyp.md](docs/schemas/train_hyp.md)
- [docs/schemas/dataset_recipe.md](docs/schemas/dataset_recipe.md)

## License

This project is licensed under the [Apache License 2.0](LICENSE).

[If you find this project useful, feel free to star the repository:]
[![Star on GitHub](https://img.shields.io/badge/Star_on_GitHub-AutoVisionLab-181717?style=for-the-badge&logo=github)](https://github.com/Fish0403/AutoVisionLab)
