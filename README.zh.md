# AutoVisionLab

工业视觉实验平台。当前实现聚焦图像分类，提供结构化实验、后台自动搜索和跨模型比较。

[English](README.md) | 中文

![React](https://img.shields.io/badge/前端-React-61DAFB?logo=react&logoColor=white)
![FastAPI](https://img.shields.io/badge/后端-FastAPI-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/训练-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![SQLite](https://img.shields.io/badge/数据库-SQLite-003B57?logo=sqlite&logoColor=white)
![Classification](https://img.shields.io/badge/任务-分类-2E7D32)

## 概览

AutoVisionLab 以 `run` 为实验容器，以 `experiment` 为单次训练记录，以 `task` 为后台编排单元，以结构化 `proposal`、`result` 和 `reflection` 连接 AI 搜索、训练和复盘。

当前实现支持图像分类，并已接入 `MobileNetV2`、`MobileNetV3 Small`、`GoogLeNet` 和 `ResNet18`。其他任务类型和模型仍在开发中。

当前实现包含：

- 图像分类训练闭环
- 结构化 `model_recipe`、`train_hyp`、`dataset_recipe`
- 白名单参数空间与搜索策略校验
- `Auto Train` 后台持续搜索
- `Compare Models` 后台跨模型 baseline 比较
- SQLite 元数据存储与本地产物落盘

![Task 页](docs/screenshots/task.png)

![Search 模式页](docs/screenshots/search.png)

![Compare 模式页](docs/screenshots/compare.png)

## 数据与产物

分类数据分成两层：

- `data/raw/<dataset_name>/` 存按类别名分文件夹的图片
- `data/classification/<dataset_name>/` 存由 `raw/` 生成的切分清单

训练器读取的是这些清单文件：

- 必需：`train.txt`、`val.txt`
- 可选：`test.txt`

使用 `data/prepare_classification_split.py` 扫描 `data/raw/<dataset_name>/` 下的类别文件夹，并生成或刷新切分清单。

示例：

```bash
python3 data/prepare_classification_split.py \
  --source-dir data/raw/your_dataset \
  --dataset-name your_dataset \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --seed 42
```

训练产物写入本地目录：

- `artifacts/runs/<run_id>.log`
- `artifacts/checkpoints/<experiment_id>.pt`

## 快速开始

1. 创建并激活 Python 虚拟环境。

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```

2. 安装后端依赖。

   ```bash
   pip install -r requirements.txt
   ```

3. 安装前端依赖。

   ```bash
   cd frontend/web
   npm install
   ```

4. 启动后端和前端。

   ```bash
   ./scripts/run_backend.sh
   ./scripts/run_frontend.sh
   ```

默认地址：

- Backend: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:5173`

环境配置：

- 复制 `.env.example` 为 `.env`
- 启动后先把 API key、模型和 base URL 配好
- 直接参考 `.env.example` 里的示例值开始填

## 文档导航

- [docs/overview.md](docs/overview.md)
- [docs/artifacts.md](docs/artifacts.md)
- [docs/llm.md](docs/llm.md)
- [docs/api.md](docs/api.md)
- [docs/schemas/model_recipe.md](docs/schemas/model_recipe.md)
- [docs/schemas/train_hyp.md](docs/schemas/train_hyp.md)
- [docs/schemas/dataset_recipe.md](docs/schemas/dataset_recipe.md)
