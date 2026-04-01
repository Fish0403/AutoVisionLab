# AutoVisionLab

工业视觉实验平台。当前实现聚焦图像分类，提供结构化实验、后台自动搜索和跨模型比较。

English version: [README.en.md](README.en.md)

## 项目概览

AutoVisionLab 以 `run` 为实验容器，以 `experiment` 为单次训练记录，以 `task` 为后台编排单元，以结构化 `proposal`、`result` 和 `reflection` 连接 AI 搜索、训练和复盘。

当前实现包含：

- 图像分类训练闭环
- 结构化 `model_recipe`、`train_hyp`、`dataset_recipe`
- 白名单参数空间与搜索策略校验
- `Auto Train` 后台持续搜索
- `Compare Models` 后台跨模型 baseline 比较
- SQLite 元数据存储与本地产物落盘

## 当前支持

- 前端：`React`
- 后端：`FastAPI`
- 训练框架：`PyTorch`
- 数据库：`SQLite`
- 当前模型：`MobileNetV2`、`MobileNetV3 Small`、`GoogLeNet`、`ResNet18`
- 当前任务类型：`classification`

## 数据与产物

分类数据使用以下目录：

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

训练产物写入本地目录：

- `artifacts/runs/<run_id>.log`
- `artifacts/checkpoints/<experiment_id>.pt`

## 快速开始

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

默认地址：

- Backend: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:5173`

## 文档导航

- [docs/overview.md](docs/overview.md)
- [docs/api.md](docs/api.md)
- [docs/schemas/model_recipe.md](docs/schemas/model_recipe.md)
- [docs/schemas/train_hyp.md](docs/schemas/train_hyp.md)
- [docs/schemas/dataset_recipe.md](docs/schemas/dataset_recipe.md)
