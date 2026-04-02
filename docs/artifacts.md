# 本地产物

这份文档说明 AutoVisionLab 当前会落到本地磁盘的主要产物，以及它们的目录结构。

## 目录树

```text
artifacts/
  runs/
    <run_id>/
      run.log
      llm.jsonl
      experiments/
        <experiment_id>/
          recipe.json
          checkpoint.pt
```

## 目录含义

- `artifacts/runs/<run_id>/`
  - 一个 `run` 的本地产物根目录
  - 保存该 `run` 的过程日志和实验子目录
- `artifacts/runs/<run_id>/run.log`
  - `run` 级文本日志
  - 记录实验创建、训练开始、训练结果、proposal 摘要等事件
- `artifacts/runs/<run_id>/llm.jsonl`
  - `run` 级 LLM 交互日志
  - 记录 `proposal_request`、`proposal_response`、`proposal_error`
- `artifacts/runs/<run_id>/experiments/<experiment_id>/`
  - 一个 `experiment` 的本地产物目录
  - 保存该实验的冻结配置快照和训练 checkpoint
- `artifacts/runs/<run_id>/experiments/<experiment_id>/recipe.json`
  - 该实验的冻结配置快照
  - 当前写入的是 `TrainerManifest` 视图，包含：
    - `model`
    - `train`
    - `data`
    - `search`
    - `ranking`
    - `runtime`
- `artifacts/runs/<run_id>/experiments/<experiment_id>/checkpoint.pt`
  - 该实验验证集最优时保存的模型参数

## 当前实现位置

- [backend/app/services/run_logging.py](../backend/app/services/run_logging.py)
  - `run` / `experiment` 目录和日志路径
- [backend/app/trainers/classification/base_trainer.py](../backend/app/trainers/classification/base_trainer.py)
  - `run.log`
  - `recipe.json`
  - `checkpoint.pt`
- [backend/app/services/proposal_service.py](../backend/app/services/proposal_service.py)
  - `llm.jsonl`
- [backend/app/services/persistence.py](../backend/app/services/persistence.py)
  - 清理 `run` / `experiment` 产物

## 数据目录

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
