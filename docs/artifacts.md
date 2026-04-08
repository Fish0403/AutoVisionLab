# 本地产物

这份文档说明 AutoVisionLab 当前会落到本地磁盘的主要产物，以及它们的目录结构。

## 目录树

```text
artifacts/
  runs/
    <run_id>/
      run.log
      llm.jsonl
      prompt_context.json
      proposal_prompts.md
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
  - 记录实验创建、训练开始、训练结果和 proposal 摘要等事件
- `artifacts/runs/<run_id>/llm.jsonl`
  - `run` 级 LLM 交互日志
  - 记录 `proposal_request`、`proposal_response`、`proposal_error`
- `artifacts/runs/<run_id>/prompt_context.json`
  - `run` 级结构化 prompt 上下文日志
  - 记录每次 proposal 生成时的 prompt blocks、字符数和 token 估计
- `artifacts/runs/<run_id>/proposal_prompts.md`
  - `run` 级人类可读 prompt 日志
  - 记录每次 `proposal_request` 的最终 `system_prompt` / `user_prompt`
  - 记录对应的 `proposal_response` 或 `proposal_error`
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
  - `prompt_context.json`
  - `proposal_prompts.md`
- [backend/app/services/persistence.py](../backend/app/services/persistence.py)
  - 清理 `run` / `experiment` 产物
