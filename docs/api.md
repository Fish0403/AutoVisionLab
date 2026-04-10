# AutoVisionLab API

本文档描述当前后端实际暴露的 HTTP API。接口细节以 FastAPI 生成的 OpenAPI 文档和本仓库的结构化 schema 为准。

当前后端默认开放：

- `GET /docs`
- `GET /redoc`
- `GET /openapi.json`

当前接口未统一挂在 `/api` 前缀下，以下路径均为服务根路径下的实际路由。

说明：

- 后端服务本身暴露的是根路径路由
- 本地前端开发时默认通过 `/api` 访问接口，由 Vite 开发服务器代理并重写到后端根路径
- 如果不用仓库内的前端开发服务，需要自行提供同等的反向代理，或者显式设置 `VITE_API_BASE_URL`

## 通用约定

### 统一响应信封

业务接口统一返回以下外层结构：

```json
{
  "ok": true,
  "code": "success",
  "message": "Human readable message.",
  "data": {},
  "errors": [],
  "meta": {
    "schema_version": "v1",
    "timestamp": "2026-03-29T08:00:00Z"
  }
}
```

### 常见错误码

| Code | HTTP | 含义 |
| --- | --- | --- |
| `bad_request` | `400` | 请求语义错误 |
| `not_found` | `404` | 资源不存在 |
| `conflict` | `409` | 状态冲突 |
| `validation_error` | `422` | 请求体或参数校验失败 |
| `upstream_error` | `502` | 上游 AI 服务失败 |
| `internal_error` | `500` | 未处理异常 |

### 常见枚举

| 字段 | 值 |
| --- | --- |
| `task_type` | `classification` |
| `model_name` | 由 `GET /models` 动态返回 |
| `model_family` | 由 `GET /models` 动态返回 |
| `run.status` | `draft`、`active`、`paused`、`completed`、`failed` |
| `experiment.status` | `draft`、`queued`、`running`、`success`、`failed`、`discarded` |
| `experiment.decision` | `keep`、`discard`、`crash`、`timeout` |
| `result.status` | `success`、`failed` |

## 结构化对象

| 对象 | 说明 | 参考 |
| --- | --- | --- |
| `ApiResponse<T>` | 统一响应信封 | 本文档 |
| `LocalDatasetSummary` | 本地数据集发现结果 | `GET /datasets` |
| `ExperimentConfig` | 实验配置对象 | `schemas/` |
| `EditableParameterSpace` | 模型参数白名单 | `schemas/` |
| `ModelDefaults` | 模型默认配置模板 | `GET /models/{model_name}/defaults` |
| `ModelManifestDraftResponse` | AI 生成的 manifest 草稿预览 | `POST /models/draft` |
| `ProposalSchema` | AI proposal | `schemas/` |
| `ResultSchema` | 训练结果 | `schemas/` |

## 数据集

| 方法 | 路径 | 用途 | 响应 |
| --- | --- | --- | --- |
| `GET` | `/datasets` | 扫描本地数据目录并返回数据集可训练状态 | `LocalDatasetSummary[]` |

## 模型

| 方法 | 路径 | 用途 | 响应 |
| --- | --- | --- | --- |
| `GET` | `/models` | 返回当前后端已注册的模型清单 | `ModelSummary[]` |
| `POST` | `/models/draft` | 让 AI 生成一个模型 manifest 草稿预览，不直接落盘 | `ModelManifestDraftResponse` |
| `POST` | `/models/draft/commit` | 提交并落盘一个已确认的 manifest 草稿 | `ModelManifestCommitResponse` |
| `GET` | `/models/{model_name}/defaults` | 返回指定模型的默认配置模板，包括 `summary`、`parameter_space`、`default_model_recipe`、`default_train_hyp`、`default_search_policy` 和 `default_ranking_policy` | `ModelDefaults` |
| `GET` | `/models/{model_name}/parameter-space` | 返回指定模型的可编辑参数白名单 | `EditableParameterSpace` |

## Runs

| 方法 | 路径 | 用途 | 备注 |
| --- | --- | --- | --- |
| `GET` | `/runs` | 列出所有 run |  |
| `POST` | `/runs` | 创建 run | 请求体包含 `base_config` |
| `GET` | `/runs/{run_id}` | 获取 run 详情 |  |
| `GET` | `/runs/{run_id}/metrics` | 获取 run 指标趋势 | 查询参数 `metric_name`，默认 `top1_acc` |
| `GET` | `/runs/{run_id}/trend` | 获取 run 多指标趋势快照 | 返回 `RunTrendResponse` |
| `GET` | `/runs/{run_id}/summary` | 获取 run 摘要 |  |
| `POST` | `/runs/{run_id}/proposal` | 为 run 生成 proposal | 依赖当前历史和白名单 |
| `POST` | `/runs/proposal/test` | 测试 proposal provider 连通性 |  |
| `POST` | `/runs/reset` | 清空所有 run | 破坏性操作 |
| `POST` | `/runs/{run_id}/reset` | 清空单个 run | 破坏性操作 |
| `POST` | `/runs/auto-train` | 启动 auto-train 后台任务 | `202 Accepted` |
| `GET` | `/runs/auto-train/active` | 获取当前活动中的 auto-train 任务 |  |
| `GET` | `/runs/auto-train/{task_id}` | 获取指定 auto-train 任务快照 |  |
| `POST` | `/runs/auto-train/{task_id}/stop` | 请求停止 auto-train 任务 |  |
| `POST` | `/runs/model-compare` | 启动 model-compare 后台任务 | `202 Accepted` |
| `GET` | `/runs/model-compare/active` | 获取当前活动中的 model-compare 任务 |  |
| `GET` | `/runs/model-compare/{task_id}` | 获取指定 model-compare 任务快照 |  |
| `POST` | `/runs/model-compare/{task_id}/stop` | 请求停止 model-compare 任务 |  |
| `GET` | `/runs/tasks` | 列出后台任务历史 | 合并 `auto_train` 和 `model_compare` |
| `POST` | `/runs/tasks/{task_type}/{task_id}/title` | 更新任务标题 | `task_type` 取值为 `auto_train` 或 `model_compare` |
| `DELETE` | `/runs/tasks/{task_type}/{task_id}` | 删除任务 |  |

### Run 相关对象

`RunDetailResponse` 和 `RunSummaryResponse` 当前围绕唯一的主搜索状态返回：

- `best_experiment_id`

`AutoTrainTaskResponse` 和 `ModelCompareTaskResponse` 返回任务状态、日志、活动信息、停止状态和 summary 快照。

## Experiments

| 方法 | 路径 | 用途 | 备注 |
| --- | --- | --- | --- |
| `POST` | `/experiments` | 创建 experiment | 请求体包含 `run_id`、`config`、`parameter_space`，`proposal` 可选 |
| `GET` | `/experiments/{experiment_id}` | 获取 experiment 详情 |  |
| `POST` | `/experiments/{experiment_id}/result` | 保存训练结果 | 请求体为 `ResultSchema` |
| `POST` | `/experiments/{experiment_id}/train` | 启动 experiment 训练 |  |
| `POST` | `/experiments/{experiment_id}/stop` | 停止 experiment 训练 | 停止后会进入丢弃流程 |
| `POST` | `/experiments/{experiment_id}/suggestion` | 启动实验建议任务 | `202 Accepted` |
| `GET` | `/experiments/{experiment_id}/suggestion` | 获取实验建议任务 |  |

### Experiment 说明

- `ExperimentConfig` 以 `model_recipe`、`train_hyp`、`dataset_recipe` 为主配置，同时保存 `search_policy` 和 `ranking_policy`
- `ExperimentConfig.params` 是从结构化配置归一化得到的参数快照
- `parameter_space` 是本次实验的白名单快照
- `result` 保存指标、资源、参数和产物路径

## 响应对象

### `LocalDatasetSummary`

本地数据集发现结果，主要字段：

- `name`
- `source_dir`
- `classification_dir`
- `has_source_dir`
- `has_prepared_source_dir`
- `train_manifest_exists`
- `val_manifest_exists`
- `test_manifest_exists`
- `is_ready_for_training`
- `original_image_size`
- `image_width`
- `image_height`
- `image_size_options`
- `train_sample_count`
- `val_sample_count`
- `test_sample_count`
- `class_names`
- `message`

### `ModelDefaults`

模型默认配置模板，主要字段：

- `summary`
- `parameter_space`
- `default_model_recipe`
- `default_train_hyp`
- `default_search_policy`
- `default_ranking_policy`

### `ModelManifestDraftResponse`

模型 manifest 草稿预览，主要字段：

- `query`
- `resolved_model_name`
- `ai_preview_text`
- `yaml_text`
- `target_path`
- `provider_warnings`
- `validation`

### `ProposalSchema`

AI 生成的结构化 proposal，主要字段：

- `task_type`
- `model_name`
- `based_on_experiment_ids`
- `hypothesis`
- `changes`
- `train_hyp_changes`
- `recipe_changes`
- `reason`

### `ResultSchema`

训练结果结构化对象，主要字段：

- `status`
- `metrics`
- `resource`
- `params`
  - 从结构化配置和训练结果归一化得到的参数快照
- `artifacts`

## 相关文档

- [docs/overview.md](overview.md)
- [docs/schemas/model_recipe.md](schemas/model_recipe.md)
- [docs/schemas/train_hyp.md](schemas/train_hyp.md)
- [docs/schemas/dataset_recipe.md](schemas/dataset_recipe.md)
- [docs/policies/rank.md](policies/rank.md)
- [docs/policies/search.md](policies/search.md)
