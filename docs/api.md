# AutoVisionLab API 文档

本文档描述当前后端实际暴露的 HTTP API，用于前端联调、手工调试和后续补测试。

当前后端基于 `FastAPI`，启动后默认可访问：

- `GET /docs`：Swagger UI
- `GET /redoc`：ReDoc
- `GET /openapi.json`：OpenAPI 描述

当前接口未统一挂在 `/api` 前缀下，以下路径均为服务根路径下的实际路由。

## 1. 通用约定

### 1.1 统一响应信封

除框架自动文档页面外，业务接口统一返回同一外层结构：

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

字段说明：

- `ok`：是否成功
- `code`：稳定结果码，成功时常见为 `success`、`created`、`updated`、`queued`、`running`
- `message`：面向人类阅读的信息
- `data`：实际业务数据
- `errors`：错误详情数组；成功时通常为空数组
- `meta`：响应元信息，当前包含 `schema_version` 与 UTC 时间戳

### 1.2 常见错误码

当前全局异常处理会把常见错误映射为以下 `code`：

- `bad_request`：请求语义错误，通常对应 `400`
- `not_found`：资源不存在，通常对应 `404`
- `conflict`：状态冲突，通常对应 `409`
- `validation_error`：请求体或参数校验失败，对应 `422`
- `upstream_error`：上游 AI 服务请求失败，对应 `502`
- `internal_error`：未处理异常，对应 `500`

`validation_error` 响应中的 `errors` 会额外携带 `field` 字段，例如：

```json
{
  "ok": false,
  "code": "validation_error",
  "message": "Request validation failed.",
  "data": null,
  "errors": [
    {
      "message": "Field required",
      "code": "validation_error",
      "field": "body.run_id"
    }
  ],
  "meta": {
    "schema_version": "v1",
    "timestamp": "2026-03-29T08:00:00Z"
  }
}
```

### 1.3 核心枚举

接口中常见的状态和枚举如下：

- `task_type`：`classification`
- `model_name`：`mobilenet_v3_small`、`googlenet`、`resnet18`
- `model_family`：`mobilenet`、`googlenet`、`resnet`
- `run.status`：`draft`、`active`、`paused`、`completed`、`failed`
- `experiment.status`：`draft`、`queued`、`running`、`success`、`failed`、`discarded`
- `experiment.decision`：`keep`、`discard`、`crash`、`timeout`
- `proposal.risk`：`low`、`medium`、`high`
- `result.status`：`success`、`failed`

## 2. 对象速览

### 2.1 ApiResponse<T>

所有业务接口的统一返回外层。

### 2.2 LocalDatasetSummary

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
- `image_size_options`
- `message`

### 2.3 ExperimentConfig

训练器实际消费的结构化配置，主要字段：

- `task_type`
- `dataset`
- `model_family`
- `model_name`
- `parameter_space_version`
- `use_demo_mode`
- `participates_in_ranking`
- `search_policy`
- `ranking_policy`
- `params`

其中：

- `search_policy` 控制 AI 可搜索的参数范围
- `ranking_policy` 控制同一 run 内实验的比较规则
- `params` 为训练参数，如 `optimizer`、`learning_rate`、`batch_size`、`image_size`、`epochs`

### 2.4 EditableParameterSpace

模型参数白名单定义：

- `model_name`
- `version`
- `editable_params`

`editable_params` 的 value 目前支持三种定义：

- `enum`
- `number_range`
- `discrete_values`

### 2.5 ProposalSchema

AI 生成的结构化 proposal，字段包括：

- `task_type`
- `model_name`
- `based_on_experiment_ids`
- `hypothesis`
- `changes`
- `reason`
- `risk`

### 2.6 ResultSchema

训练结果结构化对象，字段包括：

- `status`
- `metrics`
- `resource`
- `params`
- `artifacts`

其中：

- `metrics` 当前包含 `train_loss`、`val_loss`、`top1_acc`、`best_epoch`
- `resource` 当前包含 `gpu_memory_mb`、`training_seconds`
- `artifacts` 当前包含 `log_path`、`checkpoint_path`

## 3. 系统接口

### 3.1 健康检查

`GET /health`

用途：

- 返回服务最小健康状态

成功响应：

```json
{
  "ok": true,
  "code": "success",
  "message": "Service is healthy.",
  "data": {
    "status": "ok"
  },
  "errors": [],
  "meta": {
    "schema_version": "v1",
    "timestamp": "2026-03-29T08:00:00Z"
  }
}
```

## 4. 数据集接口

### 4.1 列出本地数据集

`GET /datasets`

用途：

- 扫描本地数据目录
- 返回每个数据集是否已具备训练所需清单文件

成功响应 `data` 类型：

- `LocalDatasetSummary[]`

## 5. Run 接口

### 5.1 列出所有 run

`GET /runs`

成功响应 `data` 类型：

- `RunListItem[]`

`RunListItem` 主要字段：

- `id`
- `name`
- `dataset`
- `model_name`
- `status`
- `experiment_count`

### 5.2 创建 run

`POST /runs`

请求体：

```json
{
  "name": "neu-resnet18-baseline",
  "dataset": "neu",
  "model_name": "resnet18",
  "notes": "Initial baseline run.",
  "base_config": {
    "task_type": "classification",
    "dataset": "neu",
    "model_family": "resnet",
    "model_name": "resnet18",
    "parameter_space_version": "resnet18@v1",
    "use_demo_mode": false,
    "participates_in_ranking": true,
    "search_policy": {
      "allow_basic_hparam_search": true,
      "allowed_basic_hparam_fields": [
        "optimizer",
        "learning_rate",
        "batch_size",
        "weight_decay",
        "scheduler",
        "label_smoothing"
      ],
      "allow_strategy_search": false,
      "allow_loss_search": false,
      "allow_augmentation_search": false,
      "require_manual_approval_for_high_impact_changes": true
    },
    "ranking_policy": {
      "primary_metric": "top1_acc",
      "primary_metric_mode": "max",
      "min_primary_metric_improvement": 0.01,
      "primary_metric_parity_epsilon": 0.0005,
      "tie_breaker_metric": "val_loss",
      "tie_breaker_mode": "min",
      "min_tie_breaker_metric_improvement": 0.01,
      "max_image_size": null
    },
    "params": {
      "optimizer": "adamw",
      "learning_rate": 0.003,
      "batch_size": 64,
      "image_size": 224,
      "epochs": 10,
      "weight_decay": 0.0001,
      "scheduler": "cosine",
      "augmentation_policy": "basic",
      "augmentation_params": {
        "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0,
        "random_erasing_prob": 0.0
      },
      "loss_name": "cross_entropy_with_label_smoothing",
      "loss_params": {
        "focal_gamma": 2.0
      },
      "label_smoothing": 0.1,
      "aux_logits": false
    }
  }
}
```

成功状态码：

- `201 Created`

成功响应 `data` 类型：

- `RunDetailResponse`

常见错误：

- `400 bad_request`：基础配置不合法

### 5.3 获取单个 run 详情

`GET /runs/{run_id}`

路径参数：

- `run_id`：run 标识

成功响应 `data` 类型：

- `RunDetailResponse`

`RunDetailResponse` 主要字段：

- `id`
- `name`
- `dataset`
- `model_name`
- `status`
- `notes`
- `baseline_experiment_id`
- `best_experiment_id`
- `frontier_experiment_id`
- `experiments`

常见错误：

- `404 not_found`：run 不存在

### 5.4 获取 run 指标趋势

`GET /runs/{run_id}/metrics`

查询参数：

- `metric_name`：可选，默认 `top1_acc`

成功响应 `data` 类型：

- `RunMetricsResponse`

`RunMetricsResponse` 主要字段：

- `run_id`
- `metric_name`
- `available_metrics`
- `points`

`points` 中每个点包含：

- `experiment_id`
- `experiment_index`
- `metric_name`
- `metric_value`

常见错误：

- `404 not_found`：run 不存在

### 5.5 获取 run 摘要

`GET /runs/{run_id}/summary`

成功响应 `data` 类型：

- `RunSummaryResponse`

`RunSummaryResponse` 主要字段：

- `run_id`
- `baseline_experiment_id`
- `best_experiment_id`
- `frontier_experiment_id`
- `keep_count`
- `discard_count`
- `crash_count`
- `timeout_count`

常见错误：

- `404 not_found`：run 不存在

### 5.6 生成 AI proposal

`POST /runs/{run_id}/proposal`

用途：

- 基于当前 run 历史生成一个新的结构化 proposal

成功响应 `data` 类型：

- `ProposalSchema`

常见错误：

- `400 bad_request`：run 状态不允许生成 proposal，或当前上下文不足
- `502 upstream_error`：AI 服务请求失败

### 5.7 测试 proposal provider 连通性

`POST /runs/proposal/test`

用途：

- 对 AI proposal provider 发起一次最小连通性测试

成功响应 `data` 类型：

```json
{
  "status": "ok"
}
```

常见错误：

- `502 upstream_error`：AI 服务不可用或返回异常

### 5.8 清空所有 run

`POST /runs/reset`

用途：

- 删除所有持久化的 runs、experiments、results

成功响应 `data` 类型：

```json
{
  "deleted_runs": 1,
  "deleted_experiments": 4,
  "deleted_results": 4,
  "deleted_artifact_files": 5
}
```

说明：

- 该接口是破坏性操作，调用前应明确确认环境

### 5.9 清空单个 run

`POST /runs/{run_id}/reset`

用途：

- 删除一个 run 及其下属 experiments 和 results

成功响应 `data` 类型：

```json
{
  "deleted_runs": 1,
  "deleted_experiments": 4,
  "deleted_results": 4,
  "deleted_artifact_files": 5
}
```

常见错误：

- `404 not_found`：run 不存在

### 5.10 启动 auto-train 后台任务

`POST /runs/auto-train`

请求体：

```json
{
  "run_id": "run_ab12cd34",
  "run_name": "neu-resnet18-baseline",
  "dataset": "neu",
  "model_name": "resnet18",
  "config": {
    "task_type": "classification",
    "dataset": "neu",
    "model_family": "resnet",
    "model_name": "resnet18",
    "parameter_space_version": "resnet18@v1",
    "use_demo_mode": false,
    "participates_in_ranking": true,
    "search_policy": {
      "allow_basic_hparam_search": true,
      "allowed_basic_hparam_fields": [
        "optimizer",
        "learning_rate",
        "batch_size",
        "weight_decay",
        "scheduler",
        "label_smoothing"
      ],
      "allow_strategy_search": true,
      "allow_loss_search": true,
      "allow_augmentation_search": true,
      "require_manual_approval_for_high_impact_changes": true
    },
    "ranking_policy": {
      "primary_metric": "top1_acc",
      "primary_metric_mode": "max",
      "min_primary_metric_improvement": 0.01,
      "primary_metric_parity_epsilon": 0.0005,
      "tie_breaker_metric": "val_loss",
      "tie_breaker_mode": "min",
      "min_tie_breaker_metric_improvement": 0.01,
      "max_image_size": null
    },
    "params": {
      "optimizer": "adamw",
      "learning_rate": 0.003,
      "batch_size": 64,
      "image_size": 224,
      "epochs": 10,
      "weight_decay": 0.0001,
      "scheduler": "cosine",
      "augmentation_policy": "basic",
      "augmentation_params": {
        "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0,
        "random_erasing_prob": 0.0
      },
      "loss_name": "cross_entropy_with_label_smoothing",
      "loss_params": {
        "focal_gamma": 2.0
      },
      "label_smoothing": 0.1,
      "aux_logits": false
    }
  },
  "parameter_space": {
    "model_name": "resnet18",
    "version": "resnet18@v1",
    "editable_params": {
      "optimizer": {
        "type": "enum",
        "choices": ["sgd", "adam", "adamw"]
      }
    }
  }
}
```

说明：

- `run_id` 可为空；为空时由后台按请求内容创建或延续实际执行上下文
- `config` 与 `parameter_space` 均为必填

成功状态码：

- `202 Accepted`

成功响应 `data` 类型：

- `AutoTrainTaskResponse`

`AutoTrainTaskResponse` 主要字段：

- `task_id`
- `status`
- `run_id`
- `current_round`
- `elapsed_seconds`
- `current_experiment_id`
- `logs`
- `summary`
- `error`
- `stop_requested`
- `stop_reason`
- 各类 token 统计字段

常见错误：

- `409 conflict`：已有活动中的 auto-train 任务

### 5.11 获取当前活动中的 auto-train 任务

`GET /runs/auto-train/active`

成功响应 `data` 类型：

- `AutoTrainTaskResponse`

常见错误：

- `404 not_found`：当前没有活动中的 auto-train 任务

### 5.12 获取指定 auto-train 任务快照

`GET /runs/auto-train/{task_id}`

成功响应 `data` 类型：

- `AutoTrainTaskResponse`

常见错误：

- `404 not_found`：任务不存在

### 5.13 请求停止 auto-train 任务

`POST /runs/auto-train/{task_id}/stop`

成功响应 `data` 类型：

- `AutoTrainTaskResponse`

说明：

- 接口语义是“请求停止”，不是同步等待任务完全结束
- 响应中的 `status` 可能先进入 `stopping`

常见错误：

- `404 not_found`：任务不存在

## 6. Experiment 接口

### 6.1 创建 experiment

`POST /experiments`

请求体：

```json
{
  "run_id": "run_ab12cd34",
  "config": {
    "task_type": "classification",
    "dataset": "neu",
    "model_family": "resnet",
    "model_name": "resnet18",
    "parameter_space_version": "resnet18@v1",
    "use_demo_mode": false,
    "participates_in_ranking": true,
    "search_policy": {
      "allow_basic_hparam_search": true,
      "allowed_basic_hparam_fields": [
        "optimizer",
        "learning_rate",
        "batch_size",
        "weight_decay",
        "scheduler",
        "label_smoothing"
      ],
      "allow_strategy_search": false,
      "allow_loss_search": false,
      "allow_augmentation_search": false,
      "require_manual_approval_for_high_impact_changes": true
    },
    "ranking_policy": {
      "primary_metric": "top1_acc",
      "primary_metric_mode": "max",
      "min_primary_metric_improvement": 0.01,
      "primary_metric_parity_epsilon": 0.0005,
      "tie_breaker_metric": "val_loss",
      "tie_breaker_mode": "min",
      "min_tie_breaker_metric_improvement": 0.01,
      "max_image_size": null
    },
    "params": {
      "optimizer": "adamw",
      "learning_rate": 0.003,
      "batch_size": 64,
      "image_size": 224,
      "epochs": 10,
      "weight_decay": 0.0001,
      "scheduler": "cosine",
      "augmentation_policy": "basic",
      "augmentation_params": {
        "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0,
        "random_erasing_prob": 0.0
      },
      "loss_name": "cross_entropy_with_label_smoothing",
      "loss_params": {
        "focal_gamma": 2.0
      },
      "label_smoothing": 0.1,
      "aux_logits": false
    }
  },
  "parameter_space": {
    "model_name": "resnet18",
    "version": "resnet18@v1",
    "editable_params": {
      "optimizer": {
        "type": "enum",
        "choices": ["sgd", "adam", "adamw"]
      }
    }
  },
  "proposal": {
    "task_type": "classification",
    "model_name": "googlenet",
    "based_on_experiment_ids": ["exp_1234abcd"],
    "hypothesis": "Enable auxiliary heads and add light mixup for the next round.",
    "changes": {
      "mixup_alpha": 0.2,
      "aux_logits": true
    },
    "train_hyp_changes": {
      "augmentation": {
        "mixup": 0.2
      }
    },
    "recipe_changes": {
      "modules": {
        "aux_logits": true
      }
    },
    "reason": "The previous run is stable enough to try one augmentation plus one strategy change.",
    "risk": "low"
  }
}
```

说明：

- `proposal` 可为空
- `parameter_space` 为本次实验的白名单快照

成功状态码：

- `201 Created`

成功响应 `data` 类型：

- `ExperimentDetailResponse`

常见错误：

- `400 bad_request`：请求体不合法
- `404 not_found`：`run_id` 不存在

### 6.2 获取 experiment 详情

`GET /experiments/{experiment_id}`

成功响应 `data` 类型：

- `ExperimentDetailResponse`

`ExperimentDetailResponse` 主要字段：

- `id`
- `run_id`
- `status`
- `decision`
- `decision_reason`
- `baseline_experiment_id`
- `is_best_so_far`
- `config`
- `parameter_space`
- `proposal`
- `result`
- `reflection`

常见错误：

- `404 not_found`：experiment 不存在

### 6.3 保存训练结果

`POST /experiments/{experiment_id}/result`

请求体：

```json
{
  "status": "success",
  "metrics": {
    "train_loss": 0.42,
    "val_loss": 0.37,
    "top1_acc": 0.93,
    "best_epoch": 8
  },
  "resource": {
    "gpu_memory_mb": 4096,
    "training_seconds": 125
  },
  "params": {
    "optimizer": "adamw",
    "learning_rate": 0.001,
    "batch_size": 64,
    "image_size": 224,
    "epochs": 10,
    "weight_decay": 0.0001,
    "scheduler": "cosine",
    "augmentation_policy": "basic",
    "augmentation_params": {
      "mixup_alpha": 0.0,
      "cutmix_alpha": 0.0,
      "random_erasing_prob": 0.0
    },
    "loss_name": "cross_entropy_with_label_smoothing",
    "loss_params": {
      "focal_gamma": 2.0
    },
    "label_smoothing": 0.1,
    "aux_logits": false
  },
  "artifacts": {
    "log_path": "artifacts/runs/run_ab12cd34.log",
    "checkpoint_path": "artifacts/checkpoints/exp_1234abcd.pt"
  }
}
```

成功响应 `data` 类型：

- `ExperimentDetailResponse`

常见错误：

- `404 not_found`：experiment 不存在

### 6.4 写入研究决策

`POST /experiments/{experiment_id}/decision`

请求体：

```json
{
  "decision": "keep",
  "decision_reason": "Improved top1_acc while keeping validation loss stable."
}
```

成功响应 `data` 类型：

- `ExperimentDetailResponse`

常见错误：

- `404 not_found`：experiment 不存在

### 6.5 启动 experiment 训练

`POST /experiments/{experiment_id}/train`

用途：

- 启动单个实验的后台训练

成功响应 `data` 类型：

- `ExperimentDetailResponse`

说明：

- 返回对象为刷新后的 experiment 详情
- 成功时 `code` 可能直接等于 experiment 当前状态，如 `queued` 或 `running`

常见错误：

- `404 not_found`：experiment 不存在
- `409 conflict`：当前状态不允许重复启动

### 6.6 停止 experiment 训练

`POST /experiments/{experiment_id}/stop`

用途：

- 请求停止当前训练中的 experiment，并在持久化层将其标记为丢弃

成功响应 `data` 类型：

- `ExperimentDetailResponse`

说明：

- 当前接口会继续调用 discard 流程，因此返回对象通常已处于停止后的持久化状态

常见错误：

- `404 not_found`：experiment 不存在
- `409 conflict`：当前状态不允许停止

### 6.7 启动实验建议任务

`POST /experiments/{experiment_id}/suggestion`

用途：

- 为已完成 experiment 启动或复用一个后台建议任务

成功状态码：

- `202 Accepted`

成功响应 `data` 类型：

- `ExperimentSuggestionTaskResponse`

`ExperimentSuggestionTaskResponse` 主要字段：

- `task_id`
- `experiment_id`
- `run_id`
- `status`
- `suggestion`
- `error`
- provider token 统计字段

常见错误：

- `409 conflict`：当前 experiment 不满足发起建议任务的条件

### 6.8 获取实验建议任务

`GET /experiments/{experiment_id}/suggestion`

成功响应 `data` 类型：

- `ExperimentSuggestionTaskResponse`

常见错误：

- `404 not_found`：建议任务不存在

## 7. 模型接口

### 7.1 获取模型 parameter space

`GET /models/{model_name}/parameter-space`

路径参数：

- `model_name`：支持的模型名

成功响应 `data` 类型：

- `EditableParameterSpace`

成功响应示例：

```json
{
  "ok": true,
  "code": "success",
  "message": "Parameter space loaded.",
  "data": {
    "model_name": "resnet18",
    "version": "resnet18@v1",
    "editable_params": {
      "optimizer": {
        "type": "enum",
        "choices": ["sgd", "adam", "adamw"]
      },
      "learning_rate": {
        "type": "number_range",
        "min": 0.0001,
        "max": 0.01
      }
    }
  },
  "errors": [],
  "meta": {
    "schema_version": "v1",
    "timestamp": "2026-03-29T08:00:00Z"
  }
}
```

常见错误：

- `404 not_found`：模型不受支持

## 8. 当前文档边界

本文档当前聚焦：

- 实际路由
- 请求体结构
- 成功响应对象
- 常见错误码

本文档暂未展开：

- 每个字段的完整 schema 约束细则
- 所有对象的逐字段示例全集
- 训练产物内部文件格式
- AI 搜索与停止策略细节

相关文档：

- [docs/plan.md](plan.md)：项目定位与边界
- [docs/schemas/README.md](schemas/README.md)：结构化对象说明
- [docs/policies/README.md](policies/README.md)：策略文档总览
- [docs/policies/experiment_policy.md](policies/experiment_policy.md)：实验运行规则
- [docs/policies/auto_train_search_policy.md](policies/auto_train_search_policy.md)：Auto Train 搜索策略
- [docs/policies/auto_train_stop_policy.md](policies/auto_train_stop_policy.md)：Auto Train 停止策略
