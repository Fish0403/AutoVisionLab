# 参数 Schema 设计

## 1. 目标

这份文档定义第一版自主训练平台里最关键的几类结构化数据：

- `experiment config`
- `editable parameter space`
- `proposal`
- `result`
- `reflection`

第一版约束：

- 固定模型实现
- 不允许修改模型结构
- AI 只能在参数白名单内提案

## 2. 设计原则

- 所有实验都必须有完整参数快照
- 参数空间和实验参数分开存
- proposal 必须能被 schema 校验
- trainer 只能读取结构化 config，不读取自由文本
- 前端点选图表后，必须能直接展示该 experiment 的参数

## 3. Experiment Config

`experiment config` 表示一次实验最终实际执行的参数。

### 字段说明

- `task_type`
  固定为 `classification`
- `dataset`
  数据集名称
- `model_family`
  模型族
- `model_name`
  具体模型名
- `params`
  本次实验真正生效的参数
- `parameter_space_version`
  当前参数空间版本号

### 示例

```json
{
  "task_type": "classification",
  "dataset": "cifar10",
  "model_family": "mobilenet",
  "model_name": "mobilenet_v2",
  "parameter_space_version": "mobilenet_v2@v1",
  "params": {
    "optimizer": "adamw",
    "learning_rate": 0.003,
    "batch_size": 128,
    "image_size": 64,
    "epochs": 30,
    "weight_decay": 0.0001,
    "scheduler": "cosine",
    "augmentation_level": "medium",
    "label_smoothing": 0.1
  }
}
```

## 4. Editable Parameter Space

`editable parameter space` 定义某个模型允许 AI 修改哪些参数，以及参数的合法范围。

### 字段说明

- `model_name`
  适用模型
- `version`
  参数空间版本
- `editable_params`
  可调参数定义

每个参数建议支持 3 种类型：

- `enum`
- `number_range`
- `discrete_values`

### 通用定义示例

```json
{
  "model_name": "mobilenet_v2",
  "version": "mobilenet_v2@v1",
  "editable_params": {
    "optimizer": {
      "type": "enum",
      "choices": ["sgd", "adam", "adamw"]
    },
    "learning_rate": {
      "type": "number_range",
      "min": 0.0001,
      "max": 0.01
    },
    "batch_size": {
      "type": "discrete_values",
      "choices": [32, 64, 128, 256]
    }
  }
}
```

## 5. MobileNetV2 Parameter Space

第一版 `mobilenet_v2` 建议开放这些参数：

- `optimizer`
- `learning_rate`
- `batch_size`
- `image_size`
- `epochs`
- `weight_decay`
- `scheduler`
- `augmentation_level`
- `label_smoothing`

### 示例

```json
{
  "model_name": "mobilenet_v2",
  "version": "mobilenet_v2@v1",
  "editable_params": {
    "optimizer": {
      "type": "enum",
      "choices": ["sgd", "adam", "adamw"]
    },
    "learning_rate": {
      "type": "number_range",
      "min": 0.0001,
      "max": 0.01
    },
    "batch_size": {
      "type": "discrete_values",
      "choices": [32, 64, 128, 256]
    },
    "image_size": {
      "type": "discrete_values",
      "choices": [32, 64, 96]
    },
    "epochs": {
      "type": "discrete_values",
      "choices": [10, 20, 30, 50]
    },
    "weight_decay": {
      "type": "number_range",
      "min": 0.0,
      "max": 0.01
    },
    "scheduler": {
      "type": "enum",
      "choices": ["none", "step", "cosine"]
    },
    "augmentation_level": {
      "type": "enum",
      "choices": ["low", "medium", "high"]
    },
    "label_smoothing": {
      "type": "number_range",
      "min": 0.0,
      "max": 0.2
    }
  }
}
```

## 6. GoogLeNet Parameter Space

第一版 `googlenet` 建议开放这些参数：

- `optimizer`
- `learning_rate`
- `batch_size`
- `image_size`
- `epochs`
- `weight_decay`
- `scheduler`
- `augmentation_level`
- `label_smoothing`
- `aux_logits`

### 示例

```json
{
  "model_name": "googlenet",
  "version": "googlenet@v1",
  "editable_params": {
    "optimizer": {
      "type": "enum",
      "choices": ["sgd", "adam", "adamw"]
    },
    "learning_rate": {
      "type": "number_range",
      "min": 0.0001,
      "max": 0.01
    },
    "batch_size": {
      "type": "discrete_values",
      "choices": [32, 64, 128, 256]
    },
    "image_size": {
      "type": "discrete_values",
      "choices": [32, 64, 96]
    },
    "epochs": {
      "type": "discrete_values",
      "choices": [10, 20, 30, 50]
    },
    "weight_decay": {
      "type": "number_range",
      "min": 0.0,
      "max": 0.01
    },
    "scheduler": {
      "type": "enum",
      "choices": ["none", "step", "cosine"]
    },
    "augmentation_level": {
      "type": "enum",
      "choices": ["low", "medium", "high"]
    },
    "label_smoothing": {
      "type": "number_range",
      "min": 0.0,
      "max": 0.2
    },
    "aux_logits": {
      "type": "enum",
      "choices": [true, false]
    }
  }
}
```

## 7. Proposal

`proposal` 表示 AI 给出的下一轮参数调整建议。

它不直接等于最终执行配置，只是候选提案。后端要先校验，再生成 `experiment config`。

### 字段说明

- `task_type`
- `model_name`
- `based_on_experiment_ids`
  参考了哪些历史实验
- `hypothesis`
  这轮调整的假设
- `changes`
  打算修改哪些参数
- `reason`
  为什么这样改
- `risk`
  风险等级

### 示例

```json
{
  "task_type": "classification",
  "model_name": "googlenet",
  "based_on_experiment_ids": ["exp_0012", "exp_0015", "exp_0018"],
  "hypothesis": "slightly higher learning rate may improve early convergence",
  "changes": {
    "learning_rate": 0.004,
    "scheduler": "cosine",
    "label_smoothing": 0.05
  },
  "reason": "recent experiments show underfitting in early epochs",
  "risk": "low"
}
```

## 8. Result

`result` 表示训练执行后的结构化输出。

### 字段说明

- `status`
  `success` 或 `failed`
- `metrics`
  当前任务指标
- `resource`
  时间和显存等资源信息
- `params`
  实际使用参数
- `artifacts`
  日志和 checkpoint 地址

### 示例

```json
{
  "status": "success",
  "metrics": {
    "train_loss": 0.42,
    "val_loss": 0.51,
    "top1_acc": 0.84
  },
  "resource": {
    "gpu_memory_mb": 2100,
    "training_seconds": 320
  },
  "params": {
    "optimizer": "adamw",
    "learning_rate": 0.004,
    "batch_size": 128,
    "image_size": 64,
    "epochs": 30,
    "weight_decay": 0.0001,
    "scheduler": "cosine",
    "augmentation_level": "medium",
    "label_smoothing": 0.05,
    "aux_logits": true
  },
  "artifacts": {
    "log_path": "artifacts/logs/exp_0019.log",
    "checkpoint_path": "artifacts/checkpoints/exp_0019.pt"
  }
}
```

## 9. Reflection

`reflection` 表示 AI 对实验结果的分析。

### 字段说明

- `outcome`
  `improved`、`neutral`、`degraded`、`failed`
- `analysis`
  对这轮结果的简短分析
- `confidence`
  0 到 1
- `next_action`
  下一步建议
- `recommended_changes`
  建议继续尝试的参数方向

### 示例

```json
{
  "outcome": "improved",
  "analysis": "higher learning rate improved convergence without obvious instability",
  "confidence": 0.78,
  "next_action": "explore nearby learning rates and keep cosine scheduler",
  "recommended_changes": {
    "learning_rate": 0.005,
    "label_smoothing": 0.08
  }
}
```

## 10. 校验流程

后端执行时建议严格走这条链路：

1. 读取模型对应的 `editable parameter space`
2. 校验 proposal 里的 `changes` 是否都在白名单中
3. 校验 proposal 的值是否合法
4. 将 proposal 合并到基础 `experiment config`
5. 生成最终执行 config
6. trainer 用最终 config 启动训练
7. 训练结束后保存 `result`
8. AI 读取 `result` 生成 `reflection`

## 11. 前端展示建议

前端至少要能展示这些字段：

- model name
- dataset
- experiment id
- 所有实际执行参数
- 核心指标
- proposal
- reflection

图表点选后，建议把 `params` 原样展示，而不是只展示 diff。第一版先保证信息完整，后续再加参数对比视图。
