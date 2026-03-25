# Schema Notes

这份文档只描述当前仍然有效的结构化对象与字段方向，不再记录已经废弃的参数设计。

## 1. 核心对象

当前平台围绕五类结构化对象工作：

- `experiment config`
- `editable parameter space`
- `search policy`
- `proposal`
- `result`

## 2. Experiment Config

`experiment config` 表示一次实验真正执行的配置快照。

关键字段：

- `task_type`
- `dataset`
- `model_family`
- `model_name`
- `parameter_space_version`
- `participates_in_ranking`
- `search_policy`
- `params`

示例：

```json
{
  "task_type": "classification",
  "dataset": "cifar10",
  "model_family": "mobilenet",
  "model_name": "mobilenet_v2",
  "parameter_space_version": "mobilenet_v2@v1",
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
  "params": {
    "optimizer": "adamw",
    "learning_rate": 0.003,
    "batch_size": 128,
    "image_size": 32,
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
```

## 3. Editable Parameter Space

`editable parameter space` 定义某个模型允许 AI 修改哪些字段，以及每个字段的合法范围。

当前参数定义类型：

- `enum`
- `number_range`
- `discrete_values`

示例：

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
    "scheduler": {
      "type": "enum",
      "choices": ["none", "step", "cosine"]
    },
    "augmentation_policy": {
      "type": "enum",
      "choices": ["none", "basic"]
    },
    "mixup_alpha": {
      "type": "number_range",
      "min": 0.0,
      "max": 1.0
    },
    "cutmix_alpha": {
      "type": "number_range",
      "min": 0.0,
      "max": 1.0
    },
    "random_erasing_prob": {
      "type": "number_range",
      "min": 0.0,
      "max": 0.5
    },
    "loss_name": {
      "type": "enum",
      "choices": ["cross_entropy", "cross_entropy_with_label_smoothing", "focal_loss"]
    },
    "focal_gamma": {
      "type": "number_range",
      "min": 0.5,
      "max": 5.0
    },
    "label_smoothing": {
      "type": "number_range",
      "min": 0.0,
      "max": 0.2
    }
  }
}
```

## 4. Search Policy

`search policy` 决定当前 run 中 AI 到底能动哪些字段。

示例：

```json
{
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
  "allow_loss_search": true,
  "allow_augmentation_search": true,
  "require_manual_approval_for_high_impact_changes": true
}
```

搜索类别含义：

- basic
  - `optimizer`
  - `learning_rate`
  - `batch_size`
  - `weight_decay`
  - `scheduler`
  - `label_smoothing`
  - `image_size`（仅在满足特殊规则时）
- loss
  - `loss_name`
  - `focal_gamma`
- augmentation
  - `augmentation_policy`
  - `mixup_alpha`
  - `cutmix_alpha`
  - `random_erasing_prob`
- strategy
  - `aux_logits`

## 5. Proposal

proposal 是 AI 输出的结构化参数变更建议。

关键约束：

- `task_type` 固定为 `classification`
- `model_name` 必须与 run 保持一致
- `based_on_experiment_ids` 必须可追溯
- `changes` 至少一个字段非空
- 只能修改 `search_policy` 放开的字段

示例：

```json
{
  "task_type": "classification",
  "model_name": "mobilenet_v2",
  "based_on_experiment_ids": ["exp_8c9e7467"],
  "hypothesis": "当前基础超参数已接近稳定，可尝试增加少量 mixup 缓解过拟合。",
  "changes": {
    "mixup_alpha": 0.2,
    "random_erasing_prob": 0.1
  },
  "reason": "最近几轮学习率和权重衰减微调没有刷新 best，适合切换到增强维度。",
  "risk": "low"
}
```

## 6. Result

result 表示一次训练完成后的结构化输出。

关键字段：

- `status`
- `metrics`
- `resource`
- `params`
- `artifacts`

示例：

```json
{
  "status": "success",
  "metrics": {
    "train_loss": 0.576,
    "val_loss": 0.6173,
    "top1_acc": 0.9667,
    "best_epoch": 7
  },
  "resource": {
    "gpu_memory_mb": 0,
    "training_seconds": 22
  },
  "params": {
    "optimizer": "adamw",
    "learning_rate": 0.003,
    "batch_size": 128,
    "image_size": 32,
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
    "checkpoint_path": "artifacts/checkpoints/exp_9019a748.pt"
  }
}
```

## 7. 运行时边界

- trainer 不读取自由文本
- proposal 不能越过 parameter space
- 数据库存结构化记录与路径索引
- 本地文件存 run log 与 checkpoint

更具体的运行规则见 [docs/experiment_policy.md](/home/fish/AutoVisionLab/docs/experiment_policy.md)。
