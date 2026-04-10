# Train Hyp

这份文档定义当前使用的 `train_hyp` 结构。

## 作用

- 统一表达训练超参数、增强和 loss 相关配置
- 作为当前分类训练链路的主训练参数对象
- 为搜索、训练执行和结果归档提供统一训练配置来源

## 字段

- `version`
  - 固定为 `train_hyp@v1`
- `task_type`
  - 当前固定为 `classification`
- `optimizer`
- `lr0`
- `lrf`
- `momentum`
- `weight_decay`
- `warmup_epochs`
- `scheduler`
- `epochs`
- `batch_size`
- `image_size`
- `dropout`
- `label_smoothing`
- `fl_gamma`
- `augmentation`
- `loss`
- `runtime`
- `metadata`

## 子结构

- `augmentation`
  - `mixup`
  - `cutmix`
  - `random_erasing`
- `loss`
  - `name`
- `runtime`
  - `amp`
  - `grad_clip_norm`

## 归一化参数快照

`TrainHyp.to_experiment_params()` 会把结构化字段投影为统一参数快照，当前映射为：

- `optimizer` -> `optimizer`
- `lr0` -> `learning_rate`
- `batch_size` -> `batch_size`
- `image_size` -> `image_size`
- `epochs` -> `epochs`
- `weight_decay` -> `weight_decay`
- `scheduler` -> `scheduler`
- `augmentation.mixup` -> `augmentation_params.mixup_alpha`
- `augmentation.cutmix` -> `augmentation_params.cutmix_alpha`
- `augmentation.random_erasing` -> `augmentation_params.random_erasing_prob`
- `loss.name` -> `loss_name`
- `fl_gamma` -> `loss_params.focal_gamma`
- `label_smoothing` -> `label_smoothing`

## 当前可搜索字段

- `optimizer`
- `lr0`
- `epochs`
- `weight_decay`
- `scheduler`
- `batch_size`
- `image_size`
- `label_smoothing`
- `augmentation.mixup`
- `augmentation.cutmix`
- `augmentation.random_erasing`
- `loss.name`
- `fl_gamma`

说明：

- `epochs` 当前已进入后端基础搜索字段集合，但是否实际对 AI 开放，仍取决于当前 run 的 `search_policy.allowed_basic_hparam_fields`

## 当前值范围

- `scheduler`
  - `none`、`step`、`cosine`
- `loss.name`
  - `cross_entropy`、`cross_entropy_with_label_smoothing`、`focal_loss`
- `image_size`
  - 当前分类参数空间使用数值区间
- `batch_size`
  - 当前分类参数空间使用离散值

## 输出位置

- `ExperimentConfig.train_hyp`
- `ExperimentConfig.params`
- 实验 result 的 `params` 字段

其中 `ExperimentConfig.params` 和 result 内的 `params` 都是从 `train_hyp`、`model_recipe` 等结构化字段归一化得到的参数快照，不是主配置入口。
