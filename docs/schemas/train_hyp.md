# Train Hyp

这份文档定义当前使用的 `train_hyp` 结构。

## 作用

- 统一表达训练超参数、增强和 loss 相关配置
- 作为当前分类训练链路的主训练参数对象
- 作为 `ExperimentConfig.params` 的结构化替代视图

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
  - `policy`
  - `mixup`
  - `cutmix`
  - `random_erasing`
- `loss`
  - `name`
- `runtime`
  - `amp`
  - `grad_clip_norm`

## 映射

`TrainHyp.to_experiment_params()` 会把结构化字段回填为旧的 `params` 视图，当前映射为：

- `optimizer` -> `optimizer`
- `lr0` -> `learning_rate`
- `batch_size` -> `batch_size`
- `image_size` -> `image_size`
- `epochs` -> `epochs`
- `weight_decay` -> `weight_decay`
- `scheduler` -> `scheduler`
- `augmentation.policy` -> `augmentation_policy`
- `augmentation.mixup` -> `augmentation_params.mixup_alpha`
- `augmentation.cutmix` -> `augmentation_params.cutmix_alpha`
- `augmentation.random_erasing` -> `augmentation_params.random_erasing_prob`
- `loss.name` -> `loss_name`
- `fl_gamma` -> `loss_params.focal_gamma`
- `label_smoothing` -> `label_smoothing`

## 当前可搜索字段

- `optimizer`
- `lr0`
- `weight_decay`
- `scheduler`
- `batch_size`
- `image_size`
- `dropout`
- `label_smoothing`
- `augmentation.mixup`
- `augmentation.cutmix`
- `augmentation.random_erasing`
- `loss.name`
- `fl_gamma`

## 当前值范围

- `scheduler`
  - `none`、`step`、`cosine`
- `augmentation.policy`
  - `none`、`basic`
- `loss.name`
  - `cross_entropy`、`cross_entropy_with_label_smoothing`、`focal_loss`
- `image_size`
  - 当前分类参数空间使用离散值
- `batch_size`
  - 当前分类参数空间使用离散值

## 返回位置

- `ExperimentConfig.train_hyp`
- `ExperimentConfig.params`
- 实验 result 的 `params` 字段
