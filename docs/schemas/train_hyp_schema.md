# Train Hyp Schema

这份文档定义 AutoVisionLab 后续引入的 `train_hyp` 结构。

## 1. 目标

`train_hyp` 用来统一表达训练超参数、增强和 loss 相关配置，并尽量靠近 YOLO 系列 `hyp.yaml` 的命名风格。

`v1` 范围明确限制为：

- 任务类型：`classification`
- base backbone：`MobileNetV3 Small`

## 2. 设计原则

- 字段命名尽量贴近 YOLO 习惯
- 同时兼容当前分类训练代码已支持的能力
- `v1` 先做分类可用字段
- 顶层对未来检测/分割常见字段预留命名口径，但不要求立刻实现

## 3. 顶层结构

建议的 `train_hyp.yaml` 如下：

```yaml
version: train_hyp@v1
task_type: classification
optimizer: adamw
lr0: 0.003
lrf: 0.01
momentum: 0.9
weight_decay: 0.0001
warmup_epochs: 0.0
scheduler: cosine
epochs: 30
batch_size: 64
image_size: 96
dropout: 0.2
label_smoothing: 0.1
fl_gamma: 0.0
augmentation:
  policy: basic
  mixup: 0.0
  cutmix: 0.0
  random_erasing: 0.0
loss:
  name: cross_entropy_with_label_smoothing
runtime:
  amp: false
  grad_clip_norm: null
metadata:
  notes: null
```

## 4. 字段说明

### 4.1 优化器与学习率

- `optimizer`
  - `sgd | adam | adamw`
- `lr0`
  - 初始学习率
- `lrf`
  - 最终学习率比例
- `momentum`
  - 主要给 `sgd` 预留
- `weight_decay`
  - 权重衰减
- `warmup_epochs`
  - warmup 轮数
- `scheduler`
  - `none | step | cosine`

说明：

- `v1` 已经有 `scheduler` 和 `weight_decay` 的实现基础
- `lrf` 与 `warmup_epochs` 当前更偏设计稿字段，后续接入训练器

### 4.2 训练规模

- `epochs`
- `batch_size`
- `image_size`

`v1` 推荐白名单：

- `epochs`
  - `10 | 20 | 30 | 50`
- `batch_size`
  - `32 | 64 | 128 | 256`
- `image_size`
  - `32 | 64 | 96`

### 4.3 正则与分类 loss

- `dropout`
  - 分类 head dropout
- `label_smoothing`
  - 标签平滑
- `fl_gamma`
  - focal gamma

`loss` 下当前建议只开放：

- `name`
  - `cross_entropy`
  - `cross_entropy_with_label_smoothing`
  - `focal_loss`

### 4.4 `augmentation`

`v1` 建议结构：

```yaml
augmentation:
  policy: basic
  mixup: 0.0
  cutmix: 0.0
  random_erasing: 0.0
```

字段说明：

- `policy`
  - `none | basic`
- `mixup`
- `cutmix`
- `random_erasing`

这里是对当前实现的结构化收口，对应现有代码里的：

- `augmentation_policy`
- `mixup_alpha`
- `cutmix_alpha`
- `random_erasing_prob`

### 4.5 `runtime`

`v1` 建议先预留：

- `amp`
- `grad_clip_norm`

说明：

- 当前实现未完整接入这些能力
- 但这类运行时字段很适合作为后续训练工程增强的入口

## 5. 面向未来多任务的预留字段

如果后续要支持检测和分割，建议现在就在 schema 设计中保留这些字段口径：

- `warmup_bias_lr`
- `box`
- `cls`
- `dfl`
- `hsv_h`
- `hsv_s`
- `hsv_v`
- `translate`
- `scale`
- `flipud`
- `fliplr`
- `mosaic`
- `copy_paste`

说明：

- `v1` 不要求分类训练器实现这些字段
- 但建议后续统一仍挂在 `train_hyp` 体系下，而不是另起一套完全不同的命名

## 6. 与当前参数结构的映射建议

当前 `ExperimentConfig.params` 里已经有一部分字段，可以先做如下映射：

- `learning_rate` -> `lr0`
- `scheduler` -> `scheduler`
- `weight_decay` -> `weight_decay`
- `label_smoothing` -> `label_smoothing`
- `loss_name` -> `loss.name`
- `focal_gamma` -> `fl_gamma`
- `augmentation_policy` -> `augmentation.policy`
- `mixup_alpha` -> `augmentation.mixup`
- `cutmix_alpha` -> `augmentation.cutmix`
- `random_erasing_prob` -> `augmentation.random_erasing`

## 7. `v1` 推荐允许 AI 搜索的字段

推荐白名单：

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

## 8. API / artifact 建议

后续建议：

- `ExperimentConfig` 增加 `train_hyp`
- 每次实验把 `train_hyp` 落盘到：
  - `artifacts/recipes/<experiment_id>_hyp.yaml`
- proposal 支持直接返回 `train_hyp_changes`

## 9. 一句话总结

`train_hyp@v1` 的目标不是重新发明一套训练参数命名，而是用接近 YOLO 的习惯把当前分类训练能力收口，并为后续检测/分割共用同一超参数体系打底。
