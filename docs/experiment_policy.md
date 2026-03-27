# 实验运行规则

这份文档描述平台现在实际按什么规则运行。

它不负责解释项目愿景。整体定位见 [plan.md](/home/fish/AutoVisionLab/plan.md)，执行事项见 [tasks.md](/home/fish/AutoVisionLab/tasks.md)，结构化对象见 [schemas.md](/home/fish/AutoVisionLab/schemas.md)。

当前默认的硬约束参数已集中定义在：

- [backend/app/schemas/run_policy.py](/home/fish/AutoVisionLab/backend/app/schemas/run_policy.py)
- [backend/app/services/run_policy.py](/home/fish/AutoVisionLab/backend/app/services/run_policy.py)

## 1. 适用范围

- 任务类型：图像分类
- 一个 run 固定一个数据集和一个模型
- AI 只能在结构化白名单参数内搜索
- 不开放模型结构搜索

当前可选模型：

- `MobileNetV2`
- `GoogLeNet`
- `ResNet18`
- `ResNet34`
- `DenseNet121`

## 2. Run 规则

- 一个 `run` 内只允许同一数据集、同一模型的实验
- 跨模型比较应通过多个 run 完成
- run 显式维护：
  - `baseline_experiment_id`
  - `best_experiment_id`
  - `frontier_experiment_id`

### 2.1 晋级 / 回退规则

- `baseline` 的首个成功实验会先成为当前 `best` 与 `frontier`
- 后续成功实验不会因为“略好一点”就自动晋级
- 当前实现的默认晋级阈值：
  - `top1_acc` 至少提升 `0.01`
  - 或在 `top1_acc` 近似持平时，`val_loss` 至少下降 `0.01`
- 如果未达到阈值：
  - 当前实验记为 `discard`
  - `best` 和 `frontier` 保持不变
  - 后续 auto-train 默认回到当前 `best / frontier` 继续分支

## 3. 参数分层

### 3.1 主表单参数

前端 `Training Setup` 直接展示：

- `run_name`
- `dataset`
- `model_name`
- `learning_rate`
- `batch_size`
- `epochs`
- `image_size`
- `auto_train_rounds`

### 3.2 隐藏管理参数

这些参数通常不放在主表单里逐个展开，但系统仍会保存和沿用：

- `optimizer`
- `weight_decay`
- `scheduler`
- `augmentation_policy`
- `label_smoothing`
- `aux_logits`

### 3.3 高影响策略参数

这些参数需要更明确的搜索开关或更严格的控制：

- `loss_name`
- `focal_gamma`
- `augmentation_policy`
- `mixup_alpha`
- `cutmix_alpha`
- `random_erasing_prob`

## 4. AI 搜索字段

### 4.1 基础超参数

当 `allow_basic_hparam_search = true` 时，允许搜索：

- `optimizer`
- `learning_rate`
- `batch_size`
- `weight_decay`
- `scheduler`
- `label_smoothing`

### 4.2 Loss / Augmentation / Strategy

- `allow_loss_search = true`
  - `loss_name`
  - `focal_gamma`
- `allow_augmentation_search = true`
  - `augmentation_policy`
  - `mixup_alpha`
  - `cutmix_alpha`
  - `random_erasing_prob`
- `allow_strategy_search = true`
  - `aux_logits`

### 4.3 默认不允许 AI 搜索

- `epochs`
- `dataset`
- `model_name`

## 5. image_size 规则

`image_size` 不作为普通默认搜索项处理。

规则：

- 当 `image_size` 等于原图大小时，AI 不搜索 `image_size`
- 当用户把 `image_size` 改成非原图大小时，AI 才允许搜索 `image_size`
- 搜索范围为“原图大小到当前设置值之间”的离散值

例如：

- `cifar10` 原图大小是 `32`
- 当前设置为 `32` 时，不搜索 `image_size`
- 当前设置为 `64` 时，只允许在 `32, 64` 内搜索
- 当前设置为 `96` 时，只允许在 `32, 64, 96` 内搜索

## 6. Proposal 规则

- proposal 必须是结构化 JSON
- proposal 必须通过 schema 校验
- proposal 必须通过 parameter space 校验
- proposal 必须通过 `search_policy` 校验
- proposal 不能为空，必须包含至少一个有效参数变化
- proposal 默认只允许修改 `1` 个字段
- 只有当 run 已连续至少 `2` 轮没有晋级时，proposal 才最多允许修改 `2` 个字段
- 同一个字段连续至少 `2` 轮参与失败实验后，会进入临时冷却
- 处于冷却中的字段，在接下来至少 `2` 轮 proposal 中不允许再次提出
- 如果最近至少 `3` 轮停滞都停留在同一个搜索维度，下一轮必须切换维度
- proposal 文本不允许引用当前未开放的字段或取值

proposal 生成时必须参考：

- 完整 run 历史
- `baseline`
- `best`
- `frontier`

这里的“搜索维度”当前分为：

- `basic`
- `augmentation`
- `loss`
- `strategy`

## 7. Auto Train 规则

`Auto Train` 采用分阶段搜索：

- 前 `50%` 轮次：
  - 允许基础超参数主导
- 后 `50%` 轮次：
  - proposal 不能只修改基础超参数
  - 必须至少包含一个 augmentation / loss / strategy 字段变化

后半程允许的非基础字段：

- `augmentation_policy`
- `mixup_alpha`
- `cutmix_alpha`
- `random_erasing_prob`
- `loss_name`
- `focal_gamma`
- `aux_logits`

除此之外，`Auto Train` 还遵循两条实验纪律：

- 默认优先单变量实验，避免一次混入过多变化
- 当连续 `2` 轮没有晋级时，下一轮才允许双变量组合变更

## 8. 前端显示规则

### 8.1 Training Setup

- 主表单只保留少量高价值手动参数
- 低价值或高噪音参数默认下沉到 AI 搜索策略层

### 8.2 AI Search Policy

必须明确展示：

- 当前 AI 会搜索哪些参数
- `image_size` 当前是否会搜索
- 若会搜索，允许的离散范围是什么

### 8.3 Results

结果区优先展示：

- 当前选中 run 的最佳实验
- 关键指标
- run 锚点摘要
- 搜索策略摘要

## 9. 数据库存储

数据库负责结构化索引和摘要：

- `run`
- `experiment`
- `result`
- `proposal`
- `reflection`
- 产物路径索引

数据库不直接存：

- 完整日志全文
- checkpoint 二进制文件
- 大型中间产物

## 10. 本地文件

本地文件负责原始过程和大体积产物。

当前规则：

- `artifacts/runs/<run_id>.log`
  - 聚合同一个 run 下的实验创建、AI proposal、训练开始、epoch 摘要和结果摘要
- `artifacts/checkpoints/<experiment_id>.pt`
  - 按 experiment 保存 checkpoint

## 11. 数据目录

统一分类目录规则：

```text
data/
  <dataset_name>/
    raw/
    classification/
      train/
        <class_name>/
      val/
        <class_name>/
```

规则：

- trainer 读取标准化后的分类目录
- 原始下载内容保留在 `raw/`
- 类别标签由子目录名自动推断
- 非标准分类格式必须先通过准备脚本转换

## 12. 文档边界

- [README.md](/home/fish/AutoVisionLab/README.md)
  - 环境、启动、数据准备、目录结构
- [plan.md](/home/fish/AutoVisionLab/plan.md)
  - 项目定位与核心设计
- [tasks.md](/home/fish/AutoVisionLab/tasks.md)
  - 当前任务状态
- [schemas.md](/home/fish/AutoVisionLab/schemas.md)
  - 结构化对象说明
