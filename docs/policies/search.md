# Search

这份文档描述 `Auto Train` 的搜索规则。

## 基本规则

- `Auto Train` 在同一个 `run` 内继续追加实验
- 一个 `run` 固定一个数据集和一个模型
- `AI` 只能在当前 `search_policy` 和 parameter space 白名单内搜索
- proposal 优先保持单变量变化

## 搜索阶段

- 前 `3` 轮允许基础超参数主导
- 第 `4` 轮起优先考虑 augmentation、loss 和 model module 变化
- 如果当前可行动作里没有足够的非基础字段，系统仍然保持 proposal 可执行

## 搜索维度

- `basic`
  - `optimizer`
  - `learning_rate`
  - `batch_size`
  - `weight_decay`
  - `scheduler`
  - `label_smoothing`
  - `image_size`
- `augmentation`
  - `augmentation_policy`
  - `mixup_alpha`
  - `cutmix_alpha`
  - `random_erasing_prob`
- `loss`
  - `loss_name`
  - `focal_gamma`
- `model_module`
  - `aux_logits`
  - `neck_name`
  - `head_name`

说明：

- `allow_strategy_search` 目前存在，但没有对应的可搜索字段
- `model_module` 是否生效取决于当前 run 的 `search_policy` 和模型的 editable parameter space
- `GoogLeNet` 主要使用 `aux_logits`
- `MobileNetV3 Small` 主要使用 `neck_name` 和 `head_name`

## Proposal 约束

- proposal 必须是结构化 JSON
- proposal 必须通过 schema 校验
- proposal 必须通过 parameter space 校验
- proposal 必须通过 `search_policy` 校验
- proposal 不能为空
- 默认只允许修改 `1` 个字段
- 连续停滞后最多允许修改 `2` 个字段

## 去重与切维

- 同一个字段连续失败后进入冷却
- 冷却中的字段在存在其他可行动作时会降为低优先级
- 如果最近连续停滞都落在同一个维度，下一轮优先切维
- 无效 proposal 会被记录并继续之后轮次
