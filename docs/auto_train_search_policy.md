# Auto Train 搜索策略

这份文档只描述 `Auto Train` 如何继续搜索，不讨论停止条件。

停止条件见 [docs/auto_train_stop_policy.md](/home/fish/AutoVisionLab/docs/auto_train_stop_policy.md)。

## 1. 基本原则

- `Auto Train` 在同一个 run 内继续追加实验
- 一个 run 固定一个数据集和一个模型
- AI 只能在当前 run 的 `search_policy` 和 parameter space 白名单内搜索
- 默认优先单变量实验，避免一次混入过多变化

## 2. 搜索阶段

当前实现按已消耗时间比例切换搜索阶段。

- 前 `50%` 时间预算
  - 允许基础超参数主导
- 后 `50%` 时间预算
  - proposal 不能只修改基础超参数
  - 必须至少包含一个 augmentation / loss / strategy 字段变化

## 3. 搜索维度

当前维度分为：

- `basic`
  - `optimizer`
  - `learning_rate`
  - `batch_size`
  - `weight_decay`
  - `scheduler`
  - `label_smoothing`
  - `image_size`（仅在满足当前特殊规则时）
- `augmentation`
  - `augmentation_policy`
  - `mixup_alpha`
  - `cutmix_alpha`
  - `random_erasing_prob`
- `loss`
  - `loss_name`
  - `focal_gamma`
- `strategy`
  - `aux_logits`

## 4. Proposal 约束

- proposal 必须是结构化 JSON
- proposal 必须通过 schema 校验
- proposal 必须通过 parameter space 校验
- proposal 必须通过 `search_policy` 校验
- proposal 不能为空，必须包含至少一个有效参数变化
- proposal 默认只允许修改 `1` 个字段
- 只有当 run 已连续至少 `2` 轮没有晋级时，proposal 才最多允许修改 `2` 个字段

## 5. 去重与切维

- 同一个字段连续至少 `2` 轮参与失败实验后，会进入临时冷却
- 处于冷却中的字段，在接下来至少 `2` 轮 proposal 中不允许再次提出
- 如果最近至少 `3` 轮停滞都停留在同一个搜索维度，下一轮必须切换维度

## 6. 一句话总结

当前 `Auto Train` 的搜索策略是：前半段先用基础超参数找稳定区间，后半段必须切到 augmentation / loss / strategy，且整个过程中持续受字段预算、字段冷却和切维规则约束。
