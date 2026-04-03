# Rank

这份文档说明当前 run 内实验的晋级规则，以及 `ranking_policy` 在实际搜索中的作用。

## 作用范围

- 当前任务类型：`classification`
- 一个 `run` 固定一个数据集和一个模型
- ranking 只决定成功实验是否晋级为当前主搜索状态

## 主状态

- `best_experiment_id`
  - run 内唯一的主搜索状态
  - 下一轮 follow-up 默认围绕它继续推进
- `baseline_experiment_id`
  - run 的起点实验
  - 不表示当前最优，也不是并列主状态

## 执行状态与研究决策

### 执行状态

- `queued`
- `running`
- `success`
- `failed`
- `discarded`

### 研究决策

- `keep`
- `discard`
- `crash`
- `timeout`

说明：

- `timeout` 仍保留在 schema 中
- 当前后端执行链路不会自动写入 `timeout`

## 晋级规则

成功实验进入晋级比较时，后端按以下顺序处理：

1. 先检查 `max_image_size`
2. 再比较 `primary_metric`
3. 主指标达到最小提升阈值则晋级
4. 主指标落在持平灰区时，再比较 `tie_breaker_metric`
5. tie-breaker 未达到最小提升阈值则记为 `discard`

也就是说，`image_size` 当前不是普通排序指标，而是先执行的成本 gate。

## ranking_policy 字段

- `primary_metric`
  - `top1_acc` 或 `val_loss`
- `primary_metric_mode`
  - `max` 或 `min`
- `min_primary_metric_improvement`
  - 主指标需要达到的最小改进量
- `primary_metric_parity_epsilon`
  - 主指标视为持平的灰区阈值
- `tie_breaker_metric`
  - `top1_acc`、`val_loss`、`training_seconds`、`latency_ms`、`parameter_count_million`
- `tie_breaker_mode`
  - `max` 或 `min`
- `min_tie_breaker_metric_improvement`
  - tie-breaker 需要达到的最小改进量
- `max_image_size`
  - 先执行的成本 gate；超出上限则不参与晋级

## `discard` 的语义

当前 `decision = discard` 需要分开理解：

### 成功但未晋级

- 实验训练成功
- 但没有通过 `ranking_policy`
- 会写成 `decision = discard`
- checkpoint 不会因为“没晋级”而自动清掉

### 被中断或显式丢弃

- 用户停止训练，或系统主动 discard 实验
- 状态会变成 `discarded`
- 同时会写 `decision = discard`
- 这种情况下会清理该实验的 checkpoint 和 result

## 产物保留

- `discard_experiment()` 会删除该实验已落盘的 checkpoint 和对应结果记录
- 成功但未晋级的实验，不会因为 `decision = discard` 自动删除 checkpoint
- 如果要只保留部分 checkpoint，需要单独设计 retention policy，而不是修改 ranking 规则本身

## 实现位置

- [backend/app/services/run_policy.py](../../backend/app/services/run_policy.py)
- [backend/app/schemas/ranking_policy.py](../../backend/app/schemas/ranking_policy.py)
- [frontend/web/src/features/training/config.ts](../../frontend/web/src/features/training/config.ts)
