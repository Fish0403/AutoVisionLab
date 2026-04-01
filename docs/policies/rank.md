# Rank

这份文档描述 run 内实验的晋级、回退和比较规则。

## 适用范围

- 当前任务类型：`classification`
- 一个 `run` 固定一个数据集和一个模型
- `AI` 只能在结构化白名单参数和受限 recipe 变化内搜索
- 不开放自由代码生成式模型搜索

## Run 锚点

- `baseline_experiment_id`
- `best_experiment_id`
- `frontier_experiment_id`
- `best_quality_experiment_id`
- `best_efficiency_experiment_id`
- `best_tradeoff_experiment_id`

## 决策状态

- 执行状态：
  - `queued`
  - `running`
  - `success`
  - `failed`
  - `discarded`
- 研究决策：
  - `keep`
  - `discard`
  - `crash`
  - `timeout`

说明：

- `timeout` 仍保留在 schema 中
- 后端执行链路当前不会自动把实验写成 `timeout`

## 晋级规则

- `baseline` 的首个成功实验会成为当前 `best` 与 `frontier`
- 成功实验先经过 `ranking_policy` 比较
- 只有跨过主指标改进阈值，或者在主指标近似持平时赢下 tie-breaker，实验才会晋级
- 未晋级的成功实验会记录为 `discard`

## 比较字段

- `primary_metric`
  - `top1_acc` 或 `val_loss`
- `primary_metric_mode`
  - `max` 或 `min`
- `min_primary_metric_improvement`
  - 主指标需要达到的最小改进量
- `primary_metric_parity_epsilon`
  - 主指标视为持平的灰区阈值
- `tie_breaker_metric`
  - `val_loss`、`top1_acc`、`training_seconds`、`latency_ms`、`parameter_count_million`
- `tie_breaker_mode`
  - `max` 或 `min`
- `min_tie_breaker_metric_improvement`
  - tie-breaker 需要达到的最小改进量
- `max_image_size`
  - 成本 gate，上限之外的实验不参与晋级

## 默认值

- `primary_metric = top1_acc`
- `primary_metric_mode = max`
- `min_primary_metric_improvement = 0.01`
- `primary_metric_parity_epsilon = 0.0005`
- `tie_breaker_metric = val_loss`
- `tie_breaker_mode = min`
- `min_tie_breaker_metric_improvement = 0.01`
- `max_image_size = null`

## 比较顺序

1. 先检查 `max_image_size`
2. 再比较 `primary_metric`
3. 主指标达到最小提升阈值时晋级
4. 主指标落在持平灰区时比较 `tie_breaker_metric`
5. tie-breaker 未达到最小提升阈值时记为 `discard`

## 失败与中断

- 训练失败但未产出有效结果时，决策记为 `crash`
- 被用户停止或被系统丢弃的实验，执行状态记为 `discarded`
- 这些实验不会晋级到 `best` 或 `frontier`

## 结果视图

- `best_quality_experiment_id`
  - 只按主指标选择的最佳实验
- `best_efficiency_experiment_id`
  - 按 `latency_ms`、`parameter_count_million`、`training_seconds` 和 `top1_acc` 综合排序得到的效率最佳实验
- `best_tradeoff_experiment_id`
  - 在主指标持平灰区内，优先挑选更高效的实验
