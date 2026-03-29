# Ranking Policy

这份文档只描述 run 内实验如何按可配置评价体系进行晋级。

## 1. 基本目标

- `Ranking Policy` 决定成功实验如何比较
- 它不负责 proposal 合法性校验
- 它不负责 auto-train 停止条件
- 它只负责回答：当前实验能否晋级为新的 `best / frontier`

## 2. 当前 V1 结构

当前 `ranking_policy` 包含：

- `primary_metric`
  - 当前支持：`top1_acc`、`val_loss`
- `primary_metric_mode`
  - `top1_acc` 通常为 `max`
  - `val_loss` 通常为 `min`
- `min_primary_metric_improvement`
  - 主指标达到“明显更好”所需的最小改进量
- `primary_metric_parity_epsilon`
  - 主指标落在该灰区内时，视为“近似持平”
- `tie_breaker_metric`
  - 当前支持：`val_loss`、`top1_acc`、`training_seconds`、`latency_ms`、`parameter_count_million`
- `tie_breaker_mode`
  - 决定 tie-breaker 是越大越好还是越小越好
- `min_tie_breaker_metric_improvement`
  - tie-breaker 达到“足够更好”所需的最小改进量
- `max_image_size`
  - 当前 V1 的成本 gate
  - 候选实验若超过该上限，则不能晋级

## 3. 当前默认值

当前默认值保持与旧版口径接近：

- `primary_metric = top1_acc`
- `primary_metric_mode = max`
- `min_primary_metric_improvement = 0.01`
- `primary_metric_parity_epsilon = 0.0005`
- `tie_breaker_metric = val_loss`
- `tie_breaker_mode = min`
- `min_tie_breaker_metric_improvement = 0.01`
- `max_image_size = null`

这意味着：

- 明显更高的 `top1_acc` 会自动晋级
- 若 `top1_acc` 近似持平，则再看 `val_loss`
- 默认不额外限制 `image_size`

当前前端对新 run 的推荐默认值更偏工业场景：

- `primary_metric = top1_acc`
- `tie_breaker_metric = latency_ms`

这意味着：

- 精度仍然是第一优先级
- 当精度差距落在持平灰区内时，优先选择推理更快的模型
- 如果后续需要更保守的学术式比较，可手动切回 `val_loss`

## 4. 当前比较顺序

当前实现按以下顺序比较：

1. 先检查成本 gate
2. 再比较 `primary_metric`
3. 若主指标达到最小提升阈值，则晋级
4. 若主指标落在持平灰区，则比较 `tie_breaker_metric`
5. 若 tie-breaker 也未达到最小提升阈值，则记为 `discard`

## 5. 为什么先只做轻量成本 gate

当前系统的可搜索字段里，大部分变化主要影响训练过程，而不是部署期推理成本。

当前真正与推理成本更直接相关的字段，主要是：

- `image_size`
- 网络结构相关字段
- `latency_ms`
- `parameter_count_million`

因此，V1 先把 `max_image_size` 作为成本 gate 接入。

后续若结果中补充了更稳定的推理指标，可继续扩展：

- `latency_ms`
- `flops`
- `parameter_count_million`
- `peak_inference_memory`

## 6. 前端开放策略

当前前端只开放 V1 中用户容易理解、且当前实现真正生效的部分：

- `primary_metric`
- `min_primary_metric_improvement`
- `primary_metric_parity_epsilon`
- `tie_breaker_metric`
- `min_tie_breaker_metric_improvement`
- `max_image_size`

其余更细粒度或尚未稳定产出的成本指标，暂不在界面上暴露。
