# Rank

这份文档描述当前 run 内实验的晋级、回退和结果视图规则。

本文档只回答三件事：

1. 哪个实验会成为搜索主状态
2. `ranking_policy` 实际怎么比较
3. 其余历史信息和主搜索状态是什么关系

## Checklist

- [x] 明确 `best` 只有一个
- [x] 区分主搜索状态和分析视图
- [x] 写清前端默认值和后端兜底值不是同一套
- [x] 写清 `discard` 和 checkpoint 的关系

## 适用范围

- 当前任务类型：`classification`
- 一个 `run` 固定一个数据集和一个模型
- `AI` 只能在结构化白名单参数和受限 recipe 变化内搜索
- 不开放自由代码生成式模型搜索

## 核心结论

### `best` 只有一个

当前代码和后续策略应都按这个原则理解：

- `best_experiment_id` 是 run 内唯一的主搜索状态
- 下一轮 proposal 默认应围绕这个 `best` 继续改
- `best` 是晋级链路里的唯一 incumbent，不应存在多个并列 `best`

如果后续要做自动搜索收敛，主线应接近单分支爬山：

- 基于当前 `best` 生成下一轮 proposal
- 候选实验赢了就替换 `best`
- 候选实验没赢就记为 `discard`

### 除 `best` 外不再扩展主状态

除了 `best_experiment_id` 之外，不应再扩展新的搜索主状态。

- `baseline_experiment_id`
  - 只表示该 run 的起点实验
  - 它不是当前最优，也不是并列主状态

### 其他信息只作为历史解释

除了 `best_experiment_id` 之外，不再维护并列的 run 级 “best” 状态。

如果前端、summary 或 LLM 需要更多上下文，应直接从实验历史中提炼：

- 哪些尝试带来了最高质量
- 哪些尝试更高效
- 哪些尝试在当前阶段更稳

这些信息可以存在于总结文本、研究记忆或关键实验摘要里，但不应再落成新的 run 主状态字段。

## 配置入口

- 前端新建实验时默认提交的 `ranking_policy` 在 [frontend/web/src/features/training/config.ts](../../frontend/web/src/features/training/config.ts) 的 `buildRankingPolicy()`
- 后端 schema 的兜底默认值在 [backend/app/schemas/ranking_policy.py](../../backend/app/schemas/ranking_policy.py)
- 实际晋级比较逻辑在 [backend/app/services/run_policy.py](../../backend/app/services/run_policy.py)
- run 级别的搜索轮次、变更幅度和冷却约束在 [backend/app/schemas/run_policy.py](../../backend/app/schemas/run_policy.py)

这里需要特别注意：

- “当前默认 ranking policy”不是唯一一套值
- 正常从前端创建实验时，以前端提交值为准
- 请求没有显式带 `ranking_policy` 时，后端才会落到 schema 兜底值

## 决策状态

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
- 后端执行链路当前不会自动把实验写成 `timeout`

## 晋级规则

### 主比较逻辑

成功实验进入晋级比较时，后端按以下顺序处理：

1. 先检查 `max_image_size`
2. 再比较 `primary_metric`
3. 主指标达到最小提升阈值则晋级
4. 主指标落在持平灰区时，比较 `tie_breaker_metric`
5. tie-breaker 未达到最小提升阈值则记为 `discard`

也就是说，`image_size` 当前不是“参与排序的普通指标”，而是先执行的硬性 cost gate。

### 晋级后的主状态

- `baseline` 的首个成功实验会成为当前 `best`
- 后续成功实验只有在通过 `ranking_policy` 时，才会替换当前 `best`
- 没有通过晋级规则的成功实验会记录为 `discard`

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
  - `top1_acc`、`val_loss`、`training_seconds`、`latency_ms`、`parameter_count_million`
- `tie_breaker_mode`
  - `max` 或 `min`
- `min_tie_breaker_metric_improvement`
  - tie-breaker 需要达到的最小改进量
- `max_image_size`
  - 先执行的成本 gate；超出上限则不参与晋级

## 默认值

### 当前前端默认提交值

- `primary_metric = top1_acc`
- `primary_metric_mode = max`
- `min_primary_metric_improvement = 0.01`
- `primary_metric_parity_epsilon = 0.0005`
- `tie_breaker_metric = latency_ms`
- `tie_breaker_mode = min`
- `min_tie_breaker_metric_improvement = 0.5`
- `max_image_size = null`

### 后端 schema 兜底默认值

- `primary_metric = top1_acc`
- `primary_metric_mode = max`
- `min_primary_metric_improvement = 0.01`
- `primary_metric_parity_epsilon = 0.0005`
- `tie_breaker_metric = val_loss`
- `tie_breaker_mode = min`
- `min_tie_breaker_metric_improvement = 0.01`
- `max_image_size = null`

### 默认值解读

- 这两套默认值不一致，尤其 tie-breaker 的口径不同
- 因此文档中的“默认值”必须明确说明是“前端默认提交值”还是“后端兜底值”
- 正常从前端创建 experiment 时，应以提交到后端的那套 `ranking_policy` 为准

## `discard` 的真实语义

当前 `discard` 需要分开理解：

### 成功但没晋级

- 实验训练成功
- 但没有通过 `ranking_policy`
- 这种情况下会写成 `decision = discard`
- checkpoint 不会因为“没晋级”而自动清掉

### 被中断或显式丢弃

- 用户停止训练，或系统主动 discard 实验
- 状态会变成 `discarded`
- 同时会写 `decision = discard`
- 这种情况下会清理该实验的 checkpoint 和 result

所以当前 `decision = discard` 实际上混合了两类不同语义：

- 成功但没赢
- 被中断或丢弃

这点在阅读 run 历史或给 LLM 提供上下文时需要特别注意。

## 产物保留

- `discard_experiment()` 会删除该实验已落盘的 checkpoint 以及对应结果记录
- 通过晋级规则判定为 `discard` 的成功实验，不会因为“没晋级”而自动删除 checkpoint
- 因此当轮数很多时，成功实验的 `pt` 文件通常会继续累积
- 如果要只保留部分 checkpoint，需要单独设计 artifact retention policy，而不是修改 ranking policy 本身

## 对 Search 的建议解释

如果后续要把 search 逻辑收敛得更稳定，推荐按下面方式理解 ranking：

- `best_experiment_id` 是唯一搜索主状态
- proposal LLM 可以看到：
  - 当前 `best`
  - 最近实验历史
  - 从历史中归纳出的有效 / 无效尝试总结
- 但默认 follow-up source 应围绕唯一 `best` 继续推进，不再在多个 run 主状态间摇摆

## 一句话总结

当前 ranking policy 的核心不是“系统里有很多个 best”，而是：

- 搜索主状态只有一个 `best`
- `baseline` 只表示起点
- 其余信息只作为历史解释
- 晋级由显式 `ranking_policy` 决定
- `discard` 和产物保留需要与“是否成功”“是否显式丢弃”区分理解
