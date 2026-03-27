# Run 晋级与回退规则

这份文档只描述 run 内实验结果如何晋级、回退和记录决策。

## 1. 基本对象

一个 `run` 围绕同一数据集和同一模型持续追加实验。

当前 run 显式维护三个锚点：

- `baseline_experiment_id`
- `best_experiment_id`
- `frontier_experiment_id`

## 2. 决策状态

实验完成后，执行状态和研究决策分开记录。

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

## 3. 默认晋级规则

- `baseline` 的首个成功实验会先成为当前 `best` 与 `frontier`
- 后续成功实验不会因为“略好一点”就自动晋级

当前默认晋级阈值：

- `top1_acc` 至少提升 `0.01`
- 或在 `top1_acc` 近似持平时，`val_loss` 至少下降 `0.01`

## 4. 晋级后的处理

若实验达到晋级阈值：

- 当前实验记为 `keep`
- `best_experiment_id` 更新到该实验
- `frontier_experiment_id` 更新到该实验

## 5. 未晋级时的回退规则

若实验未达到晋级阈值：

- 当前实验记为 `discard`
- `best_experiment_id` 保持不变
- `frontier_experiment_id` 保持不变
- 后续 auto-train 默认回到当前 `best / frontier` 继续分支

## 6. 失败与中断

- 训练失败但未产出有效结果时，决策记为 `crash`
- 被用户停止或被系统中断后丢弃结果时，执行状态可记为 `discarded`
- 这些实验不会晋级到 `best` 或 `frontier`

## 7. 一句话总结

当前 run policy 的核心不是“谁排第一谁晋级”，而是“只有跨过明确阈值的实验才晋级，否则记录为 `discard` 并回到当前最佳分支继续搜索”。
