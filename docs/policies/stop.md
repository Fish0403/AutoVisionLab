# Stop

这份文档描述 `Auto Train` 的停止规则。

## 基本规则

- 用户可以随时点 `Stop`
- 未人工停止时，系统会持续追加实验
- 当搜索空间已完成最低限度探索且长期没有新的 `keep` 时，系统会提前停止，并记为 `stopped_by_policy`
- 单轮训练不会因为策略停止而被立刻强制打断
- 人工 `Stop` 会对当前实验发送中断请求

## 提前停止条件

- `available_dimensions` 已全部被探索
- 每个已开放维度至少成功执行 `2` 次
- 连续 `6` 轮没有产生新的 `keep`

## 维度口径

- `available_dimensions`
  - 来自当前 run 的 `search_policy`
- `explored_dimensions`
  - 当前 run 中实际成功执行过的 proposal 所覆盖的维度

## 停机权

- 停机权由系统规则和人工 `Stop` 控制
- `AI` 不直接决定停机
