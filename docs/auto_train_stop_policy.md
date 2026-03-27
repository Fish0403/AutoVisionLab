# Auto Train 停止策略

当前实现采用时间预算优先的停止方式。

## 1. 基本规则

- 用户设置 `max_wall_clock_minutes`
- 系统按总时间预算持续追加实验
- 当前实现会在每一轮开始前检查累计已耗时
- 若预算已耗尽，则不再创建下一轮，并记为 `stopped_by_budget`
- 若搜索空间已被最低限度探索且长期没有新的 `keep`，则提前停止并记为 `stopped_by_policy`

补充说明：

- 当前已启动的单轮训练不会因为预算耗尽而立刻写成 `timeout`
- 现有实现是在当前轮结束后停止后续轮次，而不是在训练中途强制预算打断

## 2. 提前停止条件

当前 `stopped_by_policy` 的判断为：

- `available_dimensions` 已全部被探索
- 每个已开放维度至少成功执行 `2` 次
- 连续 `6` 轮没有产生新的 `keep`

## 3. 维度口径

- `available_dimensions`
  - 来自当前 run 的 `search_policy`
- `explored_dimensions`
  - 当前 run 中实际成功执行过的 proposal 所覆盖的维度

维度只统计当前 run 允许 AI 搜索的部分，不要求固定四类维度全部开放。

## 4. AI 的作用

AI 可以给出停止建议，但不直接拥有停机权。

当前停止权仍由系统规则和时间预算控制。
