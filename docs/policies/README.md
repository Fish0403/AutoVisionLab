# 策略文档总览

这个目录集中放置 AutoVisionLab 当前与“实验规则、搜索规则、晋级规则”相关的文档。

这样做的目的不是压缩文档数量，而是把同类文档放到同一个位置，减少顶层 `docs/` 的混杂感，同时保留每份文档的独立长度与可读性。

## 1. 当前文档

- [experiment_policy.md](experiment_policy.md)
  - 实验系统总边界与入口说明
- [run_promotion_policy.md](run_promotion_policy.md)
  - run 内实验如何晋级、回退与记录决策
- [ranking_policy.md](ranking_policy.md)
  - 可配置的实验比较规则与成本 gate
- [auto_train_search_policy.md](auto_train_search_policy.md)
  - Auto Train 如何继续搜索
- [auto_train_stop_policy.md](auto_train_stop_policy.md)
  - Auto Train 何时停止

## 2. 阅读顺序

推荐从这条顺序开始：

1. [experiment_policy.md](experiment_policy.md)
2. [auto_train_search_policy.md](auto_train_search_policy.md)
3. [auto_train_stop_policy.md](auto_train_stop_policy.md)
4. [ranking_policy.md](ranking_policy.md)
5. [run_promotion_policy.md](run_promotion_policy.md)

## 3. 与其他文档的边界

- 上层定位见 [../plan.md](../plan.md)
- 执行状态见 [../tasks.md](../tasks.md)
- 结构化对象见 [../schemas.md](../schemas.md)
- HTTP 接口见 [../api.md](../api.md)
