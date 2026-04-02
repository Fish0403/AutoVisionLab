# AutoVisionLab Overview

## 项目定位

AutoVisionLab 是一个面向工业视觉实验的结构化平台。当前实现聚焦图像分类，但对象模型、策略层和产物管理都按实验平台的方式组织。

## 边界

- 任务类型：`classification`
- 一个 `run` 固定一个数据集和一个模型
- `AI` 只能在结构化白名单参数和受限 recipe 变化内搜索
- `proposal`、`result`、`reflection`、`ranking_policy`、`search_policy` 都以结构化对象保存
- 数据库保存元数据和摘要，本地磁盘保存日志和 checkpoint

## 核心对象

- `run`
  - 维护当前唯一主搜索状态 `best_experiment_id`
  - `baseline` 只作为起点参考保留在实验记录里
- `experiment`
  - 保存配置、参数空间快照、proposal、result、reflection 和决策状态
- `task`
  - 承载 `Auto Train`、`Compare Models` 和实验建议等后台流程
- `proposal`
  - 保存 AI 生成的结构化变更
- `result`
  - 保存指标、资源信息、参数快照和产物路径

## 策略入口

- [docs/policies/rank.md](policies/rank.md)
- [docs/policies/search.md](policies/search.md)

## 目录

```text
data/
  raw/
    <dataset_name>/
  classification/
    <dataset_name>/
      train.txt
      val.txt
      test.txt
artifacts/
  runs/
```

## 相关文档

- [README.md](../README.md)
- [docs/api.md](api.md)
- [docs/schemas/model_recipe.md](schemas/model_recipe.md)
- [docs/schemas/train_hyp.md](schemas/train_hyp.md)
- [docs/schemas/dataset_recipe.md](schemas/dataset_recipe.md)
