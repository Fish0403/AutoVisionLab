# AutoVisionLab Overview

## 项目定位

AutoVisionLab 是一个面向工业视觉实验的结构化工作台。当前实现聚焦图像分类，核心目标是把 baseline、对比、搜索、结果总结和历史追踪放进同一套实验闭环里。

## 当前边界

- 当前任务类型：`classification`
- 一个 `run` 固定一个数据集和一个模型
- `AI` 只能在结构化白名单字段内搜索，不生成自由代码
- `proposal`、`result`、`reflection`、`ranking_policy`、`search_policy` 都以结构化对象保存
- 数据库存元数据和任务快照，本地磁盘存日志、配置快照和 checkpoint

## 核心对象

- `run`
  - 一组同数据集、同模型的连续实验
  - 维护当前主搜索状态 `best_experiment_id`
- `experiment`
  - 一次具体训练尝试
  - 保存配置、参数空间快照、proposal、result、reflection 和决策状态
- `task`
  - 工作台里的后台流程
  - 当前包括 `Auto Train`、`Compare Models` 和实验建议任务
- `proposal`
  - AI 生成的结构化变更建议
- `result`
  - 训练结果快照，包含指标、资源信息、参数快照和产物路径

## 当前配置口径

- `ExperimentConfig` 以 `model_recipe`、`train_hyp`、`dataset_recipe` 为主配置结构
- `search_policy` 决定 AI 可搜索的字段范围和约束
- `ranking_policy` 决定实验排序与晋级指标
- `params` 用作归一化参数快照，便于结果展示、历史摘要和实验对比

## 策略入口

- [docs/policies/rank.md](policies/rank.md)
- [docs/policies/search.md](policies/search.md)

## 相关文档

- [README.md](../README.md)
- [docs/artifacts.md](artifacts.md)
- [docs/api.md](api.md)
- [docs/llm.md](llm.md)
- [docs/schemas/model_recipe.md](schemas/model_recipe.md)
- [docs/schemas/train_hyp.md](schemas/train_hyp.md)
- [docs/schemas/dataset_recipe.md](schemas/dataset_recipe.md)
