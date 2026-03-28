# 实验运行总览

这份文档只保留项目当前实验系统的总边界和文档入口，不再展开所有细则。

## 1. 适用范围

- 任务类型：图像分类
- 一个 run 固定一个数据集和一个模型
- AI 只能在结构化白名单参数内搜索
- 不开放模型结构搜索

当前可选模型：

- `MobileNetV2`
- `GoogLeNet`
- `ResNet18`
- `ResNet34`
- `DenseNet121`

## 2. Run 基本边界

- 一个 `run` 内只允许同一数据集、同一模型的实验
- 跨模型比较应通过多个 run 完成
- run 当前维护：
  - `baseline_experiment_id`
  - `best_experiment_id`
  - `frontier_experiment_id`

## 3. 主表单口径

前端 `Training Setup` 当前直接展示：

- `run_name`
- `dataset`
- `model_name`
- `learning_rate`
- `batch_size`
- `epochs`
- `image_size`
- `max_wall_clock_minutes`

## 4. 数据与产物边界

数据库负责结构化索引和摘要：

- `run`
- `experiment`
- `result`
- `proposal`
- `reflection`
- 产物路径索引

本地文件负责大体积产物：

- `artifacts/runs/<run_id>.log`
- `artifacts/checkpoints/<experiment_id>.pt`

## 5. 数据目录规则

统一分类目录规则：

```text
data/
  <dataset_name>/
    raw/
    classification/
      train/
        <class_name>/
      val/
        <class_name>/
```

## 6. 细则文档

- [docs/run_promotion_policy.md](run_promotion_policy.md)
  - run 晋级与回退规则
- [docs/ranking_policy.md](ranking_policy.md)
  - run 内实验的可配置评价体系
- [docs/auto_train_search_policy.md](auto_train_search_policy.md)
  - Auto Train 搜索策略
- [docs/auto_train_stop_policy.md](auto_train_stop_policy.md)
  - Auto Train 停止策略
- [docs/schemas.md](schemas.md)
  - 结构化对象说明
