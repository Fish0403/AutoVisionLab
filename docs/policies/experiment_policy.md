# 实验运行总览

这份文档只保留项目当前实验系统的总边界和文档入口，不再展开所有细则。

## 1. 适用范围

- 任务类型：图像分类
- 一个 run 固定一个数据集和一个模型
- AI 只能在结构化白名单参数内搜索
- 不开放自由代码生成式模型搜索
- 允许白名单内的结构化 recipe 模块变化

当前可选模型：

- `MobileNetV3 Small`
- `GoogLeNet`
- `ResNet18`

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
- `use_demo_mode`

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
  raw/
    <dataset_name>/
  classification/
    <dataset_name>/
      train.txt
      val.txt
      test.txt
```

## 6. 细则文档

- [README.md](README.md)
  - 策略文档总览
- [run_promotion_policy.md](run_promotion_policy.md)
  - run 晋级与回退规则
- [ranking_policy.md](ranking_policy.md)
  - run 内实验的可配置评价体系
- [auto_train_search_policy.md](auto_train_search_policy.md)
  - Auto Train 搜索策略
- [auto_train_stop_policy.md](auto_train_stop_policy.md)
  - Auto Train 停止策略
- [../schemas/README.md](../schemas/README.md)
  - 结构化对象说明
