# Schema 文档总览

这个目录只负责回答两类问题：

- 当前有哪些结构化对象文档
- 每份文档的主责任边界是什么

这里不再重复放长篇对象示例和接口返回示例；这类内容统一以 [../api.md](../api.md) 和各 schema 专项文档为准，避免和总览页重复维护。

## 1. 当前主文档

- [model_recipe_schema.md](model_recipe_schema.md)
  - `model_recipe` 的主说明
- [train_hyp_schema.md](train_hyp_schema.md)
  - `train_hyp` 的主说明
- [dataset_recipe_schema.md](dataset_recipe_schema.md)
  - `dataset_recipe` 的主说明

## 2. 阅读顺序

建议从这条顺序开始：

1. [model_recipe_schema.md](model_recipe_schema.md)
2. [train_hyp_schema.md](train_hyp_schema.md)
3. [dataset_recipe_schema.md](dataset_recipe_schema.md)
4. [../api.md](../api.md)

已删除的旧草案内容已经并回主文档，不再单独保留 schema 入口。

## 3. 与其他文档的边界

- 上层定位见 [../plan.md](../plan.md)
- 执行状态见 [../tasks.md](../tasks.md)
- 规则说明见 [../policies/README.md](../policies/README.md)
- HTTP 接口与 payload 示例见 [../api.md](../api.md)
