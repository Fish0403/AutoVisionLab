# MobileNet 组件级架构搜索 v1

这份文档记录当前已经确认下来的 `MobileNetV3 Small` 组件级架构搜索实施方案。

它不是泛化到所有模型 family 的最终设计，而是当前阶段的落地口径。

## 1. 当前目标

在不改变现有 run 语义的前提下，把 `MobileNetV3 Small` 的架构搜索从旧的细粒度字段收口到组件级：

- `backbone`
- `neck`
- `head`

其中第一版真正开放搜索的只有：

- `neck`
- `head`

## 2. 边界

- 一个 run 仍然固定 `model_name = mobilenet_v3_small`
- 不在同一个 run 内跨模型 family 切换
- `backbone` 第一版冻结为 native，不开放搜索
- 不开放自由 layer / graph 改写
- trainer 只消费结构化 `model_recipe`

## 3. 第一版组件槽位

### 3.1 Backbone

固定为：

- `mobilenet_v3_small_native`

它保留在 schema 中，但不是第一版的搜索对象。

### 3.2 Neck

开放候选：

- `avg_pool`
- `gem_pool`

对应分类模型里最直接、最稳的全局聚合差异。

### 3.3 Head

开放候选：

- `native_classifier`
- `linear`
- `dropout_linear`

说明：

- `native_classifier`
  - 保持当前 MobileNet 风格的分类头
- `linear`
  - 全局池化后直接线性分类
- `dropout_linear`
  - 全局池化后 `Dropout(0.2) + Linear`

当前 `dropout_linear` 的 dropout 固定为 `0.2`，不额外暴露为搜索字段。

## 4. 搜索字段

第一版组件搜索字段收敛为：

- `neck_name`
- `head_name`

`backbone_name` 保留在 schema 中，但第一版不进入 `parameter_space` 白名单。

## 5. 旧字段处理

以下旧字段不再作为主搜索入口：

- `width_multiple`
- `pooling_type`
- `classifier_dropout`

当前处理原则：

- 保留 schema 兼容
- 保留旧实验回放能力
- 不再作为 AI proposal 的主字段

也就是说，系统可以继续读懂它们，但新的架构搜索应优先走组件字段。

## 6. Builder 口径

`MobileNetV3 Small` builder 改为 `components-first`：

- `components` 是主决策来源
- 现有 layer list 继续保留，用于 native 构建和兼容回放
- 当 `components` 和旧字段同时存在时，以 `components` 为准

第一版不删除 layer list，因为当前 native backbone 仍然依赖它稳定构建。

## 7. 验证重点

至少覆盖以下验证：

- 默认 recipe 自动补出 `components`
- `neck_name=avg_pool|gem_pool` 能正确构建并 forward
- `head_name=native_classifier|linear|dropout_linear` 能正确构建并 forward
- 旧字段 proposal 仍能映射到兼容的 `recipe_changes`
- 新的 `parameter_space` 和前端展示只强调组件字段

## 8. 一句话总结

当前阶段不是继续扩展细粒度 recipe 搜索，而是把 `MobileNetV3 Small` 架构搜索收敛成：

- 冻结 native backbone
- 搜索 neck
- 搜索 head

后续如果要做真正的 backbone 搜索，需要先引入新的组合模型入口或新的 run 语义。
