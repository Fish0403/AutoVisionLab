# Component Search Schema

这份文档定义 AutoVisionLab 从“细粒度 layer 搜索”转向“组件级架构搜索”的第一版口径。

目标不是开放自由 NAS，而是把模型结构搜索收敛成更稳定、更容易校验的三个槽位：

- `backbone`
- `neck`
- `head`

## 1. 为什么从 layer 搜索改成 component 搜索

当前平台已经有统一的 `parser / validator / builder / factory` 入口，但如果继续把搜索粒度放到单层、单卷积甚至 `from` 连接级别，会带来几个问题：

- proposal 很容易生成无效结构
- validator 会快速膨胀成图结构检查器
- builder 很难保证每次都能稳定构图
- 同一个 run 内的实验会越来越难解释和复现

对于当前分类任务来说，更合理的粒度是组件替换：

- 换一个 `backbone`
- 换一个 `neck`
- 换一个 `head`

这样做的收益是：

- 搜索空间更可控
- 失败率更低
- 每次实验更容易解释
- 更适合当前“结构化实验平台”定位

## 2. 当前边界

这一版仍然坚持以下边界：

- 不开放自由代码生成式模型搜索
- 不允许 AI 自由增删层或改连接关系
- trainer 只消费结构化 config
- proposal 只能改白名单字段

换句话说，系统允许“替换组件”，不允许“自由改图”。

## 3. 组件槽位定义

### 3.1 Backbone

`backbone` 表示主干特征提取器。

候选通常来自：

- `torchvision.models.resnet18`
- `torchvision.models.mobilenet_v3_small`
- `torchvision.models.efficientnet_b0`
- `torchvision.models.convnext_tiny`

但在当前实现阶段，还不直接开放跨家族自由互换。原因是：

- 现在一个 run 仍然固定一个 `model_name`
- 不同 backbone 的输出接口还没有完全统一成共享特征适配层

因此第一版只先把 `backbone` 槽位写进 schema，作为后续扩展位。

### 3.2 Neck

`neck` 表示连接 backbone 和 head 的中间适配层。

对当前分类任务，`neck` 不一定像检测模型那样复杂，第一版更偏向轻量适配件，例如：

- `identity`
- `avg_pool`
- `gem_pool`
- `conv1x1_bn_act`

当前最先开放的就是这一槽位，因为它最容易稳定落地，也最接近当前已有的 `pooling_type` 能力。

### 3.3 Head

`head` 表示分类头。

对当前分类任务，第一版 head 候选保持有限白名单：

- `native_classifier`
- `linear`
- `dropout_linear`

说明：

- `native_classifier`
  - 保持模型原生风格的分类头
- `linear`
  - 全局池化后直接线性分类
- `dropout_linear`
  - 全局池化后 `Dropout(0.2) + Linear`

## 4. 第一版实施口径

当前推荐的分阶段口径如下：

### 阶段 1

- `backbone` 槽位进入 schema，但默认固定为 native
- `neck` 开始成为真正可搜索的组件位
- `head` 允许少量稳定候选进入白名单

对应当前分类实现：

- `mobilenet_v3_small`
  - `backbone`: `mobilenet_v3_small_native`
  - `neck`: `avg_pool | gem_pool`
  - `head`: `native_classifier | linear | dropout_linear`
- `googlenet`
  - `backbone`: `googlenet_native`
  - `neck`: `avg_pool`
  - `head`: `native_classifier`
- `resnet18`
  - `backbone`: `resnet18_native`
  - `neck`: `avg_pool`
  - `head`: `native_classifier`

### 阶段 2

- 引入共享 backbone feature adapter
- 允许少量跨 family backbone 候选进入同一个“组合式分类器”入口
- 开始开放更通用的 `head` 候选

### 阶段 3

- 再考虑更丰富的 neck/head registry
- 再考虑是否需要独立的 `component_classifier` 逻辑模型名

## 5. Schema 建议

`model_recipe` 在保留现有 YOLO 风格 `backbone / neck / head` layer list 的同时，增加一个更高层的组件视图：

```yaml
components:
  backbone:
    name: mobilenet_v3_small_native
    params: {}
  neck:
    name: gem_pool
    params: {}
  head:
    name: native_classifier
    params: {}
```

说明：

- layer list 仍然可以保留，用于 builder 的受限表达
- `components` 则给 AI 和策略层一个更粗粒度、更稳定的搜索视图
- 对于当前 `v1`，优先让 AI 改 `components`，而不是直接改 layer

## 6. Proposal 字段建议

为了兼容当前 proposal 仍以扁平字段返回，组件级搜索建议逐步切到：

- `backbone_name`
- `neck_name`
- `head_name`

第一版实际放开的重点是：

- `neck_name`
- `head_name`

旧字段例如：

- `pooling_type`
- `classifier_dropout`
- `width_multiple`

可以短期兼容，但方向上不再继续扩展成更细粒度结构搜索。

## 7. Builder 约束

Builder 层建议遵循以下规则：

- `backbone` 必须来自注册表
- `neck` 必须来自注册表
- `head` 必须来自注册表
- 不允许未知组件名
- 不允许组件之间的输出维度不匹配
- 在真正训练前至少做一次 `dry build`，必要时补 `dummy forward`

## 8. 一句话总结

新的方向不是让 AI 去改某一层卷积，而是让 AI 在白名单内替换 `backbone / neck / head` 这三个组件槽位。

当前已经落地的 `v1` 口径是：

- `mobilenet_v3_small`
  - 冻结 native backbone
  - 开放 `neck_name = avg_pool | gem_pool`
  - 开放 `head_name = native_classifier | linear | dropout_linear`
- `googlenet / resnet18 / mobilenet_v2`
  - 当前仍保持 native backbone + native head，不开放组件搜索

后续如果要开放真正的 `backbone` 搜索，需要先引入共享 feature adapter，或新的跨模型组合入口。
