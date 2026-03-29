# Model Recipe Schema

这份文档定义 AutoVisionLab 的通用 `model_recipe` 结构。

它的目标不是为某一个具体模型单独造一套私有 schema，而是提供一套可复用的、面向多模型家族的结构化模型架构表达。后续不同 family 只是在这套通用结构之上附加自己的模块注册表、校验规则和搜索白名单。

## 1. 目标

`model_recipe` 要解决的问题是：

- 让模型结构以结构化对象表达，而不是隐含在 trainer 代码里
- 让 AI 看到的不只是超参数，还包括模型架构本身
- 让架构修改可以被校验、落盘、回放和搜索
- 让分类、检测、分割后续尽量复用同一套顶层 schema

它明确不做的事情是：

- 自由代码生成
- 无约束 NAS
- 任意 Python 模块注入
- 没有 registry 和 validator 的随意拓扑改写

## 2. 设计原则

- 顶层组织显式对齐 YOLOv5 的 recipe 约定：全局字段 + `backbone / head` + layer list
- 通用 schema 只描述“网络怎么组织”，不把 family-specific 约束写死到总 schema 里
- family-specific 变化放到 registry、builder、validator 和 search whitelist 里处理
- 同一个 recipe 应能稳定序列化为 YAML / JSON，并回写到 API、数据库和 artifact
- AI 只能修改白名单中的结构字段，不能脱离 schema 自由发挥

## 3. 参考基线

这一版 schema 明确参考 YOLOv5 的 `yolov5s.yaml` 组织方式：

- 顶层保留少量全局字段，例如 `nc`、`depth_multiple`、`width_multiple`
- 主体结构拆成 `backbone` 和 `head`
- 每一层统一使用 `[from, number, module, args]`

参考文件：

- https://github.com/ultralytics/yolov5/blob/master/models/yolov5s.yaml

这里的重点不是复制 YOLO 的检测模块，而是复用它那种“结构表达和模块实现分离”的组织方式。

## 4. 通用顶层结构

建议的通用 `model_recipe.yaml` 顶层如下：

```yaml
version: model_recipe@v1
task_type: classification
model_family: mobilenet
base_model: mobilenet_v3_small
nc: 6
depth_multiple: 1.0
width_multiple: 1.0
backbone:
  - [-1, 1, StemConv, [16, 3, 2, HS]]
  - [-1, 1, IR, [16, 3, 16, 16, true, RE, 2, 1]]
  - [-1, 1, IR, [16, 3, 72, 24, false, RE, 2, 1]]
head:
  - [-1, 1, PointwiseConv, [576, HS]]
  - [-1, 1, GlobalPool, [avg]]
  - [-1, 1, Classifier, [1024, 0.2, linear]]
modules: {}
metadata:
  notes: null
```

说明：

- `backbone / head` 统一采用 layer list 表达
- `neck` 不是顶层必填字段；分类模型通常没有 neck
- 每个 layer 只负责描述结构，不直接等同于某个 Python 类实现
- `modules` 用于承载不适合展开为 block 的结构化插件信息

## 5. 通用字段说明

### 5.1 顶层通用字段

- `version`
  - 当前固定为 `model_recipe@v1`
- `task_type`
  - 例如 `classification | detection | segmentation`
- `model_family`
  - 例如 `mobilenet | resnet | inception | yolov5`
- `base_model`
  - 当前 recipe 所依附的基线模型名
- `nc`
  - 类别数
- `depth_multiple`
  - 控制重复层数或结构深度缩放
- `width_multiple`
  - 控制通道宽度缩放

可选补充字段例如 `input_channels`、`variant`、`metadata` 可以保留，但不应破坏主结构的统一性。

### 5.2 结构段

- `backbone`
  - 主干特征提取部分
- `head`
  - 任务头部；分类、检测、分割各自 family 的实现不同

如果某个 family 确实需要 neck，可以作为可选顶层字段增加，但 `v1` 的通用约定先以 `backbone + head` 为主，尽量靠拢 YOLOv5 的外观。

这些字段统一使用 layer list，而不是一开始就拆成每个模型私有字段。这样后续从分类扩到检测/分割时，顶层 schema 不需要再翻一次。

### 5.3 `modules`

`modules` 用于承载下面这类内容：

- 开关型模块
- 插件注入位点
- 不适合直接建成 block 列表的轻量结构配置

例如：

```yaml
modules:
  aux_logits: false
  attention:
    type: se
    positions: [stage4, stage5]
```

### 5.4 `metadata`

`metadata` 只放人类可读或辅助追踪信息，例如：

- `notes`
- `source_template`
- `builder_version`

不要把真正影响模型结构语义的字段塞到 `metadata` 里。

## 6. Layer 表示法

核心约定直接对齐 YOLOv5：

- `[from, number, module, args]`

建议在 YAML 中优先使用短数组形式：

```yaml
- [-1, 1, Conv, [32, 3, 2]]
- [[-1, 4], 1, Concat, [1]]
```

字段语义：

- `from`
  - 输入来源，兼容单输入和多输入
- `number`
  - 模块重复次数
- `module`
  - 模块类型名，必须能在 family registry 中解析
- `args`
  - 结构参数，含通道、kernel、stride、激活等

说明：

- `from` / `number` / `module` / `args` 的组织方式是为了兼容 YOLOv5 类配置习惯
- 这不意味着所有 family 都要支持自由拓扑改写
- 是否允许修改某个 layer，取决于 family-specific validator 和 whitelist

如果内部运行时对象更适合使用具名字段，例如：

```yaml
- from: -1
  number: 1
  module: Conv
  args: [32, 3, 2]
```

也应保证和短数组形式可一一对应，并能稳定互转。

## 7. Module Registry 与校验分层

通用 schema 只负责表达，真正决定“什么是合法 recipe”的，是 family-specific 约束层。

建议后续拆成四类职责：

### 7.1 Module Registry

按 `model_family` 维护可用模块注册表。这里的重点是：YAML 里的 `module` 只是名字，必须由后端 registry 映射到真实实现。

例如：

- `mobilenet`
  - `StemConv`
  - `IR`
  - `PointwiseConv`
  - `GlobalPool`
  - `Classifier`
- `resnet`
  - `StemConv`
  - `BasicBlock`
  - `Bottleneck`
  - `GlobalPool`
  - `Classifier`
- `inception`
  - `StemConv`
  - `Inception`
  - `AuxLogits`
  - `GlobalPool`
  - `Classifier`

如果后续支持检测 family，再在对应 registry 中加入：

- `Conv`
- `C3`
- `SPPF`
- `Concat`
- `Detect`

### 7.2 Parser

需要一个统一 parser，负责把：

- `backbone`
- `head`
- 每层的 `[from, number, module, args]`

解析成内部运行时 layer 对象。

### 7.3 Builder

不同 family 分别实现：

- `build_model_from_recipe(model_recipe)`

builder 负责把结构化 recipe 转成实际模型对象。

### 7.4 Validator

validator 负责校验：

- `module` 是否存在
- `args` 是否合法
- `from` 连接是否有效
- `number` 是否在允许范围内
- 某些 task 是否必须要求特定 head 非空
- 通道和 shape 是否能传播 through 全图

### 7.5 Search Whitelist

AI 可改哪些字段，不应由总 schema 决定，而应由 family-specific whitelist 决定，例如：

- 允许调 `width_multiple`
- 允许改 `backbone[3][3]`
- 允许改 `head[2][3][1]`
- 允许把 `modules.aux_logits` 从 `false` 切到 `true`
- 不允许自由增删 layer
- 不允许跨 family 替换模块

## 8. 当前仓库的 `v1` 落地约束

虽然总 schema 应该是通用的，但当前仓库的真实落地范围仍然有限。

当前 `v1` 约束建议明确写成单独 profile：

- 任务类型：`classification`
- 当前优先打通：`MobileNetV3 Small`
- 当前 recipe builder 仍处于最小实现阶段
- 当前后端还没有开放自由 block 列表搜索

这意味着：

- 通用 schema 是未来长期方向
- 当前实现只是先在 `MobileNetV3 Small` 上验证 recipe 链路
- 目前已有的 `backbone.stem_variant`、`head.pooling_type` 这类字段，更适合视为 `mobilenet_v3` family 的临时 profile，而不是总 schema 的最终形态

## 9. 当前 `mobilenet_v3_small` Profile 示例

在通用 schema 之上，当前仓库的 `mobilenet_v3_small` profile 应该尽量长成下面这样：

```yaml
version: model_recipe@v1
task_type: classification
model_family: mobilenet
base_model: mobilenet_v3_small
nc: 6
depth_multiple: 1.0
width_multiple: 1.0
backbone:
  - [-1, 1, StemConv, [16, 3, 2, HS]]
  - [-1, 1, IR, [16, 3, 16, 16, true, RE, 2, 1]]
  - [-1, 1, IR, [16, 3, 72, 24, false, RE, 2, 1]]
  - [-1, 1, IR, [24, 3, 88, 24, false, RE, 1, 1]]
  - [-1, 1, IR, [24, 5, 96, 40, true, HS, 2, 1]]
head:
  - [-1, 1, PointwiseConv, [576, HS]]
  - [-1, 1, GlobalPool, [avg]]
  - [-1, 1, Classifier, [1024, 0.2, linear]]
modules: {}
metadata:
  notes: null
```

这个 profile 仍然可以继续限制 AI 只改很小一部分字段，例如：

- `width_multiple`
- `backbone[0][3]`
- `head[1][3]`
- `head[2][3]`

但重要的是：这些限制应来自 `mobilenet` family 的 whitelist，而不是总 schema 本身。

## 10. 与 `train_hyp` 的边界

`model_recipe` 负责表达：

- 网络结构
- 模块类型
- 通道与层级组织
- 任务头结构

`train_hyp` 负责表达：

- optimizer
- learning rate
- weight decay
- scheduler
- loss
- augmentation
- runtime flags

这两者必须分开，否则 AI 会把“结构修改”和“训练策略修改”混成一个不可审计的大对象。

## 11. API / Artifact 建议

后续建议：

- `ExperimentConfig` 持有完整 `model_recipe`
- experiment detail API 直接返回 recipe 快照
- 本地产物稳定落盘：
  - `artifacts/recipes/<experiment_id>_model.yaml`
- proposal 后续应优先返回结构化 `recipe_changes`，而不是只返回平铺的 `changes`

## 12. 一句话总结

`model_recipe` 应该尽量长成 YOLOv5 那种“全局字段 + layer list + module registry”的样子；`MobileNetV3 Small` 只是当前仓库拿来验证这套表达的第一个 family-specific profile，而不应该反过来定义总 schema。
