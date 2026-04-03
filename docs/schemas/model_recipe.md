# Model Recipe

这份文档定义当前使用的 `model_recipe` 结构。

## 作用

- 用结构化对象表达模型结构和兼容配置
- 为训练器、配置持久化和受限搜索提供统一入口
- 让结构变化可以落盘、回放和做受限校验

## 字段

- `version`
  - 固定为 `model_recipe@v1`
- `task_type`
  - schema 支持 `classification`、`detection`、`segmentation`
  - 当前项目实际使用的是 `classification`
- `model_family`
- `base_model`
- `nc`
  - 类别数
- `input_channels`
- `width_multiple`
- `components`
  - 当前主要使用的粗粒度组件视图
- `backbone_config`
  - 主干兼容配置
- `head_config`
  - 头部兼容配置
- `modules`
  - 轻量结构开关
- `metadata`
  - 人类可读的辅助信息

## 当前主结构

当前分类链路主要围绕下面这些字段工作：

- `components.backbone.name`
- `components.neck.name`
- `components.head.name`
- `backbone_config.stem_variant`
- `backbone_config.attention_module`
- `backbone_config.last_channel_multiplier`
- `head_config.pooling_type`
- `head_config.classifier_dropout`
- `head_config.classifier_type`
- `modules`

其中：

- `components` 是对外最稳定的粗粒度结构视图
- `backbone_config` 和 `head_config` 是当前 builder 兼容层
- `modules` 用于轻量结构开关，例如 `aux_logits`

## 展开后的架构字段

`backbone`、`neck`、`head` 三个字段仍保留在 schema 中，用于展开后的架构表示和模板兼容，但它们不是当前工作台和搜索策略的主要入口。

## 当前分类口径

- 标准 torchvision 分类模型
  - 当前主要使用 `neck_name`、`head_name`
  - 包括 `MobileNet`、`EfficientNet`、`ResNet` 这些族
- `GoogLeNet`
  - 当前主要使用 `aux_logits`
  - 不开放 `neck_name` / `head_name` 替换

具体允许值以当前模型的 editable `parameter_space` 为准，不在这份文档里重复维护一张静态白名单表。

## 默认模板

当前仓库提供的内置模型模板文件示例：

- [backend/app/trainers/recipes/classification/mobilenet_v3_small.yaml](../../backend/app/trainers/recipes/classification/mobilenet_v3_small.yaml)
