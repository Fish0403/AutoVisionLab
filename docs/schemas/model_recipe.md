# Model Recipe

这份文档定义当前使用的 `model_recipe` 结构。

## 作用

- 用结构化对象表达模型结构，而不是把结构隐含在 trainer 代码里
- 让模型结构可以落盘、回放和做受限搜索
- 为当前分类模型保留统一的 recipe 入口

## 字段

- `version`
  - 固定为 `model_recipe@v1`
- `task_type`
  - 当前支持 `classification`、`detection`、`segmentation`
- `model_family`
  - 例如 `mobilenet`、`googlenet`、`resnet`
- `base_model`
  - 例如 `mobilenet_v3_small`、`mobilenet_v3_large`、`mobilenet_v2`、`efficientnet_b0`、`efficientnet_b1`、`googlenet`、`resnet18`、`resnet34`、`resnet50`
- `nc`
  - 类别数
- `input_channels`
  - 输入通道数，当前默认 `3`
- `width_multiple`
  - 宽度缩放
- `components`
  - 当前使用的粗粒度组件视图
- `backbone_config`
  - 主干兼容配置
- `backbone`
  - YOLO 风格 layer list
- `neck`
  - YOLO 风格 layer list，当前分类模型通常为空
- `head_config`
  - 头部兼容配置
- `head`
  - YOLO 风格 layer list
- `modules`
  - 轻量结构开关
- `metadata`
  - 人类可读的辅助信息

## Layer 格式

每一层使用 `[from, repeat, module, args]` 表达，序列化时对应字段为：

- `from`
- `repeat`
- `module`
- `args`
- `tag`

## 当前分类实现

当前分类训练器会把 recipe 解析成实际模型对象。当前能直接使用的 recipe 口径如下：

- `mobilenet_v2`
  - 直接使用内置分类结构
  - 当前允许的受限结构字段包括 `neck_name`、`head_name`
- `efficientnet_b0`
  - 使用内置分类结构
  - 当前允许的受限结构字段包括 `neck_name`、`head_name`
- `efficientnet_b1`
  - 使用内置分类结构
  - 当前允许的受限结构字段包括 `neck_name`、`head_name`
- `mobilenet_v3_small`
  - 使用内置分类结构
  - 当前允许的受限结构字段包括 `neck_name`、`head_name`
- `mobilenet_v3_large`
  - 使用内置分类结构
  - 当前允许的受限结构字段包括 `neck_name`、`head_name`
- `googlenet`
  - 使用内置分类结构
  - `aux_logits` 通过 `modules` 保存
- `resnet18`
  - 使用内置分类结构
  - 当前允许的受限结构字段包括 `neck_name`、`head_name`
- `resnet34`
  - 使用内置分类结构
  - 当前允许的受限结构字段包括 `neck_name`、`head_name`
- `resnet50`
  - 使用内置分类结构
  - 当前允许的受限结构字段包括 `neck_name`、`head_name`

## 当前兼容层

- `components.backbone.name`
- `components.neck.name`
- `components.head.name`
- `backbone_config.stem_variant`
- `backbone_config.attention_module`
- `backbone_config.last_channel_multiplier`
- `head_config.pooling_type`
- `head_config.classifier_dropout`
- `head_config.classifier_type`

这些字段用于当前分类 builder 和 UI 的兼容层，其中真正参与当前 editable parameter space 的字段取决于模型白名单。

## 当前可搜索字段

- `mobilenet_v3_small`
  - `neck_name`
  - `head_name`
- `mobilenet_v2`
  - `neck_name`
  - `head_name`
- `efficientnet_b0`
  - `neck_name`
  - `head_name`
- `efficientnet_b1`
  - `neck_name`
  - `head_name`
- `googlenet`
  - `aux_logits`
- `mobilenet_v3_large`
  - `neck_name`
  - `head_name`
- `resnet18`
  - `neck_name`
  - `head_name`
- `resnet34`
  - `neck_name`
  - `head_name`
- `resnet50`
  - `neck_name`
  - `head_name`
- 当前基础超参数和增强字段不在 `model_recipe` 里，而在 `train_hyp` 里

## 默认模板

当前仓库提供的内置模型模板文件是：

- [backend/app/trainers/recipes/classification/mobilenet_v3_small.yaml](../../backend/app/trainers/recipes/classification/mobilenet_v3_small.yaml)
