# Search

这份文档描述 `Auto Train` 的搜索规则。

## 基本规则

- `Auto Train` 在同一个 `run` 内继续追加实验
- 一个 `run` 固定一个数据集和一个模型
- `AI` 在当前模型 `parameter_space` 声明的可编辑字段内自主搜索
- 前端不再要求用户手动勾选搜索维度
- proposal 可以自由决定修改一个或多个字段，只要字段和值都合法
- `image_size` 不再使用固定候选集合；若 proposal 修改它，只能相对当前 source experiment 严格减小

## 搜索维度

- `basic`
  - `optimizer`
  - `learning_rate`
  - `batch_size`
  - `weight_decay`
  - `scheduler`
  - `label_smoothing`
  - `image_size`
- `augmentation`
  - `augmentation_policy`
  - `mixup_alpha`
  - `cutmix_alpha`
  - `random_erasing_prob`
- `loss`
  - `loss_name`
  - `focal_gamma`
- `model_module`
  - `aux_logits`
  - `neck_name`
  - `head_name`

说明：

- `allow_strategy_search` 目前存在，但没有对应的可搜索字段
- `model_module` 是否生效取决于模型的 editable `parameter_space`
- `GoogLeNet` 主要使用 `aux_logits`
- `MobileNetV2`、`MobileNetV3 Small`、`MobileNetV3 Large`、`EfficientNet-B0`、`EfficientNet-B1`、`ResNet18`、`ResNet34`、`ResNet50` 主要使用 `neck_name` 和 `head_name`

### 当前结构白名单

- `neck_name` 和 `head_name` 的允许值以后端 `parameter_space` 白名单为准

| 模型 | 可搜索结构字段 | `neck_name` 允许值 | `head_name` 允许值 | 备注 |
| --- | --- | --- | --- | --- |
| `mobilenet_v2` | `neck_name`, `head_name` | `avg_pool`, `gem_pool` | `native_classifier`, `linear`, `dropout_linear` | 使用当前模型自己的 backbone，不是跨模型复用 head |
| `mobilenet_v3_small` | `neck_name`, `head_name` | `avg_pool`, `gem_pool` | `native_classifier`, `linear`, `dropout_linear` | 使用当前模型自己的 backbone，不是跨模型复用 head |
| `mobilenet_v3_large` | `neck_name`, `head_name` | `avg_pool`, `gem_pool` | `native_classifier`, `linear`, `dropout_linear` | 使用当前模型自己的 backbone，不是跨模型复用 head |
| `efficientnet_b0` | `neck_name`, `head_name` | `avg_pool`, `gem_pool` | `native_classifier`, `linear`, `dropout_linear` | 使用当前模型自己的 backbone，不是跨模型复用 head |
| `efficientnet_b1` | `neck_name`, `head_name` | `avg_pool`, `gem_pool` | `native_classifier`, `linear`, `dropout_linear` | 使用当前模型自己的 backbone，不是跨模型复用 head |
| `resnet18` | `neck_name`, `head_name` | `avg_pool`, `gem_pool` | `native_classifier`, `linear`, `dropout_linear` | 使用当前模型自己的 backbone，不是跨模型复用 head |
| `resnet34` | `neck_name`, `head_name` | `avg_pool`, `gem_pool` | `native_classifier`, `linear`, `dropout_linear` | 使用当前模型自己的 backbone，不是跨模型复用 head |
| `resnet50` | `neck_name`, `head_name` | `avg_pool`, `gem_pool` | `native_classifier`, `linear`, `dropout_linear` | 使用当前模型自己的 backbone，不是跨模型复用 head |
| `googlenet` | `aux_logits` | - | - | 当前不开放 `neck_name` / `head_name` 替换 |

| 字段 | 含义 | 当前允许值 |
| --- | --- | --- |
| `neck_name` | backbone 输出后的 pooling / neck 类型 | `avg_pool`, `gem_pool` |
| `head_name` | 当前模型特征后接的分类头类型，不是跨模型复用别的模型 head | `native_classifier`, `linear`, `dropout_linear` |
| `aux_logits` | GoogLeNet 的辅助分类头开关 | `true`, `false` |

## Proposal 约束

- proposal 必须是结构化 JSON
- proposal 必须通过 schema 校验
- proposal 必须通过 parameter space 校验
- proposal 若修改 `image_size`，新值必须是正整数，且严格小于当前 source experiment 的 `image_size`
- proposal 不能为空
- proposal 不再受单变量 / 双变量预算约束
- proposal 不再受“基础字段优先”或“非基础字段优先”这类人工引导约束

## 失败与延续

- 无效 proposal 会持续重试，直到生成一版合法 proposal 或用户手动停止
- 单轮 experiment 失败会先做局部重试
- 超出单轮 experiment 重试预算后，该轮作为失败样本保留在历史里，后续 proposal 可继续参考它
- 搜索是否停止不再由去重、切维或维度覆盖规则决定

## 停止与异常

- 用户可以随时点 `Stop`
- 未人工停止时，系统会持续追加实验
- `AI` 不直接决定停机
- 搜索策略只影响“下一轮试什么”，不影响“什么时候停”
- 单轮训练不会因为外层停止意图被立刻强制打断
- 人工 `Stop` 会对当前实验发送中断请求
- 单次 `proposal` 生成失败不会终止整个 `Auto Train`
- 单次 experiment 创建失败、训练失败或 timeout 不会终止整个 `Auto Train`
- `LLM` 请求错误会持续重试，并逐步拉长重试间隔
- 只有真正不可恢复的后台错误才会将 task 标为 `failed`
