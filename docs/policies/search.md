# Search

这份文档说明当前 `Auto Train` 的搜索规则。

## 基本规则

- `Auto Train` 在同一个 `run` 内继续追加实验
- 一个 `run` 固定一个数据集和一个模型
- `AI` 只能在当前模型 `parameter_space` 声明的可编辑字段内搜索
- 当前搜索状态由实验配置中的 `train_hyp` 和 `model_recipe` 投影得到
- 当前 run 实际允许搜索哪些字段，以最新实验配置里的 `search_policy` 为准，不再一律退回模型默认全量搜索
- proposal 使用扁平 `changes` 字段表达本轮可搜索变更，键名必须落在白名单内
- 前端不再要求用户手动勾选搜索维度
- proposal 可以同时修改一个或多个字段，只要字段和值都合法
- 若 proposal 修改 `image_size`，新值必须是正整数并落在当前参数空间内；通常优先选择不大于当前 source experiment 的值，但这不是硬性拒绝条件
- `use_demo_mode` 属于运行时选项，不属于 AI 搜索字段，也不进入 proposal 白名单

## 可搜索字段分组

- `basic`
  - `optimizer`
  - `learning_rate`
  - `batch_size`
  - `epochs`
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

## 结构字段口径

- `train_hyp`
  - `optimizer` -> `optimizer`
  - `lr0` -> `learning_rate`
  - `batch_size` -> `batch_size`
  - `weight_decay` -> `weight_decay`
  - `scheduler` -> `scheduler`
  - `label_smoothing` -> `label_smoothing`
  - `image_size` -> `image_size`
  - `augmentation.policy` -> `augmentation_policy`
  - `augmentation.mixup` -> `mixup_alpha`
  - `augmentation.cutmix` -> `cutmix_alpha`
  - `augmentation.random_erasing` -> `random_erasing_prob`
  - `loss.name` -> `loss_name`
  - `fl_gamma` -> `focal_gamma`
- `model_recipe`
  - `modules.aux_logits` -> `aux_logits`
  - `components.neck.name` -> `neck_name`
  - `components.head.name` -> `head_name`

## 分类模型口径

- 标准 torchvision 分类模型
  - 当前有效的模块搜索字段主要是 `neck_name`、`head_name`
- `GoogLeNet`
  - 当前有效的模块搜索字段主要是 `aux_logits`

具体允许值和有效字段集合以当前模型的 `editable parameter_space` 为准。

## `epochs` 口径

- `epochs` 现在是可声明的 `basic` 搜索字段，但默认不要求开启
- 当 `search_policy.allowed_basic_hparam_fields` 包含 `epochs` 时，AI 才允许修改它
- 当 `epochs` 被允许搜索时，系统把它视为 `training budget` 维度，而不是纯策略字段
- search scope summary 会额外标记 `Training Budget`
- 若 `epochs` 未开启，proposal 中的 `epochs` 变更会被拒绝，并作为 rejection reason 回喂给 AI

## Proposal 约束

- proposal 必须是结构化 JSON
- proposal 必须通过 schema 校验
- proposal 必须通过 parameter space 校验
- proposal 的 `changes` 至少包含一个非空字段
- proposal 必须能构建出当前训练器可接受的 follow-up config

## 失败与重试

- 无效 proposal 会持续重试，直到生成一版合法 proposal 或用户手动停止
- proposal 的校验错误会回喂给 AI，作为下一次重试的 rejection reason
- proposal 因 search policy 或 follow-up config 校验被拒绝时，workspace status warning 会显示最近一次 rejection reason
- 单轮 experiment 失败会先做局部重试
- 超出单轮 experiment 重试预算后，该轮作为失败样本保留在历史里，后续 proposal 可继续参考它

## 停止与异常

- 用户可以随时点 `Stop`
- 未人工停止时，系统会持续追加实验
- `AI` 不直接决定停机
- 单次 proposal 生成失败不会终止整个 `Auto Train`
- 单次 experiment 创建失败或训练失败不会直接终止整个 `Auto Train`
- `LLM` 请求错误会持续重试，并逐步拉长重试间隔
- baseline 失败会直接结束 `Auto Train`，不会进入 proposal 阶段

## 实现位置

- [backend/app/services/auto_train_service.py](../../backend/app/services/auto_train_service.py)
- [backend/app/services/proposal_service.py](../../backend/app/services/proposal_service.py)
- [backend/app/config_spaces/classification.py](../../backend/app/config_spaces/classification.py)
