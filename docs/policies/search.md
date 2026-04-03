# Search

这份文档说明当前 `Auto Train` 的搜索规则。

## 基本规则

- `Auto Train` 在同一个 `run` 内继续追加实验
- 一个 `run` 固定一个数据集和一个模型
- `AI` 只能在当前模型 `parameter_space` 声明的可编辑字段内搜索
- 前端不再要求用户手动勾选搜索维度
- proposal 可以同时修改一个或多个字段，只要字段和值都合法
- 若 proposal 修改 `image_size`，新值必须是正整数，且严格小于当前 source experiment 的 `image_size`

## 可搜索字段分组

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

## 结构字段口径

- 标准 torchvision 分类模型
  - 主要使用 `neck_name`、`head_name`
- `GoogLeNet`
  - 主要使用 `aux_logits`

具体允许值以当前模型的 editable `parameter_space` 为准。

## Proposal 约束

- proposal 必须是结构化 JSON
- proposal 必须通过 schema 校验
- proposal 必须通过 parameter space 校验
- proposal 必须能构建出当前训练器可接受的 follow-up config
- proposal 不能为空

## 失败与重试

- 无效 proposal 会持续重试，直到生成一版合法 proposal 或用户手动停止
- proposal 的校验错误会回喂给 AI，作为下一次重试的 rejection reason
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
