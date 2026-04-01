# 产品流程草案

这份文档用于沉淀 AutoVisionLab 下一阶段准备落地的目标产品流程。

当前先把流程单独写在这里，作为后续开发的锚点；等实现完成并确认口径稳定后，再同步回 `README.md`、`docs/plan.md` 和 `docs/tasks.md`。

## 1. 目标

当前产品流程收敛为两个并列入口：

1. `Models Compare`
2. `Model Search`

这样设计的原因是：

- 当用户还没有决定模型时，应该先做跨模型 baseline 比较
- 当用户已经决定模型时，应该可以直接进入单模型搜索
- `train base` 不再作为默认独立入口暴露给用户

需要强调的是：

- `baseline` 仍然存在
- 只是它从“用户手动执行的一步”变成“流程内部自动执行的一步”

## 2. 总体原则

### 2.1 Run 语义不变

- 一个 `run` 仍然固定一个数据集和一个模型
- 跨模型比较不能把多个模型塞进同一个 run
- compare 阶段的跨模型能力由更上层 task 负责

### 2.2 Compare 和 Search 分工明确

- `Models Compare`
  - 负责先看候选模型的横向分布
- `Model Search`
  - 负责围绕一个确定模型继续做策略搜索

### 2.3 Baseline 仍是必要步骤

- compare 阶段需要 shared baseline 才能公平比较
- 单模型搜索阶段也需要 baseline 作为 run 的起点
- 因此去掉的是 `train base` 这个显式入口，不是 baseline 这个步骤本身

## 3. Models Compare

`Models Compare` 面向“还没决定模型”的用户。

### 3.1 入口行为

- 用户填写一套公共训练配置
- 系统自动对当前 compare 支持集的全部模型执行 shared baseline
- 当前先不开放候选模型勾选

当前 compare 支持集先按现有实现落地：

- `MobileNetV2`
- `MobileNetV3 Small`
- `GoogLeNet`

后续模型变多后，再考虑开放候选模型选择。

### 3.2 共享配置原则

除 `model_name` 以及少数模型特例外，其余训练配置保持公共口径。

shared baseline 至少保持以下字段一致：

- `dataset`
- `use_demo_mode`
- `epochs`
- `batch_size`
- `image_size`
- `optimizer`
- `learning_rate`
- `weight_decay`
- `scheduler`
- `augmentation_policy`
- `augmentation_params`
- `loss_name`
- `loss_params`
- `label_smoothing`

模型专属差异由系统在 compare 阶段内部归一化处理，而不是要求用户分别配置。

### 3.3 执行过程

系统按候选模型逐个执行：

1. 为该模型创建独立 run
2. 用 shared baseline config 创建 baseline experiment
3. 启动训练
4. 写入 result 和资源摘要
5. 汇总到 compare 结果页

### 3.4 结果页

compare 完成后，结果页需要同时提供：

- 横向散点图
  - `x = latency_ms`
  - `y = top1_acc`
- 简表
  - `model_name`
  - `status`
  - `top1_acc`
  - `latency_ms`
  - `parameter_count_million`
  - `run_id`
- AI 解读
- AI 推荐继续优化的模型

当前 compare 阶段不自动进入第二阶段。

用户在看完结果后，手动选择一个模型继续进入后续优化。

## 4. Model Search

`Model Search` 面向“已经决定模型”的用户。

### 4.1 入口行为

- 用户直接选择一个模型
- 使用当前单模型表单配置 baseline
- 使用当前已有的 `allow_*` 搜索开关约束后续 AI 搜索范围

当前不单独重新设计一套 `Model Search` 表单规则，先沿用现有单模型训练与自动搜索配置能力。

### 4.2 执行过程

系统按单模型 run 执行：

1. 创建该模型对应的 run
2. 自动创建 baseline experiment
3. 启动 baseline 训练
4. baseline 成功后，进入该 run 的 `Auto Train`
5. 后续 proposal、校验、训练、晋级、停止都继续沿用现有单模型 run 规则

### 4.3 搜索边界

`Model Search` 只在单个 run 内进行，不承担跨模型筛选职责。

当前继续沿用已有搜索控制：

- `allow_basic_hparam_search`
- `allow_strategy_search`
- `allow_loss_search`
- `allow_augmentation_search`
- `allow_model_module_search`

## 5. AI 在流程中的职责

### 5.1 Compare 阶段

AI 在 `Models Compare` 阶段负责：

- 解读各模型结果
- 总结优势、短板和风险
- 给出“下一步优先继续哪个模型”的建议

AI 不直接替用户做最终选择。

### 5.2 单模型搜索阶段

AI 在 `Model Search` / `Auto Train` 阶段负责：

- 生成结构化 proposal
- 在允许的搜索范围内继续调参与策略搜索
- 总结本轮优化结果和下一步建议

AI 仍然要受 schema、parameter space 和 search policy 约束。

## 6. train base 的新语义

当前目标流程下，不再保留 `train base` 作为默认独立产品入口。

但 baseline 仍然有两种内部落点：

- 在 `Models Compare` 中
  - 它是每个候选模型的 shared baseline experiment
- 在 `Model Search` 中
  - 它是该单模型 run 的首个 baseline experiment

因此，后续界面和文档里要避免继续把“baseline 步骤”和“独立 Train Base 按钮”混为一谈。

## 7. 当前暂定项

以下口径先按当前阶段固定：

- 首页采用双入口并列，而不是单入口分支
- `Models Compare` 先固定当前 compare 支持集全量执行
- compare 完成后，用户手动选择模型继续
- compare 阶段由 AI 解读结果并给建议，但不自动续跑
- `Model Search` 继续沿用当前 `allow_*` 搜索开关

## 8. 后续同步

当实现完成且流程确认稳定后，再把这里的稳定结论同步到：

- [../README.md](../README.md)
- [plan.md](plan.md)
- [tasks.md](tasks.md)

在那之前，这份文档作为当前目标流程的主参考。
