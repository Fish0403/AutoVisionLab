# 跨模型比较策略

这份文档描述当前 `Compare Models` 的第一版实施口径。

它和现有 `Auto Train` 的区别是：

- `Compare Models`
  - 在多个候选模型之间跑统一 baseline
  - 输出横向可比的 `top1_acc / latency_ms`
  - 不自动继续调参
- `Auto Train`
  - 只在单个 run 内继续调参和组件搜索

## 1. 当前目标

当前目标不是做带预算的自动模型筛选，而是先把小候选集的真实分布跑出来。

第一版只回答一个问题：

- 在同一套训练协议下，不同模型的 `acc` 和 `latency` 分布是什么样

## 2. 当前边界

- 第一版只覆盖：
  - `mobilenet_v2`
  - `mobilenet_v3_small`
  - `googlenet`
- 第一版只跑每个模型的 baseline experiment
- 第一版不自动选 winner
- 第一版不自动进入第二阶段 `Auto Train`
- 一个 run 的语义不变，仍然固定一个模型和一个数据集

也就是说，跨模型能力由一个更上层的 compare task 负责，而不是把不同模型塞进同一个 run。

## 3. 公平比较原则

跨模型比较阶段必须使用共享 baseline 配置。

除 `model_name` 外，以下训练配置保持一致：

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

模型专属搜索项不进入第一阶段：

- `mobilenet_v3_small`
  - 不启用 `neck_name / head_name` 搜索
- `googlenet`
  - 固定 `aux_logits = false`

## 4. 当前结果视图

第一版比较结果以二维图为主：

- `x = latency_ms`
- `y = top1_acc`

同时保留一张简表，至少展示：

- `model_name`
- `status`
- `top1_acc`
- `latency_ms`
- `parameter_count_million`
- `run_id`

`training_seconds` 当前不是主比较指标，不放到第一屏主图和主表里。

## 5. 与后续优化阶段的关系

`Compare Models` 跑完后，用户手动选择一个模型进入下一阶段。

后续优化仍走现有 `Auto Train`：

- 在单个 run 内继续调参
- 在当前已开放的模型上继续组件搜索

第一版不在 compare task 内自动串联第二阶段。

## 6. 一句话总结

当前 `Compare Models` 的定位是：

- 先用统一 baseline 跑完几个候选模型
- 直观看 `acc-latency` 分布
- 再由用户手动选一个模型进入单模型优化
