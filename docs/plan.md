# AutoVisionLab Plan

## 1. 项目定位

AutoVisionLab 是一个面向图像分类实验的自主训练 Web 平台。

核心目标不是让 AI 自由修改训练代码，而是把实验流程收敛成结构化闭环：

1. AI 生成结构化 proposal
2. 系统校验 proposal、参数空间和搜索权限
3. 训练执行器按结构化 config 启动实验
4. 训练结果写入数据库和本地产物目录
5. AI 基于 run 历史继续提出下一步方案

平台当前坚持两个边界：

- 不开放模型结构搜索
- AI 只能在白名单参数内搜索

## 2. 任务范围

当前任务范围：

- 任务类型：图像分类
- 前端：`Streamlit`
- 后端：`FastAPI + SQLAlchemy + Pydantic`
- 训练框架：`PyTorch`
- 数据库：`SQLite`
- 本地产物：`artifacts/`

当前可选模型：

- `MobileNetV2`
- `GoogLeNet`
- `ResNet18`
- `ResNet34`
- `DenseNet121`

当前数据读取方式：

- 平台统一读取 `data/<dataset>/classification/train|val/<class>/`
- 原始下载内容保留在 `data/<dataset>/raw/`

不在本项目当前范围内的内容：

- 检测、分割、多模态任务
- 自动修改模型结构
- NAS 或自由代码生成式训练
- 多机分布式调度
- 在线推理服务

## 3. 系统原则

### 3.1 结构化优先

- trainer 只能读取结构化 config
- proposal 必须是结构化 JSON
- 参数空间必须显式声明范围和可选值
- 前端展示优先读取数据库中的结构化结果

### 3.2 同一个 run 内保持可比性

- 一个 run 固定一个数据集和一个模型
- 追加实验只在该 run 内继续调参
- 跨模型比较应通过多个 run 完成，而不是在同一个 run 内混合实验

### 3.3 数据库与本地产物分层

数据库负责：

- run / experiment 元数据
- proposal / result / reflection
- 指标、状态、决策
- 产物路径索引

本地文件负责：

- run 聚合日志
- checkpoint
- 后续需要的大文件产物

## 4. 核心对象

### 4.1 Run

`run` 表示围绕某个数据集和模型的一组连续实验。

每个 run 显式维护三个锚点：

- `baseline_experiment_id`
- `best_experiment_id`
- `frontier_experiment_id`

作用：

- 给 AI proposal 提供稳定参考
- 给前端结果区提供默认展示对象
- 为后续把“当前最好”和“当前继续探索的分支”分开保留独立锚点

当前实现补充说明：

- `frontier_experiment_id` 当前默认与最新晋级的 `keep` 实验同步
- 独立的 `frontier` 分支语义仍保留在数据结构中，但尚未扩展出与 `best` 长期分离的推进逻辑

### 4.2 Experiment

`experiment` 表示 run 下的一次具体训练。

每个 experiment 保存：

- 执行配置
- 参数空间快照
- proposal
- 训练结果
- 决策状态

执行状态和研究决策分开：

- 执行状态：`queued`、`running`、`success`、`failed`、`discarded`
- 当前执行链路稳定写入的研究决策：`keep`、`discard`、`crash`
- `timeout` 仍保留在 schema 中，但当前后端执行链路尚未自动写入该决策

### 4.3 Proposal

proposal 是 AI 生成的结构化参数变更方案。

proposal 必须满足：

- 只修改白名单字段
- 值必须落在 parameter space 允许范围内
- 不允许空 proposal
- 不允许引用当前未开放的字段或取值

### 4.4 Result

result 表示一次训练完成后的结构化输出，包括：

- 指标
- 资源耗时
- 最终参数快照
- 本地产物路径

## 5. AI 搜索设计

AI 搜索不是无限制调参，而是受 `search_policy` 控制。

当前搜索维度分为三类：

- 基础超参数
  - `optimizer`
  - `learning_rate`
  - `batch_size`
  - `weight_decay`
  - `scheduler`
  - `label_smoothing`
  - `image_size`（仅在满足特殊规则时）
- 增强相关
  - `augmentation_policy`
  - `mixup_alpha`
  - `cutmix_alpha`
  - `random_erasing_prob`
- loss / strategy
  - `loss_name`
  - `focal_gamma`
  - `aux_logits`

当前数据增强设计：

- 基础增强只保留 `augmentation_policy = none | basic`
- 额外增强单独由 `mixup_alpha`、`cutmix_alpha`、`random_erasing_prob` 控制

## 6. Auto Train 设计

`Auto Train` 的目标不是无脑重复微调基础超参数，而是分阶段搜索。

当前规则：

- 前 `50%` 轮次允许基础超参数主导
- 后 `50%` 时间预算优先考虑 augmentation / loss / strategy 变化
- 若当前 run 的可行动作仍不足以支持切维或非 `basic` 变化，系统会保留可执行性，而不是强行制造无解约束

这样做的原因：

- 前半段先快速找到稳定区间
- 后半段尽量避免一直围绕 `lr / batch_size / wd / scheduler` 打转
- 让 auto-train 的后半程更像真正的策略探索，而不是重复微调

## 7. 预算与可比性设计

### 7.1 当前问题

当前系统主要按 `epochs` 控制训练轮数。

这能保证单次实验有稳定的训练终止条件，但不能完全保证实验成本可比。

原因是：

- 不同模型的单 epoch 耗时不同
- 不同 `image_size` 会直接改变训练与推理成本
- 不同 `batch_size`、augmentation、loss / strategy 会改变训练时间

因此，“固定 epochs” 更接近固定训练流程长度，而不是固定真实预算。

### 7.2 为什么不直接采用固定训练时长

参考部分自动研究项目时，一个常见做法是给每轮实验固定 wall-clock budget，例如“每轮只训练 5 分钟”。

这个策略适合追求研究效率最大化的场景，但不一定适合本项目当前目标。

本项目更关心的是：

- 在相近推理成本下比较模型效果
- 允许训练期开销增加，只要部署期开销没有失控

因此，如果只固定训练时间，容易把“研究效率”当成主要目标，而弱化“部署可比性”。

### 7.3 后续更偏向的方向

后续如果要继续强化 run 内可比性，优先考虑的是“推理预算”而不是“训练时长预算”。

更适合本项目的约束方式是：

- 先确定 baseline 的推理成本
- 再限制后续实验的推理成本不能超过该 baseline 太多
- 只有在推理成本仍落在预算内时，才继续比较 `top1_acc`、`val_loss` 等效果指标

### 7.4 可能的推理预算字段

后续可考虑在 `result` 或 run policy 中增加：

- `inference_latency_ms`
- `throughput_samples_per_second`
- `parameter_count`
- `flops`
- `peak_inference_memory_mb`
- `input_image_size`

以及相对 baseline 的预算约束，例如：

- `max_latency_ratio_vs_baseline`
- `max_flops_ratio_vs_baseline`
- `max_parameter_count_ratio_vs_baseline`
- `max_image_size`

### 7.5 当前结论

当前阶段暂不修改实现。

设计上先明确这条判断：

- `epochs` 不等于真正预算
- 训练时间与推理时间有关联，但不是同一个量
- 本项目后续更应该优先约束“推理成本下的效果优化”，而不是机械收敛到固定训练时长

## 8. 数据与产物

### 8.1 数据目录

统一目录规则：

```text
data/
  <dataset_name>/
    raw/
    classification/
      train/
        <class_name>/
      val/
        <class_name>/
```

当前推荐：

- `cifar10`
- `neu-cls`

### 8.2 产物目录

当前本地产物规则：

- `artifacts/runs/<run_id>.log`
  - 统一记录该 run 下的实验创建、AI proposal、训练开始、epoch 摘要和结果摘要
- `artifacts/checkpoints/<experiment_id>.pt`
  - 按 experiment 保存 checkpoint

## 9. 前端目标

前端保持两个核心工作区：

- 左侧训练与策略设置
- 右侧结果、建议与训练记录

前端应尽量做到：

- 主表单只保留少量高价值手动参数
- AI Search Policy 明确显示“当前 AI 会搜索哪些参数”
- Results 默认跟随当前 run 的 best experiment
- Training Records 既能回看历史，也能做趋势比较

## 10. 文档分工

- [README.md](/home/fish/AutoVisionLab/README.md)
  - 环境、启动、数据准备、目录结构
- [docs/plan.md](/home/fish/AutoVisionLab/docs/plan.md)
  - 项目定位、系统边界、核心设计
- [docs/tasks.md](/home/fish/AutoVisionLab/docs/tasks.md)
  - 已完成 / 进行中 / 待做事项
- [docs/schemas.md](/home/fish/AutoVisionLab/docs/schemas.md)
  - 结构化数据与示例
- [docs/experiment_policy.md](/home/fish/AutoVisionLab/docs/experiment_policy.md)
  - 实验运行总览
- [docs/run_promotion_policy.md](/home/fish/AutoVisionLab/docs/run_promotion_policy.md)
  - run 晋级与回退规则
- [docs/auto_train_search_policy.md](/home/fish/AutoVisionLab/docs/auto_train_search_policy.md)
  - Auto Train 搜索策略
- [docs/auto_train_stop_policy.md](/home/fish/AutoVisionLab/docs/auto_train_stop_policy.md)
  - Auto Train 停止策略
- [CHANGELOG.md](/home/fish/AutoVisionLab/CHANGELOG.md)
  - 历史变更记录
