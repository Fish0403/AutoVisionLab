# AutoVisionLab Tasks

这个文件只记录执行层信息，不重复解释整体设计。系统定位与边界见 [plan.md](/home/fish/AutoVisionLab/plan.md)。

## 1. 执行原则

- 固定模型实现，不做模型结构搜索
- 所有可调参数必须结构化
- trainer 只读取结构化 config
- 一个 run 内固定一个模型和一个数据集
- 数据库存结构化摘要，本地文件存大体积产物

## 2. 已完成

### 平台骨架

- `FastAPI` 后端、`Streamlit` 前端、本地启动脚本已就位
- `SQLite` 持久化已接入
- `runs / experiments / results` 基础链路已打通

### 分类训练闭环

- 已支持统一分类数据目录读取：
  - `data/<dataset>/classification/train|val/<class>/`
- 已接入模型：
  - `MobileNetV2`
  - `GoogLeNet`
  - `ResNet18`
  - `ResNet34`
  - `DenseNet121`
- 已支持基础训练参数：
  - `optimizer`
  - `learning_rate`
  - `batch_size`
  - `image_size`
  - `epochs`
  - `weight_decay`
  - `scheduler`
  - `label_smoothing`
  - `aux_logits`

### 训练策略组件

- 已接入结构化 `loss registry`
- 已接入结构化 `augmentation registry`
- 当前支持：
  - `loss_name`: `cross_entropy` / `cross_entropy_with_label_smoothing` / `focal_loss`
  - `augmentation_policy`: `none` / `basic`
  - `mixup_alpha`
  - `cutmix_alpha`
  - `random_erasing_prob`

### AI proposal 与 auto-train

- proposal 已改为参考完整 run 历史
- run 已维护：
  - `baseline_experiment_id`
  - `best_experiment_id`
  - `frontier_experiment_id`
- proposal 已有字段白名单与参数空间校验
- auto-train 已支持后台任务、停止、日志轮询和结果刷新
- auto-train 已接入按总轮数比例切换的阶段策略

### 数据与产物

- `CIFAR-10` 已有准备脚本：
  [data/cifar10/prepare_classification_split.py](/home/fish/AutoVisionLab/data/cifar10/prepare_classification_split.py)
- `NEU-CLS` 已有准备脚本：
  [data/neu-cls/prepare_classification_split.py](/home/fish/AutoVisionLab/data/neu-cls/prepare_classification_split.py)
- 本地日志已改为按 run 聚合：
  - `artifacts/runs/<run_id>.log`
- checkpoint 继续按 experiment 保存：
  - `artifacts/checkpoints/<experiment_id>.pt`

## 3. 进行中

### 文档与口径统一

- 收敛 `README / plan / tasks / schemas / experiment_policy`
- 清理过时表述，统一当前实现与规则

### AI 搜索策略收敛

- 继续观察 auto-train 后半程是否真的切到 augmentation / loss 维度
- 继续减少无效或重复 proposal

## 4. 待做

### 前端

- 给 `AI Search Policy` 增加更细的字段级配置，而不只是分类开关
- 在结果区增加本地 run log 查看入口
- 在结果区增加 checkpoint / artifact 路径查看入口

### 后端

- 增加 `reflection` 真正生成链路
- 增加更稳定的 proposal 去重 / 失败回退逻辑
- 增加更清晰的 artifact manifest，而不只是一对路径

### 数据

- 清理前端默认值与 fallback demo 文案中残留的 `cifar10` 假设，使 `NEU-CLS` 的现有支持在界面和文档里表达得更准确
- 根据统一目录规范增加更多工业分类数据集

### 研究决策

- 继续完善 `keep / discard / crash / timeout` 的前端表达
- 明确人工决策与自动排名之间的边界

## 5. 建议的下一步顺序

1. 补 run log 查看入口，让本地产物可直接在界面里追踪
2. 清理默认值与 fallback demo 文案中残留的 `cifar10` 假设
3. 完成 `reflection` 链路
4. 细化字段级 AI 搜索策略
