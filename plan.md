# 自主训练 Web 平台计划

## 1. 项目目标

做一个带网页界面的自主训练平台。核心思路不是让 AI 直接无限制修改训练代码，而是把整个流程拆成可控的结构化阶段：

1. AI 先生成结构化实验方案
2. 系统校验方案是否合法
3. 执行器按方案启动训练
4. 训练结果结构化落盘
5. AI 基于结果做分析
6. AI 再输出下一步结构化 action

第一阶段先聚焦在**图像分类**，先把闭环跑通，不一上来做太复杂的训练任务。

第一版的约束很明确：**先不改模型结构，只改训练参数，并且所有参数都要结构化管理。**

## 2. 第一阶段范围

### 任务范围

- 任务类型：图像分类
- 首批模型：
  - `MobileNet`
  - `GoogLeNet`
- 首批能力：
  - 手动创建训练任务
  - 网页查看训练状态和指标
  - AI 生成实验方案
  - 执行训练
  - AI 分析结果并给出下一步建议

### 暂不做

- 检测、分割、多模态任务
- 多机分布式训练
- 自动改整份训练代码
- 自动修改模型结构
- 复杂 NAS
- 在线推理服务

## 3. 技术选型

### 后端

- 语言：`Python 3.10+`
- Web 框架：`FastAPI`
- ORM：`SQLAlchemy`
- 数据校验：`Pydantic`
- 任务队列：`Celery` 或 `RQ`
- 消息中间件：
  - 开发期可先用 `Redis`
  - 后续如果调度更复杂，可升级
- 训练框架：`PyTorch`

选择 Python 的原因：

- 训练和 AI 编排都更顺手
- 与 PyTorch、实验管理、数据处理生态兼容最好
- LLM 生成结构化 proposal 和后端执行逻辑可以放在同一套语言里

### 前端

- 语言：`Python`
- 展示框架：`Streamlit`
- 图表：优先使用 `Streamlit` 内建图表能力

选择 Streamlit 的原因：

- 更适合 demo 和单人快速迭代
- 不需要额外维护一套前后端分离工程
- 对 Python 为主的项目更容易上手
- 足够支撑 run 列表、趋势图和参数详情展示

### 数据存储

- 主数据库：
  - demo 阶段使用 `SQLite`
  - 后续需要并发和长期运行时切换 `PostgreSQL`
- 缓存 / 队列：`Redis`
- 文件存储：
  - 本地开发先用磁盘目录
  - 后续可换 `MinIO` 或对象存储

### AI 接入

- LLM 接口层单独封装成 `llm service`
- 只允许输出结构化 JSON
- 所有 AI 输出必须经过 schema 校验

## 4. 为什么先做分类

先做分类是因为它最容易形成稳定闭环：

- 数据格式成熟
- 指标简单明确，比如 `top1 accuracy`、`loss`
- 模型基线清晰
- 训练脚本容易标准化
- 更适合先验证“AI 提方案 -> 系统执行 -> AI 复盘”这条链路

`MobileNet` 和 `GoogLeNet` 适合做第一批固定模型：

- `MobileNet` 代表轻量模型
- `GoogLeNet` 代表经典卷积结构
- 两者差异足够明显，便于比较实验策略

## 5. 系统核心设计

系统不让 AI 直接自由修改整个仓库，而是采用“参数提案驱动”的执行模式。

### 核心对象

#### 1. Project

一个项目，定义任务类型、数据集、可用模型范围、默认训练模板和参数空间。

#### 2. Run

一次自主研究会话。比如“在 CIFAR-10 上优化 MobileNet 的分类效果”。

#### 3. Experiment

Run 下面的一次单独实验，包含：

- 本轮 proposal
- 本轮配置
- 本轮训练结果
- 本轮训练产物
- AI 分析结论

#### 4. Proposal

AI 输出的结构化参数方案，不直接给自由文本命令。

#### 5. Result

训练执行后的结构化结果，包括：

- 状态
- 指标
- 耗时
- 资源占用
- 日志路径
- checkpoint 路径
- 本轮实际训练参数快照

#### 6. Reflection

AI 对结果的结构化分析和下一步建议。

## 6. 训练任务先做分类

第一版只支持 `classification`，后续再扩展。

### 分类任务配置建议

- 数据集：
  - 第一优先 `CIFAR-10`
  - 第二阶段加 `CIFAR-100`
  - 再后面再考虑 `Tiny-ImageNet`
- 输入尺寸：
  - 开发期统一 32 或 64
- 指标：
  - `train_loss`
  - `val_loss`
  - `top1_acc`
  - `training_time`
  - `best_epoch`

### 分类模型范围

第一版先把模型固定住，只支持两个系列，不做结构改动：

#### A. MobileNet 系列

- `mobilenet_v2`
- 先暴露少量可调参数：
  - learning rate
  - batch size
  - optimizer
  - image size
  - augmentation level
  - epochs
  - weight decay
  - scheduler
  - label smoothing

#### B. GoogLeNet 系列

- `googlenet`
- 首批可调参数：
  - learning rate
  - batch size
  - optimizer
  - image size
  - augmentation level
  - epochs
  - weight decay
  - scheduler
  - label smoothing
  - aux logits 是否开启

### 固定层与参数层

网络相关内容拆成两层：

#### 固定层

这一层先不让 AI 动：

- 模型实现
- trainer 主流程
- 数据加载逻辑
- 验证逻辑
- 指标计算逻辑

#### 参数层

这一层由 AI 输出结构化 proposal：

- optimizer
- learning_rate
- batch_size
- image_size
- epochs
- weight_decay
- scheduler
- augmentation_level
- label_smoothing
- aux_logits

第一版目标不是让 AI 改网络，而是让 AI 在固定模型上做参数实验。

## 7. AI 输出必须结构化

AI 不直接返回“我觉得你改一下这个代码”。必须输出固定 schema。

### 参数管理原则

参数管理拆成两部分：

- `experiment config`
  表示这次实验实际使用的参数
- `editable parameter space`
  表示哪些参数允许被 AI 修改，以及各自的取值范围

这样可以避免 AI 越权修改不该动的内容。

### Proposal Schema

```json
{
  "task_type": "classification",
  "model_family": "mobilenet",
  "model_name": "mobilenet_v2",
  "hypothesis": "increase learning rate may improve early convergence",
  "changes": {
    "optimizer": "adamw",
    "learning_rate": 0.003,
    "batch_size": 128,
    "augmentation_level": "medium",
    "epochs": 30,
    "weight_decay": 0.0001,
    "scheduler": "cosine",
    "label_smoothing": 0.1
  },
  "risk": "low",
  "reason": "current runs underfit in early epochs"
}
```

### Editable Parameter Space Schema

```json
{
  "model_name": "googlenet",
  "editable_params": {
    "optimizer": ["sgd", "adam", "adamw"],
    "learning_rate": { "min": 0.0001, "max": 0.01 },
    "batch_size": [32, 64, 128, 256],
    "image_size": [32, 64, 96],
    "epochs": [10, 20, 30, 50],
    "weight_decay": { "min": 0.0, "max": 0.01 },
    "scheduler": ["none", "step", "cosine"],
    "augmentation_level": ["low", "medium", "high"],
    "label_smoothing": { "min": 0.0, "max": 0.2 },
    "aux_logits": [true, false]
  }
}
```

### Result Schema

```json
{
  "status": "success",
  "metrics": {
    "train_loss": 0.42,
    "val_loss": 0.51,
    "top1_acc": 0.84
  },
  "resource": {
    "gpu_memory_mb": 2100,
    "training_seconds": 320
  },
  "params": {
    "optimizer": "adamw",
    "learning_rate": 0.003,
    "batch_size": 128,
    "image_size": 64,
    "epochs": 30,
    "weight_decay": 0.0001,
    "scheduler": "cosine",
    "label_smoothing": 0.1
  },
  "artifacts": {
    "log_path": "artifacts/logs/exp_001.log",
    "checkpoint_path": "artifacts/checkpoints/exp_001.pt"
  }
}
```

### Reflection Schema

```json
{
  "outcome": "improved",
  "analysis": "higher learning rate improved convergence without obvious instability",
  "confidence": 0.78,
  "next_action": "explore nearby learning rates",
  "recommended_changes": {
    "learning_rate": 0.004
  }
}
```

## 8. 项目架构

建议采用前后端分离，加一个独立训练执行层。

### 总体结构

1. `streamlit frontend`
   负责 demo 页面展示、任务创建、实验查看、结果对比

2. `api backend`
   负责项目管理、实验调度、状态存储、AI 接口编排

3. `trainer worker`
   负责真正跑训练

4. `llm service`
   负责参数 proposal 和 reflection

5. `database`
   负责结构化状态存储

6. `artifact storage`
   负责日志、checkpoint、图表、导出文件

## 9. 目录结构建议

```text
project-root/
  frontend/
    streamlit_app.py
  backend/
    app/
      api/
      core/
      db/
      models/
      schemas/
      services/
      workers/
      trainers/
      config_spaces/
      llm/
  artifacts/
    logs/
    checkpoints/
    reports/
  configs/
  scripts/
  docs/
  plan.md
```

## 10. 每部分做什么

### `frontend/`

基于 `Streamlit` 的 demo 控制台。

主要页面：

- 项目列表页
- Run 详情页
- Experiment 列表页
- Experiment 详情页
- 指标趋势页
- 日志查看页
- 设置页

主要职责：

- 创建训练任务
- 选择模型和数据集
- 查看实验进度
- 展示准确率、loss、耗时、显存等指标
- 展示 AI proposal 和 AI reflection
- 展示指标趋势图
- 通过选择 run / experiment 查看该次训练参数
- 在 demo 阶段优先保持单页式交互，不引入复杂前端工程

### `backend/app/api/`

HTTP API 层。

主要职责：

- 提供前端接口
- 创建项目、run、experiment
- 查询状态
- 触发新实验
- 返回图表和日志索引

### `backend/app/models/`

数据库模型定义。

主要职责：

- 定义 `Project`
- 定义 `Run`
- 定义 `Experiment`
- 定义 `Proposal`
- 定义 `Result`
- 定义 `Reflection`

### `backend/app/schemas/`

Pydantic schema。

主要职责：

- 校验前端请求
- 校验 AI 返回 JSON
- 统一 API 输出格式

### `backend/app/services/`

业务编排层。

主要职责：

- 创建 run
- 生成参数实验计划
- 决定是否继续下一轮
- 维护实验历史
- 提供指标查询和趋势图数据
- 控制 run 状态机

### `backend/app/llm/`

LLM 封装层。

主要职责：

- 生成 proposal
- 分析 result
- 生成 reflection
- 提供 prompt template
- 做 schema 校验和失败重试

### `backend/app/trainers/`

训练执行模板层。

第一版只需要：

- `classification/base_trainer.py`
- `classification/mobilenet_trainer.py`
- `classification/googlenet_trainer.py`

主要职责：

- 根据结构化配置实例化模型
- 启动训练
- 输出结构化结果
- 保存 checkpoint 和日志

### `backend/app/config_spaces/`

参数空间定义层。

主要职责：

- 定义每个模型允许被修改的参数
- 定义参数类型、范围和候选值
- 给 proposal 校验提供白名单

### `backend/app/workers/`

异步执行层。

主要职责：

- 拉取 experiment 任务
- 调 trainer
- 更新 experiment 状态
- 失败重试
- 写入结果

### `artifacts/`

实验产物目录。

主要职责：

- 保存日志
- 保存 checkpoint
- 保存训练曲线图片
- 保存报告 JSON

## 11. 建议的状态机

### Run 状态

- `draft`
- `running`
- `paused`
- `completed`
- `failed`

### Experiment 状态

- `planned`
- `queued`
- `running`
- `success`
- `failed`

这个状态机要由后端硬控制，不要交给 AI 自己决定流程。

## 12. 实验存档与展示

第一版先不要引入回滚、discard、keep 这些复杂决策。所有做过的实验都存下来，重点先把“看得清楚”做好。

### 存档原则

- 每次实验都生成唯一 `experiment_id`
- 每次实验都保存 proposal、训练参数、训练结果、日志、checkpoint
- 每次实验都保存参数空间版本
- 不管结果好坏都保留记录
- 后续是否做筛选、排序、推荐，可以在展示层处理

### 指标展示方式

网页里提供一个统一的指标趋势区域，用户选择哪个指标，就画哪个指标的折线图。

例如：

- 选择 `train_loss`，就显示 `train_loss` 的趋势折线
- 选择 `top1_acc`，就显示 `top1_acc` 的趋势折线
- 选择 `val_loss`，就显示 `val_loss` 的趋势折线

横轴建议统一为：

- 实验序号，或
- 实验创建时间

纵轴为当前选中的指标。

### 图表交互

折线图上每个点都代表一次实验。

当用户选中某个点时，右侧或弹窗展示该次实验的详细信息：

- experiment_id
- 模型名称
- 数据集
- 训练参数
- 关键指标
- AI proposal
- AI reflection
- 日志链接
- checkpoint 链接

### 参数展示建议

选中折线图上的一个点后，至少显示这些参数：

- model family
- model name
- optimizer
- learning rate
- batch size
- image size
- epoch
- augmentation level
- scheduler
- weight decay

这样用户可以很直观地看到：某个指标上升或下降时，对应实验到底改了什么。

## 13. 第一版训练执行约束

为了避免系统一开始失控，第一版建议加白名单约束。

### 允许 AI 调的内容

- 模型族：`mobilenet` / `googlenet`
- 优化器：`sgd` / `adam` / `adamw`
- 学习率
- batch size
- weight decay
- augmentation level
- scheduler
- epoch 数
- image size
- label smoothing
- `aux_logits`，仅 `googlenet` 可用

### 暂不允许 AI 动的内容

- 模型结构
- 数据读取主逻辑
- 评测逻辑
- 任意 Python 代码执行
- pip 安装新依赖
- 改数据库
- 改网页逻辑

## 14. 风险点

### 风险 1

AI proposal 太发散，导致实验无效。

解决方式：

- 用严格 schema
- 参数白名单
- 变更幅度限制
- 参数空间版本化

### 风险 2

训练脚本不统一，导致不同模型结果难比较。

解决方式：

- 用统一 trainer 接口
- 固定数据集和评测逻辑

### 风险 3

不同任务的指标体系不同，前端图表容易做死。

解决方式：

- 前端做成“指标选择器 + 通用折线图”
- 指标字段由后端动态返回
- 点选后统一展示该实验的参数快照

### 风险 4

网页先做太大，项目推进慢。

解决方式：

- 先做管理后台风格页面
- 不做复杂设计
- 先以功能完整为主

## 15. 最终建议

第一版不要做“AI 自由改代码平台”，而要做“AI 结构化提出参数方案的平台”。

先专注这条最小闭环：

1. 选择分类任务
2. 选择 `MobileNet` 或 `GoogLeNet`
3. AI 生成参数 proposal
4. 系统执行训练
5. 系统保存结果
6. 网页展示指标趋势和参数详情
7. AI 分析结果并生成下一步 action

这条链路一旦跑通，后续再扩展到更多网络、更多任务类型、更多自治能力会稳很多。
