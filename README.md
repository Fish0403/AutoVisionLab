# AutoVisionLab

Industrial Vision Experiment Platform for Structured Model Development

面向工业视觉研发的结构化实验平台，当前公开实现先落地图像分类闭环。

## 中文

### 项目概述

AutoVisionLab 不是一个只服务于单一图像分类任务的小工具，而是一个面向工业视觉场景的实验平台。

它关注的是把模型研发过程收敛成可校验、可追踪、可复盘的结构化闭环，而不是让 AI 自由改训练代码。

当前公开实现先覆盖图像分类，但底层设计已经按工业视觉平台的方向组织：

- 结构化 proposal
- 参数空间白名单与 policy 校验
- run / experiment 历史沉淀
- 自动继续搜索与停止规则
- 跨模型 baseline 比较
- 前后端联动的实验工作台

适用方向包括但不限于：

- 工业缺陷识别
- 质量检测与质量分析
- 产线视觉实验管理
- 需要保留实验依据与策略边界的模型迭代场景

### 当前阶段

项目当前的口径应该理解为：

- 平台定位：工业视觉实验平台
- 当前已完成任务类型：图像分类
- 当前公开前端：`React`
- 当前后端：`FastAPI`
- 当前训练框架：`PyTorch`
- 当前数据库：`SQLite`
- 当前已接入模型：`MobileNetV2`、`MobileNetV3 Small`、`GoogLeNet`、`ResNet18`
- 当前自动化入口：`Auto Train`、`Compare Models`

这意味着“分类”是当前实施范围，不是项目最终边界。

### 建议插图

#### 图 1. 平台总览图

> TODO(screenshot placeholder)
> Replace this block with: `![Platform Overview](docs/assets/readme/platform-overview.png)`
> Suggested image:
> A high-level platform overview that shows industrial vision inputs, the AutoVisionLab core loop, and research outputs.
> Suggested layout:
> left = industrial scenarios or project inputs
> center = proposal, policy validation, training, run history, decision support
> right = comparable results, artifacts, next-step recommendation

### 核心能力

#### 1. Structured Experiment Loop

系统把一次实验拆成稳定对象和稳定流程：

1. AI 生成结构化 proposal
2. 后端校验 proposal、参数空间和搜索权限
3. 训练器按结构化 config 执行实验
4. 数据库存储结果、决策和索引
5. 本地目录保存日志与 checkpoint
6. 前端展示 run、实验记录、趋势和下一步建议

#### 图 2. 实验闭环流程图

> TODO(screenshot placeholder)
> Replace this block with: `![Structured Experiment Loop](docs/assets/readme/experiment-loop.png)`
> Suggested image:
> A workflow diagram for the 6-step experiment loop.
> Suggested steps:
> AI proposal -> policy and parameter validation -> training execution -> structured result storage -> artifact persistence -> next-step suggestion
> Suggested note:
> Make validation and next-step decision visually prominent, because they are core product differentiators.

#### 2. Policy-Governed Search

AI 不直接改训练代码，只能在 policy 和 parameter space 允许的范围内搜索。

这使平台更适合工业场景下常见的要求：

- 搜索边界明确
- 实验可复现
- 变更原因可追踪
- 结果能回到 run 历史中解释

#### 3. Comparable Runs

平台强调 run 内可比性与跨模型比较的边界：

- 同一个 run 固定一个数据集和一个模型
- `Auto Train` 只负责单模型 run 内继续优化
- `Compare Models` 负责在统一 baseline 下做横向比较

#### 图 3. Compare Models vs Auto Train 关系图

> TODO(screenshot placeholder)
> Replace this block with: `![Compare Models vs Auto Train](docs/assets/readme/compare-vs-search.png)`
> Suggested image:
> A two-lane product flow that explains the difference between Compare Models and Auto Train.
> Suggested layout:
> left lane = Compare Models with shared baseline across candidate models
> center = user selects one model to continue
> right lane = Auto Train continues governed optimization within a single run
> Suggested caption inside image:
> Compare across models. Optimize within a run.

#### 4. Local-First Artifact Management

数据库保存结构化元数据，本地文件保存大体积训练产物：

- run 日志：`artifacts/runs/<run_id>.log`
- checkpoint：`artifacts/checkpoints/<experiment_id>.pt`

### 为什么这样设计

工业视觉研发往往不缺“再跑一次训练”的脚本，真正缺的是：

- 可被约束的实验生成方式
- 能长期积累的实验上下文
- 明确的搜索边界与停止条件
- 对团队成员可解释的结果沉淀

AutoVisionLab 想解决的是这部分基础设施问题。

#### 图 4. 工作台界面截图

> TODO(screenshot placeholder)
> Replace this block with: `![Experiment Workspace](docs/assets/readme/workspace-screenshot.png)`
> Suggested image:
> A screenshot or annotated mockup of the main workspace.
> Suggested visible regions:
> run list, experiment table, metrics or comparison chart, and AI suggestion or task status
> Suggested fallback:
> If the current UI is still evolving, use a labeled product mockup first and replace it with a real screenshot later.

### 快速开始

环境要求：

- `Python 3.10+`
- `Node.js 20+`

安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

前端依赖：

```bash
cd frontend/web
npm install
```

如需配置 AI provider：

```bash
cp .env.example .env
```

启动后端：

```bash
./scripts/run_backend.sh
```

启动前端：

```bash
./scripts/run_frontend.sh
```

默认地址：

- Backend: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:5173`

### 文档导航

建议按下面顺序阅读：

- 平台定位与边界：
  - [docs/plan.md](docs/plan.md)
- 当前实现进度：
  - [docs/tasks.md](docs/tasks.md)
- HTTP 接口：
  - [docs/api.md](docs/api.md)
- 运行规则与搜索规则：
  - [docs/policies/README.md](docs/policies/README.md)
- 结构化对象设计：
  - [docs/schemas/README.md](docs/schemas/README.md)

## English

### Overview

AutoVisionLab is an industrial vision experimentation platform, not a single-purpose image-classification app.

The current public implementation starts with image classification, but the platform is designed around a broader goal: turning model development into a structured, governed, and reviewable experiment loop for industrial vision workloads.

Core directions:

- Structured AI proposals
- Policy and parameter-space validation
- Persistent run and experiment history
- Automated follow-up search and stop rules
- Cross-model baseline comparison
- A web workspace for experiment operations

Typical target scenarios include:

- Defect inspection
- Quality analysis
- Vision experimentation for production environments
- Teams that need reproducibility, reviewability, and controlled search boundaries

### Current Status

- Platform positioning: industrial vision experiment platform
- Current implemented task type: image classification
- Frontend: `React`
- Backend: `FastAPI`
- Training stack: `PyTorch`
- Database: `SQLite`
- Integrated models: `MobileNetV2`, `MobileNetV3 Small`, `GoogLeNet`, `ResNet18`
- Automated entry points: `Auto Train`, `Compare Models`

Image classification is the current implementation scope, not the long-term product boundary.

### Core Principles

- Structured over ad hoc: trainers consume structured config, not free-form code edits
- Governed AI search: proposals must stay within allowed policy and parameter-space boundaries
- Comparable experiments: a run stays tied to one dataset and one model
- Clear storage split: metadata in the database, heavy artifacts on local disk

### Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd frontend/web
npm install
./scripts/run_backend.sh
./scripts/run_frontend.sh
```

Default endpoints:

- Backend: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:5173`

### Documentation

- Platform scope: [docs/plan.md](docs/plan.md)
- Implementation status: [docs/tasks.md](docs/tasks.md)
- API reference: [docs/api.md](docs/api.md)
- Policies: [docs/policies/README.md](docs/policies/README.md)
- Schemas: [docs/schemas/README.md](docs/schemas/README.md)
