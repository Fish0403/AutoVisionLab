# AutoVisionLab

AutoVisionLab 是一个面向图像分类实验的自主训练 Web 平台。

它的目标不是让 AI 自由改训练代码，而是把实验流程做成结构化闭环：

1. AI 生成结构化 proposal
2. 后端校验 proposal、参数空间和搜索权限
3. 训练器按结构化 config 执行实验
4. 数据库保存结构化结果
5. 本地目录保存日志与 checkpoint
6. 前端展示 run、实验记录、趋势和 AI 建议

当前实现：

- 任务类型：图像分类
- 前端：`Streamlit`
- 后端：`FastAPI`
- 数据库：`SQLite`
- 训练框架：`PyTorch`
- 可选模型：
  - `MobileNetV2`
  - `GoogLeNet`
  - `ResNet18`
  - `ResNet34`
  - `DenseNet121`

## 1. 环境要求

- `Python 3.10+`
- 建议使用独立虚拟环境

## 2. 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

如果需要配置 AI provider：

```bash
cp .env.example .env
```

## 3. 启动

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
- Frontend: `http://localhost:8501`

## 4. 数据目录规则

平台统一读取标准分类目录：

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

说明：

- `raw/` 存原始下载内容
- `classification/train|val/` 存可直接训练的数据
- trainer 不直接依赖原始压缩包格式

## 5. 数据准备

### CIFAR-10

将 `CIFAR-10` 整理成统一分类目录：

```bash
python3 data/cifar10/prepare_classification_split.py
```

默认会导出平衡抽样子集：

- `train`: 每类 `500`
- `val`: 每类 `100`

如果需要导出全量：

```bash
python3 data/cifar10/prepare_classification_split.py --full
```

输出目录：

```text
data/
  cifar10/
    raw/
    classification/
      train/
      val/
```

### NEU-CLS

`NEU-CLS` 是较适合本项目的工业分类数据集，常见描述为：

- `6` 类
- 每类 `300` 张
- 总计 `1800` 张
- 图像大小 `200x200`

下载后可用以下脚本整理为统一分类目录：

```bash
python3 data/neu-cls/prepare_classification_split.py
```

输出目录：

```text
data/
  neu-cls/
    raw/
    classification/
      train/
      val/
```

## 6. 本地产物

当前本地产物规则：

- run 日志：
  - `artifacts/runs/<run_id>.log`
- checkpoint：
  - `artifacts/checkpoints/<experiment_id>.pt`

run 日志会聚合同一个 run 下的：

- 实验创建
- AI proposal
- 训练开始
- epoch 摘要
- 最终结果摘要

## 7. 常用文档

- 项目计划：[docs/plan.md](/home/fish/AutoVisionLab/docs/plan.md)
- 任务清单：[docs/tasks.md](/home/fish/AutoVisionLab/docs/tasks.md)
- Schema 说明：[docs/schemas.md](/home/fish/AutoVisionLab/docs/schemas.md)
- 运行规则：[docs/experiment_policy.md](/home/fish/AutoVisionLab/docs/experiment_policy.md)
- Ranking Policy：[docs/ranking_policy.md](/home/fish/AutoVisionLab/docs/ranking_policy.md)
- Run 晋级与回退规则：[docs/run_promotion_policy.md](/home/fish/AutoVisionLab/docs/run_promotion_policy.md)
- Auto Train 搜索策略：[docs/auto_train_search_policy.md](/home/fish/AutoVisionLab/docs/auto_train_search_policy.md)
- Auto Train 停止策略：[docs/auto_train_stop_policy.md](/home/fish/AutoVisionLab/docs/auto_train_stop_policy.md)
- 变更记录：[CHANGELOG.md](/home/fish/AutoVisionLab/CHANGELOG.md)

## 8. 仓库结构

```text
AutoVisionLab/
├── backend/
│   └── app/
│       ├── api/            # FastAPI 路由
│       ├── config_spaces/  # 模型参数空间白名单
│       ├── db/             # 数据库配置与 session
│       ├── models/         # SQLAlchemy ORM 模型
│       ├── schemas/        # Pydantic schema
│       ├── services/       # proposal、持久化、auto-train 等业务逻辑
│       ├── trainers/       # 各分类模型 trainer
│       └── workers/        # experiment 执行入口
├── frontend/
│   └── streamlit_app.py    # Streamlit 前端入口
├── data/                   # 数据集与准备脚本
├── artifacts/              # 日志、checkpoint 和其他训练产物
├── docs/                   # 补充说明文档
├── scripts/                # 本地启动脚本
├── README.md
├── CHANGELOG.md
└── requirements.txt
```
