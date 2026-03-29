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
- `v1` base backbone：`MobileNetV3 Small`
- 前端：`Streamlit`
- 后端：`FastAPI`
- 数据库：`SQLite`
- 训练框架：`PyTorch`
- 当前训练主配置已收敛到：
  - `model_recipe`
  - `train_hyp`
  - `dataset_recipe`
- 训练结果仍保留 `params` 摘要，作为现有结果存储与搜索策略的兼容输出
- 可选模型：
  - `MobileNetV3 Small`
  - `GoogLeNet`
  - `ResNet18`

当前演进方向：

- `v1` 先把分类 + `MobileNetV3 Small` 的 recipe 链路做完整
- 底层 schema / parser / registry 从现在开始为后续目标检测和分割预留统一抽象

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

数据相关的处理细节、脚本说明和数据集示例统一放在：

- [data/README.md](data/README.md)

## 5. 数据准备

常用数据命令：

```bash
python3 data/prepare_classification_split.py --source-dir data/raw/KDSC --dataset-name KDSC
python3 data/raw/neu/prepare_classification_source.py --force
python3 data/prepare_classification_split.py --dataset-name neu --force
```

更详细的数据目录、manifest 格式和 `neu` 预处理流程见：

- [data/README.md](data/README.md)

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

- 项目计划：[docs/plan.md](docs/plan.md)
- API 文档：[docs/api.md](docs/api.md)
- 任务清单：[docs/tasks.md](docs/tasks.md)
- Schema 总览：[docs/schemas/README.md](docs/schemas/README.md)
- 策略总览：[docs/policies/README.md](docs/policies/README.md)
- Model Recipe 设计：[docs/schemas/model_recipe_schema.md](docs/schemas/model_recipe_schema.md)
- Train Hyp 设计：[docs/schemas/train_hyp_schema.md](docs/schemas/train_hyp_schema.md)
- Dataset Recipe 设计：[docs/schemas/dataset_recipe_schema.md](docs/schemas/dataset_recipe_schema.md)
- 运行规则：[docs/policies/experiment_policy.md](docs/policies/experiment_policy.md)
- Ranking Policy：[docs/policies/ranking_policy.md](docs/policies/ranking_policy.md)
- Run 晋级与回退规则：[docs/policies/run_promotion_policy.md](docs/policies/run_promotion_policy.md)
- Auto Train 搜索策略：[docs/policies/auto_train_search_policy.md](docs/policies/auto_train_search_policy.md)
- Auto Train 停止策略：[docs/policies/auto_train_stop_policy.md](docs/policies/auto_train_stop_policy.md)
- 变更记录：[CHANGELOG.md](CHANGELOG.md)

## 8. 仓库结构

```text
AutoVisionLab/
├── backend/
│   └── app/
│       ├── api/            # FastAPI 路由
│       ├── config_spaces/  # 当前模型参数空间白名单
│       ├── db/             # 数据库配置与 session
│       ├── models/         # SQLAlchemy ORM 模型
│       ├── schemas/        # Pydantic schema，含 ExperimentConfig / recipe 相关对象
│       ├── services/       # proposal、持久化、auto-train、run policy 等业务逻辑
│       ├── trainers/       # 当前分类 trainer；训练主链路已切到 recipe / train_hyp 驱动
│       └── workers/        # experiment 执行入口
├── frontend/
│   └── streamlit_app.py    # Streamlit 前端入口
├── data/                   # 数据集与准备脚本
├── artifacts/              # 日志、checkpoint 和其他训练产物
├── docs/                   # 补充说明文档与 recipe 设计稿
├── scripts/                # 本地启动脚本
├── README.md
├── CHANGELOG.md
└── requirements.txt
```
