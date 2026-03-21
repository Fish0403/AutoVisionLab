# 自主训练 Web 平台任务清单

## 1. 任务原则

这个文件只负责落地执行，不重复解释整体愿景。产品目标、架构、技术选型请看 [`plan.md`](/home/fish/AutoVisionLab/plan.md)。

当前开发原则：

- 第一版先做图像分类
- 第一版先支持 `MobileNet` 和 `GoogLeNet`
- 第一版所有实验都存档，不做回滚和 discard/keep 机制
- 第一版固定模型实现，只允许调整训练参数
- 所有可调参数必须结构化管理
- 先把“能跑、能看、能分析”做通，再增加自治程度

## 2. MVP 目标

MVP 需要完成这条最小闭环：

1. 用户在网页创建一个分类实验 run
2. 选择数据集和模型
3. 系统创建 experiment
4. worker 启动训练
5. 训练结果写入数据库和产物目录
6. 网页展示指标趋势图
7. 点击图上的点可以看到该次训练参数
8. AI 可以基于历史实验生成 proposal 和 reflection

MVP 不包含模型结构搜索，只做参数实验。

## 3. 开发阶段

### Phase 1: 项目骨架

状态：`已完成`

目标：

- 建好前后端目录
- 跑通本地开发环境
- 确定基础依赖

任务：

- 创建 `frontend/` 项目
- 创建 `backend/` 项目
- 初始化 `FastAPI`
- 初始化 `Streamlit`
- 初始化数据库连接
- 预留 `Redis` 配置入口
- 建立基础配置文件

完成标准：

- 前端能启动
- 后端能启动
- 数据库能连接
- 基础配置可本地加载

### Phase 2: 分类训练闭环

状态：`部分完成`

目标：

- 不依赖 AI，先手动跑通分类实验

任务：

- 实现 `CIFAR-10` 数据加载
- 封装统一分类 trainer 接口
- 接入 `MobileNetV2`
- 接入 `GoogLeNet`
- 固定模型实现，不开放结构改动入口
- 定义统一 `experiment config`
- 定义 `editable parameter space`
- 支持基础训练参数：
  - optimizer
  - learning_rate
  - batch_size
  - image_size
  - epochs
  - weight_decay
  - augmentation_level
  - scheduler
  - label_smoothing
  - aux_logits，仅 `GoogLeNet`
- 保存训练日志
- 保存 checkpoint
- 输出结构化结果 JSON

完成标准：

- 可以通过 API 创建一次实验
- worker 能成功训练 `MobileNetV2`
- worker 能成功训练 `GoogLeNet`
- 实验结果能保存到数据库

### Phase 3: 实验管理 API

状态：`部分完成`

目标：

- 让前端能查询和展示实验数据

任务：

- 设计并实现数据库表
- 提供 Run API
- 提供 Experiment API
- 提供指标趋势 API
- 提供单次实验详情 API
- 提供参数空间查询 API
- 提供日志和 checkpoint 索引 API
- 视需要补 `Project API`

完成标准：

- 前端能拿到实验列表
- 前端能拿到某个 run 的趋势数据
- 前端能拿到某个 experiment 的参数和结果

### Phase 4: 前端可视化

状态：`部分完成`

目标：

- 把实验结果看清楚

任务：

- 实现 Run 创建区
- 实现 Run 详情页
- 实现 Experiment 比较视图
- 实现指标趋势图组件
- 实现指标选择器
- 实现点选图表查看 experiment 详情
- 实现日志查看入口
- 实现追加实验入口
- 实现 proposal 带入表单入口

完成标准：

- 用户可以切换指标，比如 `train_loss`、`val_loss`、`top1_acc`
- 图表能显示趋势线
- 点击任一点可以看到该次训练参数

### Phase 5: AI Proposal

状态：`部分完成`

目标：

- 让 AI 基于历史实验生成下一轮实验方案

任务：

- 定义 proposal schema
- 定义 editable parameter space schema
- 实现 LLM 调用封装
- 设计 proposal prompt
- 做 proposal schema 校验
- 将 proposal 转换为 experiment 配置

完成标准：

- 给定历史实验后，系统能生成合法 proposal
- proposal 能被后端转换成可执行 experiment

### Phase 6: AI Reflection

状态：`未开始`

目标：

- 让 AI 对实验结果做结构化分析

任务：

- 定义 reflection schema
- 设计 reflection prompt
- 将训练结果喂给 LLM
- 保存 reflection 到数据库

完成标准：

- 每次实验结束后都有 reflection 记录
- 前端能看到 AI 的分析结果

### Phase 7: 半自治循环

状态：`未开始`

目标：

- 系统自动推进下一轮实验

任务：

- 增加 run 调度逻辑
- 根据历史实验自动生成下一轮 proposal
- 支持暂停和恢复
- 支持手动审批 proposal

完成标准：

- 在开启自动模式时，run 能自动创建新 experiment
- 用户可以随时暂停

## 4. 数据库任务

第一版建议至少建这些表：

- `projects`
- `runs`
- `experiments`
- `proposals`
- `results`
- `reflections`
- `artifacts`

如果不想单独建表，也至少要保证下面两类数据能持久化：

- experiment config
- editable parameter space

每张表需要补这些工作：

- 字段设计
- 主外键关系
- 创建时间和更新时间
- 状态字段
- 索引设计

优先级：

1. `runs`
2. `experiments`
3. `results`
4. `proposals`
5. `reflections`
6. `artifacts`
7. `projects`

## 5. 后端任务

### API 层

- `GET /runs`
- `POST /runs`
- `GET /runs/{run_id}`
- `GET /runs/{run_id}/metrics`
- `POST /runs/{run_id}/proposal`
- `POST /runs/proposal/test`
- `POST /experiments`
- `GET /experiments/{experiment_id}`
- `POST /experiments/{experiment_id}/train`
- `POST /experiments/{experiment_id}/result`
- `GET /models/{model_name}/parameter-space`
- `POST /experiments/{experiment_id}/retry`
  当前尚未实现，可在 `train` 语义稳定后补齐
- `POST /runs/{run_id}/pause`
- `POST /runs/{run_id}/resume`

### Service 层

- run 创建服务
- experiment 创建服务
- 指标聚合服务
- proposal 生成服务
- 参数空间校验服务
- reflection 生成服务
- run 调度服务

### Worker 层

- experiment 执行 worker
- 日志落盘
- checkpoint 落盘
- 状态更新
- 异常处理

## 6. 前端任务

### 页面

- Run 创建区
- Run 列表页
- Run 详情页
- Experiment 详情页

### 组件

- 指标选择器
- 趋势折线图
- experiment 参数面板
- AI proposal 卡片
- AI reflection 卡片
- 状态标签
- 日志查看器入口

### 关键交互

- 切换指标时刷新折线图
- 点击图上点位时加载 experiment 详情
- 在右侧面板展示训练参数和指标

## 7. Trainer 任务

### 通用部分

- base trainer 抽象
- 配置解析
- 参数白名单校验
- 指标记录
- checkpoint 保存
- 训练结果 JSON 输出

### MobileNet

- `mobilenet_v2` 初始化
- 参数映射
- 训练配置适配
- 不开放结构改动

### GoogLeNet

- `googlenet` 初始化
- `aux_logits` 配置支持
- 训练配置适配
- 不开放结构改动

## 8. 参数管理任务

第一版把“参数配置”和“参数可编辑范围”分开。

任务：

- 定义 `experiment config` schema
- 定义 `editable parameter space` schema
- 为 `mobilenet_v2` 建参数空间
- 为 `googlenet` 建参数空间
- 对 proposal 做参数空间校验
- 将最终生效参数完整存档

建议首批开放参数：

- optimizer
- learning_rate
- batch_size
- image_size
- epochs
- weight_decay
- scheduler
- augmentation_level
- label_smoothing
- aux_logits，仅 `googlenet`

## 9. 指标展示任务

第一版图表层要支持通用指标，不要为某个任务写死。

任务：

- 后端返回可选指标列表
- 前端根据指标列表生成 selector
- 图表组件根据选择的字段动态渲染
- 每个点都带 `experiment_id`

点选后展示：

- 基础信息
- 指标信息
- 参数信息
- proposal
- reflection

## 10. 里程碑

### Milestone 1

训练平台可用：

- 能新建 run
- 能跑 `MobileNetV2`
- 能跑 `GoogLeNet`
- 能保存实验结果
- 能保存结构化训练参数

### Milestone 2

实验展示可用：

- 能看实验列表
- 能看趋势图
- 能点选图表查看参数

### Milestone 3

AI proposal 可用：

- 能生成结构化 proposal
- 能将 proposal 变成 experiment
- proposal 不能超出参数白名单

### Milestone 4

AI reflection 可用：

- 能自动分析实验结果
- 能在前端展示分析内容

### Milestone 5

半自治运行可用：

- 能自动推进下一轮实验
- 能暂停和恢复

## 11. 当前建议的第一批实际任务

按顺序先做这些：

1. `已完成` 建 `backend/` 和 `frontend/` 骨架
2. `已完成` 建数据库表：`runs`、`experiments`、`results`
3. `已完成` 定义 `experiment config` 和 `editable parameter space`
4. `已完成` 实现分类 trainer 基类
5. `已完成` 接入 `MobileNetV2`
6. `已完成` 接入 `GoogLeNet`
7. `已完成` 做参数空间校验
8. `已完成` 做创建 run / experiment 的 API
9. `已完成` 做结果查询 API
10. `已完成` 做前端趋势图页
11. `已完成` 做点击点位查看参数详情
12. `进行中` 接通真实 proposal 到追加实验流程
13. `待做` 落地 reflection 生成与展示
14. `待做` 增加 pause / resume 与半自治调度
