# Changelog

## 2026-03-26

- 新增统一的 run policy 层：[`backend/app/schemas/run_policy.py`](/home/fish/AutoVisionLab/backend/app/schemas/run_policy.py) 负责声明晋级阈值、停滞轮数、单变量 / 双变量预算等硬约束默认值，[`backend/app/services/run_policy.py`](/home/fish/AutoVisionLab/backend/app/services/run_policy.py) 负责承接这些约束的计算逻辑；同时将 [`backend/app/services/proposal_service.py`](/home/fish/AutoVisionLab/backend/app/services/proposal_service.py)、[`backend/app/services/auto_train_service.py`](/home/fish/AutoVisionLab/backend/app/services/auto_train_service.py) 与 [`backend/app/services/persistence.py`](/home/fish/AutoVisionLab/backend/app/services/persistence.py) 改为统一从这层读取，减少策略散落。
- 调整 [`backend/app/services/persistence.py`](/home/fish/AutoVisionLab/backend/app/services/persistence.py)，为 run 引入明确的晋级 / 回退规则：不再把“当前排名第一”直接视为 `keep`，而是要求新实验至少跨过最小提升阈值后才晋级 `best / frontier`；若未达到阈值，则自动记为 `discard` 并回退到当前最佳分支，减少噪声实验误晋级。
- 同步把 run 晋级阈值进一步收紧为：`top1_acc` 至少提升 `0.01`，或在 `top1_acc` 近似持平时 `val_loss` 至少下降 `0.01`，避免把过小波动误判成有效晋级。
- 调整 [`backend/app/services/proposal_service.py`](/home/fish/AutoVisionLab/backend/app/services/proposal_service.py) 与 [`backend/app/services/auto_train_service.py`](/home/fish/AutoVisionLab/backend/app/services/auto_train_service.py)，把 proposal 默认收敛为单变量实验；只有当 run 已连续至少两轮没有晋级时，才允许最多两个字段的组合变更，同时在 auto-train 日志里显式记录每轮是“晋级”还是“回退”。
- 新增 [`tests/test_run_promotion_policy.py`](/home/fish/AutoVisionLab/tests/test_run_promotion_policy.py)，覆盖“小幅提升不晋级”“显著提升才晋级”“停滞两轮后放宽到双变量”三类关键策略测试，防止后续回归。
- 把统一 JSON 响应信封从 `runs` 主链路继续扩到 [`backend/app/api/routes/experiments.py`](/home/fish/AutoVisionLab/backend/app/api/routes/experiments.py) 与 [`backend/app/api/routes/models.py`](/home/fish/AutoVisionLab/backend/app/api/routes/models.py)，让创建/读取/更新实验、训练启动/停止，以及模型参数空间接口都返回稳定的 `ok / code / message / data / errors / meta` 结构；同时把 [`backend/app/main.py`](/home/fish/AutoVisionLab/backend/app/main.py) 的 `/health` 也统一到相同外层，减少系统接口的例外形状。
- 同步更新 [`schemas.md`](/home/fish/AutoVisionLab/schemas.md)、[`tasks.md`](/home/fish/AutoVisionLab/tasks.md) 与 [`docs/cli_anything_learning.md`](/home/fish/AutoVisionLab/docs/cli_anything_learning.md)，把 API 响应收敛状态从“`runs` 主链路已完成第一轮收敛”更新为“主要现有 API 已统一成功响应外层，后续重点转向 `ArtifactManifest` 与固定验收文档”，避免文档继续落后于实现。

## 2026-03-25

- 统一 `runs` 相关 FastAPI 接口的 JSON 成功响应外层，新增通用 `ApiResponse` 信封与异常处理器，让 `list/create/get run`、`run summary/metrics`、`proposal`、`auto-train`、`reset` 等接口都返回稳定的 `ok / code / message / data / errors / meta` 结构；同时调整 [`frontend/streamlit_app.py`](/home/fish/AutoVisionLab/frontend/streamlit_app.py) 的请求封装层，自动解包成功响应并归一化错误响应，避免页面逻辑直接依赖裸 JSON 形状。
- 同步更新 [`docs/cli_anything_learning.md`](/home/fish/AutoVisionLab/docs/cli_anything_learning.md)、[`schemas.md`](/home/fish/AutoVisionLab/schemas.md) 与 [`tasks.md`](/home/fish/AutoVisionLab/tasks.md)，把“统一 JSON 响应信封”的文档口径从纯待落地改为“`runs` 主链路已完成第一轮收敛，其余接口与 `ArtifactManifest` 仍待继续推进”，避免文档与当前代码状态不一致。

## 2026-03-24

- 重写现行文档 [`README.md`](/home/fish/AutoVisionLab/README.md)、[`plan.md`](/home/fish/AutoVisionLab/plan.md)、[`tasks.md`](/home/fish/AutoVisionLab/tasks.md)、[`schemas.md`](/home/fish/AutoVisionLab/schemas.md) 与 [`docs/experiment_policy.md`](/home/fish/AutoVisionLab/docs/experiment_policy.md)，清理过时的“第一阶段 / 第一版 / Phase / MVP”叙事，统一为当前实现、当前规则和当前任务状态的口径；历史演进继续保留在 `CHANGELOG` 中。
- 调整本地产物日志组织方式：分类训练日志改为按 `run` 聚合，统一写入 `artifacts/runs/<run_id>.log`，并在其中追加实验创建、AI proposal、训练开始、epoch 摘要和最终结果；checkpoint 继续保留为 `artifacts/checkpoints/<experiment_id>.pt`。对应更新见 [`backend/app/services/run_logging.py`](/home/fish/AutoVisionLab/backend/app/services/run_logging.py)、[`backend/app/trainers/classification/base_trainer.py`](/home/fish/AutoVisionLab/backend/app/trainers/classification/base_trainer.py)、[`backend/app/services/training_runner.py`](/home/fish/AutoVisionLab/backend/app/services/training_runner.py)、[`backend/app/services/proposal_service.py`](/home/fish/AutoVisionLab/backend/app/services/proposal_service.py) 与 [`backend/app/services/persistence.py`](/home/fish/AutoVisionLab/backend/app/services/persistence.py)。
- 调整 [`backend/app/services/auto_train_service.py`](/home/fish/AutoVisionLab/backend/app/services/auto_train_service.py) 与 [`backend/app/services/proposal_service.py`](/home/fish/AutoVisionLab/backend/app/services/proposal_service.py)，为 `Auto Train` 增加按总轮数比例切换的阶段策略：前 `50%` 轮允许基础超参数主导，后 `50%` 轮 proposal 必须包含至少一个 augmentation / loss / strategy 字段变化，避免自动调优后半程一直重复搜索同一组基础超参数。
- 调整 [`backend/app/services/proposal_service.py`](/home/fish/AutoVisionLab/backend/app/services/proposal_service.py)，补全 proposal prompt 中可搜索的 augmentation / loss 字段 schema，并增加“至少一个有效参数变更”“禁止文本引用未开放选项”的校验与一次自动重试，避免出现“建议里写了未开放策略但 changes 为空”的无效 proposal。
- 更新 [`docs/experiment_policy.md`](/home/fish/AutoVisionLab/docs/experiment_policy.md)，补充 `Auto Train` 的阶段式搜索规则，明确总轮数前后两段的搜索边界。
- 调整数据增强参数设计：移除前后端运行路径中的 `augmentation_level`，将基础增强统一收敛到 `augmentation_policy=none|basic`，额外增强继续保留 `mixup_alpha`、`cutmix_alpha`、`random_erasing_prob`。对应更新见 [`backend/app/schemas/parameter_space.py`](/home/fish/AutoVisionLab/backend/app/schemas/parameter_space.py)、[`backend/app/trainers/classification/components/augmentations.py`](/home/fish/AutoVisionLab/backend/app/trainers/classification/components/augmentations.py)、[`backend/app/config_spaces/classification.py`](/home/fish/AutoVisionLab/backend/app/config_spaces/classification.py)、[`backend/app/services/proposal_service.py`](/home/fish/AutoVisionLab/backend/app/services/proposal_service.py) 与 [`frontend/streamlit_app.py`](/home/fish/AutoVisionLab/frontend/streamlit_app.py)。
- 调整 [`data/cifar10/prepare_classification_split.py`](/home/fish/AutoVisionLab/data/cifar10/prepare_classification_split.py)，默认按类别平衡抽样导出 `CIFAR-10`，当前默认 `train=500/class`、`val=100/class`，同时支持 `--full` 导出全量数据，避免开发期生成过大的图片目录。
- 调整 [`backend/app/trainers/classification/base_trainer.py`](/home/fish/AutoVisionLab/backend/app/trainers/classification/base_trainer.py)，分类训练器改为统一只读取 `data/<dataset>/classification/train|val` 目录，不再内置 `CIFAR-10` 特判下载逻辑，简化数据读取层。
- 新增 [`data/cifar10/prepare_classification_split.py`](/home/fish/AutoVisionLab/data/cifar10/prepare_classification_split.py)，用于把 `CIFAR-10` 整理成统一的 `classification/train/<class>` 与 `classification/val/<class>` 结构，使其与 `NEU-CLS` 一样走同一套训练入口。
- 更新 [`README.md`](/home/fish/AutoVisionLab/README.md) 与 [`docs/experiment_policy.md`](/home/fish/AutoVisionLab/docs/experiment_policy.md)，将统一数据目录规则进一步收敛为按任务类型分层，当前分类任务使用 `data/<dataset>/classification/train|val/<class>`，原始下载内容保留在 `raw/`，为后续 `detection/`、`segmentation/` 扩展预留空间。
- 新增 [`data/neu-cls/prepare_classification_split.py`](/home/fish/AutoVisionLab/data/neu-cls/prepare_classification_split.py)，用于把当前 `NEU-CLS` 解压后的 `images/` 目录按文件名前缀整理成 `classification/train/<class>` 与 `classification/val/<class>` 结构，便于后续按分类任务接入。
- 更新 [`README.md`](/home/fish/AutoVisionLab/README.md)，新增“项目结构”章节，说明根目录及 `backend/`、`frontend/`、`data/`、`artifacts/`、`scripts/`、`docs/` 等目录的职责，方便后续维护和新人阅读。
- 新增 [`docs/experiment_policy.md`](/home/fish/AutoVisionLab/docs/experiment_policy.md)，集中整理实验运行规则，明确模型与 run 的关系、AI 搜索参数白名单、`image_size` 搜索逻辑、前端显示边界，以及数据库 / 本地文件的落盘职责。
- 新增根目录 [`README.md`](/home/fish/AutoVisionLab/README.md)，补充项目简介、环境安装、前后端启动方式、当前 `CIFAR-10` 数据准备说明，以及推荐工业分类数据集 `NEU-CLS` 的下载链接、目录规划和接入前准备建议。
- 扩展分类模型白名单：新增 [`resnet18`](/home/fish/AutoVisionLab/backend/app/trainers/classification/resnet_trainer.py)、[`resnet34`](/home/fish/AutoVisionLab/backend/app/trainers/classification/resnet_trainer.py) 和 [`densenet121`](/home/fish/AutoVisionLab/backend/app/trainers/classification/densenet_trainer.py) 三个常用预设网络，作为可选模型接入平台，不开放网络结构搜索。
- 调整 [`backend/app/config_spaces/classification.py`](/home/fish/AutoVisionLab/backend/app/config_spaces/classification.py)、[`backend/app/services/parameter_space.py`](/home/fish/AutoVisionLab/backend/app/services/parameter_space.py)、[`backend/app/workers/experiment_worker.py`](/home/fish/AutoVisionLab/backend/app/workers/experiment_worker.py) 与 [`frontend/streamlit_app.py`](/home/fish/AutoVisionLab/frontend/streamlit_app.py)，同步补齐新模型的参数空间、执行入口和前端下拉选项。
- 调整 [`frontend/streamlit_app.py`](/home/fish/AutoVisionLab/frontend/streamlit_app.py) 的 `Image Size` 交互：默认显示数据集原图大小；当用户手动改成非原图大小时，AI 只允许在“原图大小到当前设置值之间”的离散范围内搜索，而不是在全量 `image_size` 候选上搜索。
- 调整 [`backend/app/schemas/parameter_space.py`](/home/fish/AutoVisionLab/backend/app/schemas/parameter_space.py) 与 [`backend/app/services/parameter_space.py`](/home/fish/AutoVisionLab/backend/app/services/parameter_space.py)，将 `search_policy` 从粗粒度开关细化为“开关 + 字段白名单”组合；当前默认允许 AI 搜索 `learning_rate`、`batch_size`、`optimizer`、`weight_decay`、`scheduler`、`augmentation_level`、`label_smoothing`，继续默认禁止 `epochs` 与 `image_size`。
- 调整 [`backend/app/services/proposal_service.py`](/home/fish/AutoVisionLab/backend/app/services/proposal_service.py)，proposal prompt 会显式告诉 AI 当前 run 允许修改的字段以及当前 run 保存下来的 `image_size` 可选范围，并在返回后按该白名单和最新实验参数空间清洗、校验 proposal，避免 UI 已收窄但后端仍放行未授权字段。
- 在 [`tasks.md`](/home/fish/AutoVisionLab/tasks.md) 中补充 `Training Setup` 参数收敛方向，明确主表单应继续减少低价值手动参数，把 `optimizer`、`weight_decay`、`scheduler`、`augmentation_level`、`label_smoothing` 默认下沉到 AI 搜索策略层。
- 在 [`tasks.md`](/home/fish/AutoVisionLab/tasks.md) 中补充 `learning_rate` 与 `batch_size` 的搜索定位：`learning_rate` 作为默认开放的第一优先级搜索项，`batch_size` 允许在有限离散值内搜索，但优先级低于 `learning_rate`。
- 在 [`tasks.md`](/home/fish/AutoVisionLab/tasks.md) 中补充 `image_size` 的阶段性策略：不把它当作普通超参数默认开放搜索；当前以 `CIFAR-10` 默认固定 `32` 为主，若后续开放，也应先限制在小范围离散值内，并按模型算力成本区别对待。

## 2026-03-23

- 调整 [`frontend/streamlit_app.py`](/home/fish/AutoVisionLab/frontend/streamlit_app.py) 的训练面板，主表单继续只保留高频训练参数，不直接展开 loss / augmentation 细项。
- 前端新增 `AI Search Policy` 折叠区，用于控制是否允许 AI 自动搜索普通超参数、训练策略、loss 与 augmentation，以及高影响改动是否需要人工审批。
- 在 [`backend/app/schemas/parameter_space.py`](/home/fish/AutoVisionLab/backend/app/schemas/parameter_space.py) 中新增 `SearchPolicy`，并将其纳入 `ExperimentConfig`，使搜索权限可以随实验配置一起持久化。
- 结果区会显示当前实验记录的 AI 搜索权限摘要，便于回溯自动调优时允许 AI 探索到什么范围。

- 新增 [`backend/app/trainers/classification/components/losses.py`](/home/fish/AutoVisionLab/backend/app/trainers/classification/components/losses.py)，实现结构化 loss registry，当前提供 `cross_entropy`、`cross_entropy_with_label_smoothing`、`focal_loss`。
- 新增 [`backend/app/trainers/classification/components/augmentations.py`](/home/fish/AutoVisionLab/backend/app/trainers/classification/components/augmentations.py)，实现结构化 augmentation registry，当前提供 `basic`、`strong`、`autoaugment_cifar`，并支持 `mixup`、`cutmix`、`random erasing`。
- 调整 [`backend/app/trainers/classification/base_trainer.py`](/home/fish/AutoVisionLab/backend/app/trainers/classification/base_trainer.py)，训练主循环改为通过固定组件层构建 loss 与 augmentation，而不是把策略硬编码在 trainer 内部。
- 扩展 [`backend/app/schemas/parameter_space.py`](/home/fish/AutoVisionLab/backend/app/schemas/parameter_space.py) 与 [`backend/app/config_spaces/classification.py`](/home/fish/AutoVisionLab/backend/app/config_spaces/classification.py)，为 `loss_name`、`augmentation_policy`、`mixup_alpha`、`cutmix_alpha`、`random_erasing_prob`、`focal_gamma` 预留结构化配置入口。
- 本次更新只完成后端算法组件和结构化参数入口，尚未接入前端表单、proposal 审批流和 `manual approval` 执行链路。

- 在 [`plan.md`](/home/fish/AutoVisionLab/plan.md) 中补充“结构化训练策略与分级审批”设计，明确 loss、augmentation、sampler 等能力应通过白名单 registry 接入，而不是让 AI 直接改训练代码。
- 在 [`tasks.md`](/home/fish/AutoVisionLab/tasks.md) 中新增 `Phase 9: 结构化训练策略与审批流`，拆分 schema、registry、proposal 审批状态和前端审批入口等任务。
- 在 [`schemas.md`](/home/fish/AutoVisionLab/schemas.md) 中补充下一阶段策略型参数设计，增加 `approval_level` / `component_group` 概念，并补充 `loss_name`、`augmentation_policy` 等结构化字段示例。

## 2026-03-23

- 调整 [`backend/app/services/proposal_service.py`](/home/fish/AutoVisionLab/backend/app/services/proposal_service.py) 的 proposal prompt 构造逻辑，不再只向 AI 发送 latest experiment。
- proposal 上下文改为包含完整 run 历史摘要，按时间顺序提供每轮实验的状态、decision、指标、训练耗时、参数快照和 proposal 摘要。
- proposal prompt 现在显式提供 run 的 `baseline_experiment_id`、`best_experiment_id`、`frontier_experiment_id`，要求 AI 结合整体历史而不是只盯最后一轮。
- prompt 中新增约束：`based_on_experiment_ids` 必须填写实际参考的实验 id，可包含多个，以便后续回溯 proposal 的依据。
- 调整 [`backend/app/services/auto_train_service.py`](/home/fish/AutoVisionLab/backend/app/services/auto_train_service.py) 的 follow-up 分支选择策略，auto-train 追加实验时优先遵循 proposal 的 `based_on_experiment_ids`。
- 当 proposal 未给出可用来源时，auto-train 现在按 `frontier_experiment_id` -> `best_experiment_id` -> 当前 run 最新实验的顺序回退，而不再默认沿着刚完成的上一轮继续追加。
- 后台 auto-train 日志会显式记录每一轮是从哪个 experiment 分支出去，便于排查 proposal 依据与执行路径是否一致。

## 2026-03-21

- 在 [`plan.md`](/home/fish/AutoVisionLab/plan.md) 中补充下一阶段优先增强方案，明确 `baseline_experiment_id`、`best_experiment_id`、`frontier_experiment_id` 三类 run 级锚点。
- 在 [`plan.md`](/home/fish/AutoVisionLab/plan.md) 中补充 experiment 决策层设计，区分训练状态与研究决策状态，新增 `keep` / `discard` / `crash` / `timeout` 的结构化语义。
- 在 [`plan.md`](/home/fish/AutoVisionLab/plan.md) 中补充统一排名与保留策略设计，并明确当前阶段继续以固定 `epochs` 为主，不额外引入固定时长预算。
- 在 [`tasks.md`](/home/fish/AutoVisionLab/tasks.md) 中新增 `Phase 8: 研究决策与可比性增强`，拆分 run 指针、experiment 决策层、排序规则与前端展示任务。
- 在 [`tasks.md`](/home/fish/AutoVisionLab/tasks.md) 中同步修正“暂不做 discard/keep 机制”的旧表述，改为下一阶段应补齐结构化决策层，避免规划与当前方向冲突。
- 本次更新属于设计记录与任务规划收敛，尚未实现对应后端 schema、API、数据库迁移和前端交互。

## 2026-03-20

- 初始化 `backend/` 和 `frontend/` 目录骨架，保持第一版只覆盖分类任务。
- 后端预留 `experiment config`、`editable parameter space`、`proposal`、`result`、`reflection` 的严格结构化 schema。
- 后端增加最小 API 路由占位：`/runs`、`/runs/{run_id}`、`/runs/{run_id}/metrics`、`/experiments/{experiment_id}`、`/models/{model_name}/parameter-space`。
- 为 `mobilenet_v2` 和 `googlenet` 建立静态参数空间白名单，未开放任何模型结构改动入口。
- 前端预留 Run 列表、Run 详情、Experiment 详情和趋势图页面，趋势图已支持点选点位并展示参数快照。
- 默认数据库方案明确收敛为 `SQLite` demo 模式，并补充根目录 `requirements.txt` 方便安装依赖。
- 新增最小数据库初始化脚本和 `runs`、`experiments`、`results` 三张基础表定义。
- `POST /runs` 和 `POST /experiments` 已接入 SQLite 持久化，不再只是返回静态占位数据。
- 新增 `POST /experiments/{experiment_id}/result`，训练结果可写回数据库，趋势图接口可以开始读取真实 metrics。
- 展示层方案从 `Next.js` 切换为 `Streamlit`，删除原前端骨架，新增单文件 demo UI。
- 新增 `scripts/run_backend.sh` 和 `scripts/run_frontend.sh`，统一本地启动方式。
- Streamlit 首页改为创建流程，支持先选模型、数据集和结构化训练参数，再创建 run 并排入首个 experiment。
- Streamlit 布局调整为“侧边栏创建 + 主区两列详情”，并明确当前仅支持 queued，不代表真实训练已启动。
- 训练改为后台执行；训练期间创建按钮禁用，主界面保持可用，并在训练结束后自动刷新状态。
- `Run Detail` 改为以实验对比为主，新增 comparison 表，将多次训练的状态、指标和关键参数并排展示。
- `Run Detail` 新增 experiment 多选比较，默认全选当前 run 下所有实验，也可手动筛选部分实验做趋势和参数对比。
- 去掉单独的 `Selected Experiment` 下拉，改为在 comparison 表内直接用 `Compare` / `Inspect` 列控制对比和右侧详情。
- 手工入口语义已调整为：`Manual New Run` 新建 run，`Append Experiment` 在已有 run 下追加参数实验；为后续 LLM proposal 追加实验预留了流程入口。
- 新增最小假的 `LLM proposal` 入口：可基于当前选中 run 的最新实验生成结构化 proposal，并自动带入追加实验表单。
- 新增根目录 `.env` / `.env.example`，将 AI provider 本地配置落盘，并把 `.env` 加入 `.gitignore`。
- 真实 proposal provider 从 `OpenRouter` 切换为 `AIHubMix`，默认通过 `https://aihubmix.com/v1` 的 OpenAI-compatible 接口生成 proposal。
- Streamlit 主页面重构为左右双栏：左侧专注 run 选择与训练记录筛选，右侧专注参数编辑与动作区，不再使用侧边栏训练表单。
- 右侧训练动作收敛为两个主按钮：`按当前参数训练` 与 `以当前页面为参考通过AI多次训练测试`；AI 多轮测试会先创建基线 run，再在同一 run 下连续追加实验。
- 同步更新 `tasks.md` 与 `plan.md` 中的过时描述，统一前端方案为 `Streamlit`，修正分类指标示例、仓库内文档链接和当前阶段状态。
- Streamlit 主界面调整为“左侧训练，右侧结果”，训练区只保留 `Train` 和 `Auto Train` 两个主按钮，并将训练日志固定放在训练区下方。
- 移除训练表单中的 `Notes`、`Proposal Hypothesis`、`Proposal Reason` 输入项，参数编辑区改为多列排版，降低表单噪音。
- `Train` 完成后会自动生成单次实验建议；`Auto Train` 完成后会展示本轮自动调优过程、每轮结果和下一步建议，中间不再需要人工确认。
- 修复训练状态管理：按钮点击后先锁定 UI 再排队执行，去掉整页自动刷新，避免重复触发 `Auto Train` 和日志区域闪烁丢失。
- 自动调优链路禁止 AI 修改 `epochs`：后端 proposal prompt 与校验同时收紧，前端自动执行时也会过滤 `epochs` 变更，手动训练仍可单独调整训练轮数。
- proposal 服务改为自动清洗 AI 返回的被禁字段；当 AIHubMix 仍返回 `epochs` 时，后端会忽略该字段继续执行，而不是直接报错中断自动调优。
- 修复 proposal 清洗过程中的类型错误，避免 `changes` 在后端被错误降级成普通 `dict` 后触发 `'dict' object has no attribute 'model_dump'`。
- 新增 `POST /runs/reset` 和前端 `Clear Database` 按钮，可一键清空 `runs`、`experiments`、`results`，并同步重置前端会话状态。
- 修复清空数据库后的空页面问题：当没有任何 run 记录时，左侧训练表单仍保持可用，只清空右侧结果与记录展示。
- 优化根目录 `.gitignore`，补充 SQLite、本地产物目录 `artifacts/`、测试缓存和 Python 打包缓存等开发期噪音忽略规则。
- 调整前端训练完成后的刷新逻辑：训练或自动调优结束后会主动刷新页面状态，新建 run 可立即出现在 `Run Selector` 中，不再出现训练完成后选择器暂时空白的问题。
- `Auto Train` 的 AI 总结区增加调优趋势图，用图表展示 baseline 到各轮实验的 `top1_acc`、`val_loss`、`train_loss` 变化，而不只保留文字摘要。
- 简化训练记录区交互：移除名称、状态、优化器、模型等过滤器；当用户已选定 run 时，默认直接展示全部实验，只通过勾选决定哪些实验参与趋势图比较。
- 调整训练动作语义：当 `Run Selector` 选中已有 run 时，`Train` 和 `Auto Train` 都会把新的 experiment 追加到该 run 下；只有在 `All Runs` 状态下才新建 run。
- 训练按钮文案改为随上下文动态变化：在已有 run 下显示 `Append Train` / `Append Auto Train`，减少“是否会新建 run”的歧义。
- `Run Selector` 区域的清理动作改为针对当前选中 run：新增 `POST /runs/{run_id}/reset`，按钮样式改为更轻的行内小按钮，不再使用整行的全局清库按钮。
- 自动调优总结图的横轴统一为整数 `round_index`，明确表示 baseline 和各轮实验的离散轮次序号。
- `Auto Train Rounds` 默认值调整为 `10`，上限提升到 `20`，便于连续调优。
- 调整前端渲染顺序并为 `AI Suggestion` 区增加实时占位；自动训练过程中每完成一轮，右侧趋势图都会立即更新，无需等整轮结束。
- 新增 `Stop Training` 终止链路：后端为当前 running experiment 注册可中断 stop event，前端可一键停止并将当前实验标记为 `discarded`，同时丢弃部分日志、checkpoint 和结果，不再继续生成 AI suggestion。
- `Auto Train` 从前端同步循环迁移为后端后台任务：新增启动、轮询、停止接口，前端改为轮询任务状态更新日志与右侧趋势图，从而使 `Stop Training` 在自动调优过程中可见且可点击。
- 日志区与 AI 建议区改为固定占位式展示：训练开始前也保留稳定高度，不再因为空内容或增量内容频繁跳动；训练进行中按钮区会直接切换为 `Stop Training`，而不是额外再增加一个停止按钮。
- 进一步修正训练按钮状态机：始终保留左右两个控件，当前被触发的动作按钮切换为 `Stop Training`，另一个按钮保持原位但禁用，避免训练开始后按钮布局突变。
- 去掉训练启动后的多余主动刷新，并增加按钮状态恢复兜底；当后台 auto task 或 running experiment 已存在时，训练区会稳定回到对应的 `Stop Training` 态，不再错误回落为 `Append ...`。
- 进一步收敛训练启动时的刷新节奏：手动训练恢复为点击后立即切换状态，自动训练在启动后的第一帧跳过轮询刷新，减少“没反应”与“连续多刷一次”的体感问题。
- 前端训练完成收尾逻辑统一改为单一清理函数：后台任务结束、手动 stop、实验完成三种路径都会同步解除锁定、清空当前训练态并触发结果刷新，减少“训练已结束但界面还停在旧状态”的问题。
- 日志区改为可直接选中复制的文本区域，同时补充 AI / 网络错误的详细日志落盘；当 AIHubMix 或网络请求失败时，前端与后台 auto-train 日志都会记录更具体的错误信息。
- 修复后台 `auto_train_service` 的轮次执行缩进错误，并补充 baseline / 各轮 proposal 生成、训练启动的阶段日志，避免自动训练状态推进异常时前端长期看不到新日志。
- 前端自动训练监视逻辑改为基于 Streamlit `fragment` 的 2 秒轮询更新，不再依赖浏览器整页刷新；自动训练进行中会持续刷新日志区和右侧 AI 趋势图，任务完成或停止后再统一触发一次应用级刷新收尾。
- 为兼容 Streamlit `fragment` 约束，日志区和 AI 占位区改用非 widget 的代码块渲染，避免 `Fragments cannot write widgets to outside containers` 异常，同时保留文本可复制能力。
- 自动训练摘要区补充实时进度文案，明确区分“已完成轮数”和“当前正在执行的轮次”，避免第 10 轮训练尚未结束时界面只显示 9 个已完成 round 而被误判为卡住。
