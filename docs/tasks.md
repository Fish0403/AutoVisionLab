# AutoVisionLab Tasks

这个文件只记录执行层信息，不重复解释整体设计。系统定位与边界见 [docs/plan.md](plan.md)。

## 1. 执行原则

- 不开放自由代码生成式模型搜索
- 模型变化必须收敛成结构化 recipe，并进入白名单
- 所有可调参数必须结构化
- trainer 只读取结构化 config
- 一个 run 内固定一个模型和一个数据集
- 数据库存结构化摘要，本地文件存大体积产物

当前 `v1` 实施口径：

- 任务类型先只做图像分类
- 组件搜索与 recipe 深化当前先聚焦 `MobileNetV3 Small`
- 底层 schema / parser / registry 从一开始为后续检测和分割预留统一抽象

## 2. 已完成

### 平台骨架

- `FastAPI` 后端、`React` 前端、本地启动脚本已就位
- `SQLite` 持久化已接入
- `runs / experiments / results` 基础链路已打通

### 分类训练闭环

- 已支持统一分类数据目录读取：
  - `data/classification/<dataset>/train.txt|val.txt|test.txt`
- 已接入模型：
  - `MobileNetV2`
  - `MobileNetV3 Small`
  - `GoogLeNet`
  - `ResNet18`
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
- run 已开始使用明确的晋级 / 回退规则，而不只是“排第一就 keep”
- proposal 已有字段白名单与参数空间校验
- proposal 默认收敛为单变量实验；连续停滞后才允许双变量组合变更
- auto-train 已支持后台任务、停止、日志轮询和结果刷新
- auto-train 已切到无前端时间预算的持续运行模式
- auto-train 后半程策略已改为在暖启动轮次后优先考虑非 `basic` 维度，而不是硬性禁止 `basic`
- proposal 生成已开始显式提示当前可行动作与优先动作，减少规则叠加后的空 proposal
- auto-train 已增加基于维度覆盖和连续无 `keep` 的提前停止规则：
  - 当前 run 的已开放维度都已探索
  - 每个维度至少成功执行 `2` 次
  - 连续 `6` 轮没有新的 `keep`
- proposal 已开始同时输出：
  - `changes`
  - `train_hyp_changes`
  - `recipe_changes`
- auto-train follow-up config 已优先消费：
  - `train_hyp_changes`
  - `recipe_changes`
  - 仅在缺失时回退到旧 `changes`

### Recipe 与受限结构搜索

- `model_recipe_schema.md` 已重写为“通用 schema + family-specific profile”两层口径
- `model_recipe` 总体方向已明确向 YOLOv5 的 recipe 约定靠拢：
  - 顶层全局字段
  - `backbone / head`
  - 单层 `from / number / module / args`
- 已增加内置 recipe template registry，当前首个落地模板：
  - `backend/app/trainers/recipes/classification/mobilenet_v3_small.yaml`
- `MobileNetV3 Small` 已改为从内置 architecture recipe 构建，不再直接调用默认 `torchvision.mobilenet_v3_small(...)`
- 当前 `v1` 已在 `MobileNetV3 Small` 上打通受限结构搜索链路：
  - `width_multiple`
  - `pooling_type`
  - `classifier_dropout`
- `Allow model module search` 已不再是空开关，后端白名单、proposal 映射、auto-train follow-up config、trainer builder 已接通
- `MobileNetV3 Small` builder 已支持 `pooling_type=gem`

### 评价与排序

- 训练结果资源摘要已增加：
  - `latency_ms`
  - `parameter_count_million`
- ranking policy 已支持将上述资源指标作为 tie-breaker
- 前端新 run 默认排序已改为：
  - 主指标 `top1_acc`
  - tie-breaker `latency_ms`
- run 摘要已开始派生多锚点视图：
  - `best_quality_experiment_id`
  - `best_efficiency_experiment_id`
  - `best_tradeoff_experiment_id`

### 跨模型比较

- 已新增独立的 `Compare Models` 入口
- 当前第一版候选模型：
  - `MobileNetV2`
  - `MobileNetV3 Small`
  - `GoogLeNet`
- `Compare Models` 已支持：
  - 为每个候选模型创建独立 run
  - 在共享 baseline 配置下跑 baseline experiment
  - 结果页展示 `x = latency_ms`、`y = top1_acc` 的二维散点图
  - 从比较结果里手动选择一个模型继续进入后续优化
- 当前 `Compare Models` 不会自动继续进入第二阶段 `Auto Train`

### API 响应收敛

- 主要 API 已开始统一到共享 JSON 响应信封：
  - `ok`
  - `code`
  - `message`
  - `data`
  - `errors`
  - `meta`
- 当前已覆盖：
  - `GET /health`
  - `GET /runs`
  - `POST /runs`
  - `GET /runs/{run_id}`
  - `GET /runs/{run_id}/summary`
  - `GET /runs/{run_id}/metrics`
  - `POST /runs/{run_id}/proposal`
  - `POST /runs/reset`
  - `POST /runs/{run_id}/reset`
  - `POST /runs/auto-train`
  - `GET /runs/auto-train/{task_id}`
  - `POST /runs/auto-train/{task_id}/stop`
  - `POST /experiments`
  - `GET /experiments/{experiment_id}`
  - `POST /experiments/{experiment_id}/result`
  - `POST /experiments/{experiment_id}/decision`
  - `POST /experiments/{experiment_id}/train`
  - `POST /experiments/{experiment_id}/stop`
  - `GET /models/{model_name}/parameter-space`
- 全局错误响应已开始统一为同一外层格式

### 数据与产物

- 分类数据已开始收敛到通用切分脚本：
  [data/prepare_classification_split.py](../data/prepare_classification_split.py)
- `NEU-CLS` 已补原始目录到类别目录的预处理脚本：
  [data/raw/neu/prepare_classification_source.py](../data/raw/neu/prepare_classification_source.py)
- 本地日志已改为按 run 聚合：
  - `artifacts/runs/<run_id>.log`
- checkpoint 继续按 experiment 保存：
  - `artifacts/checkpoints/<experiment_id>.pt`

## 3. 进行中

### Recipe 化改造

- 收敛 `model_recipe / train_hyp / dataset_recipe` 三类结构化对象
- `v1` 先聚焦：
  - `classification`
  - `MobileNetV3 Small`
- 训练主链路已切到 `train_hyp` 驱动
- `MobileNetV3 Small` 已接入最小 `build_model_from_recipe()` 链路
- proposal / run policy / auto-train 已开始优先读取 `train_hyp / model_recipe`
- 配置命名尽量贴近 YOLO 风格，但保留当前平台的 schema / API / run 闭环
- 当前仍是“受限结构搜索”，还未进入完整 `model_recipe` 往返：
  - 虽然总体方向已切到 YOLO 风格 layer list，但当前真正落地 builder 的 family 还只有 `MobileNetV3 Small`
  - 还缺统一 `module registry / parser / validator`
  - 还不能让 AI 回整份可直接执行的架构文件并完成后端验收
  - 还不支持自由增删层或改连接关系

### 文档与口径统一

- 收敛 `README / plan / tasks / schemas / experiment_policy`
- 将过长的 auto-train 规则继续拆分为独立短文档，避免把搜索和停止策略混在一起
- 将 run 的晋级 / 回退规则独立成单独文档，避免继续堆在总览规则中
- 清理过时表述，统一当前实现与规则
- 新增 recipe 设计稿：
  - [docs/schemas/model_recipe_schema.md](schemas/model_recipe_schema.md)
  - [docs/schemas/train_hyp_schema.md](schemas/train_hyp_schema.md)
  - [docs/schemas/dataset_recipe_schema.md](schemas/dataset_recipe_schema.md)

### API 响应继续统一

- 保持后续新增接口默认接入统一响应信封
- 补齐与当前实现一致的 schema 文档和接口示例

### AI 搜索策略收敛

- 继续观察 auto-train 在暖启动轮次结束后，后半程是否真的更多转向 augmentation / loss 维度，而不是被硬规则卡死
- 继续观察“每个维度至少成功执行 `2` 次、连续 `6` 轮无 `keep`”是否合适
- 继续减少无效或重复 proposal
- 继续调优当前晋级阈值是否过松或过严
- 开始把“模型模块搜索”从细粒度 layer/args 思路转向组件级：
  - `backbone`
  - `neck`
  - `head`
  - 当前 `v1` 已在 `MobileNetV3 Small` 上开放：
    - `neck = avg_pool | gem_pool`
    - `head = native_classifier | linear | dropout_linear`
  - `backbone` 当前仍保留为槽位，尚未开放搜索
- 开始把“跨模型比较”从口头流程收敛成独立入口与独立文档：
  - 第一版只做 baseline 横向比较
  - 下一步再决定是否增加预算筛选或自动接第二阶段

## 4. 待做

### Recipe 与训练器改造

- 继续收紧 `ExperimentConfig`，逐步减少对旧 `params` 输入的依赖
- 收敛到统一 YAML manifest 方案，优先放在 `trainers/` 体系下，单文件同时表达：
  - `model`
  - `train`
  - `data`
  - `search`
- 统一入口改为：
  - 一个共享 builder 入口读取 manifest
  - 一个共享 parser 解析 YAML
  - 一个共享 validator 校验结构和搜索白名单
  - 不做成一个包打天下的巨型单文件
- 将 `classification.py` 从“按 model_name 平铺写死”重构为：
  - 公共分类参数空间
  - family-specific overlay
  - 必要时再叠加 model-specific override
- 为 `GoogLeNet` 和 `ResNet18` 增加各自的最小 `model_recipe` 白名单与 builder 约束
- 在当前统一 classification builder 之上，继续把模型构建差异往 registry 收口：
  - 把 `backbone / neck / head` 三类组件注册表单独拆清
  - 让 `model_builder_registry` 里的按 `base_model` 分支继续变薄
  - 为后续真正开放 `backbone` 搜索预留共享 feature adapter
- 将现有 recipe 从“半高层对象”继续收敛到更稳定的 YOLO 风格 layer list
- 增加统一 `module registry`：
  - 先支持分类最小模块集：
    - `StemConv`
    - `IR`
    - `PointwiseConv`
    - `GlobalPool`
    - `Classifier`
  - 后续再按 family 增加：
    - `BasicBlock`
    - `Bottleneck`
    - `Inception`
    - `AuxLogits`
- 增加统一 `recipe parser`，把 `[from, number, module, args]` 解析成内部 layer 对象
- 统一训练前数据配置读取入口，尽量从同一份 manifest 派生：
  - dataset name
  - manifests / paths
  - image size options
- 将 proposal / run policy 从旧 `params` 视角逐步迁到 `train_hyp` / `model_recipe`
- 增加 recipe artifact 落盘：
  - `artifacts/recipes/<experiment_id>_model.yaml`
  - `artifacts/recipes/<experiment_id>_hyp.yaml`
  - `artifacts/recipes/<experiment_id>_dataset.yaml`
- proposal 支持返回 `recipe_changes / train_hyp_changes`
- 继续保持 `v1` 不扩到多 backbone，不开放自由模型代码修改
- 为完整 `model_recipe` 往返补后端验收链路：
  - module exists check
  - args shape check
  - `from` reference check
  - schema validate
  - family validator
  - graph / shape propagation
  - dry build
  - dummy forward / smoke test

### 前端

- 结果区已收敛为 `Baseline / Current / Best` 三块固定视图
- 手动训练完成后，结果区已支持自动刷新，不再需要手动点刷新按钮
- auto-train 停止后会保留总结，并尝试给出一条下一步建议
- `AI Search Policy` 已增加 `Allow model module search` 开关
- 给 `AI Search Policy` 增加更细的字段级配置，而不只是分类开关
- 在结果区增加 `Quality / Efficiency / Tradeoff` 三类锚点展示
- 在结果区增加 recipe 快照查看入口
- 在结果区增加本地 run log 查看入口
- 在结果区增加 checkpoint / artifact 路径查看入口
- 训练失败时，如果后端识别到 `CUDA out of memory`，前端显示明确错误提示，而不只是留在日志里

### 后端

- 优化 backend 目录边界，把“深度学习网络相关能力”尽量收拢到 `trainers/` 体系下：
  - `trainers/recipes/`
    - 存内置 YAML manifests
  - `trainers/modules/`
    - 存 `Conv / IR / BasicBlock / Inception / Classifier` 这类模块实现或注册入口
  - `trainers/builder.py`
    - 统一构建入口
  - `trainers/parser.py`
    - 统一 manifest parser
  - `trainers/validator.py`
    - 统一结构和搜索白名单校验
  - 当前 classification trainer 已收敛到通用 trainer + model adapters
  - 下一步继续把 builder 收敛到更明确的 `backbone / neck / head` registry
- 增加 `reflection` 真正生成链路
- 增加更稳定的 proposal 去重 / 失败回退逻辑
- 增加更清晰的 artifact manifest，而不只是一对路径
- 保持后续新增主要 API 默认复用统一响应信封
- 为 recipe 相关对象补 API schema 与序列化支持
- 训练失败时识别 `CUDA out of memory` 等常见资源错误，并保留原始异常到 run log / decision reason

### 测试与验收

- 新增 `docs/testing.md`，把数据准备、proposal、手动训练、auto-train、artifact 落盘整理成固定验收清单

### 数据

- 清理前端默认值与 fallback demo 文案中的旧数据集名残留，使当前 `data/classification/<dataset_name>/` 目录名与界面显示保持一致
- 根据统一目录规范增加更多工业分类数据集

### 研究决策

- 继续完善 `keep / discard / crash / timeout` 的前端表达
- 明确人工决策与自动排名之间的边界

## 5. 建议的下一步顺序

1. 在前端结果区展示 `Quality / Efficiency / Tradeoff` 三类锚点
2. 将现有 `model_recipe` 继续收敛到统一的 YOLO 风格 layer list 运行时表示
3. 把配置收敛到统一 YAML manifest：`model + train + data + search`
4. 把 `classification.py` 重构成“公共分类空间 + family overlay + model override”
5. 补统一 `module registry + recipe parser + validator`
6. 为 `GoogLeNet` 和 `ResNet18` 补最小 recipe builder 与白名单约束
7. 将 backend 里 recipe/module/builder 尽量收拢到 `trainers/` 体系
8. 将 proposal / run policy 从旧 `params` 视角继续迁到 `train_hyp` / `model_recipe`
9. 补 recipe artifact 落盘与结果区查看入口
10. 为完整 `model_recipe` 往返补 validator / dry-run / smoke-test
11. 补 `ArtifactManifest`，让本地产物有稳定对象和接口
12. 新增 `docs/testing.md`
13. 完成 `reflection` 链路
14. 给新增接口补固定验收用例，避免后续再回退到裸 JSON
