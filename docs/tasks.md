# AutoVisionLab Tasks

这个文件只记录执行层信息，不重复解释整体设计。系统定位与边界见 [docs/plan.md](plan.md)。

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
- run 已开始使用明确的晋级 / 回退规则，而不只是“排第一就 keep”
- proposal 已有字段白名单与参数空间校验
- proposal 默认收敛为单变量实验；连续停滞后才允许双变量组合变更
- auto-train 已支持后台任务、停止、日志轮询和结果刷新
- auto-train 已切到按总时间预算驱动，而不是固定轮数驱动
- auto-train 后半程策略已改为按已消耗时间比例优先考虑非 `basic` 维度，而不是硬性禁止 `basic`
- proposal 生成已开始显式提示当前可行动作与优先动作，减少规则叠加后的空 proposal
- auto-train 已增加基于维度覆盖和连续无 `keep` 的提前停止规则：
  - 当前 run 的已开放维度都已探索
  - 每个维度至少成功执行 `2` 次
  - 连续 `6` 轮没有新的 `keep`

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

### 文档与口径统一

- 收敛 `README / plan / tasks / schemas / experiment_policy`
- 将过长的 auto-train 规则继续拆分为独立短文档，避免把搜索和停止策略混在一起
- 将 run 的晋级 / 回退规则独立成单独文档，避免继续堆在总览规则中
- 清理过时表述，统一当前实现与规则

### API 响应继续统一

- 保持后续新增接口默认接入统一响应信封
- 补齐与当前实现一致的 schema 文档和接口示例

### AI 搜索策略收敛

- 继续观察 auto-train 在时间预算模式下，后半程是否真的更多转向 augmentation / loss 维度，而不是被硬规则卡死
- 继续观察“每个维度至少成功执行 `2` 次、连续 `6` 轮无 `keep`”是否合适
- 继续减少无效或重复 proposal
- 继续调优当前晋级阈值是否过松或过严

## 4. 待做

### 前端

- 给 `AI Search Policy` 增加更细的字段级配置，而不只是分类开关
- 在结果区增加本地 run log 查看入口
- 在结果区增加 checkpoint / artifact 路径查看入口
- 训练失败时，如果后端识别到 `CUDA out of memory`，前端显示明确错误提示，而不只是留在日志里

### 后端

- 增加 `reflection` 真正生成链路
- 增加更稳定的 proposal 去重 / 失败回退逻辑
- 增加更清晰的 artifact manifest，而不只是一对路径
- 保持后续新增主要 API 默认复用统一响应信封
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

1. 补 `ArtifactManifest`，让本地产物有稳定对象和接口
2. 新增 `docs/testing.md`
3. 补 run log 查看入口，让本地产物可直接在界面里追踪
4. 完成 `reflection` 链路
5. 给新增接口补固定验收用例，避免后续再回退到裸 JSON
