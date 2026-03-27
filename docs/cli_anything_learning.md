# 从 CLI-Anything 可学习的点，以及 AutoVisionLab 可以改成什么样

这份文档不是介绍 CLI-Anything 本身，而是回答一个更实际的问题：

- AutoVisionLab 当前已经做到什么程度
- 从 CLI-Anything 身上最值得学什么
- 如果把这些做法落到当前项目，应该改成什么样
- 改完之后，前端、后端、Agent 和测试会有什么变化

## 1. 当前项目现状

结合当前仓库文档，AutoVisionLab 已经具备较完整的结构化实验闭环：

- 已有 `run / experiment / result / proposal` 等核心对象
- 已有结构化参数空间和 `search_policy`
- 已有 `Auto Train`、后台任务、停止、日志轮询
- 已有本地产物规则：
  - `artifacts/runs/<run_id>.log`
  - `artifacts/checkpoints/<experiment_id>.pt`
- 已有统一分类数据目录

当前更像是“核心功能已经成型，但机器消费能力和验证规范还不够统一”的阶段。

主要差距不是“能不能跑”，而是下面两件事还没有完全收敛：

1. 关键状态虽然已经开始统一，但还没有全部收敛成稳定、可复用、面向机器的 JSON 契约
2. 核心链路还缺少一份清晰的、可重复执行的验收清单

截至当前代码状态，已经落地的部分是：

- `runs` 主链路的成功响应已开始统一到 `ok / code / message / data / errors / meta`
- `experiments`、`models` 和 `/health` 成功响应也已经切到相同外层
- `auto-train` 的启动、查询、停止接口也已经使用相同外层结构
- 全局错误响应已开始统一为同一 JSON 外层

但还没有完全收敛的部分是：

- `ArtifactManifest` 仍未形成独立对象与独立接口
- 还没有 `docs/testing.md` 这类固定验收文档

这正是 CLI-Anything 最值得学习的地方。

## 2. 最值得学习的两点

### 2.1 统一机器可消费的输出契约

CLI-Anything 强调每个命令都应尽量具备：

- 清晰的 `--help`
- 稳定的 `--json`
- 自描述、可发现、可组合的结构化输出

它的价值不只是“命令行友好”，而是让 Agent、脚本、测试、人工调试都面对同一套事实来源。

对 AutoVisionLab 来说，这个思想可以翻译成：

- 不让前端从多个接口和页面状态里反推真实状态
- 不让自动化脚本靠字符串匹配判断任务是否成功
- 不让未来的 Agent 集成依赖 Streamlit 页面结构

应该把关键状态统一成标准 JSON 对象。

### 2.2 先定义验证，再说功能完成

CLI-Anything 不是只做“功能生成”，它很强调测试计划、测试文档和真实后端验证。

这个习惯的核心不是多写文档，而是提前回答：

- 这个功能怎样才算真的完成
- 验收时应该检查数据库、日志、产物还是页面
- 出现回归时，应该从哪几个固定步骤排查

对 AutoVisionLab，这意味着：

- 每一条关键实验链路都应该有固定验证步骤
- 验证对象不只是 API 返回值，也包括数据库状态和本地产物
- 前端展示是否正确，也应该纳入验收定义

## 3. 当前项目可以改成什么样

下面的目标不是推翻现有系统，而是在现有设计上补齐“可消费”和“可验证”两层。

### 3.1 统一 JSON 输出口径

建议把系统里的关键查询和关键动作结果统一成稳定返回格式。

当前仓库已经把 `runs`、`experiments`、`models` 与 `/health` 收敛到同一成功响应外层，但 `ArtifactManifest` 和固定验收文档仍未落地。

统一信封结构建议：

```json
{
  "ok": true,
  "code": "success",
  "message": "Human readable summary.",
  "data": {},
  "errors": [],
  "meta": {
    "schema_version": "v1",
    "timestamp": "2026-03-24T10:00:00Z"
  }
}
```

其中：

- `ok`：机器快速判断是否成功
- `code`：稳定结果码，避免靠自然语言判断
- `message`：给人看的简洁说明
- `data`：真正业务对象
- `errors`：字段级错误或失败细节
- `meta`：版本、时间戳、分页等补充信息

### 3.2 优先统一的三个对象

当前最值得先收敛的不是全部接口，而是这三个高价值对象：

- `RunDetail`
- `AutoTrainTaskStatus`
- `ArtifactManifest`

当前进度可以概括为：

- `RunDetail`：已落地第一轮统一响应外层，但 `data` 内部仍主要沿用现有 `RunDetailResponse`
- `AutoTrainTaskStatus`：已落地第一轮统一响应外层，但字段仍是当前任务快照模型
- `ArtifactManifest`：仍未落地，是下一阶段最明显的缺口

原因：

- `RunDetail` 决定前端结果区是否需要自行拼装状态
- `AutoTrainTaskStatus` 决定前端和 Agent 能否稳定读取训练进度
- `ArtifactManifest` 决定日志、checkpoint 和后续产物能否被一致展示和消费

### 3.3 改造后的对象应长什么样

下面这些结构更适合作为后续目标，而不是说当前代码已经完全长成这样。

#### RunDetail

```json
{
  "ok": true,
  "code": "success",
  "message": "Run detail loaded.",
  "data": {
    "run": {
      "run_id": "run_001",
      "run_name": "cifar10-resnet18-baseline",
      "dataset": "cifar10",
      "model_name": "resnet18",
      "status": "active",
      "baseline_experiment_id": "exp_001",
      "best_experiment_id": "exp_004",
      "frontier_experiment_id": "exp_006"
    },
    "search_policy": {
      "allow_basic_hparam_search": true,
      "allow_loss_search": true,
      "allow_augmentation_search": true,
      "allow_strategy_search": false,
      "allowed_basic_hparam_fields": [
        "optimizer",
        "learning_rate",
        "batch_size",
        "weight_decay",
        "scheduler",
        "label_smoothing"
      ]
    },
    "experiments": [
      {
        "experiment_id": "exp_001",
        "status": "success",
        "decision_status": "keep",
        "metrics": {
          "top1_acc": 0.901,
          "val_loss": 0.62
        }
      },
      {
        "experiment_id": "exp_004",
        "status": "success",
        "decision_status": "keep",
        "metrics": {
          "top1_acc": 0.926,
          "val_loss": 0.54
        }
      }
    ],
    "artifacts": {
      "run_log_path": "artifacts/runs/run_001.log",
      "best_checkpoint_path": "artifacts/checkpoints/exp_004.pt"
    }
  },
  "errors": [],
  "meta": {
    "schema_version": "v1"
  }
}
```

#### AutoTrainTaskStatus

```json
{
  "ok": true,
  "code": "running",
  "message": "Auto-train is in progress.",
  "data": {
    "task_id": "auto_001",
    "run_id": "run_001",
    "status": "running",
    "rounds_total": 10,
    "rounds_completed": 4,
    "current_round_index": 5,
    "baseline_experiment_id": "exp_001",
    "latest_experiment_id": "exp_005",
    "best_experiment_id": "exp_004",
    "latest_log_excerpt": [
      "Round 5 proposal generated.",
      "Experiment exp_005 queued."
    ]
  },
  "errors": [],
  "meta": {
    "schema_version": "v1"
  }
}
```

#### ArtifactManifest

```json
{
  "ok": true,
  "code": "success",
  "message": "Artifacts loaded.",
  "data": {
    "run_id": "run_001",
    "experiment_id": "exp_004",
    "artifacts": [
      {
        "artifact_type": "run_log",
        "path": "artifacts/runs/run_001.log",
        "exists": true
      },
      {
        "artifact_type": "checkpoint",
        "path": "artifacts/checkpoints/exp_004.pt",
        "exists": true
      }
    ]
  },
  "errors": [],
  "meta": {
    "schema_version": "v1"
  }
}
```

### 3.4 未来如果再往前走一步

如果后面希望更适合 Agent 直接接入，可以在现有 API 之外补一个轻量 CLI：

- `avl run inspect --run-id run_001 --json`
- `avl auto-train status --task-id auto_001 --json`
- `avl artifacts list --experiment-id exp_004 --json`

这不是为了“追求 CLI”，而是为了让系统事实可以脱离 Streamlit 页面被稳定消费。

## 4. 验证文档应该改成什么样

建议新增一份 `docs/testing.md`，不写成松散笔记，而写成固定格式的验收清单。

每条测试链路建议统一包含：

- 场景
- 前置条件
- 操作步骤
- 预期数据库结果
- 预期本地产物结果
- 预期前端可见性
- 通过判定

## 5. 一个适合当前项目的 testing 文档骨架

下面是建议的最小结构。

### 5.1 数据准备链路

- 场景：将原始数据整理为统一分类目录
- 前置条件：`raw/` 目录存在原始数据
- 操作步骤：执行数据准备脚本
- 预期数据库结果：无
- 预期本地产物结果：
  - `classification/train/` 存在
  - `classification/val/` 存在
  - 类别目录数量正确
- 预期前端可见性：
  - 数据集可被识别
- 通过判定：
  - 目录结构符合规则
  - 文件数量符合预期

### 5.2 Proposal 校验链路

- 场景：AI 生成 proposal 并通过约束校验
- 前置条件：已有 run 和至少一个 experiment
- 操作步骤：触发 proposal 生成
- 预期数据库结果：
  - proposal 被保存
  - `based_on_experiment_ids` 可追溯
- 预期本地产物结果：
  - run log 追加 proposal 记录
- 预期前端可见性：
  - proposal 摘要可显示
- 通过判定：
  - `changes` 非空
  - 只包含允许字段
  - 所有值都在参数空间内

### 5.3 手动训练链路

- 场景：用户手动创建或追加一次 experiment
- 前置条件：数据集已准备完成
- 操作步骤：提交训练请求
- 预期数据库结果：
  - experiment 从 `queued` 进入 `running`，最终进入终态
  - result 被正确保存
- 预期本地产物结果：
  - `artifacts/runs/<run_id>.log` 存在
  - 成功训练时 checkpoint 存在
- 预期前端可见性：
  - Training Records 可看到新实验
  - Results 可读取最新结果
- 通过判定：
  - 数据库状态、日志、产物、页面展示一致

### 5.4 Auto Train 链路

- 场景：系统自动多轮追加 experiment
- 前置条件：baseline experiment 已存在
- 操作步骤：启动 auto-train，轮询状态，必要时停止
- 预期数据库结果：
  - 每轮 experiment 被正确创建
  - best/frontier 锚点按规则更新
- 预期本地产物结果：
  - run log 持续追加
  - 成功轮次可找到对应 checkpoint
- 预期前端可见性：
  - 进度、日志、趋势图与后端状态一致
- 通过判定：
  - 轮次推进正确
  - 停止逻辑可生效
  - 最终状态可解释

### 5.5 Artifact 落盘链路

- 场景：训练结束后检查产物登记是否完整
- 前置条件：至少一个成功 experiment
- 操作步骤：读取 artifact 列表
- 预期数据库结果：
  - artifact 路径索引存在
- 预期本地产物结果：
  - 路径真实存在
  - 文件可读取
- 预期前端可见性：
  - 页面可直接查看路径或打开日志
- 通过判定：
  - 数据库索引与文件系统一致

## 6. 改完以后会有什么效果

如果按上面的方向收敛，AutoVisionLab 会出现几个明显变化。

### 6.1 前端会更轻

前端不需要再自己推断：

- 当前 run 的真实状态
- auto-train 当前进度
- 哪个 experiment 是 best/frontier
- 日志和 checkpoint 到底在哪里

前端只需要消费稳定对象并展示。

### 6.2 Agent 接入会更自然

未来无论是 CLI、脚本还是外部 Agent，都可以直接依赖：

- `RunDetail`
- `AutoTrainTaskStatus`
- `ArtifactManifest`

而不是依赖页面结构或零散字段。

### 6.3 回归验证会更稳

当你后面继续做：

- `reflection` 链路
- `NEU-CLS` 正式接入
- 字段级 AI Search Policy
- artifact manifest

都可以放进同一份验证清单里，不会每次都靠人工回忆“这次应该检查什么”。

## 7. 建议的最小落地顺序

不建议一次性做完，建议按下面顺序推进：

1. 先把统一响应信封从 `runs` 主链路扩到 `experiments`、`models` 等剩余接口
2. 再补 `ArtifactManifest` 对象与对应接口
3. 再新增 `docs/testing.md`，写 5 条核心链路验收清单
4. 再让前端结果区和训练状态区进一步直接消费这些稳定对象
5. 最后视需要补轻量 CLI 或 agent-facing 命令入口

## 8. 一句话总结

CLI-Anything 最值得学的，不是“把一切做成 CLI”，而是这套工程纪律：

- 用稳定结构化输出作为系统事实
- 用明确验收清单定义功能完成

AutoVisionLab 如果吸收这两点，系统会从“已经能跑的实验平台”进一步收敛成“更适合前端、脚本和 Agent 共同消费的实验基础设施”。
