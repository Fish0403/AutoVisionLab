# AutoVisionLab Codex 规则

这些规则只在当前仓库内生效。

## 默认风格

- 在适用时遵循 Google 风格，但必须根据当前文件所用语言调整，不要跨语言生搬硬套。
- 代码中的标识符、注释、docstring、面向提交记录的文本、`TODO` / `FIXME` 默认使用英文；除非用户明确要求代码内使用其他语言。
- 与用户的对话可以使用中文。
- 仓库中的 Markdown 文档默认使用中文；除非用户明确要求文档内容使用其他语言。

## 命名

- 优先使用清晰、可搜索的名称，避免不必要缩写。
- 函数名应表达动作或行为。
- 布尔命名在自然情况下优先使用 `is`、`has`、`can`、`should` 这类前缀。
- 避免使用含义空泛的名字，如 `data`、`temp`、`obj`、`util`、`handler2`，除非作用域确实非常小且语义明显。
- 不使用拼音命名，不使用中英混合命名。
- 命名风格要随语言调整：
  - Python：函数、方法、变量使用 `lower_with_under`；类使用 `CapWords`；常量使用 `CAPS_WITH_UNDER`。
  - JavaScript / TypeScript：函数、方法、参数、局部变量使用 `lowerCamelCase`；类和类型使用 `UpperCamelCase`；常量使用 `CONSTANT_CASE`。
  - C++：函数优先使用 `PascalCase`，但如果周围代码已经形成明确本地风格，则保持与仓库现有风格一致。
  - Shell：函数和变量使用 `lower_case_with_underscores`；常量和导出的环境变量使用 `UPPER_CASE_WITH_UNDERSCORES`。

## 注释与文档

- 注释必须使用英文。
- 注释应解释意图、限制条件、不变量或不明显的设计决策。
- 不要添加只是重复代码表面含义的注释。
- 对公共函数、类、模块，在符合该语言和文件习惯的前提下补充简洁英文文档。
- 使用 `TODO(name): ...` 或 `FIXME(name): ...` 这类格式记录待办或问题，适合时带上责任人或上下文。
- 文档注释应在签名本身不够清晰时，说明用途、关键参数、返回值和副作用。
- 仓库内 Markdown 文档中的链接默认使用相对路径，不要写机器相关的绝对文件系统路径。

## 函数

- 函数应聚焦于单一、清晰的职责。
- 优先使用显式输入和输出，避免隐藏副作用。
- 如果一个函数混合了不相关动作，应拆分。
- 查询类函数的命名应像查询、判断或读取。
- 命令类函数的命名应像执行动作。
- 避免使用过于泛化的函数名，如 `handle`、`process`、`run`、`doStuff`，除非上下文已经把动作限定得非常明确。
- 在可行时，尽量把查询函数与修改状态的函数分离。

## 测试

- 当仓库对应区域已经有测试目录或测试模式时，行为变更应同步新增或更新测试。
- 测试名称应能表达被测行为和具体场景。

## 上下文加载

- 进入本仓库开始新任务时，默认先建立最小上下文，不要一次性全文读取所有文档。
- 默认优先阅读：
  - [README.md](README.md)
  - [docs/plan.md](docs/plan.md)
  - [docs/tasks.md](docs/tasks.md)
  - [docs/policies/README.md](docs/policies/README.md)
  - [docs/schemas/README.md](docs/schemas/README.md)
- 完成上述默认阅读后，应在首次回复或首次进度更新中显式告诉用户已经读了这些启动文档，避免让用户猜测是否已建立上下文。
- 如果任务涉及 `auto-train`、proposal、搜索行为、停止条件或晋级逻辑，再按需阅读：
  - [docs/policies/auto_train_search_policy.md](docs/policies/auto_train_search_policy.md)
  - [docs/policies/auto_train_stop_policy.md](docs/policies/auto_train_stop_policy.md)
  - [docs/policies/ranking_policy.md](docs/policies/ranking_policy.md)
  - [docs/policies/run_promotion_policy.md](docs/policies/run_promotion_policy.md)
  - [docs/policies/experiment_policy.md](docs/policies/experiment_policy.md)
- 如果任务涉及 `ExperimentConfig`、API payload、recipe、trainer 配置、数据对象或结构化字段，再按需阅读：
  - [docs/schemas/model_recipe_schema.md](docs/schemas/model_recipe_schema.md)
  - [docs/schemas/train_hyp_schema.md](docs/schemas/train_hyp_schema.md)
  - [docs/schemas/dataset_recipe_schema.md](docs/schemas/dataset_recipe_schema.md)
- 如果任务涉及接口定义、请求体、响应体或错误码，再按需阅读：
  - [docs/api.md](docs/api.md)
- 如果任务涉及数据准备、manifest、原始数据目录或切分脚本，再按需阅读：
  - [data/README.md](data/README.md)
- 按需加载时，只读取与当前任务直接相关的文档，不要把整个 `docs/` 目录一次性读完。

## 协作记录

- 默认将用户视为正在旁观执行过程的协作者；在无相反要求时，优先保证过程可跟随、可检查，而不是单纯追求速度。
- 默认采用小步推进，不要在一次回复或一轮连续编辑中堆入过多改动，除非用户明确要求批量完成。
- 当一轮改动已经形成可独立验证的结果时，应主动提醒用户是否需要提交 `git`。
- 只要任务包含多个可分离步骤，就应先给出 checklist 或明确的阶段列表，再开始执行。
- 每完成一个独立步骤后，应立即更新对应状态；优先直接将该项标记为 `- [x]`，让用户可以实时看到进度。
- 在完成当前步骤并更新状态前，不应静默展开下一批大改动；应保持“做一项，记一项，看一项”的节奏。
- 当一次工作跨越多个文件、持续时间较长，或用户明确要求保留过程记录时，必须维护简短的 changelog notes。
- changelog 或进度记录应聚焦已完成的事实性变更，例如修改了什么、为什么要改、是否已验证；避免写空泛描述。
- 如果某项工作未完成但已知存在阻塞，应明确记录 blocker 或 remaining work，避免仅留下看似已完成的状态。
- 新增或更新 Markdown 文档时，默认使用中文描述；代码块内标识符、注释和示例代码仍遵循对应语言习惯。
