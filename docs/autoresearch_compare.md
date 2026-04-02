# 临时对照：`autoresearch` 的晋级 / 回退机制

这是一份临时说明，目的只有一个：把 `karpathy/autoresearch` 里的“晋级 / 回退”逻辑，和本仓库当前的 `run` / `experiment` 规则对照清楚。

参考来源：

- `karpathy/autoresearch` 的 `README.md` 和 `program.md`
- 本仓库的 [docs/policies/rank.md](policies/rank.md)
- 本仓库的 [docs/api.md](api.md)

## 结论先说

`karpathy/autoresearch` 更像是“单分支实验爬山”：

- 先改代码
- 跑一轮短训练
- 如果指标更好，就保留这次提交并继续往前走
- 如果指标更差，就回退到这轮开始前的状态

本仓库当前的做法更结构化：

- 不是直接围绕 git commit 做晋级
- 而是围绕 `run`、`experiment` 和多个锚点状态做管理
- `decision` 是后端自动写入的结果标签，用来记录晋级、回退和失败归类

## `autoresearch` 的行为模型

从 `karpathy/autoresearch` 的描述看，它的核心流程可以概括成：

1. 修改 `train.py` 或相关训练代码。
2. 创建一个新的 git commit。
3. 跑一轮固定时长的训练。
4. 对比验证指标，比如 `val_bpb`。
5. 如果结果更好，就“advance the branch”。
6. 如果结果没变好，就 `git reset` 回到起点。

这意味着它的晋级和回退是绑定在 git 历史上的，流程很直接，但表达能力有限：

- 只有“保留”与“回退”两个方向
- 决策重点在当前这次实验是否赢过起点
- 不强调多个锚点同时存在
- 不强调独立存档式的决策记录

## 本仓库当前的对应机制

本仓库把类似逻辑拆成了更明确的对象和状态。

### 1. 结果标签不是靠 git 回退，而是靠实验状态记录

`experiment` 有独立的执行状态和结果标签：

- 执行状态：`queued`、`running`、`success`、`failed`、`discarded`
- 研究决策：`keep`、`discard`、`crash`、`timeout`

这和 [docs/api.md](api.md) 里的实验接口一致，也和 [docs/policies/rank.md](policies/rank.md) 的自动晋级语义一致。

### 2. 晋级由 `ranking_policy` 约束

当前晋级不是“只要更好就行”，而是按明确规则比较：

- 主指标先比
- 只有跨过最小提升阈值才算晋级
- 主指标接近持平时，再看 tie-breaker
- 成本过高的实验可以直接不参与晋级

换句话说，本仓库的“回退”更多体现在 `discard`，而不是把代码状态真的回滚掉。

### 3. 有多个锚点，而不是只看当前最好一次

`run` 里会维护多个稳定锚点：

- `baseline_experiment_id`
- `best_experiment_id`
- `frontier_experiment_id`
- `best_quality_experiment_id`
- `best_efficiency_experiment_id`
- `best_tradeoff_experiment_id`

这比 `autoresearch` 的单线推进更适合长期实验管理，因为：

- 可以保留多个维度上的“最好”
- 可以区分质量最佳和效率最佳
- 可以把持平区间内的候选留给前沿位

## 一句话对照

- `autoresearch`：指标变好就继续，变差就 `reset`
- AutoVisionLab：实验结果进入结构化决策流，按 `keep` / `discard` / `crash` / `timeout` 和多个锚点管理，不靠 git 历史本身表达晋级

## 适合怎么理解这份差异

如果你把 `autoresearch` 看成“轻量版自动爬山”，那本仓库更像“带审计记录的实验晋级系统”。

前者适合快速试错，后者适合：

- 保留实验历史
- 分离训练执行和研究决策
- 同时维护质量、效率、折中三种结果视图
- 让后续自动搜索可以在明确规则下继续演化
