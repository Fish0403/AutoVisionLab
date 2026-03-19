# CV + 大模型视觉质检方案（含 GitHub 落地）

## 1. 目标与边界
- 目标：用传统 CV 做稳定检测，用大模型做语义理解与异常解释，形成可追溯的自动质检系统。
- 适用场景：外观缺陷检测、装配完整性检查、标签/字符识别与规则核验。
- 边界：
  - 实时硬要求（<100ms）优先 CV/轻量检测模型；
  - 复杂语义判定、误报复核、报告生成交给多模态大模型。

## 2. 总体架构
1. 图像采集层
- 工业相机/视频流输入，统一做时间戳、产线位置信息绑定。

2. CV 推理层（第一道）
- 预处理：去噪、畸变矫正、ROI 裁剪、光照归一。
- 检测/分割：YOLO/RT-DETR/Mask2Former（按场景选）。
- 规则引擎：尺寸阈值、位置偏差、计数一致性。
- 输出：结构化结果（`defect_type`、`bbox`、`score`、`rule_hit`）。

3. 大模型判定层（第二道）
- 输入：原图 + CV 标注图 + 结构化检测 JSON。
- 任务：
  - 复杂缺陷复核（如“疑似划伤但反光干扰”）；
  - 语义一致性判断（标签内容与工单是否一致）；
  - 生成可读结论与处置建议。
- 输出：`final_decision`、`reasoning_summary`、`risk_level`、`action`。

4. 业务编排层
- 状态机：`PASS / REVIEW / FAIL`。
- 人工复核：仅对低置信度或高风险样本触发。
- 数据闭环：把复核结果回流训练集。

5. 可观测与追溯
- 日志：请求 ID、模型版本、prompt 版本、置信度。
- 指标：误报率、漏报率、复核占比、平均处理时延。
- 追溯：每个判定可回放“图像 + CV结果 + LLM结论”。

## 3. 推荐落地流程（4 个阶段）
1. POC（1-2 周）
- 选 2-3 个高价值缺陷类别；
- 先把 CV 跑通，建立基线指标（Precision/Recall/F1）。

2. 联调（2-4 周）
- 接入大模型复核链路；
- 定义升级条件（如 `0.4 < cv_score < 0.7` 才进 LLM）。

3. 试运行（2-4 周）
- 人机并行；
- 对比“纯 CV vs CV+LLM”误报漏报变化。

4. 生产化
- 灰度发布（按产线/班次）；
- 建立持续评估和模型更新节奏（双周/月度）。

## 4. 关键工程点
- Prompt 模板版本化：把模板当配置管理（可回滚）。
- 成本控制：只把“疑难样本”送大模型。
- 安全合规：脱敏、最小权限、日志审计。
- 稳定性：超时降级到 CV 规则判定，保证产线不中断。

## 5. 仓库建议结构
```text
vision-inspection-agent/
├─ README.md
├─ docs/
│  ├─ solution.md
│  ├─ prompt_templates.md
│  └─ metrics_definition.md
├─ data/
│  ├─ raw/
│  ├─ labels/
│  └─ samples/
├─ src/
│  ├─ cv/
│  ├─ llm/
│  ├─ pipeline/
│  └─ api/
├─ tests/
├─ scripts/
└─ .github/
   └─ workflows/
      ├─ ci.yml
      └─ deploy.yml
```

## 6. 怎么和你的 GitHub 对接（完整）

### 6.1 初始化并关联远程仓库
```bash
cd /home/fish/vision-inspection-agent
git init
git add .
git commit -m "init: cv+llm inspection solution"
git branch -M main
git remote add origin <你的GitHub仓库URL>
git push -u origin main
```

### 6.2 建议分支策略
- `main`：稳定可发布。
- `develop`：日常集成。
- `feature/*`：功能开发（如 `feature/cv-inference`）。
- `hotfix/*`：线上紧急修复。

### 6.3 建议 PR 流程
1. 从 `develop` 拉 `feature/*` 分支开发。
2. 本地通过测试后 push。
3. 发起 PR 到 `develop`。
4. 代码评审 + CI 通过后合并。
5. 周期性从 `develop` 合并到 `main` 做发布。

### 6.4 最小 CI（GitHub Actions）建议
- 触发：`pull_request`、`push` 到 `main/develop`。
- 检查项：
  - 代码格式化（ruff/black 或 eslint/prettier）
  - 单元测试（pytest/jest）
  - 基础安全扫描（依赖漏洞）

### 6.5 版本与发布
- 用 Tag 管理版本：`v0.1.0`, `v0.2.0`。
- 每次发布附 `CHANGELOG`，记录：新增能力、模型版本、指标变化。

## 7. 你可以马上执行的下一步
1. 先在 GitHub 新建一个空仓库。
2. 把上面的 `remote add` 和 `push` 命令执行掉。
3. 我再帮你补 `.github/workflows/ci.yml` 和基础 `README.md` 模板。
