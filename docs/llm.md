# LLM 通讯约定

这份文档说明 AutoVisionLab 后端与 LLM 的通讯方式、prompt 组织方式、日志落盘位置，以及常见返回示例。

## 作用范围

- `Auto Train` 的 proposal 生成
- `Compare Models` 的 compare result summary 生成
- AIHubMix 连接测试

当前实现使用的是 OpenAI-compatible chat completions 接口，具体封装在 [`backend/app/llm/aihubmix_client.py`](../backend/app/llm/aihubmix_client.py)。

## Provider 配置

后端从环境变量读取 LLM 配置，支持以下别名：

- `OPENAI_API_KEY` / `AVL_LLM_API_KEY` / `AVL_AIHUBMIX_API_KEY`
- `OPENAI_MODEL` / `AVL_LLM_MODEL` / `AVL_AIHUBMIX_MODEL`
- `OPENAI_BASE_URL` / `AVL_LLM_BASE_URL` / `AVL_AIHUBMIX_BASE_URL`

默认值如下：

- `model = minimax/minimax-m2.5`
- `base_url = https://aihubmix.com/v1`

## 请求形态

请求体固定使用两条消息：

- `system`
- `user`

并且会带上：

- `response_format: {"type": "json_object"}`
- `temperature: 0.2`

也就是说，服务端假设模型返回的是 JSON 对象，而不是自由文本。

### 基本请求示例

```json
{
  "model": "minimax/minimax-m2.5",
  "messages": [
    {
      "role": "system",
      "content": "..."
    },
    {
      "role": "user",
      "content": "..."
    }
  ],
  "response_format": {
    "type": "json_object"
  },
  "temperature": 0.2
}
```

## 返回约定

服务端只接受 JSON 对象作为最终内容。

如果模型返回了额外包裹文本，客户端会做有限清理：

- 去掉 `<think>...</think>`
- 去掉代码围栏
- 尝试提取第一个 `{...}` 片段

如果仍然无法解析，就会抛出请求错误。

## Proposal 通讯

### 目标

`proposal` 请求用于为同一个 `run` 生成下一轮结构化变更建议。

系统提示词要求模型：

- 只返回 JSON
- 严格遵循 `ProposalSchema`
- `task_type` 固定为 `classification`
- `model_name` 不可修改
- `epochs` 是否可修改取决于当前 run 的 `search_policy`
- `changes` 至少包含一个非空字段
- 只在 `changes` 中返回本轮改动；未改动字段保持 `null`
- 不返回 `train_hyp_changes`、`recipe_changes` 这类结构化 patch
- `hypothesis` 和 `reason` 使用简洁英文
- 只能使用当前参数空间和白名单字段
- 结合整个 run 的历史，而不是只看最后一轮
- 涉及增强时应直接使用当前有效字段和取值，例如 `augmentation_policy=none/basic`、`mixup_alpha`、`cutmix_alpha`、`random_erasing_prob`

补充说明：

- 当 `search_policy` 未开放 `epochs` 时，prompt 会明确要求模型不要修改 `epochs`
- 当 `search_policy` 开放 `epochs` 时，prompt 会允许模型调整 `epochs`，并要求把它当作 `training budget` 变化来解释
- 若 proposal 因字段不在白名单、值不合法或 follow-up config 校验失败被拒绝，拒绝原因会作为 `Previous rejection reason` 回喂给下一次请求

### 传给模型的数据

用户提示词里会包含四部分：

- `Run summary`
- `Experiment history`
- `Allowed AI change fields`
- `Allowed field definitions`
- `Current source experiment constraints`

其中 `Experiment history` 是按实验整理后的结构化摘要，包含：

- 实验状态和决策
- 指标
- 资源信息
- `params`
- `train_hyp`
- `model_recipe`
- 既往 `proposal`

其中 `Current source experiment constraints` 会附带当前分支源 experiment 的完整 config 摘要：

- `params`
- `train_hyp`
- `model_recipe`

也就是说，模型看到的是“当前完整配置 + 历史轨迹”，但返回时只需要给出本轮 delta，也就是 `changes`。

### 示例输出

```json
{
  "task_type": "classification",
  "model_name": "mobilenet_v3_small",
  "based_on_experiment_ids": ["exp_001", "exp_003"],
  "hypothesis": "A slightly lower learning rate with higher weight decay may improve validation stability.",
  "changes": {
    "optimizer": null,
    "learning_rate": 0.0005,
    "batch_size": null,
    "image_size": null,
    "epochs": null,
    "weight_decay": 0.0001,
    "scheduler": "cosine",
    "augmentation_policy": null,
    "mixup_alpha": null,
    "cutmix_alpha": null,
    "random_erasing_prob": null,
    "loss_name": null,
    "focal_gamma": null,
    "label_smoothing": null,
    "aux_logits": null,
    "neck_name": "avg_pool",
    "head_name": "linear"
  },
  "reason": "The current best result already sits in a stable range, so a more conservative tuning step is a reasonable next move."
}
```

## Auto Train 停止摘要通讯

### 目标

当 `Auto Train` 因用户停止或任务结束进入收尾阶段时，系统会为 workspace results panel 生成一段简短英文摘要。

系统提示词要求模型：

- 只返回一个键 `summary_text`
- 内容客观、英文、简短
- 不要说自己是 AI
- 不要给下一步建议
- 固定覆盖四个方面：
  - 停止原因和本次搜索范围
  - 最终领先 experiment 及核心指标
  - 本轮搜索里效果最好或最稳定的策略
  - 本轮搜索里无效、不稳定或反复失败的策略

说明：

- 输出仍然保持单段文本，不扩展结构化字段
- 详细 prompt 约束以代码实现为准
- 当 `Auto Train` 的已完成轮数少于 `3` 时，后端不会请求这段 summary，避免在样本过少时生成噪声结论

## Compare Summary 通讯

### 目标

`Compare Models` 任务会把每个候选模型的结果交给 LLM，总结成一个简短的 results panel 文本。

系统提示词要求模型：

- 只返回一个键 `summary_text`
- 内容简短、客观、英文
- 不要说自己是 AI
- 不要给下一步建议
- 如果有成功候选，要提到领先模型、accuracy 和 latency
- 如果全失败，要明确说明

### 传给模型的数据

用户提示词会包含两部分：

- `Shared baseline config`
- `Candidate results`

候选结果里通常包含：

- `model_name`
- `status`
- `top1_acc`
- `latency_ms`
- `parameter_count_million`
- `normalized_config_notes`

### 示例输出

```json
{
  "summary_text": "MobileNetV3 Small is the leading successful model in this comparison, with the best observed accuracy among completed runs and competitive latency. Other candidates either trailed in accuracy or failed during evaluation."
}
```

### 全部失败时的示例输出

```json
{
  "summary_text": "All comparison candidates failed, so no successful leaderboard result is available for this workspace."
}
```

## 日志落盘

与 LLM 交互相关的运行级日志会写到：

- `artifacts/runs/<run_id>/llm.jsonl`

普通文本运行日志会写到：

- `artifacts/runs/<run_id>/run.log`

说明：

- 本地仓库当前没有现成的 `artifacts` 日志文件
- 下面的示例按代码实际写入格式整理

### `proposal` 日志示例

下面是一个典型的 JSONL 事件序列，字段名与代码一致：

```json
{"timestamp":"2026-04-02T08:00:00Z","event_type":"proposal_request","payload":{"attempt":1,"history_items":4,"prompt_chars":8124,"prompt_tokens_estimate":2140,"system_prompt":"...","user_prompt":"..."}}
{"timestamp":"2026-04-02T08:00:05Z","event_type":"proposal_response","payload":{"attempt":1,"usage":{"prompt_tokens":2140,"completion_tokens":312,"total_tokens":2452},"response_model":"minimax/minimax-m2.5","response_chars":612,"raw_content":"{\"task_type\":\"classification\",...}","parsed_payload":{"task_type":"classification","model_name":"mobilenet_v3_small", "...":"..."}}}
```

如果请求失败，日志会写入：

```json
{"timestamp":"2026-04-02T08:00:05Z","event_type":"proposal_error","payload":{"attempt":1,"prompt_tokens_estimate":2140,"error":"chat completions failed with status=502 body=..."}}
```

### `proposal` 文本日志示例

运行文本日志里会出现更短的摘要行，例如：

```text
[2026-04-02T08:00:00] [proposal-meta] attempt=1 | history_items=4 | prompt_chars=8124 | prompt_tokens_estimate=2140
[2026-04-02T08:00:05] [proposal] based_on=exp_001,exp_003 | changed_fields=["learning_rate","weight_decay","scheduler"] | prompt_tokens_estimate=2140 | provider_usage={"prompt_tokens":2140,"completion_tokens":312,"total_tokens":2452} | hypothesis=A slightly lower learning rate with higher weight decay may improve validation stability. | changes={"learning_rate":0.0005,"weight_decay":0.0001,"scheduler":"cosine"} | reason=The current best result already sits in a stable range, so a more conservative tuning step is a reasonable next move.
```

### `compare summary` 日志示例

`Compare Models` 目前没有单独的 JSONL 事件分类，但会在任务日志里记录摘要生成结果，例如：

```text
[2026-04-02T08:10:00] [1/5] Comparing mobilenet_v2
[2026-04-02T08:12:35] Compare summary generated
```

如果摘要生成失败，会记录：

```text
[2026-04-02T08:12:35] Compare summary generation failed: chat completions returned non-JSON message content: ...
```

## 错误与重试

### 传输层

客户端对部分临时性错误做有限重试，包括：

- `408`
- `409`
- `429`
- `500`
- `502`
- `503`
- `504`

### 解析层

如果返回内容不是合法 JSON，对应请求会失败并抛出异常。

### 业务层

- `proposal` 生成失败时，任务会继续按重试策略重新尝试
- `compare summary` 生成失败时，只会丢弃 AI summary，不影响 compare 结果本身

## 相关代码

- [`backend/app/llm/aihubmix_client.py`](../backend/app/llm/aihubmix_client.py)
- [`backend/app/services/proposal_service.py`](../backend/app/services/proposal_service.py)
- [`backend/app/services/model_compare_service.py`](../backend/app/services/model_compare_service.py)
- [`backend/app/services/run_logging.py`](../backend/app/services/run_logging.py)
