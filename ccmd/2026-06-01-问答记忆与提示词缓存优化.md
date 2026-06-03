# RAG 记忆压缩与 Prompt Cache 优化

## 摘要

本次会话围绕 Ragent 的会话记忆摘要压缩和 LLM 输入缓存展开架构评审，并完成了一版代码落地。核心结论是：在 DeepSeek V4 这类百万上下文模型下，固定 4/5 轮摘要是负优化，应该改成原文优先、Token/上下文窗口感知、摘要作为应急压缩手段。最终提交 `f799a88 Optimize memory summarization and prompt cache telemetry`，编译验证通过。

## 背景与目标

用户最初在模拟面试中追问 Ragent 的会话记忆摘要机制，随后进一步要求从资深 Agentic RAG / LLM 应用架构视角评审当前“滑动窗口 + LLM 摘要”的合理性。

关键业务背景：

- 当前项目原配置是最近 4 轮原文、第 5 轮开始摘要。
- 用户线上模型 DeepSeek V4 具备百万级上下文，担心低 token 量就摘要会丢信息、浪费模型调用，并破坏 Prompt Cache。
- 项目主问答链路会把 system prompt、会话历史、检索 chunk、MCP 工具结果和用户问题一起发给模型，输入 token 较长。
- 用户要求基于真实代码优化，并在完成后 `git commit`。

## 关键发现

- `RAGPromptService` 当前消息顺序是 `system -> history -> user(证据+问题)`，证据和问题已经放在最后动态区，整体顺序是合理的。
- 会话摘要原理是：`conversation-summary.st` 摘要提示词 + 旧摘要 + 待压缩历史消息，一起通过 `llmService.chat()` 发给内部模型，返回新摘要后写入 `t_conversation_summary`，并记录 `lastMessageId` 做增量摘要。
- 旧方案的问题不是“用 LLM 摘要”，而是触发条件粗糙：轮次和 token 压力不等价。短闲聊 5 轮可能只有几百 token，摘要成本和信息损失都不划算。
- Prompt Cache 对前缀稳定很敏感。高频变化的 summary 放在前面，会降低缓存命中；百万上下文下原文历史比 200 字摘要更可靠。
- 官方 Anthropic Messages API 仍以顶层 `system` 为主，第三方 OpenAI 兼容代理可能支持 `messages` 中的 system。项目自己的 `AbstractAnthropicStyleChatClient` 原先会覆盖多条 system，只保留最后一条，这和协议争议无关，是本地 adapter 的确定性风险。
- `mcp` 工具定义和意图树不进入主问答 prompt。它们分别出现在 MCP 参数提取和意图分类内部 LLM 调用，不应误认为主问答 prompt cache 的核心收益来源。

## 实现细节

本次提交：`f799a88 Optimize memory summarization and prompt cache telemetry`

主要文件：

- `bootstrap/src/main/java/com/nageoffer/ai/ragent/rag/config/MemoryProperties.java`
- `bootstrap/src/main/java/com/nageoffer/ai/ragent/rag/core/memory/JdbcConversationMemoryStore.java`
- `bootstrap/src/main/java/com/nageoffer/ai/ragent/rag/core/memory/JdbcConversationMemorySummaryService.java`
- `bootstrap/src/main/java/com/nageoffer/ai/ragent/rag/core/prompt/RAGPromptService.java`
- `infra-ai/src/main/java/com/nageoffer/ai/ragent/infra/chat/AbstractOpenAIStyleChatClient.java`
- `infra-ai/src/main/java/com/nageoffer/ai/ragent/infra/chat/AbstractAnthropicStyleChatClient.java`
- `infra-ai/src/main/java/com/nageoffer/ai/ragent/infra/chat/OpenAIStyleSseParser.java`
- `infra-ai/src/main/java/com/nageoffer/ai/ragent/infra/http/AnthropicStyleSseParser.java`
- `infra-ai/src/main/java/com/nageoffer/ai/ragent/infra/chat/log/LLMRequestLogger.java`
- `bootstrap/src/main/resources/application.yaml`

记忆策略改造：

- 新增 `history-token-budget`，历史加载时按 token budget 保留原文，超预算才裁掉更早消息。
- `history-keep-turns` 默认调为 `100`，避免数据库无限查询，但不再作为主要压缩触发逻辑。
- 新增 `summary-trigger-token-threshold`、`summary-trigger-context-ratio`、`summary-min-token-threshold`、`summary-debounce-seconds`。
- 摘要触发从固定轮次变为 token-aware：首次摘要按 `max(absoluteThreshold, maxContextTokens * ratio)` 计算，且有 90% context hard cap。
- 已存在摘要后，只有新增待压缩历史超过 `summary-min-token-threshold` 才增量更新。
- 摘要更新增加 debounce，避免连续追问导致频繁重写 summary。
- 摘要不再装饰为 `SYSTEM` 消息，而是作为动态 `USER` 上下文块，避免覆盖主规则和破坏稳定 system 前缀。

当前核心配置含义：

```yaml
memory:
  history-keep-turns: 100
  history-token-budget: 80000
  summary-start-turns: 20
  summary-enabled: true
  summary-trigger-token-threshold: 200000
  summary-trigger-context-ratio: 0.7
  summary-min-token-threshold: 20000
  summary-debounce-seconds: 60
  summary-max-chars: 800
```

Prompt Cache / 观测改造：

- `RAGPromptService` 不再让意图节点自定义 prompt 整体替换默认 RAG prompt，而是作为“场景补充规则”追加，保留稳定公共前缀。
- `AbstractAnthropicStyleChatClient` 将多条 system 合并，避免后面的 system 覆盖前面的 system。
- `AIModelProperties.ProviderConfig` 新增 `promptCacheKeyEnabled` 和 `streamUsageEnabled`，默认关闭，方便对支持的 OpenAI 兼容代理灰度开启。
- OpenAI 兼容请求在开启配置后可透传 `prompt_cache_key`，流式请求可加 `stream_options.include_usage`。
- OpenAI / Anthropic 风格 SSE parser 增加 usage 解析，`LLMRequestLogger` 会记录流式 usage。后续如果代理返回 `cached_tokens`、`cache_read_input_tokens` 等字段，日志里可以看到。

验证命令：

```bash
./mvnw -pl bootstrap -am -DskipTests compile
```

结果：`BUILD SUCCESS`。只有 Maven 插件版本缺失、依赖 metadata checksum 等既有 warning。

## 决策与权衡

- 没有新增数据库字段 `token_count` 到会话消息表。原因是当前需要快速落地、降低迁移成本；本版用已有 `TokenCounterService` 在读取和摘要判断时估算 token。
- 不默认开启 `prompt_cache_key` 和 `stream_options.include_usage`。原因是主力模型经过 MoYu/NewAPI 等 OpenAI 兼容代理，是否支持这些字段必须实测；默认关闭可以避免不兼容代理报错。
- 不做“chunk 进入模型侧 prompt cache”。原因是检索结果每次不同，chunk 组合、顺序和引用编号变化大，放进稳定前缀反而破坏命中。chunk 更适合做应用层格式化缓存，而不是 LLM prefix cache。
- 不做语义答案缓存。原因是 RAG 答案受知识库版本、用户问题细节、多轮上下文影响，答错风险较高，应等 usage/命中率观测稳定后再灰度。
- 摘要 `summary-max-chars` 从 200 调到 800。因为摘要只在很长会话时触发，太短会显著丢失约束、否定条件和任务状态。

## 未完成事项 / 后续行动

- 对 MoYu/NewAPI 代理做 P0 实测：两次相同稳定 system 前缀请求，观察第二次 response usage 是否暴露 `cached_tokens` 或类似字段。
- 若代理支持，按 provider 打开：
  - `prompt-cache-key-enabled: true`
  - `stream-usage-enabled: true`
- 建议在日志或 Trace 中进一步结构化记录 `cached_tokens / input_tokens`、TTFT、total latency、provider、model、scene，用于判断缓存收益。
- 可继续做应用层 chunk 格式化缓存，key 可设计为 `chunk:{chunkId}:{contentHash}:{citationEnabled}:{formatVersion}`。
- 如果未来要更精确控制历史 token，考虑给 `t_message` 增加 `token_count` 字段，并在消息落库时计算。
- 如果启用原生 Anthropic 显式缓存，需要把 content 改成 content block 数组并注入 `cache_control`，当前提交只修复多 system 合并和 usage 观测，没有直接上 Claude 显式 cache block。

## 备注与提醒

- 当前工作区提交后是干净的，但 `git status` 曾显示 `.run/` 为未跟踪目录，本次没有触碰。
- 项目中 `DeepSeek V4` 的 `max-context-tokens` 已调为 `1000000`，这会影响摘要触发阈值：`summary-trigger-context-ratio: 0.7` 下首次摘要更接近 70 万 token。
- 摘要仍然是 LLM 摘要，不是本地规则压缩；只是触发变得更保守、更适配大上下文和 Prompt Cache。
