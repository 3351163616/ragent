# Ragent MCP、UI 与 Prompt 调整备忘

## 摘要

本次会话围绕 Ragent 的 MCP 工具链路、聊天 UI 细节、查询改写 Prompt 与结构化 LLM 输出兜底机制展开。已完成 MCP 示例意图写入、mcp-server 清理重编译与联通验证、前端侧栏和假数据清理、开场回复换行修复、查询改写错字修正规则补充，并整理了 MCP 工具生产化与 JSON 兜底机制的关键结论。

## 背景与目标

用户通过 `D:\project\prompts\Prompt1.md` 作为唯一需求入口，主流程需要持续监听该文件变化并执行新需求，监听间隔已从 60 秒改为 5 秒。

本次核心目标不是单点修 bug，而是把 Ragent 的 MCP 能力跑通并解释清楚：MCP 工具如何发现、如何被意图识别命中、如何提取参数、如何调用远端 mcp-server，以及生产环境里应如何替换示例 mock 数据。

## 关键发现

- MCP Server 现使用官方 MCP SDK 暴露 `/mcp`，工具由 `mcp-server/src/main/java/com/nageoffer/ai/ragent/mcp/executor/` 下的 `SalesMcpExecutor`、`TicketMcpExecutor`、`WeatherMcpExecutor` 注册。
- Bootstrap 侧通过 `McpClientAutoConfiguration` 连接 `rag.mcp.servers`，调用 `listTools()` 发现远端工具，再包装为 `McpClientToolExecutor` 注册到 `DefaultMcpToolRegistry`。
- MCP 调用链路为：意图识别命中 MCP 节点 -> `RetrievalEngine` 找到 `mcpToolId` -> `LLMMcpParameterExtractor` 根据工具 schema 提参 -> `McpClientToolExecutor.callTool()` 调远端 -> 结果格式化进 MCP 上下文 -> 最终回答模型生成自然语言。
- 三个 MCP 工具当前都是演示数据：销售与工单用带日期 seed 的 `Random` 生成 mock 数据，天气按城市、日期、季节模拟生成，不调用真实数据库或外部 API。
- 生产环境里 MCP 的形态类似本项目，但工具实现应替换为 CRM、工单系统、BI 平台、天气 API 等真实数据源，并补权限、脱敏、审计、超时、限流、熔断和参数二次校验。
- mcp-server 曾出现旧 target classes 导致“未发现任何工具执行器”的日志，原因是编译产物残留。执行 `.\mvnw.cmd -pl mcp-server clean compile` 后再启动即可恢复官方 SDK 工具注册。
- 敏感连接串/密钥曾用于远端 PostgreSQL 和 Redis 操作，归档中刻意不记录任何明文凭据。

## 实现细节

MCP 意图数据已写入远端 `t_intent_node`，包括一个 MCP 根节点和三个叶子意图：

- `mcp-tools`
- `mcp-sales-query` -> `sales_query`
- `mcp-ticket-query` -> `ticket_query`
- `mcp-weather-query` -> `weather_query`

Redis 意图树缓存已清理过，主应用重新加载后可命中 MCP 天气意图。日志曾验证“北京天气”类问题可以命中 `mcp-weather-query`，参数提取出 `{city=北京, queryType=current}`，并完成远端调用。

修复开场回复换行问题时发现：`StreamChatPipeline.finishOpeningReply()` 会发送 `\n\n`，但 `StreamChatEventHandler.onContent()` 用 `StrUtil.isBlank(chunk)` 把纯换行丢掉了。已改为仅丢弃 `null` 或空字符串，并只在回答开头丢弃纯空白，保留开场回复后的换行分隔。

相关文件：

- `bootstrap/src/main/java/com/nageoffer/ai/ragent/rag/service/handler/StreamChatEventHandler.java`
- `bootstrap/src/main/java/com/nageoffer/ai/ragent/rag/service/pipeline/StreamChatPipeline.java`

前端聊天侧栏曾把整个 `aside` 设置为滚动区，导致左下角“管理后台/用户信息”等底部区域随会话列表滚动并覆盖。已改为：侧栏整体 `overflow-hidden`，只有会话列表 `overflow-y-auto`，底部操作区固定在侧栏底部；移动端抽屉也补了明确高度。

相关文件：

- `frontend/src/components/chat-anthropic/AnthropicSidebar.tsx`
- `frontend/src/components/chat-anthropic/AnthropicChatView.tsx`

前端还清除了样式稿假数据和魔法值：

- 删除 `Knowledge Bases / rag-core / kb-finance / kb-engineering` 及假数量。
- 删除顶部默认 `claude-sonnet-4-6 · routing healthy`。
- 清理 `AnthropicInspector` 里的 Claude 默认名、假延迟与假路由列表，改为只展示真实选择模型或说明后端路由决定。

相关文件：

- `frontend/src/components/chat-anthropic/AnthropicTopNav.tsx`
- `frontend/src/components/chat-anthropic/AnthropicInspector.tsx`

查询改写 Prompt 已新增“错误修正”规则和示例，约束只能修明显输入法、近音、漏字、多字错误，不能修改专有名词或新增条件。示例为：

```json
{
  "rewrite": "北京今天天气怎么样",
  "should_split": false,
  "sub_questions": ["北京今天天气怎么样"]
}
```

相关文件：

- `bootstrap/src/main/resources/prompt/user-question-rewrite.st`

## 结构化输出兜底机制

系统对 LLM 生成 JSON 的处理是分模块防御式解析，而不是相信模型总能合规。

公共清洗在 `infra-ai/src/main/java/com/nageoffer/ai/ragent/infra/util/LLMResponseCleaner.java`，会移除 Markdown code fence 并 trim。之后每个模块有自己的策略：

- 查询改写：`MultiQuestionRewriteService` 解析 `rewrite/sub_questions`；失败则回退到术语归一化后的原问题作为唯一子问题；开关关闭时用规则拆分。
- 意图识别：`DefaultIntentClassifier` 支持 JSON 数组和 `{results:[...]}`，标准解析失败时用正则抢救 `id/score`；未知意图 ID 跳过；最终失败返回空列表。
- MCP 参数提取：`LLMMcpParameterExtractor` 只接受工具 schema 里声明的参数，缺失字段补 schema default；JSON 失败时返回默认参数。必填且无默认的参数不会瞎猜，由工具返回业务错误。
- 歧义判断：`AmbiguityLLMChecker` 解析失败或缺字段时返回 `true`，即保守触发澄清。
- 入库/配置候选：`JsonResponseParser` 和 `ConfigBootstrapCandidateParser` 会抽取 JSON body、字段校验、非法项跳过，必要时返回空建议。

模型不可用和格式不规范是两类问题。模型超时、供应商失败、流式首包失败由 infra-ai 的模型路由、熔断和首包探测兜底；JSON 格式不规范主要由业务解析器降级处理。

## 验证结果

- `.\mvnw.cmd -pl bootstrap,mcp-server -am -DskipTests compile` 曾通过。
- `.\mvnw.cmd -pl mcp-server clean compile` 后 mcp-server 可正常发现三个工具。
- `npm run build` 已在前端两轮改动后通过，只有既有大 chunk 提示。
- Prompt 资源变更没有单独编译需求。

## 决策与权衡

- 对 LLM 结构化输出当前采用“解析失败即降级”的稳态策略，避免复杂 JSON repair 引入误修复。后续如果需要更高召回，可增加 schema validation + 一次修复重试。
- MCP 示例工具暂保留 mock 逻辑用于演示链路。生产化时优先替换 `handleCall()` 内部数据源，而不是改主 RAG 调用链。
- 前端删除假知识库和假模型状态，而不是用另一个静态占位替代，避免用户误以为系统已接入真实健康检查或知识库统计。
- 对错字修正 Prompt 使用保守措辞，避免把业务词、产品名或系统名误改。

## 未完成事项 / 后续行动

- 若要让 MCP 工具生产可用，需要为 `sales_query`、`ticket_query`、`weather_query` 接入真实数据源，并加鉴权、审计、脱敏、超时和限流。
- 可考虑为 `LLMResponseCleaner` 增加更可靠的 JSON repair，尤其是处理智能引号、前后多余解释文字、单引号 JSON 等场景。
- 如需展示模型路由健康状态，前端应接后端真实模型健康/候选状态 API，而不是写死 `routing healthy`。
- 如需展示知识库列表和数量，前端应接知识库 API 获取真实数据，再决定是否放回聊天侧栏。

## 备注

本次会话涉及过远端数据库连接与 Redis 操作，但所有敏感凭据均未写入本归档。继续会话时，需求来源仍是 `D:\project\prompts\Prompt1.md`，监听命令应使用 `-IntervalSeconds 5`。
