# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Ragent 是一个 Agentic RAG (Retrieval-Augmented Generation) 智能问答系统，基于 Spring Boot 3.5 + React 18 构建。覆盖多路检索、意图识别、查询重写、会话记忆、模型路由容错、MCP 工具集成、文档入库 ETL 管道等能力。

## Build & Run Commands

### Backend (Java 17, Maven multi-module)

```bash
# 完整构建（含 Spotless 代码格式化）
./mvnw clean package

# 跳过测试构建
./mvnw clean package -DskipTests

# 只编译不打包
./mvnw compile

# 运行单个测试类
./mvnw test -pl bootstrap -Dtest=SomeTestClass

# 运行单个测试方法
./mvnw test -pl bootstrap -Dtest=SomeTestClass#someMethod

# 代码格式化（Spotless，编译阶段自动执行）
./mvnw spotless:apply

# 启动主应用（bootstrap 模块）
./mvnw spring-boot:run -pl bootstrap

# 启动 MCP Server（端口 9099，独立进程）
./mvnw spring-boot:run -pl mcp-server
```

### Frontend (React + Vite + TypeScript)

```bash
cd frontend
npm install
npm run dev        # 开发服务器 localhost:5173，自动代理 /api -> localhost:9090
npm run build      # 生产构建
npm run lint       # ESLint（--max-warnings 0）
npm run format     # Prettier 格式化
```

### Infrastructure (Docker Compose)

```bash
# 开发环境全量启动（PostgreSQL + Redis + RocketMQ + RustFS + Milvus）
docker compose -f resources/docker/dev-all-in-one.compose.yaml up -d

# 生产全量栈
docker compose -f resources/docker/prod-all-in-one.compose.yaml up -d

# 仅 Milvus（独立栈）
docker compose -f resources/docker/milvus-stack-2.6.6.compose.yaml up -d

# 低配机器用轻量栈（附带内存限制）
docker compose -f resources/docker/lightweight/milvus-stack-2.6.6.compose.yaml up -d
```

数据库 schema 与初始数据位于 `resources/database/schema_pg.sql` 与 `init_data_pg.sql`，升级脚本形如 `upgrade_vX.Y_to_vX.Z.sql`。

## Architecture

### Maven 模块结构

```
ragent (parent pom)
├── framework      # 公共基础框架：异常体系、分布式ID、幂等、缓存、RocketMQ、链路追踪、SSE 封装
├── infra-ai       # AI 基础设施：多模型提供商 Chat/Embedding/Rerank 客户端 + 优先级路由 + 熔断
├── bootstrap      # 主业务应用（Spring Boot 启动入口 + 全部业务代码），端口 9090，context-path /api/ragent
└── mcp-server     # 独立 MCP 工具服务器（端口 9099，JSON-RPC over HTTP）
```

### 根包名

`com.nageoffer.ai.ragent`

### bootstrap 业务模块

- **rag/** — RAG 对话核心：AOP（限流/追踪）、controller、service/impl、core（意图/检索/记忆/Prompt/重写/引导/MCP 客户端/向量存储）
- **knowledge/** — 知识库、文档、分块 CRUD 与 S3 文件存储
- **ingestion/** — 数据摄入 ETL 管道：`IngestionEngine` + 节点族（Fetch/Parse/Chunk/Enhance/Enrich/Index）+ `ConditionEvaluator`（SpEL + JSON DSL）
- **core/** — 通用文档解析器、分块策略
- **user/** — 用户认证（Sa-Token）
- **admin/** — 管理后台仪表盘

### 配置与基础设施

- 后端端口 9090，context-path `/api/ragent`
- 前端端口 5173（Vite），`/api` 反向代理到 9090
- MCP Server 端口 9099
- 主配置 `bootstrap/src/main/resources/application.yaml`：DataSource、Redis、RocketMQ、Milvus、rag.* 业务开关、ai.providers/chat/embedding/rerank 候选列表
- `rag.vector.type` 切换向量后端（`pg` / `milvus`）
- `ai.chat.candidates` 定义带 `priority`/`supports-thinking` 的多模型候选清单

## Core Implementation Details

### RAG 对话管线（bootstrap/rag/）

入口：`RAGChatServiceImpl.streamChat()`，完整流程：

1. **记忆加载** — `ConversationMemoryService` 并行加载历史消息+摘要；摘要由 LLM 按 `conversation-summary.st` 压缩，超过 `rag.memory.summary-start-turns` 触发
2. **查询改写** — `MultiQuestionRewriteService`：先用 `QueryTermMappingService` 做术语归一化（DB 映射表，长词优先匹配，`QueryTermMappingCacheManager` 负责 Redis 缓存），再 LLM 按 `user-question-rewrite.st` 改写+拆分（带最近 2 轮历史做指代消解），失败兜底规则拆分
3. **意图识别** — `DefaultIntentClassifier`：`IntentTreeFactory` + `IntentTreeCacheManager` 从 DB/Redis 加载意图树（`IntentKind.KB` / `SYSTEM` / `MCP` 三种类型），LLM 低温度分类返回 `[{id, score, reason}]`；`IntentResolver` 并行对多子问题识别，保底+竞争策略控总意图数
4. **多通道检索** — `MultiChannelRetrievalEngine`：`IntentDirectedSearchChannel`（定向，高优先级）+ `VectorGlobalSearchChannel`（全局兜底）；`AbstractParallelRetriever<T>` 模板方法封装通道内并行
5. **后处理链** — 责任链模式：`DeduplicationPostProcessor`（按通道优先级去重）→ `RerankPostProcessor`（语义重排）；实现 `SearchResultPostProcessor` 接口自动加入链
6. **Prompt 构建** — `RAGPromptService` 按 `PromptScene`（KB_ONLY / MCP_ONLY / MIXED）选模板；单意图时可走节点级 Prompt
7. **流式生成** — SSE 推送，`StreamCancellationHandles` + Redis Pub/Sub 实现跨实例取消

`ChatRateLimit` + `ChatQueueLimiter` 在入口层做并发限流：请求入 Redis ZSET 排队，Lua 脚本原子出队（`queue_claim_atomic.lua`），Pub/Sub 广播唤醒其他实例，队列状态 SSE 推送给前端。

### 模型路由（infra-ai/）

三层：Service 接口 → `RoutingXxxService`（@Primary） → 具体 `ChatClient` / `EmbeddingClient` / `RerankClient`

- **优先级调度**（`ModelSelector`）：过滤不可用 → priority 升序 → 首选模型置顶 → 熔断器过滤
- **三态熔断器**（`ModelHealthStore`）：CLOSED→OPEN→HALF_OPEN，连续失败 `ai.selection.failure-threshold` 次熔断 `open-duration-ms` 毫秒，半开仅放行 1 个探测请求，`ConcurrentHashMap.compute()` 保证状态转换原子
- **流式首包探测**：`ProbeStreamBridge` + `ProbeBufferingCallback`（装饰器模式缓冲事件，commit 后回放）；首包失败透明切换下一候选，用户端不会看到半截脏数据
- **提供商**（`ModelProvider` 枚举）：
  - `OLLAMA` — 本地（Chat + Embedding）
  - `BAI_LIAN` — 阿里云百炼（Chat + Rerank）
  - `SILICON_FLOW` — SiliconFlow（Chat + Embedding）
  - `MOYU` — OpenAI 兼容代理（Chat）
  - `NOOP` — 占位/测试兜底
- **SSE 解析**：`OpenAIStyleSseParser` 被 `AbstractOpenAIStyleChatClient` 的子类共用（BaiLian、SiliconFlow、MoYu）

### 知识库 & ETL（knowledge/ + ingestion/）

三层数据模型：`KnowledgeBase` 1:N `KnowledgeDocument` 1:N `KnowledgeChunk`

- **双处理模式**：`ProcessMode.CHUNK`（直接分块入库）与 `ProcessMode.PIPELINE`（走 ETL 管道）
- **事务消息驱动**：RocketMQ 半消息保证"文档状态变更 + 任务投递"原子性，`KnowledgeDocumentChunkTransactionChecker` 回查 DB
- **原子持久化**：`persistChunksAndVectorsAtomically()` 在编程式事务中统一完成 DB 写入 + 向量库写入，任一失败整体回滚
- **ETL 引擎**（`IngestionEngine`）：6 种节点（Fetcher / Parser / Chunker / Enhancer / Enricher / Indexer），`nextNodeId` 链表拓扑 + 环检测，Spring IoC 自动扫描注册
- **条件执行**（`ConditionEvaluator`）：支持 SpEL 表达式 + JSON DSL（`all` / `any` / `not` + `eq` / `regex` / `contains` 等操作符）
- **Fetcher 策略族**：S3Fetcher、HttpUrlFetcher、FeishuFetcher 按 `SourceType` 分派
- **skipIndexerWrite 双模式**：pipeline 独立运行时直接写向量库；被 knowledge 调用时跳过写入，由调用方在事务中统一持久化
- **增量刷新**：定时扫描 + DB 乐观锁 + 锁续期；变更检测走 ETag → Last-Modified → SHA-256 三级降级

### 向量存储（rag/core/vector/）

两种后端由 `rag.vector.type` 切换：

- `pg` — `PgVectorStoreService` + `PgVectorStoreAdmin`（基于 PostgreSQL + pgvector 扩展，默认）
- `milvus` — `MilvusVectorStoreService` + `MilvusVectorStoreAdmin`

统一接口：`VectorStoreService`（检索）、`VectorStoreAdmin`（Collection 创建/删除/索引管理）。`VectorSpaceId` + `VectorSpaceSpec` 封装集合命名与维度/度量类型。

### 基础框架（framework/）

- **三级异常体系**：`ClientException`(A) / `ServiceException`(B) / `RemoteException`(C) 继承 `AbstractException`，`GlobalExceptionHandler` 统一拦截转 `Result<T>`；`errorcode/` 内含 KB 模块等专用错误码
- **分布式 ID**：Redis Lua 脚本原子分配 workerId/datacenterId（最多 1024 节点轮转），注入 Hutool Snowflake 单例，MyBatis-Plus 用 `CustomIdentifierGenerator` 自动填充
- **幂等双方案**：`@IdempotentSubmit`（Redisson 分布式锁 + SpEL 动态 Key） + `@IdempotentConsume`（Redis SET NX 两阶段状态机）
- **链路追踪**：`@RagTraceRoot` / `@RagTraceNode` 注解声明式埋点，`RagTraceAspect` AOP 拦截，TransmittableThreadLocal 跨线程池传播，Deque 栈管理嵌套深度
- **用户上下文**：`UserContext` 基于 TTL，异步线程池安全透传
- **MQ 封装**：`MessageQueueProducer` → `RocketMQProducerAdapter`，`DelegatingTransactionListener` 全局单例按 topic 路由回查逻辑，`MessageWrapper<T>` 统一消息信封（keys + body + uuid + timestamp）
- **SSE**：`SseEmitterSender` 用 AtomicBoolean CAS 保证连接只关闭一次

### 线程池分层（`rag/config/ThreadPoolExecutorConfig`）

10 个按工作负载特征独立配置的线程池，均用 `TtlExecutors` 包装以透传用户上下文与 Trace：

| Bean | 用途 |
|------|------|
| `chatEntryExecutor` | 对话入口调度 |
| `modelStreamExecutor` | 流式 Chat 异步执行 |
| `intentClassifyThreadPoolExecutor` | 意图识别并行 |
| `ragRetrievalThreadPoolExecutor` | 检索通道间并行 |
| `ragInnerRetrievalThreadPoolExecutor` | 通道内多 Collection 并行 |
| `ragContextThreadPoolExecutor` | 多子问题上下文构建 |
| `mcpBatchThreadPoolExecutor` | MCP 工具批量并行调用 |
| `memoryLoadThreadPoolExecutor` | 会话记忆并行加载 |
| `memorySummaryThreadPoolExecutor` | 记忆摘要压缩 |
| `knowledgeChunkExecutor` | 文档分块异步处理 |

### 前端结构（frontend/）

React 18 + TypeScript + Vite + TailwindCSS + shadcn/ui (Radix) + Zustand + react-hook-form + react-markdown。

```
frontend/src/
├── pages/       # ChatPage（问答）、LoginPage、admin/*（仪表板/KB/意图树/入库/Trace/设置）
├── components/  # chat/、ui/（shadcn）、layout/、session/
├── services/    # axios API 调用层，统一前缀 /api/ragent
├── stores/      # Zustand stores（auth、chat、theme）
├── hooks/       # useAuth、useChat、useStreamResponse 等
└── router.tsx   # 路由配置
```

路径别名 `@` → `src/`。

## Extension Points

新增扩展实现 Spring Bean 即可自动生效：

- **检索通道**：实现 `SearchChannel` 接口
- **后处理器**：实现 `SearchResultPostProcessor` 接口（按 `getOrder()` 串联）
- **MCP 工具**：bootstrap 侧实现 `MCPToolExecutor`（被 `DefaultMCPToolRegistry` 发现）；mcp-server 侧在 `mcp/executor/` 新增 `MCPToolExecutor` 实现
- **入库节点**：实现 `IngestionNode` 接口，配置 `nextNodeId` 加入 Pipeline
- **模型提供商**：在 `infra-ai/chat` / `embedding` / `rerank` 下实现对应 Client 接口，新增 `ModelProvider` 枚举，在 `application.yaml` 的候选列表注册

## Code Conventions

- 所有 Java 源文件必须包含 Apache 2.0 License 头（Spotless 在编译阶段自动应用，模板在 `resources/format/copyright.txt`）
- Lombok：`@Data` / `@Builder` / `@RequiredArgsConstructor`，配置见 `lombok.config`
- MyBatis-Plus ORM，自动填充 `createTime` / `updateTime`
- 分布式 ID 用雪花算法（`CustomIdentifierGenerator`）
- 统一返回结构 `Result<T>`，通过 `Results` 工具类构建
- 异常继承 `AbstractException`，按业务域划入 `ClientException` / `ServiceException` / `RemoteException`
- MCP 协议约定：mcp-server 端使用 JSON-RPC 2.0（`JsonRpcRequest` / `JsonRpcResponse` / `JsonRpcError`）
