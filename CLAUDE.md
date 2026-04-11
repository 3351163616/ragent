# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ragent 是一个 Agentic RAG (Retrieval-Augmented Generation) 智能问答系统，基于 Spring Boot 3.5 + React 18 构建。支持多路检索、意图识别、查询重写、会话记忆、模型路由容错、MCP 工具集成和文档入库 ETL 管道。

## Build & Run Commands

### Backend (Java 17, Maven multi-module)

```bash
# 完整构建（含代码格式化 spotless）
./mvnw clean package

# 跳过测试构建
./mvnw clean package -DskipTests

# 只编译不打包
./mvnw compile

# 运行单个测试类
./mvnw test -pl bootstrap -Dtest=SomeTestClass

# 运行单个测试方法
./mvnw test -pl bootstrap -Dtest=SomeTestClass#someMethod

# 代码格式化（Spotless，编译时自动执行）
./mvnw spotless:apply

# 启动应用（bootstrap 模块）
./mvnw spring-boot:run -pl bootstrap
```

### Frontend (React + Vite + TypeScript)

```bash
cd frontend
npm install
npm run dev        # 开发服务器 localhost:5173，自动代理 /api -> localhost:9090
npm run build      # 生产构建
npm run lint       # ESLint 检查
npm run format     # Prettier 格式化
```

### Infrastructure (Docker Compose)

```bash
# 开发环境全量启动（PostgreSQL + Redis + RustFS + Milvus）
docker compose -f resources/docker/dev-all-in-one.compose.yaml up -d
```

## Architecture

### Maven 模块结构

```
ragent (parent pom)
├── framework      # 公共基础框架：异常体系、分布式ID、幂等、Redis Stream MQ、链路追踪
├── infra-ai       # AI 基础设施：多模型提供商路由（Chat/Embedding/Rerank）
├── bootstrap      # 主业务应用（Spring Boot 启动入口 + 全部业务代码）
└── mcp-server     # 独立 MCP 工具服务器（端口 9099，JSON-RPC 协议）
```

### 根包名

`com.nageoffer.ai.ragent`

### 后端核心业务模块（均在 bootstrap 中）

- **rag/** - RAG 对话核心：意图识别 → 查询重写/拆分 → 多通道检索 → 去重/Rerank → Prompt 构建 → LLM 流式生成
- **knowledge/** - 知识库管理：知识库 CRUD、文档上传/分块、向量化写入
- **ingestion/** - 数据摄入管道：可编排节点流水线（Fetch → Parse → Chunk → Enhance → Enrich → Index）
- **user/** - 用户认证（Sa-Token）
- **admin/** - 管理仪表盘
- **core/** - 文档解析器、分块策略

### AI 模型路由（infra-ai）

通过 `RoutingLLMService` / `RoutingEmbeddingService` / `RoutingRerankService` 实现多模型优先级调度、首包探测、健康检查和自动降级。支持的提供商：百炼（阿里云）、SiliconFlow、Ollama。

### 向量存储

支持两种后端，通过配置切换：
- **pg** - PostgreSQL + pgvector 扩展（当前默认）
- **milvus** - Milvus 向量数据库

接口：`VectorStoreService`（检索）、`VectorStoreAdmin`（集合管理）

### 消息队列

基于 RocketMQ 实现事务消息队列，用于文档分块异步处理等场景。核心类在 `framework/mq/`。
- `MessageQueueProducer` → `RocketMQProducerAdapter`：支持普通同步消息和事务消息
- `DelegatingTransactionListener`：全局唯一事务监听器，通过 ConcurrentHashMap 按 topic 路由回查逻辑
- `MessageWrapper<T>`：统一消息信封（keys + body + uuid + timestamp）

### 前端结构

React 18 + TypeScript + Vite + TailwindCSS + shadcn/ui (Radix) + Zustand 状态管理

```
frontend/src/
├── pages/          # 页面：ChatPage（问答）、LoginPage、admin/*（管理后台）
├── components/     # 组件：chat/（对话）、ui/（shadcn基础组件）、layout/、session/
├── services/       # API 调用层（axios，统一前缀 /api/ragent）
├── stores/         # Zustand stores（auth、chat、theme）
└── hooks/          # 自定义 hooks（useAuth、useChat、useStreamResponse）
```

## Key Configuration

- 后端端口：9090，context-path：`/api/ragent`
- 前端端口：5173，代理 `/api` → `localhost:9090`
- MCP Server 端口：9099
- 数据库：PostgreSQL 5432
- Redis：6379
- 对象存储：RustFS（S3 兼容）9000

## Core Implementation Details

### RAG 对话管线（rag/）

入口：`RAGChatServiceImpl.streamChat()`，完整流程：
1. **记忆加载**：`ConversationMemoryService` 并行加载历史消息+摘要，摘要通过 LLM 压缩（`conversation-summary.st` 模板）
2. **查询改写**：`MultiQuestionRewriteService` — 先做术语归一化（DB 映射规则，长词优先），再 LLM 改写+拆分（传入最近 2 轮历史做指代消解），兜底规则拆分
3. **意图识别**：`DefaultIntentClassifier` — 从 Redis/DB 加载意图树（KB/MCP/SYSTEM 三种类型），LLM 低温度分类返回 `[{id, score, reason}]`；`IntentResolver` 对多子问题并行识别，保底+竞争策略控制总意图数
4. **多通道检索**：`MultiChannelRetrievalEngine` — `IntentDirectedSearchChannel`(优先级1,定向) + `VectorGlobalSearchChannel`(优先级10,兜底)；`AbstractParallelRetriever<T>` 模板方法封装并行检索
5. **后处理**：责任链模式 — `DeduplicationPostProcessor`(order=1,按通道优先级去重) → `RerankPostProcessor`(order=10,语义重排)
6. **Prompt 构建**：`RAGPromptService` 按场景选模板（KB_ONLY/MCP_ONLY/MIXED），单意图可用节点级 Prompt
7. **流式生成**：SSE 推送，`StreamTaskManager` 用 Redis Pub/Sub 实现分布式取消

### 模型路由（infra-ai/）

三层架构：Service 接口 → RoutingXxxService(@Primary) → 具体 Client
- **优先级调度**（`ModelSelector`）：过滤不可用 → priority 升序排序 → 首选模型置顶 → 熔断过滤
- **断路器**（`ModelHealthStore`）：CLOSED→OPEN→HALF_OPEN 三态，连续失败 2 次熔断 30s，半开只允许 1 个探测请求，ConcurrentHashMap.compute() 保证原子性
- **流式首包探测**：`FirstPacketAwaiter`(CountDownLatch 等待首包) + `ProbeBufferingCallback`(装饰器缓冲事件，commit 后回放)，失败透明切换下一个候选
- **提供商**：Ollama(Chat+Embedding)、百炼(Chat+Rerank)、SiliconFlow(Chat+Embedding)、Noop(Rerank 兜底)
- **SSE 解析**：`OpenAIStyleSseParser` 被百炼和 SiliconFlow 共用

### 知识库 & ETL（knowledge/ + ingestion/）

三层数据模型：KnowledgeBase 1:N KnowledgeDocument 1:N KnowledgeChunk
- **双处理模式**：`ProcessMode.CHUNK`（直接分块）和 `ProcessMode.PIPELINE`（走 ETL 管道）
- **事务消息驱动**：RocketMQ 半消息保证"状态变更+任务投递"原子性，`KnowledgeDocumentChunkTransactionChecker` 回查 DB 状态
- **原子持久化**：`persistChunksAndVectorsAtomically()` 在编程式事务中统一完成 DB+向量库写入
- **ETL 引擎**（`IngestionEngine`）：6 节点（Fetcher→Parser→Chunker→Enhancer→Enricher→Indexer），nextNodeId 链表拓扑+环检测，Spring IoC 自动注册
- **条件评估器**（`ConditionEvaluator`）：支持 SpEL + JSON DSL（all/any/not + eq/regex/contains 等操作符）
- **Fetcher 策略族**：S3Fetcher、HttpUrlFetcher、FeishuFetcher 按 SourceType 路由
- **skipIndexerWrite 双模式**：pipeline 独立执行时直接写向量，被 knowledge 调用时跳过写入由调用方在事务中统一持久化
- **定时刷新**：DB 乐观锁+锁续期，ETag→Last-Modified→SHA-256 三级变更检测

### 基础框架（framework/）

- **异常体系**：ClientException(A)/ServiceException(B)/RemoteException(C) 三分类，`GlobalExceptionHandler` 统一拦截转 Result
- **分布式 ID**：Redis Lua 脚本原子分配 workerId/datacenterId（最多 1024 节点轮转），注入 Hutool Snowflake 单例
- **幂等双方案**：`@IdempotentSubmit`(Redisson 分布式锁+SpEL 动态 Key) + `@IdempotentConsume`(Redis SET NX 两阶段状态机)
- **链路追踪**：`@RagTraceRoot`/`@RagTraceNode` 注解声明式埋点，TransmittableThreadLocal 跨线程池传播，Deque 栈管理嵌套深度
- **用户上下文**：`UserContext` 基于 TTL，异步线程池安全传播
- **SSE 封装**：`SseEmitterSender` 用 AtomicBoolean CAS 保证连接只关闭一次

### 线程池分层

5 个专用线程池（配置在 `ThreadPoolExecutorConfig`）：
- `intentClassifyExecutor`：意图识别并行
- `ragRetrievalExecutor`：通道间并行检索
- `ragInnerRetrievalExecutor`：通道内多 Collection 并行
- `ragContextExecutor`：多子问题上下文构建
- `mcpBatchExecutor`：MCP 工具并行调用
- `modelStreamExecutor`：流式 Chat 异步执行

## Code Conventions

- Java 源码需包含 Apache 2.0 License 头（Spotless 在编译阶段自动应用，模板在 `resources/format/copyright.txt`）
- 使用 Lombok（`@Data`、`@Builder` 等），配置见 `lombok.config`
- MyBatis-Plus 作为 ORM，自动填充 `createTime`/`updateTime`
- 分布式 ID 使用雪花算法（`CustomIdentifierGenerator`）
- 统一返回结构 `Result<T>`，通过 `Results` 工具类构建
- 异常体系：`ClientException` / `ServiceException` / `RemoteException` 继承 `AbstractException`
- 前端路径别名：`@` → `src/`
