# 🎯 Ragent 智能问答系统面试押题精讲

> 来源：基于 Ragent（Agentic RAG）项目的模拟面试押题整理（20 题全覆盖）
> 涵盖领域：RAG 架构设计、AI 模型路由、向量检索、流式通信、消息队列、幂等与链路追踪、分布式设计、设计模式、前端工程化、软素质

---

## 📋 目录导航

| 序号 | 主题 | 考察方向 |
|:---:|------|---------|
| 1 | [RAG 核心流程与架构设计](#1-rag-核心流程与架构设计) | 架构设计 |
| 2 | [多模型路由与自动降级机制](#2-多模型路由与自动降级机制) | 原理深度 |
| 3 | [流式首包探测机制](#3-流式首包探测机制) | 原理深度 |
| 4 | [向量检索与 pgvector 实现](#4-向量检索与-pgvector-实现) | 技术应用 |
| 5 | [多通道检索与 Rerank 重排序](#5-多通道检索与-rerank-重排序) | 架构设计 |
| 6 | [查询改写与意图识别](#6-查询改写与意图识别) | 原理深度 |
| 7 | [文档分块策略设计](#7-文档分块策略设计) | 技术应用 |
| 8 | [Redis Stream 消息队列](#8-redis-stream-消息队列) | 原理深度 |
| 9 | [幂等机制设计](#9-幂等机制设计) | 技术应用 |
| 10 | [链路追踪实现](#10-链路追踪实现) | 技术应用 |
| 11 | [SSE 流式通信机制](#11-sse-流式通信机制) | 原理深度 |
| 12 | [知识库文档 ETL 管道](#12-知识库文档-etl-管道) | 架构设计 |
| 13 | [断路器模式与健康检查](#13-断路器模式与健康检查) | 原理深度 |
| 14 | [会话记忆与上下文管理](#14-会话记忆与上下文管理) | 技术应用 |
| 15 | [MCP 工具集成](#15-mcp-工具集成) | 架构设计 |
| 16 | [前端流式响应与状态管理](#16-前端流式响应与状态管理) | 技术应用 |
| 17 | [Spring Boot 多模块架构](#17-spring-boot-多模块架构) | 架构设计 |
| 18 | [异常体系与统一返回](#18-异常体系与统一返回) | 基础知识 |
| 19 | [条件装配与多后端切换](#19-条件装配与多后端切换) | 原理深度 |
| 20 | [项目中的设计模式应用](#20-项目中的设计模式应用) | 综合应用 |

---

## 1. RAG 核心流程与架构设计

> 💡 **考察重点**：架构设计 — 对 RAG 系统端到端流程的理解深度，能否清晰表达多阶段 Pipeline 的设计思路

### 📌 面试官提问

> 你好，我看你做了一个 Agentic RAG 智能问答系统。能先整体介绍一下，当用户发起一个问题时，系统内部的处理流程是什么样的？从接收请求到最终返回答案，经历了哪些阶段？

### ✅ 回答思路

- **核心概念**：RAG = 检索增强生成，先从知识库中检索相关文档片段，再将其作为上下文交给 LLM 生成回答
- **关键特性**：Ragent 采用多阶段 Pipeline 设计，每个阶段职责单一、可插拔
- **应用场景**：企业内部知识库问答、文档智能检索、客服辅助等
- **深度理解**：区别于简单 RAG，Agentic RAG 引入了意图识别、多通道检索、MCP 工具调用等能力

### 🗣️ 参考答案

面试官您好。Ragent 是一个 Agentic RAG 智能问答系统，当用户发起一个问题时，系统内部会经历以下 **六个核心阶段**：

```
用户提问 → 记忆加载 → 查询改写/拆分 → 意图识别 → 多通道检索 → Prompt 组装 → LLM 流式生成
```

具体来说：

1. **记忆加载**：从 `ConversationMemoryService` 加载当前会话的历史对话，作为上下文传递给后续环节
2. **查询改写与拆分**：通过 `MultiQuestionRewriteService` 对用户问题做术语归一化（比如缩写词映射），再调用 LLM 将复杂问题拆分为多个子问题
3. **意图识别**：`IntentResolver` 对每个子问题**并行**执行意图分类，判断应该走知识库检索（KB）、工具调用（MCP）还是系统对话（SYSTEM）
4. **多通道检索**：`RetrievalEngine` 并行执行 KB 向量检索和 MCP 工具调用，KB 侧通过 `MultiChannelRetrievalEngine` 实现意图定向检索和全局兜底检索，结果经过去重和 Rerank 重排序
5. **Prompt 组装**：`RAGPromptService` 根据场景（纯 KB / 纯 MCP / 混合）选择不同的 Prompt 模板，将检索结果、历史对话、用户问题组装成结构化消息
6. **LLM 流式生成**：通过 `RoutingLLMService` 调用大模型，以 SSE 方式实时将回答推送给前端

整个流程在 `RAGChatServiceImpl` 中编排，每个阶段通过 `@RagTraceNode` 注解实现链路追踪，便于排查性能瓶颈。

### 🔄 可能追问

1. 你提到了"意图识别"，如果用户的问题既涉及知识库又需要工具调用，系统是怎么处理的？
2. 查询改写阶段如果 LLM 调用失败了怎么办？有没有兜底策略？
3. 为什么选择 SSE 而不是 WebSocket 来做流式推送？

---

## 2. 多模型路由与自动降级机制

> 💡 **考察重点**：原理深度 — 多模型优先级调度的实现方式、降级策略、断路器模式的理解

### 📌 面试官提问

> 你提到了系统支持多个 AI 模型提供商。能详细说说你们是怎么实现模型路由和自动降级的吗？比如一个模型挂了，系统是怎么自动切换到备用模型的？

### ✅ 回答思路

- **工作机制**：通过 `ModelSelector` 按优先级排序候选模型，由 `ModelRoutingExecutor` 执行遍历式降级
- **核心原理**：断路器模式（CLOSED → OPEN → HALF_OPEN）+ 连续失败阈值触发熔断
- **实现方式**：`ModelHealthStore` 基于 `ConcurrentHashMap` + `compute` 原子操作维护健康状态
- **延伸思考**：流式场景的特殊挑战 — 首包探测机制

### 🗣️ 参考答案

我们的多模型路由由三个核心组件协同实现：

**1. ModelSelector — 候选模型筛选**

根据请求类型（Chat/Embedding/Rerank）筛选可用模型，过滤掉 `enabled=false` 和处于熔断状态的模型，按 `priority` 排序后返回候选列表。如果是深度思考模式，还会额外筛选 `supportsThinking=true` 的模型。

**2. ModelRoutingExecutor — 降级执行器**

核心方法是 `executeWithFallback()`，遍历候选模型列表依次尝试调用：

```
模型A（最高优先级）→ 调用成功 → markSuccess() → 返回结果
                   → 调用失败 → markFailure() → 尝试模型B
                                                 → 调用失败 → markFailure() → 尝试模型C
                                                                              → 全部失败 → 抛出异常
```

**3. ModelHealthStore — 断路器状态管理**

采用经典的**断路器模式**，维护三种状态：

| 状态 | 含义 | 行为 |
|------|------|------|
| `CLOSED` | 正常 | 允许所有调用 |
| `OPEN` | 熔断 | 拒绝调用，等待冷却期 |
| `HALF_OPEN` | 半开探测 | 放行一个请求试探 |

当某个模型**连续失败次数**达到 `failureThreshold` 后进入 OPEN 状态；等待 `openDurationMs` 后自动转为 HALF_OPEN，放行一个探测请求；探测成功则回到 CLOSED，失败则重回 OPEN。全部基于 `ConcurrentHashMap.compute()` 保证线程安全，无需加锁。

这套机制支持 Chat、Embedding、Rerank 三种路由服务（`RoutingLLMService`、`RoutingEmbeddingService`、`RoutingRerankService`），共享同一套路由基础设施。

### 🔄 可能追问

1. 断路器的 `failureThreshold` 和 `openDurationMs` 这两个参数你们是怎么确定的？设置不当会有什么后果？
2. `ConcurrentHashMap.compute()` 是怎么保证原子性的？和加 `synchronized` 有什么区别？
3. 如果所有模型都挂了，系统会怎么表现？有没有最终的兜底方案？

---

## 3. 流式首包探测机制

> 💡 **考察重点**：原理深度 — 流式场景下模型降级的特殊挑战，缓冲回放设计，CountDownLatch 的使用

### 📌 面试官提问

> 刚才你说了模型降级机制。但流式输出和普通请求不同，流式调用一旦开始往前端推送数据，中途切换模型就很麻烦了。你们是怎么解决这个问题的？

### ✅ 回答思路

- **核心挑战**：流式调用不同于同步调用，一旦开始向客户端推送，就无法回退
- **解决方案**：首包探测 + 缓冲回放机制
- **关键实现**：`ProbeBufferingCallback` 装饰器 + `FirstPacketAwaiter` 基于 CountDownLatch 等待

### 🗣️ 参考答案

这是一个非常好的问题。流式场景的难点在于：**如果已经开始给前端推送内容，中途发现模型有问题想切换，用户已经看到了部分内容，体验就很差**。

我们的方案是"**先探测再推送**"，通过 `ProbeBufferingCallback` 实现：

**核心流程**：

```
1. 发起流式调用 → 创建 ProbeBufferingCallback（包装真实的 callback）
2. 探测阶段：所有收到的事件（onContent/onThinking）不转发给前端，而是缓存到 buffer
3. FirstPacketAwaiter.await(60s) → 基于 CountDownLatch 阻塞等待首包
4. 首包成功 → commit()：原子切换为 committed 状态，按顺序回放缓存事件给前端
5. 首包失败（超时/错误/无内容）→ cancel() 当前 handle，降级到下一个模型重试
6. 所有模型都失败 → callback.onError() 通知客户端
```

**`FirstPacketAwaiter` 的线程安全设计**：
- `CountDownLatch(1)` — 等待首包事件
- `AtomicBoolean` — 标记是否已收到首包，防止重复触发
- `AtomicReference<Throwable>` — 捕获异步线程中的异常

**关键设计决策**：
- 为什么不在收到第一个 token 后就直接转发？因为有些模型可能返回一个空内容就断开了，需要确认首包**有实际内容**
- 为什么用 CountDownLatch 而不是 Future？因为 callback 是被动触发的，不是主动获取结果的场景

### 🔄 可能追问

1. `ProbeBufferingCallback` 的 buffer 会不会无限增长？有没有大小限制？
2. 首包等待超时设为 60 秒会不会太长？这个值是怎么确定的？
3. 如果首包探测成功后，后续流式输出中途断了怎么办？

---

## 4. 向量检索与 pgvector 实现

> 💡 **考察重点**：技术应用 — 向量数据库的选型、pgvector 的使用细节、向量相似度计算

### 📌 面试官提问

> 你们系统的知识库检索底层用了向量数据库。能说说你们为什么选择 pgvector？具体是怎么实现向量写入和检索的？

### ✅ 回答思路

- **选型理由**：复用 PostgreSQL，减少运维成本，pgvector 扩展成熟够用
- **写入流程**：文本 → Embedding 模型 → 向量 → 批量 INSERT（UPSERT）
- **检索流程**：查询文本 → Embedding → L2 归一化 → 余弦距离排序 → Top-K
- **关键细节**：HNSW 索引、ef_search 参数、向量归一化

### 🗣️ 参考答案

**选型理由**：

我们选择 pgvector 主要考虑两点：一是项目本身已经使用 PostgreSQL 作为业务数据库，加一个扩展就能支持向量检索，**无需额外运维一套独立的向量数据库**；二是对于中小规模知识库（百万级别以下），pgvector 的 HNSW 索引性能完全够用。

当然我们也保留了 **Milvus** 作为可选后端，通过 `@ConditionalOnProperty` 条件装配实现切换。

**向量写入**（`PgVectorStoreService`）：

```sql
INSERT INTO t_knowledge_vector (id, content, metadata, embedding)
VALUES (?, ?, ?::jsonb, ?::vector)
ON CONFLICT (id) DO UPDATE SET content=EXCLUDED.content, ...
```

- 使用 `JdbcTemplate.batchUpdate()` 批量写入
- metadata 存储为 JSONB 格式，包含 `kb_id`、`doc_id`、`chunk_index` 等信息
- 支持 UPSERT（`ON CONFLICT DO UPDATE`），文档重新分块时幂等覆盖

**向量检索**（`PgRetrieverService`）：

1. 调用 `embeddingService.embed(query)` 将查询文本转为向量
2. 对查询向量做 **L2 归一化**（使其长度为 1），确保余弦距离计算准确
3. 设置 `SET hnsw.ef_search = 200` 提升召回率
4. 执行检索 SQL：`SELECT *, 1 - (embedding <=> ?::vector) AS score ORDER BY embedding <=> ?::vector LIMIT ?`

其中 `<=>` 是 pgvector 提供的**余弦距离**运算符，`1 - distance` 转化为相似度分数。

### 🔄 可能追问

1. 为什么要做 L2 归一化？不归一化会怎样？
2. HNSW 索引的 `ef_search` 参数是什么意思？设大设小有什么影响？
3. pgvector 和 Milvus 相比，在性能和功能上各有什么优劣？

---

## 5. 多通道检索与 Rerank 重排序

> 💡 **考察重点**：架构设计 — 多通道检索的策略模式设计、后置处理器链、Rerank 的作用

### 📌 面试官提问

> 你提到了多通道检索。能具体说说系统有哪些检索通道？它们是怎么协同工作的？检索完之后结果又是怎么处理的？

### ✅ 回答思路

- **通道设计**：意图定向检索（精准）+ 向量全局检索（兜底），策略模式可扩展
- **后置处理**：去重 → Rerank 重排序，责任链模式
- **核心价值**：多通道提升召回率，Rerank 提升精确率

### 🗣️ 参考答案

我们的检索层 `MultiChannelRetrievalEngine` 采用了**策略模式 + 责任链模式**的组合设计。

**检索通道**（`SearchChannel` 接口，策略模式）：

| 通道 | 优先级 | 触发条件 | 作用 |
|------|:------:|---------|------|
| `IntentDirectedSearchChannel` | 1（最高） | 存在 KB 意图 | 根据意图节点关联的知识库**精准检索** |
| `VectorGlobalSearchChannel` | 10（兜底） | 无意图或置信度低 | 扫描所有知识库 collection **全局检索** |

两个通道可以**同时启用、并行执行**（使用专用线程池 `ragRetrievalExecutor`），最终合并结果。

**后置处理器链**（`SearchResultPostProcessor` 接口，责任链模式）：

```
所有通道结果合并
    │
    ▼
DeduplicationPostProcessor（顺序=1）
    → 基于 chunk ID 去重，多通道命中同一段落时保留高优先级通道的分数
    │
    ▼
RerankPostProcessor（顺序=10）
    → 调用 Rerank 模型对候选文档重新打分排序，输出最终 Top-K
```

**为什么需要 Rerank？**

向量检索基于 Embedding 相似度，可能召回"表面相似但语义不相关"的内容。Rerank 模型（如 bge-reranker）是一个**交叉编码器**，能更精确地判断 query 和 document 的相关性，相当于对粗排结果做一轮**精排**。

### 🔄 可能追问

1. 如果新增一个检索通道（比如全文检索），需要改动哪些代码？
2. Rerank 模型和 Embedding 模型的区别是什么？为什么不直接用 Embedding 相似度排序？
3. 去重处理器中"保留高优先级通道的分数"是怎么实现的？

---

## 6. 查询改写与意图识别

> 💡 **考察重点**：原理深度 — 查询改写的必要性、多子问题拆分、意图分类的并行化与限额策略

### 📌 面试官提问

> 在检索之前，你们对用户的原始问题做了改写和意图识别。这两个步骤的目的是什么？具体是怎么实现的？

### ✅ 回答思路

- **查询改写**：消除歧义、术语归一化、复杂问题拆分为可独立检索的子问题
- **意图识别**：判断每个子问题应该走什么检索通道（KB/MCP/SYSTEM）
- **工程细节**：LLM 调用失败时的规则兜底、并行分类、总意图数限额

### 🗣️ 参考答案

**查询改写**（`MultiQuestionRewriteService`）：

目的是将用户的原始问题转化为**更适合检索的形式**。分两步：

1. **术语归一化**：通过 `QueryTermMappingService.normalize()` 将缩写词映射为标准术语（比如 "KB" → "知识库"），提升检索命中率
2. **LLM 改写 + 拆分**：调用 LLM 将复杂问题改写并拆分为多个子问题，返回 JSON `{rewrite, sub_questions}`
   - 构建请求时使用低温度（`temperature=0.1, topP=0.3`），保证输出稳定
   - 保留最近 2 轮历史对话作为上下文

**关键的兜底设计**：如果 LLM 调用失败或 JSON 解析失败，会回退到 `ruleBasedSplit()` — 按标点符号（`?？。；;\n`）做规则拆分，保证流程不中断。

**意图识别**（`IntentResolver`）：

对每个子问题并行执行意图分类（使用专用线程池 `intentClassifyExecutor`），判断应该走 KB 检索、MCP 工具还是系统对话。

**总意图数限额策略**（`capTotalIntents()`）：

当子问题较多时，总意图数可能爆炸式增长。我们的限额策略是：

1. 如果总意图数未超过 `MAX_INTENT_COUNT`，直接通过
2. 超限时，**保底策略**：每个子问题至少保留 1 个最高分意图
3. 剩余配额按全局分数降序分配

最终通过 `mergeIntentGroup()` 将所有子问题的意图合并为 `IntentGroup(mcpIntents, kbIntents)`，交给检索引擎处理。

### 🔄 可能追问

1. 术语归一化的映射表是怎么维护的？能动态更新吗？
2. 意图识别用的是什么模型？是专门训练的分类模型还是通用 LLM？
3. 如果用户的问题既涉及闲聊又涉及知识库查询，意图分类会怎么处理？

---

## 7. 文档分块策略设计

> 💡 **考察重点**：技术应用 — 文档分块对 RAG 检索质量的影响、多种分块策略的适用场景、工厂模式的应用

### 📌 面试官提问

> 知识库的文档需要分块后才能做向量检索。你们实现了哪些分块策略？不同策略各有什么优缺点？是怎么组织这些策略代码的？

### ✅ 回答思路

- **分块的重要性**：分块粒度直接影响检索质量 — 太大则噪声多，太小则上下文丢失
- **四种策略**：固定大小、段落、句子、结构感知，各有适用场景
- **代码组织**：工厂模式 + 模板方法模式

### 🗣️ 参考答案

文档分块是 RAG 系统中**非常关键但容易被忽视**的环节。分块粒度直接影响检索质量：太大会引入噪声，太小会丢失上下文。

我们实现了 **四种分块策略**，通过 `ChunkingStrategyFactory`（工厂模式）统一管理：

| 策略 | 类名 | 分块方式 | 适用场景 |
|------|------|---------|---------|
| 固定大小 | `FixedSizeTextChunker` | 按字符数固定切分，支持重叠 | 格式统一的纯文本 |
| 段落分块 | `ParagraphChunker` | 按段落（空行）拆分 | 文章、博客类文档 |
| 句子分块 | `SentenceChunker` | 按句子边界拆分 | 法规、合同等对句完整性要求高的场景 |
| 结构感知 | `StructureAwareTextChunker` | 识别标题/列表等结构，保持语义完整性 | **默认策略**，适合大多数文档 |

**代码设计**：

1. **工厂模式**：`ChunkingStrategyFactory` 内部用 `EnumMap<ChunkingMode, ChunkingStrategy>` 注册所有策略，在 `@PostConstruct` 时检测重复注册
2. **模板方法模式**：所有分块器继承 `AbstractEmbeddingChunker`，基类在分块完成后自动调用 Embedding 服务生成向量，子类只需实现具体的分块逻辑

```java
// 模板方法：分块 → 自动嵌入
public List<DocumentChunk> chunkAndEmbed(String text, ChunkConfig config) {
    List<DocumentChunk> chunks = doChunk(text, config);  // 子类实现
    embedChunks(chunks);  // 基类统一调用 Embedding
    return chunks;
}
```

### 🔄 可能追问

1. 分块时的"重叠"（overlap）是什么意思？为什么需要重叠？
2. 结构感知分块是怎么识别标题和列表的？用的是正则还是其他方式？
3. 如果一篇文档很长（比如 100 页 PDF），分块过程的性能瓶颈在哪里？

---

## 8. Redis Stream 消息队列

> 💡 **考察重点**：原理深度 — Redis Stream 的使用方式、消费者组机制、与专业 MQ 的对比

### 📌 面试官提问

> 你们项目中用 Redis Stream 做消息队列来处理文档分块任务。为什么选择 Redis Stream 而不是 RabbitMQ 或 Kafka？具体是怎么实现的？

### ✅ 回答思路

- **选型理由**：项目已有 Redis，无需引入额外中间件，轻量够用
- **实现方式**：生产者发消息到 Stream，消费者组异步消费
- **工程细节**：`SmartLifecycle` 自动启动、Consumer 命名策略、优雅停机

### 🗣️ 参考答案

**选型理由**：

项目本身已经使用 Redis 做缓存和分布式锁。文档分块是一个**低频、非海量**的异步任务，不需要 Kafka 那种百万级吞吐量。用 Redis Stream 可以**零额外运维成本**地获得消息队列能力。

**框架层设计**（`framework/mq/`）：

我们在 framework 模块封装了一套轻量的 Redis Stream MQ 框架，包括三个核心组件：

**1. 生产者**（`RedisStreamProducer`）：
- 将 `MessageWrapper<?>` 序列化后通过 `StringRedisTemplate.opsForStream().add()` 发送到指定 Stream

**2. 消费者启动器**（`RedisStreamConsumerBootstrap`）：
- 实现 `SmartLifecycle` 接口，在 Spring 容器启动时自动扫描 `@MQConsumer` 注解的 Bean
- 为每个消费者创建 `StreamMessageListenerContainer`，绑定到对应的 consumer group
- Consumer 名称 = `hostname + PID`，保证集群中每个实例唯一
- **优雅停机**：先停止 Listener 容器，再关闭线程池（30 秒等待 + 强制关闭）

**3. 注解扫描器**（`MQConsumerScanner`）：
- 扫描所有 `@MQConsumer` 注解的 Bean，提取 topic、consumerGroup、payload 类型

**业务使用示例**：

```
用户上传文档 → KnowledgeDocumentChunkProducer 发送分块事件
             → Redis Stream
             → KnowledgeDocumentChunkConsumer 异步消费
             → 执行 文本提取 → 分块 → 向量化 → 写入向量库
```

一个注意点：MQ 消费者线程没有 HTTP 上下文，所以通过 `UserContext.set(LoginUser)` 手动注入操作人信息，保证审计日志和权限检查正常工作。

### 🔄 可能追问

1. Redis Stream 的消费者组（Consumer Group）和 Kafka 的消费者组有什么异同？
2. 如果消费者处理失败了，消息会怎么样？有没有重试和死信机制？
3. Redis Stream 在什么场景下不适用？什么时候应该换成 Kafka 或 RabbitMQ？

---

## 9. 幂等机制设计

> 💡 **考察重点**：技术应用 — 分布式场景下的幂等设计、两种幂等场景的区别、Redisson 分布式锁 vs Redis SET NX

### 📌 面试官提问

> 你们项目中有两种幂等机制：一种防止表单重复提交，一种防止 MQ 消息重复消费。能详细说说这两种机制的实现方式有什么区别？为什么要分开设计？

### ✅ 回答思路

- **两种场景本质不同**：HTTP 重复提交是"防并发"，MQ 重复消费是"防重入"
- **实现差异**：分布式锁 vs SET NX + 状态标记
- **AOP 切面**：通过注解无侵入地应用到业务代码

### 🗣️ 参考答案

两种幂等机制解决的是**不同场景的问题**，所以实现方式也不同：

**1. `@IdempotentSubmit` — 防止 HTTP 重复提交**

场景：用户快速双击"发送"按钮，或网络抖动导致请求重复发送。

实现：基于 **Redisson 分布式锁**
- Lock Key = SpEL 表达式计算值（如 `UserContext.getUserId()`），或 `请求路径 + userId + 参数MD5`
- `tryLock()` 非阻塞尝试，获取成功则执行业务，获取失败则直接抛出"请勿重复提交"异常
- 方法执行完自动释放锁

```
请求A → tryLock() 成功 → 执行业务 → 释放锁
请求B → tryLock() 失败 → 立即返回"请勿重复提交" ❌
```

**2. `@IdempotentConsume` — 防止 MQ 消息重复消费**

场景：Redis Stream 在网络异常时可能重复投递同一条消息。

实现：基于 **Redis SET NX PX** 原子操作（Lua 脚本）
- 维护三种状态：`CONSUMING`（消费中）/ `CONSUMED`（已消费）/ 不存在
- 消费前先 SET NX 标记为 CONSUMING，如果已存在则判断状态：
  - `CONSUMED` → 跳过（已成功处理过）
  - `CONSUMING` → 跳过（其他实例正在处理）
- 消费成功后更新为 `CONSUMED`
- 消费失败则**删除 key**，允许后续重试

**为什么分开设计？**

| | 重复提交 | 重复消费 |
|---|---|---|
| **时间窗口** | 极短（毫秒级并发） | 较长（可能跨分钟） |
| **失败处理** | 直接拒绝 | 删除标记允许重试 |
| **锁的语义** | 互斥锁（同一时刻只有一个通过） | 状态标记（记录处理进度） |

两者都通过 **AOP 切面**（`IdempotentSubmitAspect` / `IdempotentConsumeAspect`）实现，业务代码只需加一个注解，零侵入。

### 🔄 可能追问

1. Redisson 分布式锁底层是怎么实现的？和 `SET NX PX` 有什么区别？
2. 如果消费者处理到一半宕机了，CONSUMING 状态的 key 会一直存在吗？怎么处理？
3. SpEL 表达式在这里是怎么用的？能举个具体例子吗？

---

## 10. 链路追踪实现

> 💡 **考察重点**：技术应用 — 自研链路追踪的设计思路、TransmittableThreadLocal 跨线程传播、AOP 切面的嵌套

### 📌 面试官提问

> 你提到了每个 RAG 阶段都有链路追踪。这个追踪是怎么实现的？特别是在涉及线程池异步执行的场景下，追踪信息是怎么传递的？

### ✅ 回答思路

- **自研链路追踪**：基于 AOP 注解（`@RagTraceRoot` / `@RagTraceNode`）自动埋点
- **核心挑战**：线程池异步场景下 ThreadLocal 会丢失
- **解决方案**：TransmittableThreadLocal（TTL）跨线程传播

### 🗣️ 参考答案

我们自研了一套轻量的 RAG 链路追踪系统，主要解决两个问题：**性能瓶颈定位**和**问题回溯排查**。

**核心组件**：

**1. `RagTraceContext` — 追踪上下文**
- 使用 `TransmittableThreadLocal`（阿里 TTL）存储，支持在线程池中跨线程透传
- 维护三个信息：`traceId`（全局标识）、`taskId`（任务标识）、`nodeStack`（栈结构，支持嵌套节点）

**2. `RagTraceAspect` — AOP 切面**

提供两个注解：

| 注解 | 作用 | 记录内容 |
|------|------|---------|
| `@RagTraceRoot` | 标记链路根节点 | 创建 traceId，记录整个 run 的耗时和状态 |
| `@RagTraceNode` | 标记链路子节点 | 入栈/出栈，记录节点名称、类名、方法名、嵌套深度、耗时、状态 |

```
@RagTraceRoot                              ← 创建 traceId
├── @RagTraceNode("query-rewrite")         ← depth=1
├── @RagTraceNode("intent-resolve")        ← depth=1
│   └── @RagTraceNode("intent-classify")   ← depth=2（嵌套）
├── @RagTraceNode("retrieval")             ← depth=1
└── @RagTraceNode("llm-generate")          ← depth=1
```

**3. 持久化**：追踪数据写入 `t_rag_trace_run`（运行级别）和 `t_rag_trace_node`（节点级别）两张表。

**关键技术点 — TransmittableThreadLocal**：

普通的 `ThreadLocal` 在使用线程池时会丢失上下文（因为线程被复用，不会继承父线程的 ThreadLocal）。`InheritableThreadLocal` 只在创建子线程时继承，对线程池复用场景也无效。

`TransmittableThreadLocal`（TTL）通过在任务提交到线程池时**捕获当前线程的 TTL 快照**，在任务执行前**恢复快照**，执行后**还原**，完美解决了线程池场景的上下文传播问题。

### 🔄 可能追问

1. TransmittableThreadLocal 和 InheritableThreadLocal 有什么区别？为什么后者在线程池场景下失效？
2. 链路追踪数据量会不会很大？有没有采样或清理策略？
3. 为什么自研而不是用 SkyWalking 或 Jaeger 这样的成熟方案？

---

## 11. SSE 流式通信机制

> 💡 **考察重点**：原理深度 — SSE 协议原理、与 WebSocket 的对比、SseEmitter 的使用细节

### 📌 面试官提问

> 你们的对话接口用了 SSE 实现流式输出。能说说 SSE 的工作原理吗？为什么选 SSE 而不是 WebSocket？在实现过程中有没有踩过什么坑？

### ✅ 回答思路

- **SSE 原理**：基于 HTTP 的单向服务端推送，`text/event-stream` 格式
- **vs WebSocket**：SSE 更简单、天然支持断线重连、足够满足"服务端 → 客户端"单向推送
- **实现细节**：SseEmitter 超时设置、异步 Dispatch 与登录拦截的冲突

### 🗣️ 参考答案

**SSE（Server-Sent Events）工作原理**：

SSE 基于标准 HTTP 协议，客户端发起一个普通 GET 请求，服务端返回 `Content-Type: text/event-stream`，然后**保持连接不关闭**，持续以 `event: xxx\ndata: xxx\n\n` 格式推送事件。

**与 WebSocket 的对比**：

| 特性 | SSE | WebSocket |
|------|-----|-----------|
| 通信方向 | 单向（服务端 → 客户端） | 双向 |
| 协议 | 标准 HTTP | 独立协议（ws://） |
| 断线重连 | **浏览器自动重连** | 需自行实现 |
| 数据格式 | 文本 | 文本 + 二进制 |
| 复杂度 | 低 | 高 |

我们选 SSE 是因为对话场景只需要**服务端向客户端单向推送** token，不需要双向通信。SSE 更简单，天然支持断线重连，且能复用现有的 HTTP 基础设施（认证、代理、负载均衡）。

**实现中遇到的坑**：

1. **异步 Dispatch 与 Sa-Token 拦截冲突**：SSE 完成时 Spring 会触发 `asyncDispatch`，这个请求也会经过拦截器链。但此时已经没有用户 token 了，会被 Sa-Token 拦截。解决方案是在 `SaTokenConfig` 中对 `DispatcherType.ASYNC` 类型的请求跳过登录检查。

2. **事件分类设计**：我们定义了多种 SSE 事件类型来传递不同信息：
   - `meta` — 会话 ID + 任务 ID
   - `message` — 内容增量（按 `messageChunkSize` 默认 5 字符分批推送）
   - `finish` / `done` / `cancel` — 流结束信号

### 🔄 可能追问

1. SseEmitter 的超时时间是怎么设置的？超时后会发生什么？
2. 如果客户端断开连接了，服务端还在往 SseEmitter 写数据会怎样？怎么处理的？
3. "按 5 个字符分批推送"是为什么？直接一个 token 推一次不行吗？

---

## 12. 知识库文档 ETL 管道

> 💡 **考察重点**：架构设计 — 管道式数据处理的设计思路、节点编排、可扩展性

### 📌 面试官提问

> 你们的文档入库除了基本的分块模式，还有一个"管道模式"。能介绍一下这个 ETL 管道是怎么设计的？

### ✅ 回答思路

- **管道设计**：基于节点连线的链式执行引擎
- **节点类型**：Fetch → Parse → Chunk → Enhance → Enrich
- **工程细节**：条件检查、环检测、节点日志、最大执行数防御

### 🗣️ 参考答案

我们的 `IngestionEngine` 实现了一个**基于节点连线的链式执行引擎**，核心思路类似于工作流引擎。

**五种节点类型**（`IngestionNode` 接口）：

```
FetcherNode → ParserNode → ChunkerNode → EnhancerNode → EnricherNode
  数据获取      文档解析      文本分块       分块增强        分块富化
```

| 节点 | 职责 | 策略实现 |
|------|------|---------|
| `FetcherNode` | 获取原始文档 | S3 / HTTP / 本地文件 / 飞书 |
| `ParserNode` | 解析为纯文本 | Apache Tika（支持 PDF、Word、HTML 等） |
| `ChunkerNode` | 文本分块 | 四种分块策略（工厂模式） |
| `EnhancerNode` | LLM 增强分块 | 生成摘要、提取关键词 |
| `EnricherNode` | LLM 富化元数据 | 补充分类、标签等 metadata |

**引擎执行流程**（`IngestionEngine`）：

1. **构建节点映射**：`Map<String, IngestionNode>` 按 `nodeType` 注册
2. **验证无环**：检测节点引用关系中是否存在循环依赖
3. **找起始节点**：自动识别没有前驱引用的节点作为起点
4. **链式执行**：从起始节点开始，依次执行后续节点
5. **条件检查**：每个节点支持 `ConditionEvaluator`，条件不满足则跳过
6. **执行日志**：每个节点记录 `NodeLog`（耗时、输入输出、错误信息）

**防御性设计**：最大执行节点数不超过配置总数，防止因配置错误导致的无限循环。

**数据获取层的策略模式**：`DocumentFetcher` 接口有四种实现（`S3Fetcher`、`HttpUrlFetcher`、`LocalFileFetcher`、`FeishuFetcher`），新增数据源只需实现接口并注册。

### 🔄 可能追问

1. Enhance 和 Enrich 这两个阶段有什么区别？为什么要分开？
2. 如果管道执行到一半某个节点失败了，怎么处理？支持断点续传吗？
3. 管道的节点配置是写在代码里还是可以动态编排的？

---

## 13. 断路器模式与健康检查

> 💡 **考察重点**：原理深度 — 断路器三态转换原理、在 AI 模型调用场景下的具体应用

### 📌 面试官提问

> 你前面提到了断路器模式。能详细说说断路器的三个状态是怎么转换的？在 AI 模型调用这个场景下，和传统微服务熔断有什么不同？

### ✅ 回答思路

- **三态转换**：CLOSED → OPEN → HALF_OPEN → CLOSED/OPEN
- **触发条件**：连续失败计数 vs 失败率（传统微服务常用滑动窗口+失败率）
- **AI 场景特殊性**：模型响应不稳定、首包探测、降级策略是切换模型而非返回兜底值

### 🗣️ 参考答案

**`ModelHealthStore` 的三态转换**：

```
           连续失败 >= threshold
CLOSED ─────────────────────────→ OPEN
  ↑                                │
  │ 探测成功                        │ 等待 openDurationMs
  │                                ▼
  └──────────── HALF_OPEN ←────────┘
                │
                │ 探测失败
                └──────→ OPEN（重置冷却期）
```

| 状态 | `allowCall()` 返回 | 转入条件 |
|------|:------------------:|---------|
| `CLOSED` | `true` | 探测成功 / 初始状态 |
| `OPEN` | `false` | 连续失败次数 ≥ `failureThreshold` |
| `HALF_OPEN` | `true`（仅放行一个） | OPEN 状态持续超过 `openDurationMs` |

**线程安全实现**：

```java
// 全部基于 ConcurrentHashMap.compute() 原子操作
healthMap.compute(modelKey, (key, state) -> {
    if (state == null) return new ModelHealth(CLOSED);
    state.incrementFailure();
    if (state.failures >= threshold) state.trip(); // → OPEN
    return state;
});
```

`compute()` 方法保证了对同一个 key 的读-改-写操作是原子的，不需要额外加锁。

**与传统微服务熔断的区别**：

| | 传统微服务（如 Sentinel） | AI 模型场景 |
|---|---|---|
| 统计方式 | 滑动窗口 + 失败率 | 连续失败计数 |
| 降级策略 | 返回兜底值 / 限流 | **切换到备用模型** |
| 探测方式 | 定时健康检查 | **首包探测**（实际发一次请求看是否有内容返回） |
| 恢复速度 | 定时恢复 | 首包成功即恢复 |

AI 模型调用的不稳定性比传统 RPC 更高（模型过载、API 限流、网络波动），所以我们用**连续失败计数**而非失败率更为敏感和实用。

### 🔄 可能追问

1. `ConcurrentHashMap.compute()` 内部加锁的粒度是什么？是锁整个 Map 还是锁单个桶？
2. 如果 HALF_OPEN 状态放行的探测请求刚好很慢，后续请求怎么处理？
3. 有没有考虑过用 Resilience4j 等成熟库来替代自研的断路器？

---

## 14. 会话记忆与上下文管理

> 💡 **考察重点**：技术应用 — 多轮对话的记忆管理、上下文窗口限制、记忆压缩策略

### 📌 面试官提问

> 你们的系统支持多轮对话。随着对话轮数增加，历史消息会越来越长。你们是怎么管理会话记忆的？有没有做上下文长度控制？

### ✅ 回答思路

- **记忆加载**：每次对话前加载历史消息作为上下文
- **长度控制**：当历史过长时触发记忆压缩（摘要化）
- **存储设计**：持久化存储 + 按会话隔离

### 🗣️ 参考答案

我们的会话记忆管理通过 `ConversationMemoryService` 实现，主要解决两个问题：**记忆持久化**和**上下文窗口控制**。

**基本流程**：

```
用户提问
  │
  ▼
loadAndAppend(conversationId, userId, userMessage)
  │
  ├─ 1. 从数据库加载该会话的历史消息
  ├─ 2. 追加用户当前消息
  └─ 3. 返回完整历史列表
  │
  ▼
（RAG Pipeline 执行）
  │
  ▼
memoryService.append(conversationId, assistantMessage)  ← LLM 生成完成后追加回复
```

**记忆压缩**（`ConversationMemorySummaryService`）：

当历史消息轮数过多或 token 数超过阈值时，触发记忆压缩：
- 将较早的对话历史通过 LLM 生成**摘要**
- 用摘要替代原始消息，大幅减少 token 消耗
- 保留最近 N 轮完整对话，确保近期上下文不丢失

```
[摘要: "用户之前询问了知识库的创建方式和文档上传流程..."]
[完整] User: 向量检索的原理是什么？
[完整] Assistant: 向量检索是通过...
[完整] User: 那 pgvector 和 Milvus 怎么选？  ← 当前问题
```

**设计考量**：
- 记忆按 `conversationId + userId` 隔离，多用户之间互不干扰
- 查询改写阶段只使用最近 **2 轮**历史，避免过长上下文干扰改写质量
- Prompt 组装阶段使用完整（或压缩后的）历史，保证回答连贯性

### 🔄 可能追问

1. 记忆压缩的摘要质量怎么保证？如果 LLM 把关键信息压缩丢了怎么办？
2. 为什么查询改写只用最近 2 轮而不是全部历史？
3. 如果用户长时间不操作后回来继续对话，记忆是怎么恢复的？

---

## 15. MCP 工具集成

> 💡 **考察重点**：架构设计 — MCP 协议的理解、工具注册与调度机制、独立部署的设计考量

### 📌 面试官提问

> 你们系统除了知识库检索，还支持 MCP 工具调用。能说说 MCP 是什么？你们是怎么把工具能力集成到 RAG 流程中的？

### ✅ 回答思路

- **MCP 协议**：Model Context Protocol，标准化的 AI 工具调用协议
- **独立部署**：MCP Server 作为独立服务（端口 9099），通过 JSON-RPC 通信
- **集成方式**：意图识别匹配到 MCP 意图 → 并行调用工具 → 结果注入 Prompt

### 🗣️ 参考答案

**MCP（Model Context Protocol）** 是一种标准化协议，让 AI 模型能够调用外部工具获取实时数据或执行操作。

**我们的 MCP Server 架构**：

MCP Server 作为一个**独立的 Spring Boot 服务**（端口 9099），与主应用解耦部署。核心组件：

| 组件 | 类 | 职责 |
|------|-----|------|
| 协议分发 | `MCPDispatcher` | 解析 JSON-RPC 请求，路由到对应方法 |
| 工具注册 | `MCPToolRegistry` | 维护所有已注册的工具定义 |
| 工具执行 | `MCPToolExecutor` | 每个工具一个实现类 |

支持的 JSON-RPC 方法：
- `initialize` — 握手，返回协议版本和能力声明
- `tools/list` — 列出所有可用工具的名称和参数定义
- `tools/call` — 根据工具名称查找 executor，传入参数，返回执行结果

**集成到 RAG 流程的方式**：

```
用户提问: "今天北京天气怎么样？"
  │
  ▼ 意图识别
IntentResolver → 识别为 MCP 意图（WeatherTool）
  │
  ▼ 检索阶段
RetrievalEngine
  ├── KB 检索（无匹配 → 空）
  └── MCP 调用（mcpBatchExecutor 线程池并行执行）
      └── MCPClient → HTTP → MCP Server → WeatherMCPExecutor
          → 返回 "北京 23°C 晴"
  │
  ▼ Prompt 组装
RAGPromptService（MCP_ONLY 场景模板）
  → System Prompt 中注入 "## 动态数据片段: 北京 23°C 晴"
  │
  ▼ LLM 生成
"根据实时数据，今天北京天气晴朗，气温 23°C..."
```

**独立部署的设计考量**：
- MCP 工具可能依赖外部 API（天气、票务等），独立部署隔离了故障域
- 可以独立扩缩容，不影响主应用
- 支持多个主应用共享同一个 MCP Server

### 🔄 可能追问

1. MCP 和 OpenAI 的 Function Calling 有什么区别？
2. 如果 MCP Server 调用超时，会不会阻塞整个 RAG 流程？怎么处理的？
3. 如何新增一个 MCP 工具？需要改动哪些代码？

---

## 16. 前端流式响应与状态管理

> 💡 **考察重点**：技术应用 — 前端 SSE 处理、Zustand 状态管理、流式 UI 更新策略

### 📌 面试官提问

> 前端是怎么接收和展示 SSE 流式数据的？用了什么状态管理方案？在流式输出过程中，UI 更新的性能是怎么保证的？

### ✅ 回答思路

- **SSE 接收**：基于 `fetch` + `ReadableStream` 手动解析，而非 `EventSource` API
- **状态管理**：Zustand（轻量级 React 状态库）
- **性能优化**：增量更新、分批渲染

### 🗣️ 参考答案

**SSE 接收方案**（`useStreamResponse` Hook）：

我们没有用浏览器原生的 `EventSource` API，而是基于 `fetch` + `ReadableStream` 手动解析 SSE 协议。原因是 `EventSource` 只支持 GET 请求且无法自定义 Header（比如携带认证 token）。

核心实现：
1. 发起 `fetch` 请求，获取 `response.body`（ReadableStream）
2. 通过 `ReadableStream.getReader()` 逐块读取数据
3. 手动解析 `event:` 和 `data:` 字段，支持多行 data 拼接
4. 根据事件类型（meta / message / finish / done / cancel / error）分发到对应回调

**内置重试机制**：支持指数退避自动重试（默认 2 次），网络抖动时用户无感知。

**状态管理**（`chatStore.ts`，Zustand）：

```typescript
// 核心状态
{
  sessions: Session[],       // 会话列表
  messages: Message[],       // 当前会话消息
  isStreaming: boolean,      // 是否正在流式输出
  deepThinkingEnabled: boolean,  // 深度思考模式
  streamTaskId: string,      // 当前流任务ID（用于取消）
}
```

**`sendMessage()` 的流程**：

1. 创建 user 消息（立即渲染）+ assistant 消息（初始为 streaming 状态，内容为空）
2. 构建 SSE URL：`/rag/v3/chat?question=&conversationId=&deepThinking=`
3. 调用 `createStreamResponse()` 建立 SSE 连接
4. `onMessage` 回调中**增量拼接** assistant 消息内容，触发 React 重渲染
5. `onFinish` 时将 assistant 消息标记为完成状态
6. 支持用户点击"停止生成"（调用 `/rag/v3/stop` API）

**性能考量**：
- 后端按 5 字符一批推送，避免每个 token 都触发网络事件
- 前端每次 `onMessage` 只更新单条消息的 content 字段，Zustand 的细粒度订阅确保只重渲染聊天气泡组件

### 🔄 可能追问

1. 为什么选 Zustand 而不是 Redux 或 Context API？
2. 如果流式输出过程中用户切换了会话，怎么处理正在进行的流？
3. 深度思考模式在前端是怎么展示的？和普通模式有什么 UI 区别？

---

## 17. Spring Boot 多模块架构

> 💡 **考察重点**：架构设计 — 模块划分原则、依赖管理、关注点分离

### 📌 面试官提问

> 你们项目是 Maven 多模块结构。能说说为什么要这样拆分？每个模块的职责是什么？模块之间的依赖关系是怎么设计的？

### ✅ 回答思路

- **拆分原则**：按职责边界和复用性划分
- **四个模块**：framework（基础设施）、infra-ai（AI 能力）、bootstrap（业务主体）、mcp-server（工具服务）
- **依赖链**：单向依赖，避免循环

### 🗣️ 参考答案

项目拆分为四个 Maven 模块，遵循**关注点分离**和**依赖方向单一**的原则：

```
ragent (parent pom) — Spring Boot 3.5 + Java 17
├── framework      — 公共基础框架
├── infra-ai       — AI 基础设施（依赖 framework）
├── bootstrap      — 主业务应用（依赖 framework + infra-ai）
└── mcp-server     — 独立 MCP 工具服务器（独立依赖，不依赖其他模块）
```

**各模块职责**：

| 模块 | 职责 | 核心内容 |
|------|------|---------|
| `framework` | 与业务无关的**通用基础设施** | 异常体系、分布式 ID（雪花算法）、幂等机制、Redis Stream MQ、链路追踪、Sa-Token 认证 |
| `infra-ai` | **AI 能力抽象层** | 多模型路由（Chat/Embedding/Rerank）、断路器健康管理、首包探测、各提供商客户端 |
| `bootstrap` | **全部业务代码** + Spring Boot 启动入口 | RAG 对话、知识库管理、文档 ETL、用户认证、管理后台 |
| `mcp-server` | **独立的 MCP 工具服务** | JSON-RPC 分发、工具注册/执行，完全独立部署 |

**依赖方向**：

```
bootstrap → infra-ai → framework
mcp-server（独立，仅依赖 spring-boot-starter-web + gson）
```

严格遵循**单向依赖**：上层可以依赖下层，下层不能依赖上层。`framework` 是最底层，不依赖任何业务模块。

**这样拆分的好处**：
- `framework` 可以被其他项目直接复用
- `infra-ai` 封装了 AI 调用的复杂性，业务层只需调用 `RoutingLLMService` 等接口
- `mcp-server` 独立部署，故障隔离，可以独立扩缩容

### 🔄 可能追问

1. 为什么不把 infra-ai 合并到 framework 里？拆分的依据是什么？
2. bootstrap 模块是不是太大了？有没有考虑过进一步拆分？
3. mcp-server 完全独立不依赖其他模块，如果需要共享一些数据结构怎么办？

---

## 18. 异常体系与统一返回

> 💡 **考察重点**：基础知识 — 分层异常体系的设计原则、全局异常处理、统一 API 响应结构

### 📌 面试官提问

> 你们项目中有一套自定义的异常体系。能说说为什么要自定义异常？是怎么分层的？和 Spring 的全局异常处理是怎么配合的？

### ✅ 回答思路

- **分层设计**：ClientException / ServiceException / RemoteException
- **统一返回**：`Result<T>` 包装所有 API 响应
- **全局捕获**：`@RestControllerAdvice` + `@ExceptionHandler`

### 🗣️ 参考答案

**异常分层设计**：

```
AbstractException (code + message + errorCode)
├── ClientException   — 客户端错误（参数校验失败、非法请求等）
├── ServiceException  — 服务端错误（业务逻辑异常，如知识库不存在）
└── RemoteException   — 远程调用错误（AI 模型调用失败、MCP 服务超时等）
```

基类 `AbstractException` 包含三个字段：
- `code` — HTTP 状态码级别的错误码
- `message` — 面向用户的友好提示
- `errorCode` — 内部错误编码（用于日志排查）

**为什么要这样分层？** 因为不同来源的异常，处理策略不同：
- `ClientException` — 直接返回给前端，告知用户哪里填错了
- `ServiceException` — 返回业务提示 + 记录 WARN 日志
- `RemoteException` — 返回"服务暂时不可用" + 记录 ERROR 日志 + 触发告警

**统一返回结构**（`Result<T>`）：

```json
{
  "code": "0",          // "0" 表示成功，其他为错误码
  "message": "success",
  "data": { ... }       // 业务数据
}
```

通过 `Results` 工具类构建：`Results.success(data)` / `Results.failure(code, message)`。

**全局异常处理**（`GlobalExceptionHandler`）：

使用 `@RestControllerAdvice` 统一捕获，按优先级处理：

| 异常类型 | 处理方式 |
|---------|---------|
| `MethodArgumentNotValidException` | 提取校验错误信息，返回 400 |
| `AbstractException`（自定义） | 按子类型返回对应错误码和消息 |
| `NotLoginException`（Sa-Token） | 返回 401 未登录 |
| `NotRoleException`（Sa-Token） | 返回 403 无权限 |
| `MaxUploadSizeExceededException` | 返回"文件大小超过限制" |
| `Throwable`（兜底） | 返回 500 + 记录错误日志 |

### 🔄 可能追问

1. `errorCode` 和 `code` 有什么区别？为什么需要两个编码？
2. 前端是怎么根据返回结构判断请求成功或失败的？
3. 如果是异步任务（比如 MQ 消费者）中抛出异常，全局异常处理能捕获到吗？

---

## 19. 条件装配与多后端切换

> 💡 **考察重点**：原理深度 — Spring 条件装配机制、多实现切换的设计思路、`@ConditionalOnProperty` 原理

### 📌 面试官提问

> 你提到向量存储支持 pgvector 和 Milvus 两种后端，通过配置切换。这个是怎么实现的？Spring 的条件装配机制你了解多少？

### ✅ 回答思路

- **实现方式**：同一接口两套实现，通过 `@ConditionalOnProperty` 按配置激活
- **Spring 条件装配原理**：`@Conditional` 注解族在 Bean 注册阶段做条件判断
- **设计优势**：零代码修改切换后端，符合开闭原则

### 🗣️ 参考答案

**实现方式**：

我们定义了统一的接口（`VectorStoreService`、`RetrieverService`、`VectorStoreAdmin`），然后分别提供 pgvector 和 Milvus 两套实现：

```java
// pgvector 实现
@Service
@ConditionalOnProperty(name = "rag.vector.type", havingValue = "pg")
public class PgVectorStoreService implements VectorStoreService { ... }

// Milvus 实现
@Service
@ConditionalOnProperty(name = "rag.vector.type", havingValue = "milvus")
public class MilvusVectorStoreService implements VectorStoreService { ... }
```

切换时只需修改配置文件：

```yaml
rag:
  vector:
    type: pg      # 改为 milvus 即可切换
```

**Spring 条件装配原理**：

`@ConditionalOnProperty` 是 Spring Boot 提供的条件注解，底层基于 `@Conditional` 机制：

1. Spring 在 Bean 定义注册阶段（不是实例化阶段）扫描到 `@ConditionalOnProperty`
2. 调用对应的 `Condition.matches()` 方法，检查 `Environment` 中的属性值
3. 条件匹配 → 注册该 Bean 定义；不匹配 → 跳过，这个类根本不会被实例化

**同类注解还有**：
- `@ConditionalOnClass` — 类路径存在某个类时生效
- `@ConditionalOnMissingBean` — 容器中没有某个 Bean 时生效
- `@ConditionalOnExpression` — SpEL 表达式为 true 时生效
- `@Profile` — 基于 Spring Profile 激活

**设计优势**：
- 业务代码只依赖接口（`VectorStoreService`），不关心具体实现
- 切换后端无需修改任何 Java 代码，只改配置
- 符合**开闭原则**：新增后端只需添加一个新的实现类 + `@ConditionalOnProperty`

### 🔄 可能追问

1. 如果配置文件里 `rag.vector.type` 写错了（比如写了个不存在的值），会怎样？启动会报错吗？
2. `@ConditionalOnProperty` 和 `@Profile` 有什么区别？什么场景用哪个？
3. 如果想在运行时动态切换向量后端（不重启应用），有可能实现吗？

---

## 20. 项目中的设计模式应用

> 💡 **考察重点**：综合应用 — 能否结合实际项目场景说清楚设计模式的应用动机和收益

### 📌 面试官提问

> 最后，能总结一下你在这个项目中用到了哪些设计模式？每种模式解决了什么问题？不用面面俱到，挑几个你觉得用得最好的说。

### ✅ 回答思路

- 不要泛泛而谈，要结合**具体场景**说明每个模式解决了什么问题
- 重点说 3-4 个，讲清楚"为什么用"和"带来了什么好处"

### 🗣️ 参考答案

我挑四个在项目中用得最典型的设计模式来说：

**1. 策略模式 — 多处应用，解决"同一行为多种实现"**

| 应用场景 | 接口 | 实现 |
|---------|------|------|
| 分块策略 | `ChunkingStrategy` | 固定大小 / 段落 / 句子 / 结构感知 |
| 检索通道 | `SearchChannel` | 意图定向 / 向量全局 |
| 数据获取 | `DocumentFetcher` | S3 / HTTP / 本地 / 飞书 |
| AI 客户端 | `ChatClient` | 百炼 / SiliconFlow / Ollama |

好处：新增一种策略只需实现接口并注册，不用修改已有代码（开闭原则）。

**2. 责任链模式 — 检索后置处理器**

```
去重 DeduplicationPostProcessor → Rerank RerankPostProcessor → ...
```

`MultiChannelRetrievalEngine` 注入 `List<SearchResultPostProcessor>`，按顺序依次执行。好处是：
- 处理器之间**完全解耦**
- 新增处理器只需实现接口并设置 `@Order`
- 可以灵活调整处理顺序

**3. 装饰器模式 — 首包探测**

`ProbeBufferingCallback` 包装了真实的 `StreamCallback`：
- 探测阶段：拦截所有事件缓存到 buffer
- 提交后：回放缓存并透传后续事件

不修改原有 callback 逻辑，**透明地增加了探测能力**。

**4. 模板方法模式 — 分块器基类**

`AbstractEmbeddingChunker` 定义了"分块 → 嵌入"的骨架流程，子类只需实现 `doChunk()` 方法。好处是：
- 避免每个分块器都重复写 Embedding 调用逻辑
- 保证了"分块后必须嵌入"的流程约束

**其他模式**（简要提及）：
- **断路器模式**：`ModelHealthStore` 的三态状态机
- **工厂模式**：`ChunkingStrategyFactory` 管理分块策略注册
- **生产者-消费者模式**：Redis Stream MQ 的异步文档处理
- **AOP 代理模式**：幂等、链路追踪、限流等横切关注点

### 🔄 可能追问

1. 策略模式和简单的 if-else 相比，什么时候用策略模式更合适？什么时候 if-else 就够了？
2. 责任链模式中如果某个处理器出错了，整个链会怎么处理？
3. 你觉得这些设计模式中，哪个如果不用的话对代码影响最大？

---

## 📊 考察维度分布

| 考察维度 | 题号 | 数量 |
|---------|------|:----:|
| 🏗️ 架构设计 | 1, 5, 12, 15, 17 | 5 |
| 🔬 原理深度 | 2, 3, 6, 8, 11, 13, 19 | 7 |
| 🛠️ 技术应用 | 4, 7, 9, 10, 14, 16 | 6 |
| 📚 基础知识 | 18 | 1 |
| 🧩 综合应用 | 20 | 1 |
