# Query Embedding 共享复用重构 Code Review

## 摘要

对"Query Embedding 共享复用"重构（未提交的 working tree 改动）执行了 5 角度全量 code review。该重构目标是将 RAG 检索阶段的多次重复 Embedding HTTP 调用（4×）合并为 1 次，跨通道共享结果向量。Review 共产出 15 个 verified findings，其中 3 个高危：preload 异常无 try/catch 导致容错降级、SearchContext @Data setter 暴露线程安全容器、ModelRouteResult.target() 泄露 apiKey。

## 背景与目标

- 用户在分析两条 RAG 对话日志（"你好" 和 "拼多多开票信息"）时，发现同一个查询文本在 VectorGlobalSearch（3 个 collection）和 IntentDirectedSearch（1 个 collection）中被独立 embed 了 4 次
- 虽然并行执行下实测延迟没叠加（~70ms），但 token 费用理论上是 4 倍，且随 QPS 增长问题会放大
- 决定实现 `QueryEmbeddingContext` + `preload` 机制：在 `prepareQueryEmbeddings` 阶段一次性 embed，各通道通过 `getOrCreate` 复用缓存

## 关键发现

### 高危（3 项）

1. **preload 异常打破容错隔离**（`MultiChannelRetrievalEngine.java:229`）
   - `prepareQueryEmbeddings` 调用 `context.getQueryEmbeddingContext().preload(allTargets)` 无 try/catch
   - 任意一个 target 的 embedding 失败，整个 `executeSearchChannels` 抛异常，所有通道返回 0 chunks
   - 改造前每个 channel 在自己的 `CompletableFuture` try/catch 里独立 embed，一个通道失败不影响其他
   - **回归**：从"target 级容错"降级为"全局一票否决"

2. **SearchContext @Data setter 暴露线程安全容器**（`SearchContext.java:80`）
   - `@Data` 生成 `setSearchTargets(Map)`，`@Builder` 生成 `.searchTargets(Map)` — 都能替换掉 `@Builder.Default ConcurrentHashMap`
   - 若写成 `SearchContext.builder().searchTargets(new HashMap<>()).build()`，并行 channel 的读写直接踩 HashMap 并发 bug

3. **ModelRouteResult.target() 泄露 apiKey**（`ModelRouteResult.java:27`）
   - `ModelTarget` 同时暴露 `candidate()` 和 `provider()`
   - `AIModelProperties.ProviderConfig.apiKey`（`AIModelProperties.java:159`）通过 Lombok getter 完全开放
   - 任何 `log.info("result={}", result)` 或 Jackson 序列化都可能把 apiKey 写进日志

### 中危（6 项）

4. **ConcurrentHashMap.computeIfAbsent lambda 内阻塞 HTTP**（`QueryEmbeddingContext.java:128`） — bin 锁被持有整段 embedding HTTP 往返，极端情况下若 createEmbedding 递归回调同一 map 会死锁
5. **null embeddingModelId 全部映射为 `__default__` cache key**（`QueryEmbeddingContext.java:211`） — 历史遗留 KB 行 embedding_model=NULL 时，不同维度集合共用同一向量，维度不匹配
6. **`buildSearchContext` 赋 `intents=List.of()` 不可变**（`MultiChannelRetrievalEngine.java:317`） — 扩展通道对 `context.getIntents()` 做 add/removeIf 时，仅在 subIntents 为空时抛 UnsupportedOperationException
7. **`executeParallelRetrieval(SearchContext null, ...)` 传 null question 到 embed(null)**（`AbstractParallelRetriever.java:79`） — 新重载允许 context=null，question=null 泄透到 embeddingService.embed(null) NPE
8. **VecGlobal 用 `Collectors.toMap` 合并 collectionName → embeddingModelId，`(left, right) -> left` 结果随 DB 行顺序随机**（`VectorGlobalSearchChannel.java:173`） — 两条 KB 行共享 collection_name 时 embeddingModelId 会翻转
9. **fallback 路径绕过缓存重新 embed**（`AbstractParallelRetriever.java:124`） — preload/getOrCreate 失败后 resolveQueryEmbedding 返回 null，回退到 retrieverService.retrieve 重新独立 embed，退化为 N 次调用

### 低危 / 关注（6 项）

10. **QueryEmbedding.vector 是共享 `float[]`，无防御性拷贝**（`QueryEmbedding.java:53`） — 项目内已有 `PgRetrieverService.normalize()` 就地修改入参的先例，未来新增 RetrieverService 极可能污染缓存
11. **normalize() 对全零向量不处理**（`QueryEmbeddingContext.java:235`） — pgvector 的 `<=>` 算子在零向量上行为不可预测
12. **EmbeddingService.default `embedWithMetadata` 丢失 embeddingProvider**（`EmbeddingService.java:83`） — 默认实现不填 provider，非 RoutingEmbeddingService 实现日志里 provider=null
13. **IntentParallelRetriever.IntentTask 从 2 参改成 4 参**（`IntentParallelRetriever.java:45`） — 公开 record 的 canonical constructor 变了，没有 shim，外部扩展编译失败
14. **IntentParallelRetriever.createRetrievalTask 用 channel question 而非 task.query 写入 RetrieveRequest**（`IntentParallelRetriever.java:134`） — 未来若走 BM25 混合检索，BM25 侧和向量侧用的查询文本不一致
15. **IntentDirectedSearchChannel.loadKnowledgeBases 用字面字符串 column 名**（`IntentDirectedSearchChannel.java:214`） — 比原 lambdaQuery 方案少了列重命名时的编译期校验

## 实现细节

### 核心新增文件

- `bootstrap/.../retrieve/QueryEmbedding.java` — 向量+元数据 DTO
- `bootstrap/.../retrieve/QueryEmbeddingContext.java` — per-request 缓存，key=`modelIdquery`，ConcurrentHashMap 存储
- `bootstrap/.../channel/SearchTarget.java` — 检索目标描述（channelName, collectionName, embeddingModelId, query）
- `infra-ai/.../embedding/EmbeddingResult.java` — 带元数据的 embedding 结果
- `infra-ai/.../model/ModelRouteResult.java` — 路由执行结果（response + target）
- `bootstrap/src/test/.../QueryEmbeddingReuseTests.java` — 集成测试

### 关键改动文件

- `MultiChannelRetrievalEngine` — 新增 `prepareQueryEmbeddings`，在通道并行前同步调用
- `SearchContext` — 新增 `searchTargets`（ConcurrentHashMap）和 `queryEmbeddingContext` 字段
- `SearchChannel` 接口 — 新增 `default resolveSearchTargets(SearchContext)` 方法
- `AbstractParallelRetriever` — 新增 `executeParallelRetrieval(SearchContext, List<T>, int)` 重载 + `resolveQueryEmbedding` 优先走缓存
- `CollectionParallelRetriever` — 从 `AbstractParallelRetriever<String>` 改为 `<SearchTarget>`，新增 `createRetrievalTask(... QueryEmbedding)` 重载
- `IntentParallelRetriever` — IntentTask record 从 2 参改为 4 参，新增 overload
- `RoutingEmbeddingService` — `embed()` 改为 `embedWithMetadata()` delegate，新增 `buildEmbeddingResult` 提取路由元数据
- `ModelRoutingExecutor` — 新增 `executeWithFallbackResult` 返回 `ModelRouteResult<T>`

### 关键配置

```yaml
# per-KB 模型配置（t_knowledge_base.embedding_model 列）
# 为 NULL 时走全局默认路由 → 可能维度不匹配（见 finding #5）
```

## 决策与权衡

- **选择在 preload 阶段串行 embed** 而非通道内并行：换取"同一查询只 embed 一次"的确定性，代价是通道执行前增加约 70-140ms 的串行等待（preload for-loop 非并行）
- **保留旧 3 参 `createRetrievalTask` 作为 fallback**：resolveQueryEmbedding 返回 null 时自动走旧路径，保证改造向后兼容，但 fallback 会重新 embed（退化为 N 次调用）
- **`SearchChannel.resolveSearchTargets` 用 default 空实现**：不改非向量通道接口，但代价是调用方无法区分"通道不是向量通道"和"通道解析失败返回了空列表"

## 未完成事项

1. **给 `prepareQueryEmbeddings` 加 try/catch** — preload 失败后应降级为"跳过预生成，让各通道 fallback"，而非整条链路失败
2. **SearchContext 移除 `@Data`，改用 `@Getter` + 手写 `putSearchTargets`** — 封掉 `setSearchTargets` 暴露口，或在 setter 里做类型检查
3. **ModelRouteResult 不直接暴露 ModelTarget** — 新建一个不带 apiKey 的 `ModelRouteMeta` DTO，只暴露 id/provider/model/priority
4. **`QueryEmbedding.getVector()` 返回 clone** — 防止共享数组被就地修改
5. **normalize 处理全零向量** — 至少 log.error + 抛异常，不能悄悄返回零数组
6. **修复 cache key `__default__` 碰撞** — null embeddingModelId 不应和"显式配置了 __default__ 这个模型名"混淆；改用单独 sentinel 或直接 throw

## 备注

- 本次 review 的 diff 范围：11 个已修改文件 + 5 个新增文件（均为 working tree 未提交状态）
- 意图树的 `IntentNode.id` 为 nullable String，resolveSearchTargets 里对 null id 的处理隐式走 fallback（finding #9），建议在 schema 层面加 NOT NULL
- Embedding 维度问题（#5）在本次改造前已存在（旧路径也走全局默认模型），但新路径为每个 KB 单独携带 embeddingModelId 会让 NULL 值问题更难排查（日志显示 `__default__` 而非具体模型名）
