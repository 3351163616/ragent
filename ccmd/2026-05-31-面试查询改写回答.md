# Query 改写子系统面试应答材料

## 摘要

为面试问题「你的 RAG 系统是怎么进行 Query 改写的」准备了完整应答材料。通过 5 个并行 agent 精读改写子系统的全部代码（主流程、术语归一化、质量守卫、Prompt 设计、调用链与配置），交叉验证后产出一份端到端的面试回答。涵盖了改写流水线的设计哲学、关键实现细节、踩坑经验、以及可主动展示的亮点。

## 背景与目标

- 面试中被问到 RAG 系统的 Query 改写实现
- 需要从源码级深度理解改写子系统，并综合成逻辑清晰、由浅入深的面试回答
- 不只是罗列功能，而是要体现"为什么这么做"和工程化思维

## 关键发现

### 改写是流水线，不是单次 LLM 调用

完整链路：**规则术语归一化 → LLM 改写(指代消解 + 同义扩展 + 复合问句拆分) → 质量守卫 → 多级兜底**。产物 `RewriteResult(rewrittenQuestion, subQuestions)` 同时驱动下游意图识别、澄清引导、多通道检索。设计哲学一句话：**确定性的事交规则、不确定的事交 LLM、再用工程兜底为 LLM 不可靠性兜底**。

### 历史注入的踩坑经验

最初用标准多轮 `user/assistant` 消息注入历史，模型会"接着对话"去回答历史问题，破坏 JSON 输出约束。改成把最近 2 轮历史拼成纯文本块塞进单条 user 消息，前缀加「仅供指代消解参考，禁止回答历史问题」——既保留指代消解能力又稳住结构化输出。代码注释明确记录了原因（`MultiQuestionRewriteService.java:224-227`）。

### 质量守卫的反幻觉设计

`QueryRewriteGuard`（提交 `aa6e9de` 新增）是 Prompt 软约束的运行时硬兜底：

1. **语义相似度校验**：复用检索同一套 `EmbeddingService` 对"归一化输入 vs LLM 输出"算余弦，低于 0.8 阈值即丢弃。口径与检索对齐，批量优先单条降级，算不出来 fail-closed
2. **短 Query 防噪**（≤8 字才触发）：禁止拆多问、必须保留核心词、禁止引入高风险业务词
3. **退费↔收费双向硬规则**：纯余弦相似度拦不住业务语义反转（两词向量可能很近），用领域规则补 embedding 盲区

### 术语归一化前置

归一化在 LLM 之前执行，结果是真正喂给 LLM 的输入。长词优先匹配靠排序近似（priority 降序 + 词长降序），`applyMapping` 手写扫描防重复归一化，Redis 缓存 + 写后失效 + DB 回源。

### 已发现的代码问题

- **注释/实现不一致**：`QueryTermMappingDO` 注释写"priority 越小优先级越高"，但 `loadMappings` 实现是 priority 降序（越大越优先）
- **match_type 未完全落地**：DO 规划了精确/前缀/正则/整词四类，但 `normalize` 只处理 `matchType==1`（精确），其余直接 skip

## 实现细节 / 代码片段

### 核心文件清单

| 文件 | 职责 |
|---|---|
| `bootstrap/.../rag/core/rewrite/MultiQuestionRewriteService.java` | 编排主流程：开关判断 → 归一化 → LLM 改写 → 守卫 → 兜底 |
| `bootstrap/.../rag/core/rewrite/QueryTermMappingService.java:43-98` | 术语归一化：Redis 缓存 + DB 回源 + 长词优先排序 |
| `bootstrap/.../rag/core/rewrite/QueryTermMappingUtil.java:27-67` | 安全替换：手写扫描 + alreadyTarget 防重复归一化 |
| `bootstrap/.../rag/core/rewrite/QueryRewriteGuard.java:78-162` | 质量守卫：语义相似度 + 短 Query 防噪 |
| `bootstrap/.../rag/core/rewrite/QueryRewriteService.java` | 三层递进接口，default 方法逐层委托 |
| `bootstrap/.../rag/service/pipeline/StreamChatPipeline.java:89-118` | RAG 编排流水线，定位改写在管线的精确位置 |
| `bootstrap/src/main/resources/prompt/user-question-rewrite.st` | 改写 Prompt 模板（160 行，8 个 few-shot） |

### 关键配置项

```yaml
rag.query-rewrite.enabled: true          # 总开关，关 → 仅归一化 + 规则拆分
rag.query-rewrite.semantic-validation.enabled: true  # 语义相似度校验开关
rag.query-rewrite.semantic-validation.min-similarity: 0.8  # 余弦阈值
rag.query-rewrite.short-query.max-length: 8              # 短 Query 判定长度
```

LLM 请求参数（硬编码）：`temperature=0.1`, `topP=0.3`, `thinking=false`。

### 三层降级兜底

| 失败场景 | 兜底动作 |
|---|---|
| 总开关关闭 | 仅术语归一化 + `ruleBasedSplit` 规则拆分 |
| LLM 调用异常 / JSON 解析失败 | `RewriteResult(归一化问题, [归一化问题])` |
| 质量守卫不通过 | 退回归一化问题 |

兜底统一退到「归一化问题」而非裸原始问题，保留一层确定性收益。

### 面试叙事大纲

1. **一句话定位**：改写是流水线而非单次调用
2. **术语归一化**：规则前置 + 长词优先 + 防重复归一化
3. **LLM 改写四合一**：检索化改写 + 指代消解 + 术语扩展 + 问句拆分
4. **踩坑经验**：历史注入方式从多轮消息改为纯文本拼接
5. **质量守卫**：语义相似度 + 短 Query 防噪 + 退费↔收费硬规则
6. **多级降级**：任何环节失败都不让管线崩
7. **可观测**：守卫日志带 similarity/threshold/before/after，Prompt 有 contains 回归测试

## 决策与权衡

- **规则 vs LLM 分工**：已知可枚举的术语对齐用 DB 映射表（零成本零幻觉可审计），需要语义理解的指代消解/拆分交给 LLM
- **不信任 LLM 的 `should_split` 字段**：Java 端只按 `sub_questions` 数组长度判定拆分，避免标记与内容打架
- **守卫 fail-closed 而非 fail-open**：embedding 报错时视为不通过而非放行，安全优先
- **复用检索 EmbeddingService 做相似度**：保证判断口径与检索口径对齐
- **低温度低 topP**：改写是确定性结构化任务，不是创造性任务

## 备注与提醒

- 改写阶段同步执行在 `chatEntryExecutor` 线程池，本身不并行；并行发生在下游意图识别（`intentClassifyExecutor`）
- 改写前后各有 `taskManager.isCancelled()` 检查，用户中途取消可短路
- 面试延伸追问准备：线程模型、术语映射局限（match_type 未完全落地）、阈值调优策略（配合日志的 similarity 分布线上调）
