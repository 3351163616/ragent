# 评估中心自动化评测功能说明

## 摘要

评估中心是 Ragent RAG 系统的自动化质量评估后台，用于在 Prompt、模型、意图树、检索策略、分块策略或 MCP 工具逻辑变更后，批量运行标准评测题，并用自动评分器计算意图命中、文档召回、上下文精度、MCP 工具命中、答案规则等指标。

它的目标不是替代普通聊天页，而是给研发、测试和运营提供一套"上线前考试系统"：

- 用评测套件管理固定试卷。
- 用评测用例描述问题和 Ground Truth。
- 用多次 Trial 对抗 LLM 随机性。
- 用自动 Scorer 生成可量化指标。
- 用 Trace 关联失败链路，辅助定位是改写、意图、检索、MCP 还是答案阶段出了问题。

## 功能定位

### 解决的问题

当前 RAG/Agent 系统存在天然的不确定性：

- 同一个问题可能因为模型采样、改写、意图识别或 rerank 产生不同结果。
- Prompt 调优后很难靠肉眼判断整体质量是上升还是下降。
- 文档分块、OCR 清洗、模型切换、检索阈值变化都会影响召回质量。
- 线上出问题后，只看最终答案很难定位是哪一环坏了。

评估中心通过"标准题库 + 多次执行 + 自动评分 + Trace 归因"形成闭环，让质量变化可观察、可比较、可回归。

### 和 `/rag/eval` 的关系

现有 `/rag/eval` 是单题调试接口，主要用于快速查看某个问题的改写、意图和检索结果。

评估中心是在它之上的后台化能力：

- `/rag/eval`：单题即时调试。
- `/admin/eval/**`：评测套件、用例、批量运行、Trial、评分和报告。

## 当前实现范围

### 后端模块

主要代码位于：

```text
bootstrap/src/main/java/com/nageoffer/ai/ragent/rag/eval/
```

核心类：

| 类 | 作用 |
| --- | --- |
| `EvalAdminController` | 自动化评测后台 API，路径前缀 `/admin/eval` |
| `EvalAdminService` | 评测后台服务接口 |
| `EvalAdminServiceImpl` | 套件、用例、运行、Trial、评分和报告的核心实现 |
| `EvalRagRunner` | 单题 RAG 评测执行器，复用查询改写、意图识别、检索链路 |
| `EvalSnapshot` | 单次 Trial 的结构化快照 |
| `EvalGroundTruth` | 用例标准答案 JSON 的 Java 映射 |
| `EvalScorer` | 自动阅卷器扩展接口 |
| `EvalMetricResult` | 单项评分结果 |
| `IntentAccuracyScorer` | 意图叶子节点命中率 |
| `RetrievalDocRecallScorer` | 文档级召回率 |
| `RetrievalContextPrecisionScorer` | chunk 上下文来源精度 |
| `McpToolScorer` | MCP 工具命中评分 |
| `AnswerRuleScorer` | 基于 requiredFacts / forbiddenClaims 的答案规则评分 |

相关 DAO 位于：

```text
bootstrap/src/main/java/com/nageoffer/ai/ragent/rag/dao/entity/
bootstrap/src/main/java/com/nageoffer/ai/ragent/rag/dao/mapper/
```

新增实体：

```text
EvalSuiteDO
EvalCaseDO
EvalRunDO
EvalTrialDO
EvalScoreDO
```

### 前端模块

主要代码位于：

```text
frontend/src/pages/admin/eval/EvalPage.tsx
frontend/src/services/evalService.ts
```

路由和导航入口：

```text
frontend/src/router.tsx
frontend/src/pages/admin/AdminLayout.tsx
```

页面路径：

```text
/admin/eval
```

页面能力：

- 搜索、新增、编辑、删除评测套件。
- 新增、编辑、删除评测用例。
- 配置 Ground Truth JSON。
- 设置 Trial 次数并运行评测。
- 查看运行记录、Pass@K、Trial 成功率、耗时。
- 查看每个 Trial 的评分结果。
- 通过 traceId 跳转到链路追踪详情。

## 数据模型

数据库变更已加入：

```text
resources/database/schema_pg.sql
resources/database/upgrade_v1.6_to_v1.7.sql
```

### `t_eval_suite`

评测套件表，用来组织一组评测用例。

关键字段：

| 字段 | 说明 |
| --- | --- |
| `id` | 套件 ID |
| `name` | 套件名称 |
| `description` | 套件描述 |
| `version` | 套件版本 |
| `enabled` | 是否启用 |
| `deleted` | 逻辑删除标记 |

### `t_eval_case`

评测用例表，存储具体问题和 Ground Truth。

关键字段：

| 字段 | 说明 |
| --- | --- |
| `suite_id` | 所属评测套件 |
| `question` | 评测问题 |
| `category` | 分类 |
| `ground_truth` | 标准答案 JSON |
| `enabled` | 是否参与运行 |

### `t_eval_run`

一次评测运行记录。

关键字段：

| 字段 | 说明 |
| --- | --- |
| `suite_id` | 所属套件 |
| `trial_count` | 每个用例执行次数 |
| `status` | 运行状态 |
| `total_cases` | 总用例数 |
| `total_trials` | 总 Trial 数 |
| `completed_trials` | 已完成 Trial 数 |
| `passed_trials` | 通过 Trial 数 |
| `success_rate` | Trial 级成功率 |
| `pass_at_k` | Case 级 Pass@K |
| `summary_json` | 汇总报告 JSON |

### `t_eval_trial`

某个用例的一次执行记录。

关键字段：

| 字段 | 说明 |
| --- | --- |
| `run_id` | 所属运行 |
| `suite_id` | 所属套件 |
| `case_id` | 所属用例 |
| `trial_index` | 第几次 Trial |
| `status` | 执行状态 |
| `success` | 是否通过全部评分 |
| `trace_id` | 关联 RagTrace |
| `latency_ms` | RAG 执行耗时 |
| `snapshot_json` | 执行快照 |
| `error_message` | 错误信息 |

### `t_eval_score`

Trial 级指标评分。

关键字段：

| 字段 | 说明 |
| --- | --- |
| `run_id` | 所属运行 |
| `trial_id` | 所属 Trial |
| `case_id` | 所属用例 |
| `metric_name` | 指标名称 |
| `score_value` | 评分，0 到 1 |
| `passed` | 是否通过 |
| `reason` | 评分原因 |
| `detail_json` | 指标明细 |

## Ground Truth 格式

评测用例通过 `ground_truth` 字段描述标准答案，当前建议 JSON 格式：

```json
{
  "referenceDocIds": ["FAQ_VAC_001"],
  "expectedIntentLeafIds": ["intent_leave_policy"],
  "expectedTool": {
    "name": "sales_query",
    "args": {
      "region": "华东"
    }
  },
  "requiredFacts": ["年假按司龄递增"],
  "forbiddenClaims": ["年假可以无限结转"],
  "expectedAnswer": "可选，用于后续语义评估"
}
```

字段说明：

| 字段 | 说明 | 当前评分器 |
| --- | --- | --- |
| `referenceDocIds` | 期望召回的业务文档 ID | `RetrievalDocRecallScorer`、`RetrievalContextPrecisionScorer` |
| `expectedIntentLeafIds` | 期望命中的意图叶子节点 | `IntentAccuracyScorer` |
| `expectedTool.name` | 期望命中的 MCP 工具 | `McpToolScorer` |
| `requiredFacts` | 答案必须包含的事实点 | `AnswerRuleScorer` |
| `forbiddenClaims` | 答案不能包含的断言 | `AnswerRuleScorer` |
| `expectedAnswer` | 标准答案文本，后续可用于 LLM Judge | 暂未使用 |

## 执行流程

一次批量评测的大致流程：

```text
1. 管理员在评估中心创建评测套件。
2. 管理员录入或导入评测用例。
3. 点击运行评测，传入 suiteId 和 trialCount。
4. 后端创建 EvalRunDO。
5. 后端加载启用的 EvalCaseDO。
6. 每个 case 按 trialCount 执行多次。
7. EvalRagRunner 对单题执行：
   question -> rewrite -> intent -> retrieval -> EvalSnapshot
8. EvalScorer 链对 EvalSnapshot 和 Ground Truth 打分。
9. 写入 t_eval_trial 和 t_eval_score。
10. 聚合生成 successRate、Pass@K、metricAverages、failedCases。
11. 前端展示运行记录和 Trial 明细。
```

## 指标说明

### `successRate`

Trial 级成功率。

计算方式：

```text
successRate = passedTrials / completedTrials
```

例如 50 个用例，每个跑 5 次，总共 250 个 Trial，其中 200 个通过，则：

```text
successRate = 200 / 250 = 80%
```

### `passAtK`

Case 级 Pass@K。

当前实现中，一个 case 的 K 次 Trial 里只要有一次通过，该 case 就算通过。

计算方式：

```text
passAtK = passedCases / totalCases
```

例如 50 个用例，每个跑 5 次，其中 45 个用例至少有 1 次通过，则：

```text
passAtK = 45 / 50 = 90%
```

### `intent_accuracy`

意图识别准确率。

对比：

```text
groundTruth.expectedIntentLeafIds
snapshot.intentLeafIds
```

当前规则要求期望意图全部命中才通过。

### `retrieval_doc_recall`

文档级召回率。

对比：

```text
groundTruth.referenceDocIds
snapshot.retrievedDocIds
```

用于判断是否召回到了期望文档。

### `retrieval_context_precision`

chunk 上下文来源精度。

对比：

```text
groundTruth.referenceDocIds
snapshot.retrievedContextDocIds
```

用于判断召回的 chunk 是否主要来自参考文档。

### `mcp_tool_accuracy`

MCP 工具命中率。

对比：

```text
groundTruth.expectedTool.name
snapshot.mcpToolIds
```

注意：当前实现评估的是意图链路里识别到的 MCP 工具 ID，不等价于实际工具调用成功，也不校验工具参数。

### `answer_rule`

答案规则评分。

检查：

- `requiredFacts` 是否全部出现在答案中。
- `forbiddenClaims` 是否没有出现在答案中。

注意：当前 `EvalRagRunner` 只执行到检索阶段，尚未生成最终答案。因此只要配置了 requiredFacts 或 forbiddenClaims，当前阶段就可能失败，除非后续接入最终回答生成。

## API 概览

后端 API 前缀：

```text
/api/ragent/admin/eval
```

常用接口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/admin/eval/suites` | 分页查询评测套件 |
| `POST` | `/admin/eval/suites` | 创建评测套件 |
| `PUT` | `/admin/eval/suites/{id}` | 更新评测套件 |
| `DELETE` | `/admin/eval/suites/{id}` | 删除评测套件 |
| `GET` | `/admin/eval/cases` | 分页查询评测用例 |
| `POST` | `/admin/eval/cases` | 创建评测用例 |
| `POST` | `/admin/eval/suites/{suiteId}/cases/import` | 批量导入评测用例 |
| `PUT` | `/admin/eval/cases/{id}` | 更新评测用例 |
| `DELETE` | `/admin/eval/cases/{id}` | 删除评测用例 |
| `POST` | `/admin/eval/runs` | 运行评测 |
| `GET` | `/admin/eval/runs` | 分页查询运行记录 |
| `GET` | `/admin/eval/runs/{runId}` | 查询运行详情 |
| `GET` | `/admin/eval/runs/{runId}/trials` | 查询 Trial 明细 |
| `GET` | `/admin/eval/runs/{runId}/report` | 查询汇总报告 |

## 前端使用方式

访问路径：

```text
http://localhost:5173/admin/eval
```

典型使用流程：

1. 在左侧创建评测套件，例如"核心知识库问答 v1"。
2. 在用例管理区新增评测问题。
3. 填写 Ground Truth JSON。
4. 设置 Trial 次数。
5. 点击"运行评测"。
6. 查看运行记录中的 Pass@K、Trial 成功率和耗时。
7. 打开运行详情，查看每个 Trial 的评分。
8. 如果有 traceId，跳转到链路追踪页面分析失败原因。

## 配置项

配置位于：

```yaml
app:
  eval:
    enabled: true
    execution:
      max-concurrency: 2
      default-trial-count: 5
      timeout-seconds: 120
```

说明：

| 配置 | 说明 |
| --- | --- |
| `app.eval.enabled` | 是否启用评测接口 |
| `app.eval.execution.max-concurrency` | 评测 Trial 最大并发 |
| `app.eval.execution.default-trial-count` | 默认 Trial 次数 |
| `app.eval.execution.timeout-seconds` | 单次 Trial 建议超时时间 |

## 当前已知问题

### 1. 后端测试失败

当前 `EvalScorerTests.contextPrecisionUsesChunkLevelDocIds()` 使用了：

```java
List.of("FAQ_A", "FAQ_A", "FAQ_B", null)
```

`List.of` 不允许 null，会直接抛 `NullPointerException`。

建议改为：

```java
Arrays.asList("FAQ_A", "FAQ_A", "FAQ_B", null)
```

或：

```java
new ArrayList<>(Arrays.asList("FAQ_A", "FAQ_A", "FAQ_B", null))
```

### 2. 评测运行仍是同步 HTTP 长请求

当前前端点击"运行评测"后会等待后端完整执行结束。

问题：

- 大套件容易超过 HTTP 超时。
- 页面无法实时看到进度。
- 不适合 50 题 × 5 Trial 这类批量评测。

建议：

- `POST /admin/eval/runs` 只创建 run 并返回 runId。
- 后台异步执行 Trial。
- 前端轮询 `/admin/eval/runs/{runId}`。
- 增加取消接口。

### 3. Trial 提交存在队列上限风险

当前实现一次性提交所有 Trial 到线程池。

线程池队列大小为 200，如果执行 50 题 × 5 次 = 250 个 Trial，可能触发拒绝策略。

建议：

- 使用 `CompletionService`。
- 或按 case 分批提交。
- 或使用信号量控制在途任务数量。
- 不要一次性提交全部 Trial。

### 4. 答案评分当前语义不完整

`AnswerRuleScorer` 依赖 `snapshot.answer`，但当前 `EvalRagRunner` 只执行到检索阶段，不生成最终答案。

影响：

- 如果 Ground Truth 配置了 `requiredFacts` 或 `forbiddenClaims`，该用例可能被判失败。

建议：

- 第一阶段暂时禁用答案评分器。
- 或新增"最终回答评测模式"，复用完整 RAG 对话链路生成 answer。

### 5. MCP 工具评分当前只评意图，不评实际调用

当前 `mcp_tool_accuracy` 来自意图识别结果里的 MCP toolId。

它可以说明"意图上选中了这个工具"，但不能说明：

- MCP 工具真的被调用了。
- 工具参数正确。
- 工具返回结果可用。

建议后续增加：

- `mcp_call_success`
- `mcp_args_accuracy`
- `mcp_result_validity`

### 6. 空 Ground Truth 可能被误判成功

当前如果某个用例没有任何可评分项，`metricResults` 为空时会被视为成功。

建议：

- 至少要求每个用例产生一个评分项。
- 空 Ground Truth 标记为 `CONFIG_ERROR` 或 `UNSCORED`。

## 后续演进建议

### 第一阶段：让当前功能稳定可用

- 修复 `EvalScorerTests`。
- 改造评测运行方式为异步后台任务。
- 避免一次性提交过多 Trial。
- 明确 `answer_rule` 在当前阶段是否启用。
- 明确空 Ground Truth 的通过语义。

### 第二阶段：完善质量指标

- 增加 MCP 参数评分。
- 增加 rerank 前后对比指标。
- 增加 context recall / MRR / NDCG。
- 增加 chunk 质量指标，例如 OCR 空格率、乱码率。

### 第三阶段：接入最终答案评估

- 复用完整 RAG 生成链路得到 `answer`。
- 增加 LLM-as-Judge。
- 固化 Judge Prompt 和模型版本。
- 支持答案事实性、引用完整性、无幻觉评分。

### 第四阶段：版本对比和发布门禁

- 支持 baseline run。
- 支持 current vs baseline 对比。
- 支持 CI 或后台一键评测。
- P0 用例失败时阻断发布。

## 验证记录

当前验证结果：

```powershell
.\mvnw test -pl bootstrap -am '-Dtest=EvalScorerTests' '-Dsurefire.failIfNoSpecifiedTests=false' '-DforkCount=0'
```

结果：失败。

失败原因：`EvalScorerTests` 中 `List.of(..., null)` 抛出 `NullPointerException`。

前端构建：

```powershell
cd frontend
npm run build
```

结果：通过。

备注：Vite 有 chunk size warning，暂不阻断。

## 总结

评估中心当前已经具备自动化评测体系的骨架：

- 有评测套件。
- 有评测用例。
- 有多 Trial 执行。
- 有自动评分器。
- 有运行报告。
- 有前端管理页面。
- 有 Trace 关联设计。

但它还不适合作为生产级发布门禁直接使用。当前最需要优先修复的是测试失败、同步长请求、线程池队列拒绝、答案评分语义和空 Ground Truth 误判成功问题。

修完这些后，评估中心就可以作为 Ragent 的第一版质量回归平台，支撑后续 Prompt 调优、模型切换、意图树调整、检索策略修改和文档处理策略变更。
