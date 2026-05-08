# 查询改写历史消息结构优化

> 优化日期：2026-05-08
> 涉及模块：`bootstrap/.../rag/core/rewrite/MultiQuestionRewriteService`、`infra-ai/.../util/LLMResponseCleaner`

---

## 1. 问题现象

用户在多轮对话中提问时，查询改写阶段偶尔失败。LLM 不按 system prompt 要求输出 JSON，而是返回对话式文本：

```
raw=我无法直接查看或列出知识库中的具体文件列表和完整内容。我的工作方式是：
1. **你提问** → 系统根据你的问题在知识库中检索相关片段
...
```

`MultiQuestionRewriteService.parseRewriteAndSplit` 尝试 `JsonParser.parseString()` 解析失败，触发兜底逻辑（使用归一化问题替代改写结果），检索质量下降。

## 2. 排查过程

### 2.1 初步怀疑：代理注入 Claude Code system prompt

`bigmodel_proxy.py` 的 `_ensure_system_prompt` 会将 `"You are Claude Code, Anthropic's official CLI for Claude."` 前置到 system 数组中。初步分析认为这与项目的改写 prompt 角色定义冲突。

**实际验证**：用 curl 直接向代理发送完整模拟请求（`system` = 改写 prompt），同步 5 次 + 流式 5 次，**全部返回合法 JSON**。代理前置 Claude Code 块的影响可忽略，此假设被推翻。

### 2.2 第二次怀疑：模型候选路由到了其他 provider

`application.yaml` 中 `claude-opus`（moyu）priority=0 是首选，`glm-5.1` priority=1 是兜底。怀疑实际走的不是 glm-5.1。

**用户确认**：前端手动选择了 glm-5.1 模型，排除路由问题。

### 2.3 锁定根因：多轮对话历史诱导模型偏离

对比测试脚本与项目实际请求的差异：

| 维度 | 测试脚本 | 项目实际 |
|------|----------|----------|
| messages 结构 | `[system] + [user]` | `[system] + [user] + [assistant] + [user]` |
| 历史消息 | 无 | 有（从 Redis 加载最近 2 轮） |
| 结果 | 10/10 成功 | 间歇性失败 |

构造带诱导历史的请求（assistant 历史为对话式回答）后复现成功：**5 次中 1 次返回对话式文本**，与日志中的失败模式完全一致。

## 3. 根因分析

`MultiQuestionRewriteService.buildRewriteRequest` 将会话历史以标准多轮对话结构注入 messages：

```java
// 修复前
messages.add(ChatMessage.system(systemPrompt));       // [1] 改写指令
messages.addAll(recentHistory);                       // [2] 历史 user/assistant
messages.add(ChatMessage.user(question));             // [3] 当前问题
```

LLM 收到的实际请求：

```
[system]    你是查询改写助手...严格返回 JSON
[user]      你能告诉我知识库里都有什么文件吗？
[assistant] 我无法直接查看或列出知识库中的具体文件列表和完整内容...
[user]      这个文件的完整内容都能访问到吗？
```

**关键机制**：messages 结构决定了模型的推理路径。

- **多轮 user/assistant 交替** → 模型进入"对话延续"模式，按 assistant 身份继续回答
- **system + 单条 user** → 模型进入"指令执行"模式，严格按 system 约束输出

多轮对话的"惯性"压制了 system prompt 的角色定义，模型自然延续上一轮 assistant 的风格回答问题，而不是按改写指令输出 JSON。

## 4. 修复方案

### 4.1 方案选型

| 方案 | 描述 | 优劣 |
|------|------|------|
| A | 改写请求不传历史 | 最简单，但失去指代消解能力 |
| **B（采用）** | 历史拼成纯文本放入单条 user 消息 | 保留指代消解，消除对话诱导 |
| C | 保留多轮结构 + 加强格式提醒 | 膏药方案，不够稳定 |

### 4.2 方案 B 实现

将历史消息以纯文本标注形式拼接到 user 消息中，messages 结构变为 `[system] + [user(拼接文本)]`：

```java
// 修复后
StringBuilder userMsg = new StringBuilder();
if (CollUtil.isNotEmpty(history)) {
    List<ChatMessage> recentHistory = history.stream()
            .filter(m -> m.getRole() == ChatMessage.Role.USER
                    || m.getRole() == ChatMessage.Role.ASSISTANT)
            .skip(Math.max(0, history.size() - 4))
            .toList();
    if (!recentHistory.isEmpty()) {
        userMsg.append("【历史对话（仅供指代消解参考，禁止回答历史问题）】\n");
        for (ChatMessage m : recentHistory) {
            String role = m.getRole() == ChatMessage.Role.USER ? "用户" : "助手";
            userMsg.append(role).append("：").append(m.getContent()).append("\n");
        }
        userMsg.append("\n【当前需要改写的问题】\n");
    }
}
userMsg.append(question);
messages.add(ChatMessage.user(userMsg.toString()));
```

LLM 收到的修复后请求：

```
[system]    你是查询改写助手...严格返回 JSON
[user]      【历史对话（仅供指代消解参考，禁止回答历史问题）】
            用户：你能告诉我知识库里都有什么文件吗？
            助手：我无法直接查看或列出...
            
            【当前需要改写的问题】
            这个文件的完整内容都能访问到吗？
```

### 4.3 附加修复：LLMResponseCleaner 智能引号归一化

GLM 模型有时返回中文智能引号（`""''`），导致 JSON 解析失败。在 `LLMResponseCleaner.stripMarkdownCodeFence` 中增加正则替换：

```java
private static final Pattern SMART_QUOTES = Pattern.compile("[“”‘’]");
// stripMarkdownCodeFence 中：
cleaned = SMART_QUOTES.matcher(cleaned).replaceAll("'");
```

## 5. 验证结果

### 5.1 稳定性测试

用相同的诱导历史（assistant 为对话式回答），分别测试修复前后各 10 次：

| 状态 | JSON 成功率 | 对话式失败 |
|------|-------------|------------|
| 修复前 | 4/5（80%） | 1/5 |
| 修复后 | **10/10（100%）** | 0/10 |

### 5.2 指代消解验证

原问题"**这个**文件的完整内容都能访问到吗？"被改写为"**知识库**文件的完整内容是否能访问到"，模型正确从历史中识别出"这个"指代"知识库文件"，指代消解能力完整保留。

## 6. 涉及文件

| 文件 | 改动 |
|------|------|
| `bootstrap/.../rag/core/rewrite/MultiQuestionRewriteService.java` | `buildRewriteRequest` 重构为单条 user 消息结构 |
| `infra-ai/.../util/LLMResponseCleaner.java` | 增加 SMART_QUOTES 正则，归一化智能引号为单引号 |

## 7. 总结

**核心认知**：在需要 LLM 严格按固定格式（如 JSON）输出的场景中，messages 结构的设计直接影响模型的推理路径。多轮 user/assistant 交替结构会触发"对话延续"模式，导致模型忽略 system 的格式约束；将历史以纯文本形式放入单条 user 消息，则触发"指令执行"模式，模型严格遵循 system 输出格式。

这一原则适用于所有需要 LLM 在有历史上下文的情况下执行结构化任务（JSON/YAML/代码生成等）的场景。
