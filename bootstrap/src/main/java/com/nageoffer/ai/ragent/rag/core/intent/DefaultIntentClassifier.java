/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nageoffer.ai.ragent.rag.core.intent;

import cn.hutool.core.bean.BeanUtil;
import cn.hutool.core.collection.CollUtil;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.nageoffer.ai.ragent.infra.util.LLMResponseCleaner;
import com.nageoffer.ai.ragent.rag.dao.entity.IntentNodeDO;
import com.nageoffer.ai.ragent.rag.dao.mapper.IntentNodeMapper;
import com.nageoffer.ai.ragent.framework.convention.ChatMessage;
import com.nageoffer.ai.ragent.framework.convention.ChatRequest;
import com.nageoffer.ai.ragent.infra.chat.LLMService;
import com.nageoffer.ai.ragent.rag.core.prompt.PromptTemplateLoader;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static com.nageoffer.ai.ragent.rag.constant.RAGConstant.INTENT_CLASSIFIER_PROMPT_PATH;

/**
 * 默认意图分类器（LLM 串行实现）—— RAG 管线意图识别阶段的核心实现类。
 * <p>
 * 职责：将用户问题与预定义的意图树进行匹配，通过 LLM 对每个叶子意图节点打分，
 * 返回按置信度降序排列的意图识别结果。
 * <p>
 * 核心流程：
 * <ol>
 *     <li><b>意图树加载</b>：优先从 Redis 缓存读取意图树，缓存未命中时从数据库加载并回填缓存。
 *         意图树采用 N 叉树结构，只有叶子节点参与 LLM 打分。</li>
 *     <li><b>Prompt 构建</b>：将所有叶子节点的 id、路径、描述、类型（KB/MCP/SYSTEM）和示例问题
 *         拼接为结构化文本，嵌入到分类 Prompt 模板中。</li>
 *     <li><b>LLM 调用</b>：使用低温度（0.1）+ 低 topP（0.3）保证分类结果的稳定性和确定性。</li>
 *     <li><b>响应解析</b>：解析 LLM 返回的 JSON 数组 {@code [{id, score, reason}]}，
 *         容错处理外层包装（{@code {results: [...]}}）和 Markdown 代码块标记。</li>
 * </ol>
 * <p>
 * 同时实现 {@link IntentNodeRegistry} 接口，提供根据 ID 查找意图节点的能力，
 * 供检索通道和 Prompt 构建阶段使用。
 *
 * @see IntentClassifier 意图分类器接口
 * @see IntentNodeRegistry 意图节点注册表接口
 * @see IntentTreeCacheManager Redis 意图树缓存管理器
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class DefaultIntentClassifier implements IntentClassifier, IntentNodeRegistry {

    /** LLM 服务，用于调用大语言模型完成意图分类打分 */
    private final LLMService llmService;

    /** 意图节点 MyBatis-Plus Mapper，用于从数据库加载意图树的扁平节点列表 */
    private final IntentNodeMapper intentNodeMapper;

    /** Prompt 模板加载器，负责加载和渲染意图分类的提示词模板 */
    private final PromptTemplateLoader promptTemplateLoader;

    /** Redis 意图树缓存管理器，提供意图树的缓存读写能力 */
    private final IntentTreeCacheManager intentTreeCacheManager;

    /**
     * 从 Redis 缓存加载意图树并构建内存数据结构。
     * <p>
     * 加载策略（三级回退）：
     * <ol>
     *     <li>优先从 Redis 缓存读取（通过 {@link IntentTreeCacheManager}）</li>
     *     <li>缓存未命中时从数据库加载，并将结果回填到 Redis 缓存</li>
     *     <li>数据库也无数据时返回空的 {@link IntentTreeData}</li>
     * </ol>
     * <p>
     * 注意：每次调用都会重新从 Redis 读取，确保意图树数据是最新的（适用于运行时动态调整意图配置的场景）。
     * 返回的 {@link IntentTreeData} 是临时对象，包含扁平化的全部节点、叶子节点列表和 ID 索引 Map。
     *
     * @return 意图树内存数据结构，包含 allNodes、leafNodes 和 id2Node 映射
     */
    private IntentTreeData loadIntentTreeData() {
        // 第一步：从 Redis 读取（IntentTreeCacheManager 内部处理了缓存未命中时的自动加载）
        List<IntentNode> roots = intentTreeCacheManager.getIntentTreeFromCache();

        // 第二步：Redis 也没有数据时，从数据库加载并回填缓存
        if (CollUtil.isEmpty(roots)) {
            roots = loadIntentTreeFromDB();
            if (!roots.isEmpty()) {
                intentTreeCacheManager.saveIntentTreeToCache(roots);
            }
        }

        // 第三步：构建内存索引结构（每次调用创建临时对象，不做持久化缓存）
        if (CollUtil.isEmpty(roots)) {
            return new IntentTreeData(List.of(), List.of(), Map.of());
        }

        // 将树形结构扁平化为列表，便于遍历和索引
        List<IntentNode> allNodes = flatten(roots);
        // 提取叶子节点——只有叶子节点参与 LLM 意图打分
        List<IntentNode> leafNodes = allNodes.stream()
                .filter(IntentNode::isLeaf)
                .collect(Collectors.toList());
        // 构建 ID → Node 的快速查找映射，供解析 LLM 响应时使用
        Map<String, IntentNode> id2Node = allNodes.stream()
                .collect(Collectors.toMap(IntentNode::getId, n -> n));

        log.debug("意图树数据加载完成, 总节点数: {}, 叶子节点数: {}", allNodes.size(), leafNodes.size());

        return new IntentTreeData(allNodes, leafNodes, id2Node);
    }

    /**
     * 根据意图节点 ID 查找对应的 {@link IntentNode}。
     * <p>
     * 实现 {@link IntentNodeRegistry} 接口，供检索通道和 Prompt 构建阶段
     * 根据意图 ID 获取节点的详细信息（如集合名、Prompt 模板等）。
     *
     * @param id 意图节点的唯一标识（对应数据库中的 intentCode）
     * @return 对应的 {@link IntentNode}；ID 为空或节点不存在时返回 null
     */
    @Override
    public IntentNode getNodeById(String id) {
        if (id == null || id.isBlank()) {
            return null;
        }
        IntentTreeData data = loadIntentTreeData();
        return data.id2Node.get(id);
    }

    /**
     * 意图树内存数据结构（临时对象，每次调用 {@link #loadIntentTreeData()} 时创建，不做持久化）。
     *
     * @param allNodes  扁平化的全部意图节点列表
     * @param leafNodes 叶子意图节点列表（参与 LLM 打分的节点）
     * @param id2Node   意图节点 ID → IntentNode 的快速查找映射
     */
    private record IntentTreeData(
            List<IntentNode> allNodes,
            List<IntentNode> leafNodes,
            Map<String, IntentNode> id2Node
    ) {
    }

    /**
     * 将意图树的树形结构扁平化为节点列表。
     * <p>
     * 使用栈（非递归）实现深度优先遍历，避免深层嵌套时的栈溢出风险。
     *
     * @param roots 意图树的根节点列表
     * @return 扁平化后的全部节点列表（包含根节点和所有子节点）
     */
    private List<IntentNode> flatten(List<IntentNode> roots) {
        List<IntentNode> result = new ArrayList<>();
        Deque<IntentNode> stack = new ArrayDeque<>(roots);
        while (!stack.isEmpty()) {
            IntentNode n = stack.pop();
            result.add(n);
            if (n.getChildren() != null) {
                for (IntentNode child : n.getChildren()) {
                    stack.push(child);
                }
            }
        }
        return result;
    }

    /**
     * 对所有叶子分类节点进行意图识别打分（核心方法）。
     * <p>
     * 完整流程：
     * <ol>
     *     <li>从 Redis/DB 加载最新的意图树数据</li>
     *     <li>基于叶子节点信息构建分类 Prompt</li>
     *     <li>以低温度（0.1）调用 LLM，获取每个叶子节点的匹配分数</li>
     *     <li>解析 LLM 返回的 JSON 数组，按分数降序排列</li>
     * </ol>
     * <p>
     * LLM 预期返回格式：{@code [{"id": "intent_code", "score": 0.9, "reason": "匹配原因"}, ...]}
     * <p>
     * 容错机制：
     * <ul>
     *     <li>自动移除 Markdown 代码块标记（{@code ```json ... ```}）</li>
     *     <li>兼容外层包装格式（{@code {"results": [...]}}）</li>
     *     <li>跳过 LLM 返回的未知节点 ID，避免空指针</li>
     *     <li>解析失败时返回空列表，由上层做保底处理</li>
     * </ul>
     *
     * @param question 用户问题（经查询改写后的子问题）
     * @return 按 score 降序排列的 {@link NodeScore} 列表；解析失败返回空列表
     */
    @Override
    public List<NodeScore> classifyTargets(String question) {
        // 每次都从 Redis 读取最新的意图树数据，确保运行时修改能及时生效
        IntentTreeData data = loadIntentTreeData();

        // 构建分类 Prompt：列出所有叶子节点的 id/路径/描述/类型/示例
        String systemPrompt = buildPrompt(data.leafNodes);
        // 使用低温度和低 topP 保证分类结果的确定性和可复现性
        ChatRequest request = ChatRequest.builder()
                .messages(List.of(
                        ChatMessage.system(systemPrompt),
                        ChatMessage.user(question)
                ))
                .temperature(0.1D)
                .topP(0.3D)
                .thinking(false)
                .build();

        String raw = llmService.chat(request);

        try {
            // 移除 LLM 响应中可能存在的 Markdown 代码块标记（如 ```json ... ```）
            String cleanedRaw = LLMResponseCleaner.stripMarkdownCodeFence(raw);

            JsonElement root = JsonParser.parseString(cleanedRaw);

            JsonArray arr;
            if (root.isJsonArray()) {
                // 标准格式：直接返回 JSON 数组
                arr = root.getAsJsonArray();
            } else if (root.isJsonObject() && root.getAsJsonObject().has("results")) {
                // 容错：部分模型可能在外层多包一层 { "results": [...] }
                arr = root.getAsJsonObject().getAsJsonArray("results");
            } else {
                log.warn("LLM 返回了非预期的 JSON 格式, 原始响应: {}", raw);
                return List.of();
            }

            List<NodeScore> scores = new ArrayList<>();
            for (JsonElement el : arr) {
                if (!el.isJsonObject()) continue;
                JsonObject obj = el.getAsJsonObject();

                if (!obj.has("id") || !obj.has("score")) continue;

                String id = obj.get("id").getAsString();
                double score = obj.get("score").getAsDouble();

                IntentNode node = data.id2Node.get(id);
                if (node == null) {
                    // LLM 可能幻觉出不存在的节点 ID，跳过以避免空指针
                    log.warn("LLM 返回了未知的意图节点 ID: {}, 已跳过", id);
                    continue;
                }

                scores.add(new NodeScore(node, score));
            }

            // 按 score 降序排序，高置信度意图排在前面
            scores.sort(Comparator.comparingDouble(NodeScore::getScore).reversed());

            // 日志记录完整的意图识别结果（清除 children 避免日志过长）
            log.info("当前问题：{}\n意图识别树如下所示：{}\n",
                    question,
                    JSONUtil.toJsonPrettyStr(
                            scores.stream().peek(each -> {
                                IntentNode node = each.getNode();
                                node.setChildren(null);
                            }).collect(Collectors.toList())
                    )
            );
            return scores;
        } catch (Exception e) {
            log.warn("解析 LLM 响应失败, 原始内容: {}", raw, e);
            return List.of();
        }
    }

    /**
     * 获取 Top-K 个且分数高于阈值的意图分类结果。
     * <p>
     * 覆盖接口的默认实现，直接委托给 {@link #classifyTargets(String)} 后过滤。
     *
     * @param question 用户问题
     * @param topN     最多返回的结果数量
     * @param minScore 最低分数阈值
     * @return 过滤后的 {@link NodeScore} 列表
     */
    @Override
    public List<NodeScore> topKAboveThreshold(String question, int topN, double minScore) {
        return classifyTargets(question).stream()
                .filter(ns -> ns.getScore() >= minScore)
                .limit(topN)
                .toList();
    }

    /**
     * 构建发送给 LLM 的意图分类 Prompt。
     * <p>
     * 将所有叶子意图节点的元信息拼接为结构化文本，包括：
     * <ul>
     *     <li><b>id</b>：节点唯一标识，LLM 返回结果中需引用此 ID</li>
     *     <li><b>path</b>：从根到叶子的完整路径（如"业务系统 > OA系统 > 审批流程"），
     *         帮助 LLM 理解意图的层级关系</li>
     *     <li><b>description</b>：意图的详细描述</li>
     *     <li><b>type</b>：节点类型标识（KB=知识库查询, MCP=工具调用, SYSTEM=系统指令），
     *         影响后续的处理策略选择</li>
     *     <li><b>toolId</b>：MCP 类型节点关联的工具 ID（仅 MCP 类型包含）</li>
     *     <li><b>examples</b>：示例问题，帮助 LLM 理解该意图匹配的问题模式</li>
     * </ul>
     * <p>
     * 最终通过 {@link PromptTemplateLoader} 将拼接好的意图列表渲染到分类 Prompt 模板中。
     *
     * @param leafNodes 需要参与分类打分的叶子意图节点列表
     * @return 渲染完成的完整分类 Prompt 文本
     */
    private String buildPrompt(List<IntentNode> leafNodes) {
        StringBuilder sb = new StringBuilder();

        for (IntentNode node : leafNodes) {
            sb.append("- id=").append(node.getId()).append("\n");
            sb.append("  path=").append(node.getFullPath()).append("\n");
            sb.append("  description=").append(node.getDescription()).append("\n");

            // 添加节点类型标识，不同类型决定后续不同的处理策略
            if (node.isMCP()) {
                sb.append("  type=MCP\n");
                // MCP 节点需要额外标记关联的工具 ID，供后续 MCP 工具调用使用
                if (node.getMcpToolId() != null) {
                    sb.append("  toolId=").append(node.getMcpToolId()).append("\n");
                }
            } else if (node.isSystem()) {
                sb.append("  type=SYSTEM\n");
            } else {
                sb.append("  type=KB\n");
            }

            if (node.getExamples() != null && !node.getExamples().isEmpty()) {
                sb.append("  examples=");
                sb.append(String.join(" / ", node.getExamples()));
                sb.append("\n");
            }
            sb.append("\n");
        }

        return promptTemplateLoader.render(
                INTENT_CLASSIFIER_PROMPT_PATH,
                Map.of("intent_list", sb.toString())
        );
    }

    /**
     * 从数据库加载意图树并组装为树形结构。
     * <p>
     * 处理流程：
     * <ol>
     *     <li>查询所有未删除的意图节点（扁平结构的 DO 列表）</li>
     *     <li>第一遍遍历：将 DO 转换为 IntentNode，建立 ID → Node 映射</li>
     *     <li>第二遍遍历：根据 parentId 组装父子关系，形成树形结构</li>
     *     <li>填充 fullPath 字段（如"集团信息化 > 人事 > 考勤"）</li>
     * </ol>
     * <p>
     * 容错处理：
     * <ul>
     *     <li>parentId 为空的节点视为根节点</li>
     *     <li>parentId 指向不存在的父节点时，也将该节点兜底为根节点，避免节点丢失</li>
     * </ul>
     *
     * @return 意图树的根节点列表
     */
    private List<IntentNode> loadIntentTreeFromDB() {
        // 1. 查出所有未删除且已启用的节点（扁平结构）
        List<IntentNodeDO> intentNodeDOList = intentNodeMapper.selectList(
                Wrappers.lambdaQuery(IntentNodeDO.class)
                        .eq(IntentNodeDO::getDeleted, 0)
                        .eq(IntentNodeDO::getEnabled, 1)
        );

        if (intentNodeDOList.isEmpty()) {
            return List.of();
        }

        // 第一遍：DO → IntentNode，建立 ID 索引（code 字段映射为 IntentNode 的 id/parentId）
        Map<String, IntentNode> id2Node = new HashMap<>();
        for (IntentNodeDO each : intentNodeDOList) {
            IntentNode node = BeanUtil.toBean(each, IntentNode.class);
            // 数据库中的 code 映射到 IntentNode 的 id/parentId
            node.setId(each.getIntentCode());
            node.setParentId(each.getParentCode());
            node.setMcpToolId(each.getMcpToolId());
            node.setParamPromptTemplate(each.getParamPromptTemplate());
            // 确保 children 不为 null（避免后面 add NPE）
            if (node.getChildren() == null) {
                node.setChildren(new ArrayList<>());
            }
            id2Node.put(node.getId(), node);
        }

        // 第二遍：根据 parentId 组装父子关系，形成 N 叉树
        List<IntentNode> roots = new ArrayList<>();
        for (IntentNode node : id2Node.values()) {
            String parentId = node.getParentId();
            if (parentId == null || parentId.isBlank()) {
                // 没有 parentId，当作根节点
                roots.add(node);
                continue;
            }

            IntentNode parent = id2Node.get(parentId);
            if (parent == null) {
                // 找不到父节点，兜底也当作根节点，避免节点丢失
                roots.add(node);
                continue;
            }

            // 追加到父节点的 children
            if (parent.getChildren() == null) {
                parent.setChildren(new ArrayList<>());
            }
            parent.getChildren().add(node);
        }

        // 第四步：递归填充 fullPath 字段（如"集团信息化 > 人事 > 考勤"）
        fillFullPath(roots, null);

        return roots;
    }

    /**
     * 递归填充意图节点的 fullPath 字段。
     * <p>
     * fullPath 表示从根节点到当前节点的完整路径，用 " > " 分隔，例如：
     * <ul>
     *     <li>根节点："集团信息化"</li>
     *     <li>二级节点："集团信息化 > 人事"</li>
     *     <li>三级节点："业务系统 > OA系统 > 系统介绍"</li>
     * </ul>
     * <p>
     * fullPath 会嵌入到分类 Prompt 中，帮助 LLM 理解意图的层级上下文。
     *
     * @param nodes  当前层级的节点列表
     * @param parent 父节点（根节点的 parent 为 null）
     */
    private void fillFullPath(List<IntentNode> nodes, IntentNode parent) {
        if (nodes == null) return;

        for (IntentNode node : nodes) {
            if (parent == null) {
                node.setFullPath(node.getName());
            } else {
                node.setFullPath(parent.getFullPath() + " > " + node.getName());
            }

            if (node.getChildren() != null && !node.getChildren().isEmpty()) {
                fillFullPath(node.getChildren(), node);
            }
        }
    }
}
