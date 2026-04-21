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

package com.nageoffer.ai.ragent.rag.core.guidance;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.StrUtil;
import com.nageoffer.ai.ragent.rag.config.GuidanceProperties;
import com.nageoffer.ai.ragent.rag.constant.RAGConstant;
import com.nageoffer.ai.ragent.rag.dto.SubQuestionIntent;
import com.nageoffer.ai.ragent.rag.enums.IntentLevel;
import com.nageoffer.ai.ragent.rag.core.intent.IntentNode;
import com.nageoffer.ai.ragent.rag.core.intent.IntentNodeRegistry;
import com.nageoffer.ai.ragent.rag.core.intent.NodeScore;
import com.nageoffer.ai.ragent.rag.core.intent.NodeScoreFilters;
import com.nageoffer.ai.ragent.rag.core.prompt.PromptTemplateLoader;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * 意图歧义引导服务
 * <p>
 * 在 RAG 管线的意图识别阶段之后运行，负责检测用户查询是否存在跨系统的意图歧义。
 * 当多个 KB 意图分属不同业务系统且置信度得分接近时，说明系统无法确定用户真正想查询哪个系统，
 * 此时生成一条澄清提示（Clarification Prompt）返回给用户，引导其明确选择目标系统，
 * 从而避免错误检索导致答非所问。
 * <p>
 * 核心流程：
 * <ol>
 *   <li>检查引导功能是否启用（通过 {@link GuidanceProperties} 配置）</li>
 *   <li>在候选意图中查找歧义组 —— 同名意图分属不同系统且得分比值超过阈值</li>
 *   <li>若用户问题中已包含某系统的名称关键词，跳过引导（用户意图已明确）</li>
 *   <li>使用 Prompt 模板生成可读的澄清提示</li>
 * </ol>
 *
 * @see GuidanceDecision 引导决策值对象
 * @see GuidanceProperties 引导功能配置项
 */
@Service
@RequiredArgsConstructor
public class IntentGuidanceService {

    /** 引导功能配置（是否启用、歧义得分比阈值、最大选项数等） */
    private final GuidanceProperties guidanceProperties;

    /** 意图节点注册表，用于根据 ID 查找意图节点及其层级关系 */
    private final IntentNodeRegistry intentNodeRegistry;

    /** Prompt 模板加载器，用于渲染澄清提示模板 */
    private final PromptTemplateLoader promptTemplateLoader;

    /**
     * 检测意图歧义并生成引导决策
     * <p>
     * 判断逻辑：
     * <ol>
     *   <li>引导功能未启用 → 返回 NONE（继续管线）</li>
     *   <li>无法找到歧义组（候选意图不足或未跨系统） → 返回 NONE</li>
     *   <li>用户问题中已包含某系统名称关键词 → 返回 NONE（意图已明确）</li>
     *   <li>以上均不满足 → 返回 PROMPT（携带澄清提示，中断管线等待用户选择）</li>
     * </ol>
     *
     * @param question   用户原始问题
     * @param subIntents 意图识别阶段产出的子问题意图列表
     * @return {@link GuidanceDecision} —— NONE 表示继续管线，PROMPT 表示中断管线并返回澄清提示
     */
    public GuidanceDecision detectAmbiguity(String question, List<SubQuestionIntent> subIntents) {
        if (!Boolean.TRUE.equals(guidanceProperties.getEnabled())) {
            return GuidanceDecision.none();
        }

        AmbiguityGroup group = findAmbiguityGroup(subIntents);
        if (group == null || CollUtil.isEmpty(group.optionIds())) {
            return GuidanceDecision.none();
        }

        List<String> systemNames = resolveOptionNames(group.optionIds());
        if (shouldSkipGuidance(question, systemNames)) {
            return GuidanceDecision.none();
        }

        String prompt = buildPrompt(group.topicName(), group.optionIds());
        return GuidanceDecision.prompt(prompt);
    }

    /**
     * 在候选意图中查找歧义组
     * <p>
     * 查找策略：
     * <ol>
     *   <li>仅处理单个子问题的场景（多子问题歧义检测过于复杂，暂不支持）</li>
     *   <li>按意图节点名称归一化后分组，同名意图可能分属不同系统</li>
     *   <li>对每个同名组，按得分降序排列，筛选满足以下条件的组：
     *       组内至少 2 个意图、第二名/第一名得分比 >= 阈值、分属多个不同系统</li>
     *   <li>从满足条件的组中选出最高分组作为歧义组</li>
     * </ol>
     *
     * @param subIntents 子问题意图列表
     * @return 歧义组（包含主题名和可选系统 ID 列表），无歧义时返回 null
     */
    private AmbiguityGroup findAmbiguityGroup(List<SubQuestionIntent> subIntents) {
        // 仅在单个子问题时进行歧义检测，多子问题场景复杂度过高暂不处理
        if (CollUtil.isEmpty(subIntents) || subIntents.size() != 1) {
            return null;
        }

        // 过滤出满足最低分数阈值且为 KB 类型的候选意图
        List<NodeScore> candidates = filterCandidates(subIntents.get(0).nodeScores());
        if (candidates.size() < 2) {
            return null;
        }

        // 按意图节点名称归一化后分组，同名意图可能分属不同业务系统
        Map<String, List<NodeScore>> grouped = candidates.stream()
                .filter(ns -> StrUtil.isNotBlank(ns.getNode().getName()))
                .collect(Collectors.groupingBy(ns -> normalizeName(ns.getNode().getName())));

        // 从所有同名组中，选出得分最高且满足歧义条件的组
        Optional<Map.Entry<String, List<NodeScore>>> best = grouped.entrySet().stream()
                .map(entry -> Map.entry(entry.getKey(), sortByScore(entry.getValue())))
                .filter(entry -> entry.getValue().size() > 1)          // 至少 2 个候选
                .filter(entry -> passScoreRatio(entry.getValue()))      // 第二名得分足够接近第一名
                .filter(entry -> hasMultipleSystems(entry.getValue()))  // 分属不同系统才构成歧义
                .max(Comparator.comparingDouble(entry -> entry.getValue().get(0).getScore()));

        if (best.isEmpty()) {
            return null;
        }

        List<NodeScore> groupScores = best.get().getValue();
        // 取第一名的节点名称作为主题名
        String topicName = Optional.ofNullable(groupScores.get(0).getNode().getName())
                .orElse(best.get().getKey());
        // 收集歧义组中涉及的不同系统节点 ID
        List<String> optionIds = collectSystemOptions(groupScores);
        if (optionIds.size() < 2) {
            return null;
        }
        // 截取最大选项数限制
        return new AmbiguityGroup(topicName, trimOptions(optionIds));
    }

    /**
     * 过滤候选意图节点
     * <p>
     * 仅保留得分 >= 最低阈值且为 KB（知识库）类型的意图节点，
     * 排除 MCP 和 SYSTEM 类型，因为歧义引导只对知识库检索场景有意义。
     *
     * @param scores 原始意图得分列表
     * @return 满足条件的候选列表
     */
    private List<NodeScore> filterCandidates(List<NodeScore> scores) {
        if (CollUtil.isEmpty(scores)) {
            return List.of();
        }
        return NodeScoreFilters.kb(scores, RAGConstant.INTENT_MIN_SCORE);
    }

    /**
     * 收集歧义组中涉及的不同系统节点 ID
     * <p>
     * 使用 {@link LinkedHashSet} 保持插入顺序（按得分高低），
     * 保证呈现给用户的选项顺序与得分排名一致。
     *
     * @param groupScores 同名意图组，已按得分降序排列
     * @return 去重后的系统节点 ID 列表，保持得分排序
     */
    private List<String> collectSystemOptions(List<NodeScore> groupScores) {
        Set<String> ordered = new LinkedHashSet<>();
        for (NodeScore score : groupScores) {
            IntentNode node = score.getNode();
            String systemId = resolveSystemNodeId(node);
            if (StrUtil.isNotBlank(systemId)) {
                ordered.add(systemId);
            }
        }
        return new ArrayList<>(ordered);
    }

    /**
     * 判断是否应跳过引导（即用户问题中已包含某系统名称关键词）
     * <p>
     * 如果用户已经在问题中明确提到了某个系统的名称（或别名），
     * 说明用户意图已经足够明确，无需再进行歧义引导。
     *
     * @param question    用户原始问题
     * @param systemNames 歧义系统的名称列表
     * @return true 表示应跳过引导（用户意图已明确），false 表示需要引导
     */
    private boolean shouldSkipGuidance(String question, List<String> systemNames) {
        if (StrUtil.isBlank(question) || CollUtil.isEmpty(systemNames)) {
            return false;
        }
        String normalizedQuestion = normalizeName(question);
        for (String name : systemNames) {
            if (StrUtil.isBlank(name)) {
                continue;
            }
            for (String alias : buildSystemAliases(name)) {
                // 跳过长度不足 2 的别名，避免单字符误匹配（如 "A"、"B"）
                if (alias.length() < 2) {
                    continue;
                }
                if (normalizedQuestion.contains(alias)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * 根据系统节点 ID 列表解析出可读的系统名称
     *
     * @param optionIds 系统节点 ID 列表
     * @return 对应的系统名称列表，找不到节点时跳过
     */
    private List<String> resolveOptionNames(List<String> optionIds) {
        if (CollUtil.isEmpty(optionIds)) {
            return List.of();
        }
        List<String> names = new ArrayList<>();
        for (String id : optionIds) {
            IntentNode node = intentNodeRegistry.getNodeById(id);
            if (node == null) {
                continue;
            }
            String name = StrUtil.blankToDefault(node.getName(), node.getId());
            names.add(name);
        }
        return names;
    }

    /**
     * 为系统名称构建别名列表（用于问题关键词匹配）
     * <p>
     * 当前策略：仅对名称进行归一化（去标点、转小写）作为唯一别名。
     * 后续可扩展为从配置或数据库中加载更多别名。
     *
     * @param systemName 系统名称
     * @return 别名列表
     */
    private List<String> buildSystemAliases(String systemName) {
        if (StrUtil.isBlank(systemName)) {
            return List.of();
        }
        String normalized = normalizeName(systemName);
        List<String> aliases = new ArrayList<>();
        if (StrUtil.isNotBlank(normalized)) {
            aliases.add(normalized);
        }
        return aliases;
    }

    /**
     * 判断同名意图组是否满足得分接近条件
     * <p>
     * 计算第二名与第一名的得分比值，如果比值 >= 配置的歧义阈值（{@code ambiguityScoreRatio}），
     * 则认为两者得分接近，存在歧义。
     * 例如：阈值 0.8 表示第二名得分需 >= 第一名的 80% 才判定为歧义。
     *
     * @param group 同名意图组，已按得分降序排列
     * @return true 表示得分足够接近，存在歧义
     */
    private boolean passScoreRatio(List<NodeScore> group) {
        if (group.size() < 2) {
            return false;
        }
        double top = group.get(0).getScore();
        double second = group.get(1).getScore();
        if (top <= 0) {
            return false;
        }
        double ratio = second / top;
        return ratio >= Optional.ofNullable(guidanceProperties.getAmbiguityScoreRatio()).orElse(0.0D);
    }

    /**
     * 判断同名意图组中的意图是否分属多个不同的业务系统
     * <p>
     * 通过向上回溯意图节点树，将每个意图解析到其所属的系统级节点（CATEGORY 层），
     * 如果去重后系统数 > 1，说明存在跨系统歧义。
     *
     * @param group 同名意图组
     * @return true 表示分属多个系统
     */
    private boolean hasMultipleSystems(List<NodeScore> group) {
        Set<String> systems = group.stream()
                .map(NodeScore::getNode)
                .map(this::resolveSystemNodeId)
                .filter(StrUtil::isNotBlank)
                .collect(Collectors.toSet());
        return systems.size() > 1;
    }

    /**
     * 向上回溯意图节点树，解析出当前节点所属的系统级节点 ID
     * <p>
     * 回溯规则：沿 parentId 链向上查找，直到找到层级为 CATEGORY 且其父节点层级为 DOMAIN 的节点，
     * 该节点即为"系统级节点"。如果回溯到树根仍未找到，则返回当前所在的最顶层节点 ID。
     *
     * @param node 起始意图节点
     * @return 系统级节点 ID，节点为 null 时返回空字符串
     */
    private String resolveSystemNodeId(IntentNode node) {
        if (node == null) {
            return "";
        }
        IntentNode current = node;
        IntentNode parent = fetchParent(current);
        for (; ; ) {
            IntentLevel level = current.getLevel();
            if (level == IntentLevel.CATEGORY && (parent == null || parent.getLevel() == IntentLevel.DOMAIN)) {
                return current.getId();
            }
            if (parent == null) {
                return current.getId();
            }
            current = parent;
            parent = fetchParent(current);
        }
    }

    /**
     * 获取节点的父节点
     *
     * @param node 当前节点
     * @return 父节点，不存在时返回 null
     */
    private IntentNode fetchParent(IntentNode node) {
        if (node == null || StrUtil.isBlank(node.getParentId())) {
            return null;
        }
        return intentNodeRegistry.getNodeById(node.getParentId());
    }

    /**
     * 按得分降序排列意图节点
     *
     * @param scores 原始得分列表
     * @return 排序后的得分列表（最高分在前）
     */
    private List<NodeScore> sortByScore(List<NodeScore> scores) {
        return scores.stream()
                .sorted(Comparator.comparingDouble(NodeScore::getScore).reversed())
                .toList();
    }

    /**
     * 截取选项列表，确保不超过配置的最大选项数
     *
     * @param optionIds 原始选项 ID 列表
     * @return 截取后的选项 ID 列表
     */
    private List<String> trimOptions(List<String> optionIds) {
        int maxOptions = Optional.ofNullable(guidanceProperties.getMaxOptions()).orElse(optionIds.size());
        if (optionIds.size() <= maxOptions) {
            return optionIds;
        }
        return optionIds.subList(0, maxOptions);
    }

    /**
     * 构建澄清提示文本
     * <p>
     * 使用模板引擎渲染预定义的 Prompt 模板，将主题名称和可选系统列表填入模板变量。
     *
     * @param topicName 歧义主题名称（如"报表查询"）
     * @param optionIds 可选系统节点 ID 列表
     * @return 渲染后的澄清提示文本
     */
    private String buildPrompt(String topicName, List<String> optionIds) {
        String options = renderOptions(optionIds);
        return promptTemplateLoader.render(
                RAGConstant.GUIDANCE_PROMPT_PATH,
                Map.of(
                        "topic_name", StrUtil.blankToDefault(topicName, ""),
                        "options", options
                )
        );
    }

    /**
     * 渲染选项列表为编号文本（如 "1) 财务系统\n2) HR 系统"）
     *
     * @param optionIds 系统节点 ID 列表
     * @return 编号格式的选项文本
     */
    private String renderOptions(List<String> optionIds) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < optionIds.size(); i++) {
            String id = optionIds.get(i);
            IntentNode node = intentNodeRegistry.getNodeById(id);
            String name = node == null || StrUtil.isBlank(node.getName()) ? id : node.getName();
            sb.append(i + 1).append(") ").append(name).append("\n");
        }
        return sb.toString().trim();
    }

    /**
     * 名称归一化：去除标点和空白，转为小写
     * <p>
     * 用于在分组和关键词匹配时忽略大小写、标点差异。
     * 例如："HR系统" 和 "hr 系统" 归一化后均为 "hr系统"。
     *
     * @param name 原始名称
     * @return 归一化后的字符串，null 输入返回空字符串
     */
    private String normalizeName(String name) {
        if (name == null) {
            return "";
        }
        String cleaned = name.trim().toLowerCase(Locale.ROOT);
        return cleaned.replaceAll("[\\p{Punct}\\s]+", "");
    }

    /**
     * 歧义组值对象
     *
     * @param topicName 歧义主题名称（即同名意图的名称）
     * @param optionIds 歧义涉及的系统节点 ID 列表（已去重、截取）
     */
    private record AmbiguityGroup(String topicName, List<String> optionIds) {
    }
}
