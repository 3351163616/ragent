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

import lombok.Getter;

/**
 * 引导决策值对象（Value Object）
 * <p>
 * 在 RAG 管线的意图歧义检测阶段，由 {@link IntentGuidanceService} 生成，
 * 用于表示是否需要向用户输出引导式澄清提示。
 * <p>
 * 两种决策类型：
 * <ul>
 *   <li>{@link Action#NONE} —— 无歧义，管线继续执行后续的检索和生成阶段</li>
 *   <li>{@link Action#PROMPT} —— 存在歧义，管线中断，将澄清提示返回给用户，
 *       等待用户明确选择后再重新进入管线</li>
 * </ul>
 * <p>
 * 本类为不可变对象，通过静态工厂方法 {@link #none()} 和 {@link #prompt(String)} 创建实例。
 */
@Getter
public class GuidanceDecision {

    /**
     * 引导决策动作枚举
     */
    public enum Action {
        /** 无歧义，继续 RAG 管线 */
        NONE,
        /** 存在歧义，中断管线并返回澄清提示 */
        PROMPT
    }

    /** 决策动作类型 */
    private final Action action;

    /** 澄清提示文本，仅当 action 为 PROMPT 时有值 */
    private final String prompt;

    /**
     * 私有构造方法，强制通过静态工厂方法创建实例
     *
     * @param action 决策动作
     * @param prompt 澄清提示文本（NONE 时为 null）
     */
    private GuidanceDecision(Action action, String prompt) {
        this.action = action;
        this.prompt = prompt;
    }

    /**
     * 创建"无歧义"决策，管线继续执行
     *
     * @return NONE 类型的决策实例
     */
    public static GuidanceDecision none() {
        return new GuidanceDecision(Action.NONE, null);
    }

    /**
     * 创建"需要澄清"决策，携带提示文本，管线中断
     *
     * @param prompt 澄清提示文本，将通过 SSE 推送给前端展示
     * @return PROMPT 类型的决策实例
     */
    public static GuidanceDecision prompt(String prompt) {
        return new GuidanceDecision(Action.PROMPT, prompt);
    }

    /**
     * 判断当前决策是否为需要澄清的类型
     *
     * @return true 表示需要向用户展示澄清提示
     */
    public boolean isPrompt() {
        return action == Action.PROMPT;
    }
}
