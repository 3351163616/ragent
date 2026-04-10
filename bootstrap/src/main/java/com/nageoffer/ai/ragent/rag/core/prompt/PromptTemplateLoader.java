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

package com.nageoffer.ai.ragent.rag.core.prompt;

import cn.hutool.core.util.StrUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Prompt 模板加载器 —— RAG 管线中 Prompt 构建阶段的基础组件
 * <p>
 * 核心职责：
 * <ol>
 *     <li>从 classpath 下加载 .st（StringTemplate）格式的 Prompt 模板文件</li>
 *     <li>使用 {@link ConcurrentHashMap} 做本地缓存，避免重复 IO 读取</li>
 *     <li>提供模板变量填充（render）能力，将占位符替换为实际值后返回最终 Prompt</li>
 * </ol>
 * <p>
 * 模板文件路径支持两种写法：
 * <ul>
 *     <li>不带前缀（如 "prompts/rag.st"）—— 自动补全为 "classpath:prompts/rag.st"</li>
 *     <li>带 "classpath:" 前缀 —— 直接使用</li>
 * </ul>
 * <p>
 * 缓存策略：应用生命周期内永不过期，适用于模板文件在部署后不变的场景。
 * 若需热更新模板，可扩展为带 TTL 或手动刷新的缓存。
 *
 * @see PromptTemplateUtils 模板变量填充和格式清理工具类
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class PromptTemplateLoader {

    /** Spring 资源加载器，用于定位 classpath 下的模板文件 */
    private final ResourceLoader resourceLoader;
    /** 模板内容缓存：key 为模板路径，value 为模板文本。使用 ConcurrentHashMap 保证并发安全 */
    private final Map<String, String> cache = new ConcurrentHashMap<>();

    /**
     * 加载指定路径的 Prompt 模板内容
     * <p>
     * 使用 {@link ConcurrentHashMap#computeIfAbsent} 保证同一路径只读取一次，
     * 后续调用直接从缓存返回，线程安全且无锁竞争。
     *
     * @param path 模板文件路径，支持 classpath: 前缀；不带前缀时自动补全
     * @return 模板内容字符串（原始文本，未做占位符替换）
     * @throws IllegalArgumentException 当路径为空或 null 时抛出
     * @throws IllegalStateException    当模板文件不存在或读取 IO 异常时抛出
     */
    public String load(String path) {
        if (StrUtil.isBlank(path)) {
            throw new IllegalArgumentException("提示模板路径为空");
        }
        // computeIfAbsent 保证并发场景下只触发一次文件读取
        return cache.computeIfAbsent(path, this::readResource);
    }

    /**
     * 加载模板并渲染：将模板中的占位符替换为实际值，同时清理多余空行和残留标记
     * <p>
     * 渲染过程：
     * <ol>
     *     <li>调用 {@link #load} 获取原始模板文本（含缓存）</li>
     *     <li>通过 {@link PromptTemplateUtils#fillSlots} 替换占位符</li>
     *     <li>通过 {@link PromptTemplateUtils#cleanupPrompt} 清理格式</li>
     * </ol>
     *
     * @param path  模板文件路径
     * @param slots 占位符映射表：key 为占位符名称（不含定界符），value 为要替换的实际内容
     * @return 渲染并清理后的最终 Prompt 文本
     */
    public String render(String path, Map<String, String> slots) {
        String template = load(path);
        String filled = PromptTemplateUtils.fillSlots(template, slots);
        return PromptTemplateUtils.cleanupPrompt(filled);
    }

    /**
     * 从 classpath 读取模板文件的原始内容（私有方法，作为 {@link #load} 的实际读取逻辑）
     * <p>
     * 路径处理：若传入路径不以 "classpath:" 开头，自动补全前缀。
     * 使用 try-with-resources 确保输入流正确关闭。
     *
     * @param path 模板文件路径
     * @return 模板内容字符串（UTF-8 编码）
     * @throws IllegalStateException 当模板文件不存在或读取 IO 异常时抛出
     */
    private String readResource(String path) {
        // 若路径不以 classpath: 开头，自动补全，方便调用方使用相对路径
        String location = path.startsWith("classpath:") ? path : "classpath:" + path;
        Resource resource = resourceLoader.getResource(location);
        if (!resource.exists()) {
            throw new IllegalStateException("提示词模板路径不存在：" + path);
        }
        try (InputStream in = resource.getInputStream()) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            log.error("读取提示模板失败，路径：{}", path, e);
            throw new IllegalStateException("读取提示模板失败，路径：" + path, e);
        }
    }
}
