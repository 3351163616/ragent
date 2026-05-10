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

package com.nageoffer.ai.ragent.infra.config;

import com.nageoffer.ai.ragent.infra.chat.GenericAnthropicChatClient;
import com.nageoffer.ai.ragent.infra.chat.GenericOpenAIChatClient;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeansException;
import org.springframework.beans.factory.config.BeanDefinition;
import org.springframework.beans.factory.config.ConfigurableListableBeanFactory;
import org.springframework.beans.factory.support.BeanDefinitionBuilder;
import org.springframework.beans.factory.support.BeanDefinitionRegistry;
import org.springframework.beans.factory.support.BeanDefinitionRegistryPostProcessor;
import org.springframework.boot.context.properties.bind.Bindable;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.context.EnvironmentAware;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

/**
 * 配置驱动的 ChatClient 自动注册器
 * <p>
 * 读取 {@code ai.providers} 中未由硬编码 @Service 覆盖的 provider 配置，
 * 根据 {@code format} 字段（openai / anthropic）自动注册对应的 ChatClient bean。
 * <p>
 * 注册的 bean 会被 {@code RoutingLLMService} 通过 {@code List<ChatClient>} 自动收集，
 * 无需修改路由代码。
 */
@Slf4j
@Component
public class ChatClientRegistrar implements BeanDefinitionRegistryPostProcessor, EnvironmentAware {

    private static final Set<String> HARDCODED_PROVIDERS = Set.of(
            "ollama", "bailian", "siliconflow", "moyu", "newapi"
    );

    private volatile Environment environment;

    @Override
    public void setEnvironment(Environment environment) {
        this.environment = environment;
    }

    @Override
    public void postProcessBeanDefinitionRegistry(BeanDefinitionRegistry registry) throws BeansException {
        Environment env = resolveEnvironment(registry);
        if (env == null) {
            log.warn("无法获取 Environment，跳过配置驱动的 ChatClient 注册");
            return;
        }

        Binder binder = Binder.get(env);
        Map<String, AIModelProperties.ProviderConfig> providers = binder.bind(
                "ai.providers",
                Bindable.mapOf(String.class, AIModelProperties.ProviderConfig.class)
        ).orElse(Collections.emptyMap());

        for (Map.Entry<String, AIModelProperties.ProviderConfig> entry : providers.entrySet()) {
            String providerName = entry.getKey();
            if (HARDCODED_PROVIDERS.contains(providerName)) {
                log.debug("跳过已有硬编码实现的 provider: {}", providerName);
                continue;
            }

            AIModelProperties.ProviderConfig config = entry.getValue();
            String format = config.getFormat() != null ? config.getFormat() : "openai";

            Class<?> beanClass = "anthropic".equals(format)
                    ? GenericAnthropicChatClient.class
                    : GenericOpenAIChatClient.class;

            BeanDefinition beanDef = BeanDefinitionBuilder.genericBeanDefinition(beanClass)
                    .addConstructorArgValue(providerName)
                    .addConstructorArgReference("syncHttpClient")
                    .addConstructorArgReference("streamingHttpClient")
                    .addConstructorArgReference("modelStreamExecutor")
                    .addConstructorArgReference("llmRequestLogger")
                    .getBeanDefinition();

            registry.registerBeanDefinition("chatClient-" + providerName, beanDef);
            log.info("注册配置驱动的 ChatClient: provider={}, format={}", providerName, format);
        }
    }

    @Override
    public void postProcessBeanFactory(ConfigurableListableBeanFactory beanFactory) throws BeansException {
        // no-op
    }

    private Environment resolveEnvironment(BeanDefinitionRegistry registry) {
        if (this.environment != null) {
            return this.environment;
        }
        return null;
    }
}
