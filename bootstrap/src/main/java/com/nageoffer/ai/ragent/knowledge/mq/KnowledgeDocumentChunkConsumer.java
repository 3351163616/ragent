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

package com.nageoffer.ai.ragent.knowledge.mq;

import com.nageoffer.ai.ragent.framework.context.LoginUser;
import com.nageoffer.ai.ragent.framework.context.UserContext;
import com.nageoffer.ai.ragent.framework.mq.MessageWrapper;
import com.nageoffer.ai.ragent.framework.mq.consumer.MQConsumer;
import com.nageoffer.ai.ragent.framework.mq.consumer.MessageQueueConsumer;
import com.nageoffer.ai.ragent.knowledge.mq.event.KnowledgeDocumentChunkEvent;
import com.nageoffer.ai.ragent.knowledge.service.KnowledgeDocumentService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * 文档分块任务 MQ 消费者
 * 负责异步执行耗时的文本提取、分块、向量嵌入及写库操作
 */
@Slf4j
@RequiredArgsConstructor
@MQConsumer(topic = KnowledgeDocumentChunkProducer.TOPIC, consumerGroup = "knowledge-document-chunk-consumer-group")
public class KnowledgeDocumentChunkConsumer implements MessageQueueConsumer<KnowledgeDocumentChunkEvent> {

    private final KnowledgeDocumentService documentService;

    @Override
    public void consume(MessageWrapper<KnowledgeDocumentChunkEvent> message) {
        KnowledgeDocumentChunkEvent event = message.getBody();
        log.info("开始消费文档分块任务，docId={}", event.getDocId());

        // 消费者线程无 HTTP 上下文，手动注入操作人，与 ScheduleJob 保持一致
        UserContext.set(LoginUser.builder().username(event.getOperator()).build());
        try {
            documentService.executeChunk(event.getDocId());
        } finally {
            UserContext.clear();
        }
    }
}
