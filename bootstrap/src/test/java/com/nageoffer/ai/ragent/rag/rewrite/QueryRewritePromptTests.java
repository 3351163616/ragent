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

package com.nageoffer.ai.ragent.rag.rewrite;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

public class QueryRewritePromptTests {

    @Test
    public void shouldGuideSynonymExpansionForRetrievalTerms() throws IOException {
        String prompt = loadPrompt();

        Assertions.assertTrue(prompt.contains("术语扩展"));
        Assertions.assertTrue(prompt.contains("税号（纳税人识别号、统一社会信用代码）"));
        Assertions.assertTrue(prompt.contains("发票抬头（开票抬头、开票名称）"));
        Assertions.assertTrue(prompt.contains("每个子问题尽量保持原文表述，并携带该子问题相关的术语扩展"));
        Assertions.assertTrue(prompt.contains("\"公司税号（纳税人识别号、统一社会信用代码）是多少\""));
    }

    private String loadPrompt() throws IOException {
        try (InputStream in = Thread.currentThread()
                .getContextClassLoader()
                .getResourceAsStream("prompt/user-question-rewrite.st")) {
            Assertions.assertNotNull(in, "query rewrite prompt should exist");
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }
}
