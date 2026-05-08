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

import com.google.gson.JsonArray;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class DefaultIntentClassifierParsingTests {

    @Test
    void parseMarkdownJsonWithUnescapedQuotesInReason() {
        String raw = """
                ```json
                [
                  {"id": "sys-welcome", "score": 0.85, "reason": "用户以"你好"打招呼，属于欢迎与问候类交互"},
                  {"id": "sys-about", "score": 0.70, "reason": "用户询问助手能访问哪些文件，涉及助手功能能力的询问"}
                ]
                ```
                """;

        JsonArray result = DefaultIntentClassifier.parseIntentResultArray(raw);

        assertEquals(2, result.size());
        assertEquals("sys-welcome", result.get(0).getAsJsonObject().get("id").getAsString());
        assertEquals(0.85D, result.get(0).getAsJsonObject().get("score").getAsDouble());
        assertEquals("sys-about", result.get(1).getAsJsonObject().get("id").getAsString());
        assertEquals(0.70D, result.get(1).getAsJsonObject().get("score").getAsDouble());
    }

    @Test
    void parseWrappedResultsObject() {
        String raw = """
                {
                  "results": [
                    {"id": "sys-about", "score": 0.7, "reason": "能力说明"}
                  ]
                }
                """;

        JsonArray result = DefaultIntentClassifier.parseIntentResultArray(raw);

        assertEquals(1, result.size());
        assertEquals("sys-about", result.get(0).getAsJsonObject().get("id").getAsString());
        assertEquals(0.7D, result.get(0).getAsJsonObject().get("score").getAsDouble());
    }
}
