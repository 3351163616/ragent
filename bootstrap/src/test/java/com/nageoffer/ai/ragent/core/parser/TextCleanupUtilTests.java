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

package com.nageoffer.ai.ragent.core.parser;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class TextCleanupUtilTests {

    @Test
    void removesAllOcrSpacesBetweenChineseCharacters() {
        String cleaned = TextCleanupUtil.cleanup("我 学 习 新 技 术 , 提 升 竞 争 力");

        assertThat(cleaned).isEqualTo("我学习新技术 , 提升竞争力");
    }

    @Test
    void keepsParagraphBreaksAndEnglishSpaces() {
        String cleaned = TextCleanupUtil.cleanup("""
                第一段
                第二段
                Java 17 和 Spring Boot 3
                """);

        assertThat(cleaned).isEqualTo("""
                第一段
                第二段
                Java 17 和 Spring Boot 3
                """.trim());
    }
}
