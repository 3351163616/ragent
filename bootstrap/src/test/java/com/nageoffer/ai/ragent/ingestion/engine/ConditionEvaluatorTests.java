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

package com.nageoffer.ai.ragent.ingestion.engine;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.nageoffer.ai.ragent.ingestion.domain.context.DocumentSource;
import com.nageoffer.ai.ragent.ingestion.domain.context.IngestionContext;
import com.nageoffer.ai.ragent.ingestion.domain.enums.SourceType;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class ConditionEvaluatorTests {

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final ConditionEvaluator evaluator = new ConditionEvaluator(objectMapper);

    @Test
    void evaluatesOpAliasAndSnakeCaseSourceType() throws Exception {
        IngestionContext context = context();

        boolean matched = evaluator.evaluate(
                context,
                objectMapper.readTree("""
                        {"field":"source_type","op":"eq","value":"file"}
                        """)
        );

        assertThat(matched).isTrue();
    }

    @Test
    void evaluatesTemplateFieldsInCompoundCondition() throws Exception {
        IngestionContext context = context();

        boolean matched = evaluator.evaluate(
                context,
                objectMapper.readTree("""
                        {
                          "all": [
                            {"field":"mime_type","op":"contains","value":"pdf"},
                            {"field":"file_name","op":"regex","value":".*\\\\.pdf$"},
                            {"field":"file_size","op":"gt","value":1048576}
                          ]
                        }
                        """)
        );

        assertThat(matched).isTrue();
    }

    @Test
    void keepsOperatorFieldCompatible() throws Exception {
        IngestionContext context = context();

        boolean matched = evaluator.evaluate(
                context,
                objectMapper.readTree("""
                        {"field":"source.type","operator":"eq","value":"file"}
                        """)
        );

        assertThat(matched).isTrue();
    }

    private IngestionContext context() {
        return IngestionContext.builder()
                .source(DocumentSource.builder()
                        .type(SourceType.FILE)
                        .location("file:///tmp/invoice.pdf")
                        .fileName("invoice.pdf")
                        .build())
                .mimeType("application/pdf")
                .rawBytes(new byte[1048577])
                .build();
    }
}
