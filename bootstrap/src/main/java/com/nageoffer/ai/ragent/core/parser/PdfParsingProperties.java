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

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Data
@Component
@ConfigurationProperties(prefix = "rag.pdf")
public class PdfParsingProperties {

    /**
     * Default PDF OCR strategy.
     */
    private PdfOcrStrategy ocrStrategy = PdfOcrStrategy.NO_OCR;

    private CorruptionDetection corruptionDetection = new CorruptionDetection();

    private Ocr ocr = new Ocr();

    private PostIngestionCheck postIngestionCheck = new PostIngestionCheck();

    @Data
    public static class CorruptionDetection {

        /**
         * Whether to run text quality checks after PDF text extraction.
         */
        private boolean enabled = true;

        /**
         * When a document is expected to be Chinese, lower ratio means likely mojibake.
         */
        private double minChineseRatio = 0.05;

        /**
         * Long uppercase-only runs are a common symptom of broken PDF custom encodings.
         */
        private int maxUppercaseSequence = 15;

        /**
         * Skip quality checks for tiny snippets.
         */
        private int minTextLength = 100;

        /**
         * Apply the Chinese-ratio rule only when the file name or text has a Chinese signal.
         */
        private boolean requireChineseSignal = true;
    }

    @Data
    public static class Ocr {

        /**
         * Tesseract language expression. Tesseract accepts combined languages with '+'.
         */
        private String languages = "chi_sim+eng";

        /**
         * OCR timeout in seconds.
         */
        private int timeoutSeconds = 30;

        /**
         * Render DPI for PDF page OCR.
         */
        private int dpi = 300;
    }

    @Data
    public static class PostIngestionCheck {

        /**
         * Whether to mark a document as text_corrupted after chunking.
         */
        private boolean enabled = true;

        /**
         * Ratio of corrupted chunks that marks the document as text_corrupted.
         */
        private double corruptionThreshold = 0.8;
    }
}
