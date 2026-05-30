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

import org.springframework.util.StringUtils;

import java.util.regex.Pattern;

public final class TextQualityInspector {

    private TextQualityInspector() {
    }

    public static TextQualityReport inspect(String text, String fileName,
                                            PdfParsingProperties.CorruptionDetection config) {
        if (config == null || !config.isEnabled()) {
            return TextQualityReport.clean(0, 0, 0D);
        }

        String safeText = text == null ? "" : text;
        int charCount = safeText.codePointCount(0, safeText.length());
        int cjkCount = countCjk(safeText);
        double cjkRatio = charCount == 0 ? 0D : (double) cjkCount / charCount;
        boolean hasLongUppercase = hasLongUppercaseSequence(safeText, config.getMaxUppercaseSequence());
        boolean hasReplacement = safeText.indexOf('\uFFFD') >= 0;
        boolean chineseSignal = !config.isRequireChineseSignal()
                || cjkCount > 0
                || containsCjk(fileName);

        if (charCount < Math.max(1, config.getMinTextLength())) {
            boolean tooShortForChineseDocument = chineseSignal && cjkRatio < config.getMinChineseRatio();
            return new TextQualityReport(charCount, cjkCount, cjkRatio, hasLongUppercase, hasReplacement,
                    tooShortForChineseDocument, tooShortForChineseDocument ? "TEXT_TOO_SHORT" : "SHORT_TEXT_OK");
        }

        boolean lowChineseRatio = chineseSignal && cjkRatio < config.getMinChineseRatio();
        boolean likelyCorrupted = lowChineseRatio || hasLongUppercase || hasReplacement;
        String reason;
        if (hasReplacement) {
            reason = "HAS_REPLACEMENT_CHAR";
        } else if (hasLongUppercase) {
            reason = "LONG_UPPERCASE_SEQUENCE";
        } else if (lowChineseRatio) {
            reason = "LOW_CHINESE_RATIO";
        } else {
            reason = "OK";
        }
        return new TextQualityReport(charCount, cjkCount, cjkRatio, hasLongUppercase, hasReplacement,
                likelyCorrupted, reason);
    }

    public static boolean containsCjk(String value) {
        if (!StringUtils.hasText(value)) {
            return false;
        }
        return value.codePoints().anyMatch(TextQualityInspector::isCjk);
    }

    private static int countCjk(String value) {
        if (!StringUtils.hasText(value)) {
            return 0;
        }
        return (int) value.codePoints()
                .filter(TextQualityInspector::isCjk)
                .count();
    }

    private static boolean isCjk(int codePoint) {
        return Character.UnicodeScript.of(codePoint) == Character.UnicodeScript.HAN;
    }

    private static boolean hasLongUppercaseSequence(String text, int maxUppercaseSequence) {
        if (!StringUtils.hasText(text) || maxUppercaseSequence <= 0) {
            return false;
        }
        return Pattern.compile("[A-Z]{" + maxUppercaseSequence + ",}").matcher(text).find();
    }

    public record TextQualityReport(int charCount,
                                    int cjkCount,
                                    double cjkRatio,
                                    boolean hasLongUppercaseSequence,
                                    boolean hasReplacementChar,
                                    boolean likelyCorrupted,
                                    String reason) {

        private static TextQualityReport clean(int charCount, int cjkCount, double cjkRatio) {
            return new TextQualityReport(charCount, cjkCount, cjkRatio, false, false, false, "DISABLED");
        }
    }
}
