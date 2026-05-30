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

import com.nageoffer.ai.ragent.framework.exception.ServiceException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.metadata.TikaCoreProperties;
import org.apache.tika.parser.AutoDetectParser;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.parser.Parser;
import org.apache.tika.parser.ocr.TesseractOCRConfig;
import org.apache.tika.parser.pdf.PDFParserConfig;
import org.apache.tika.sax.ToTextContentHandler;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.io.StringWriter;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Apache Tika 文档解析器
 * <p>
 * 支持多种文档格式：PDF、Word、Excel、PPT、HTML、XML 等
 * 使用 Apache Tika 库进行文档解析和文本提取
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class TikaDocumentParser implements DocumentParser {

    public static final String OPTION_OCR_STRATEGY = "ocrStrategy";
    public static final String OPTION_PDF_OCR_STRATEGY = "pdfOcrStrategy";
    public static final String OPTION_FILE_NAME = "fileName";

    private final PdfParsingProperties properties;

    private final AutoDetectParser parser = new AutoDetectParser();

    @Override
    public String getParserType() {
        return ParserType.TIKA.getType();
    }

    @Override
    public ParseResult parse(byte[] content, String mimeType, Map<String, Object> options) {
        if (content == null || content.length == 0) {
            return ParseResult.ofText("");
        }

        try {
            String fileName = resolveFileName(null, options == null ? Map.of() : options);
            ExtractionResult result = extract(content, fileName, mimeType, options);
            Map<String, Object> metadata = new HashMap<>();
            metadata.put("pdfOcrStrategy", result.strategy().getCode());
            metadata.put("textQualityReason", result.qualityReport().reason());
            metadata.put("cjkRatio", result.qualityReport().cjkRatio());
            metadata.put("textCorrupted", result.qualityReport().likelyCorrupted());
            return ParseResult.of(result.text(), metadata);
        } catch (Exception e) {
            log.error("Tika 解析失败，MIME 类型: {}", mimeType, e);
            throw new ServiceException("文档解析失败: " + e.getMessage());
        }
    }

    @Override
    public String extractText(InputStream stream, String fileName) {
        return extractText(stream, fileName, Map.of());
    }

    @Override
    public String extractText(InputStream stream, String fileName, Map<String, Object> options) {
        try {
            byte[] content = stream.readAllBytes();
            return extract(content, fileName, null, options).text();
        } catch (Exception e) {
            log.error("从文件中提取文本内容失败: {}", fileName, e);
            throw new ServiceException("解析文件失败: " + fileName);
        }
    }

    @Override
    public boolean supports(String mimeType) {
        // Tika 支持大部分常见文档格式
        return mimeType != null && !mimeType.startsWith("text/markdown");
    }

    private ExtractionResult extract(byte[] content, String fileName, String mimeType, Map<String, Object> options)
            throws Exception {
        Map<String, Object> safeOptions = options == null ? Map.of() : options;
        String resolvedFileName = resolveFileName(fileName, safeOptions);
        PdfOcrStrategy requestedStrategy = resolveStrategy(safeOptions);
        boolean pdf = isPdf(mimeType, resolvedFileName);

        if (!pdf || requestedStrategy != PdfOcrStrategy.AUTO) {
            String text = parseWithStrategy(content, resolvedFileName, mimeType, requestedStrategy, pdf);
            String cleaned = TextCleanupUtil.cleanup(text);
            TextQualityInspector.TextQualityReport report = inspect(cleaned, resolvedFileName);
            return new ExtractionResult(cleaned, requestedStrategy, report);
        }

        String text = parseWithStrategy(content, resolvedFileName, mimeType, PdfOcrStrategy.NO_OCR, true);
        String cleaned = TextCleanupUtil.cleanup(text);
        TextQualityInspector.TextQualityReport report = inspect(cleaned, resolvedFileName);
        if (!report.likelyCorrupted()) {
            return new ExtractionResult(cleaned, PdfOcrStrategy.NO_OCR, report);
        }

        log.warn("PDF 文本层疑似乱码，切换 OCR 重试，fileName={}, reason={}, cjkRatio={}",
                resolvedFileName, report.reason(), String.format(Locale.ROOT, "%.4f", report.cjkRatio()));
        String ocrText = parseWithStrategy(content, resolvedFileName, mimeType, PdfOcrStrategy.OCR_ONLY, true);
        String cleanedOcr = TextCleanupUtil.cleanup(ocrText);
        TextQualityInspector.TextQualityReport ocrReport = inspect(cleanedOcr, resolvedFileName);
        return new ExtractionResult(cleanedOcr, PdfOcrStrategy.OCR_ONLY, ocrReport);
    }

    private String parseWithStrategy(byte[] content, String fileName, String mimeType,
                                     PdfOcrStrategy strategy, boolean pdf) throws Exception {
        Metadata metadata = new Metadata();
        if (StringUtils.hasText(fileName)) {
            metadata.set(TikaCoreProperties.RESOURCE_NAME_KEY, fileName);
        }
        if (StringUtils.hasText(mimeType)) {
            metadata.set("Content-Type", mimeType);
        }

        ParseContext context = new ParseContext();
        context.set(Parser.class, parser);
        if (pdf) {
            context.set(PDFParserConfig.class, pdfConfig(strategy));
        }
        if (pdf && strategy != PdfOcrStrategy.NO_OCR) {
            context.set(TesseractOCRConfig.class, tesseractConfig());
        }

        StringWriter writer = new StringWriter(Math.max(1024, content.length / 4));
        try (ByteArrayInputStream input = new ByteArrayInputStream(content)) {
            parser.parse(input, new ToTextContentHandler(writer), metadata, context);
        }
        return writer.toString();
    }

    private PDFParserConfig pdfConfig(PdfOcrStrategy strategy) {
        PDFParserConfig pdfConfig = new PDFParserConfig();
        pdfConfig.setExtractInlineImages(false);
        pdfConfig.setExtractUniqueInlineImagesOnly(true);
        pdfConfig.setOcrDPI(properties.getOcr().getDpi());
        pdfConfig.setOcrStrategy(switch (strategy) {
            case OCR_ONLY -> PDFParserConfig.OCR_STRATEGY.OCR_ONLY;
            case AUTO -> PDFParserConfig.OCR_STRATEGY.AUTO;
            case NO_OCR -> PDFParserConfig.OCR_STRATEGY.NO_OCR;
        });
        return pdfConfig;
    }

    private TesseractOCRConfig tesseractConfig() {
        TesseractOCRConfig config = new TesseractOCRConfig();
        config.setLanguage(normalizeLanguages(properties.getOcr().getLanguages()));
        config.setTimeoutSeconds(properties.getOcr().getTimeoutSeconds());
        config.setDensity(properties.getOcr().getDpi());
        return config;
    }

    private String normalizeLanguages(String languages) {
        if (!StringUtils.hasText(languages)) {
            return "chi_sim+eng";
        }
        return languages.trim().replace(",", "+").replace(" ", "");
    }

    private TextQualityInspector.TextQualityReport inspect(String text, String fileName) {
        return TextQualityInspector.inspect(text, fileName, properties.getCorruptionDetection());
    }

    private PdfOcrStrategy resolveStrategy(Map<String, Object> options) {
        Object value = options.get(OPTION_PDF_OCR_STRATEGY);
        if (value == null) {
            value = options.get(OPTION_OCR_STRATEGY);
        }
        return PdfOcrStrategy.normalize(value, properties.getOcrStrategy());
    }

    private String resolveFileName(String fileName, Map<String, Object> options) {
        if (StringUtils.hasText(fileName)) {
            return fileName;
        }
        Object fromOptions = options.get(OPTION_FILE_NAME);
        return fromOptions == null ? null : fromOptions.toString();
    }

    private boolean isPdf(String mimeType, String fileName) {
        if (StringUtils.hasText(mimeType) && mimeType.toLowerCase(Locale.ROOT).contains("pdf")) {
            return true;
        }
        return StringUtils.hasText(fileName) && fileName.toLowerCase(Locale.ROOT).endsWith(".pdf");
    }

    private record ExtractionResult(String text,
                                    PdfOcrStrategy strategy,
                                    TextQualityInspector.TextQualityReport qualityReport) {
    }
}
