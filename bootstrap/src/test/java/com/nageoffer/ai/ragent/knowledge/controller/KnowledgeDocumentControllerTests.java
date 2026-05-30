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

package com.nageoffer.ai.ragent.knowledge.controller;

import com.nageoffer.ai.ragent.knowledge.controller.vo.KnowledgeDocumentVO;
import com.nageoffer.ai.ragent.knowledge.service.KnowledgeDocumentService;
import com.nageoffer.ai.ragent.rag.service.FileStorageService;
import jakarta.servlet.ServletOutputStream;
import jakarta.servlet.http.HttpServletResponse;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.io.ByteArrayInputStream;
import java.net.SocketTimeoutException;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class KnowledgeDocumentControllerTests {

    @Mock
    private KnowledgeDocumentService documentService;
    @Mock
    private FileStorageService fileStorageService;

    private KnowledgeDocumentController controller;

    @BeforeEach
    void setUp() {
        controller = new KnowledgeDocumentController(documentService, fileStorageService);
    }

    @Test
    void fileIgnoresClientAbortDuringStreaming() throws Exception {
        KnowledgeDocumentVO document = document();
        HttpServletResponse response = mock(HttpServletResponse.class);
        ServletOutputStream outputStream = mock(ServletOutputStream.class);
        when(documentService.get("doc-1")).thenReturn(document);
        when(fileStorageService.openStream("s3://bucket/doc.pdf"))
                .thenReturn(new ByteArrayInputStream("pdf".getBytes(StandardCharsets.UTF_8)));
        when(response.getOutputStream()).thenReturn(outputStream);
        doThrow(new SocketTimeoutException("write timeout"))
                .when(outputStream)
                .write(any(byte[].class), anyInt(), anyInt());

        assertDoesNotThrow(() -> controller.file("doc-1", response));

        verify(response).setContentType("application/pdf");
    }

    @Test
    void fileRethrowsNonClientAbortFailures() throws Exception {
        KnowledgeDocumentVO document = document();
        HttpServletResponse response = mock(HttpServletResponse.class);
        when(documentService.get("doc-1")).thenReturn(document);
        when(fileStorageService.openStream("s3://bucket/doc.pdf")).thenThrow(new RuntimeException("storage down"));

        assertThrows(RuntimeException.class, () -> controller.file("doc-1", response));
    }

    private KnowledgeDocumentVO document() {
        KnowledgeDocumentVO document = new KnowledgeDocumentVO();
        document.setDocName("doc.pdf");
        document.setFileType("pdf");
        document.setFileUrl("s3://bucket/doc.pdf");
        return document;
    }
}
