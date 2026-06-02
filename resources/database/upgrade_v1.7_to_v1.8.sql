CREATE INDEX IF NOT EXISTS idx_knowledge_chunk_kb_enabled_deleted
    ON t_knowledge_chunk (kb_id, enabled, deleted);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunk_doc_enabled_deleted
    ON t_knowledge_chunk (doc_id, enabled, deleted);

CREATE INDEX IF NOT EXISTS idx_knowledge_document_kb_status_enabled_deleted
    ON t_knowledge_document (kb_id, status, enabled, deleted);

CREATE INDEX IF NOT EXISTS idx_knowledge_document_update_time
    ON t_knowledge_document (update_time);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunk_content_fts
    ON t_knowledge_chunk
    USING gin (to_tsvector('simple', coalesce(content, '')));

CREATE INDEX IF NOT EXISTS idx_knowledge_document_doc_name_fts
    ON t_knowledge_document
    USING gin (to_tsvector('simple', coalesce(doc_name, '')));

COMMENT ON INDEX idx_knowledge_chunk_content_fts IS '关键词/BM25 检索：Chunk 正文全文索引';
COMMENT ON INDEX idx_knowledge_document_doc_name_fts IS '关键词/BM25 检索：文档名称全文索引';
