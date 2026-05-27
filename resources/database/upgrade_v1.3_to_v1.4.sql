-- ragent v1.3 -> v1.4 升级脚本
-- AI 辅助初始化配置：新增任务表、术语映射候选表、意图节点候选表

CREATE TABLE t_config_bootstrap_run (
    id                     VARCHAR(20)   NOT NULL PRIMARY KEY,
    status                 VARCHAR(32)   NOT NULL,
    kb_ids                 TEXT,
    use_llm                SMALLINT      NOT NULL DEFAULT 1,
    document_count         INTEGER       NOT NULL DEFAULT 0,
    chunk_count            INTEGER       NOT NULL DEFAULT 0,
    term_candidate_count   INTEGER       NOT NULL DEFAULT 0,
    intent_candidate_count INTEGER       NOT NULL DEFAULT 0,
    summary                TEXT,
    error_message          TEXT,
    create_by              VARCHAR(64),
    update_by              VARCHAR(64),
    create_time            TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time            TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_config_bootstrap_run_status ON t_config_bootstrap_run (status);
COMMENT ON TABLE t_config_bootstrap_run IS 'AI辅助初始化配置任务表';

CREATE TABLE t_term_mapping_candidate (
    id             VARCHAR(20)   NOT NULL PRIMARY KEY,
    run_id         VARCHAR(20)   NOT NULL,
    source_term    VARCHAR(128)  NOT NULL,
    target_term    VARCHAR(128)  NOT NULL,
    confidence     DOUBLE PRECISION,
    risk_level     VARCHAR(16)   NOT NULL DEFAULT 'MEDIUM',
    evidence_json  TEXT,
    status         VARCHAR(32)   NOT NULL DEFAULT 'PENDING',
    review_comment TEXT,
    published_id   VARCHAR(20),
    create_by      VARCHAR(64),
    review_by      VARCHAR(64),
    publish_by     VARCHAR(64),
    create_time    TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time    TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    review_time    TIMESTAMP,
    publish_time   TIMESTAMP
);
CREATE INDEX idx_term_mapping_candidate_run ON t_term_mapping_candidate (run_id);
CREATE INDEX idx_term_mapping_candidate_status ON t_term_mapping_candidate (status);
COMMENT ON TABLE t_term_mapping_candidate IS 'AI辅助初始化术语映射候选表';

CREATE TABLE t_intent_node_candidate (
    id              VARCHAR(20)   NOT NULL PRIMARY KEY,
    run_id          VARCHAR(20)   NOT NULL,
    intent_code     VARCHAR(128)  NOT NULL,
    name            VARCHAR(128)  NOT NULL,
    level           SMALLINT      NOT NULL DEFAULT 2,
    parent_code     VARCHAR(128),
    description     TEXT,
    examples_json   TEXT,
    kb_id           VARCHAR(20),
    collection_name VARCHAR(128),
    top_k           INTEGER,
    kind            SMALLINT      NOT NULL DEFAULT 0,
    sort_order      INTEGER       NOT NULL DEFAULT 0,
    confidence      DOUBLE PRECISION,
    risk_level      VARCHAR(16)   NOT NULL DEFAULT 'MEDIUM',
    evidence_json   TEXT,
    status          VARCHAR(32)   NOT NULL DEFAULT 'PENDING',
    review_comment  TEXT,
    published_id    VARCHAR(20),
    create_by       VARCHAR(64),
    review_by       VARCHAR(64),
    publish_by      VARCHAR(64),
    create_time     TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time     TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    review_time     TIMESTAMP,
    publish_time    TIMESTAMP
);
CREATE INDEX idx_intent_node_candidate_run ON t_intent_node_candidate (run_id);
CREATE INDEX idx_intent_node_candidate_status ON t_intent_node_candidate (status);
COMMENT ON TABLE t_intent_node_candidate IS 'AI辅助初始化意图节点候选表';
