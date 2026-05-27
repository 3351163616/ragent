ALTER TABLE t_term_mapping_candidate
    ADD COLUMN IF NOT EXISTS generation_source VARCHAR(16) NOT NULL DEFAULT 'UNKNOWN';

COMMENT ON COLUMN t_term_mapping_candidate.generation_source IS '候选生成来源：RULE/LLM/UNKNOWN';

ALTER TABLE t_intent_node_candidate
    ADD COLUMN IF NOT EXISTS generation_source VARCHAR(16) NOT NULL DEFAULT 'UNKNOWN';

COMMENT ON COLUMN t_intent_node_candidate.generation_source IS '候选生成来源：RULE/LLM/UNKNOWN';
