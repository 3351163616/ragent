-- ragent v1.4 -> v1.5 升级脚本
-- AI 辅助初始化配置：记录每次任务实际采样的文档与 chunk 明细

ALTER TABLE t_config_bootstrap_run
    ADD COLUMN IF NOT EXISTS sample_json TEXT;

COMMENT ON COLUMN t_config_bootstrap_run.sample_json IS 'AI辅助初始化采样文档与chunk明细JSON';
