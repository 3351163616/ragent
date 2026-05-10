-- ragent v1.2 -> v1.3 升级脚本
-- t_message 表：新增回答引用来源字段

ALTER TABLE t_message ADD COLUMN citations_json TEXT DEFAULT NULL;
COMMENT ON COLUMN t_message.citations_json IS '回答引用来源JSON';
