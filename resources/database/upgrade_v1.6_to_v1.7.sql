CREATE TABLE IF NOT EXISTS t_eval_suite (
    id          VARCHAR(64) PRIMARY KEY,
    name        VARCHAR(128) NOT NULL,
    description TEXT,
    version     VARCHAR(32)  NOT NULL DEFAULT 'v1',
    enabled     SMALLINT     NOT NULL DEFAULT 1,
    create_time TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted     SMALLINT     NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_eval_suite_deleted ON t_eval_suite (deleted);
CREATE INDEX IF NOT EXISTS idx_eval_suite_update_time ON t_eval_suite (update_time);

COMMENT ON TABLE t_eval_suite IS 'RAG 自动化评测套件';
COMMENT ON COLUMN t_eval_suite.name IS '评测套件名称';
COMMENT ON COLUMN t_eval_suite.description IS '评测套件描述';
COMMENT ON COLUMN t_eval_suite.version IS '评测套件版本';
COMMENT ON COLUMN t_eval_suite.enabled IS '是否启用 0：停用 1：启用';
COMMENT ON COLUMN t_eval_suite.deleted IS '是否删除 0：正常 1：删除';

CREATE TABLE IF NOT EXISTS t_eval_case (
    id           VARCHAR(64) PRIMARY KEY,
    suite_id     VARCHAR(64) NOT NULL,
    question     TEXT        NOT NULL,
    category     VARCHAR(64),
    ground_truth TEXT,
    enabled      SMALLINT    NOT NULL DEFAULT 1,
    create_time  TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time  TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted      SMALLINT    NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_eval_case_suite_id ON t_eval_case (suite_id);
CREATE INDEX IF NOT EXISTS idx_eval_case_deleted ON t_eval_case (deleted);
CREATE INDEX IF NOT EXISTS idx_eval_case_category ON t_eval_case (category);

COMMENT ON TABLE t_eval_case IS 'RAG 自动化评测用例';
COMMENT ON COLUMN t_eval_case.suite_id IS '所属评测套件 ID';
COMMENT ON COLUMN t_eval_case.question IS '评测问题';
COMMENT ON COLUMN t_eval_case.category IS '业务分类';
COMMENT ON COLUMN t_eval_case.ground_truth IS '标准答案 JSON';
COMMENT ON COLUMN t_eval_case.enabled IS '是否启用 0：停用 1：启用';
COMMENT ON COLUMN t_eval_case.deleted IS '是否删除 0：正常 1：删除';

CREATE TABLE IF NOT EXISTS t_eval_run (
    id               VARCHAR(64) PRIMARY KEY,
    suite_id         VARCHAR(64) NOT NULL,
    trial_count      INTEGER     NOT NULL DEFAULT 5,
    status           VARCHAR(32) NOT NULL,
    total_cases      INTEGER     NOT NULL DEFAULT 0,
    total_trials     INTEGER     NOT NULL DEFAULT 0,
    completed_trials INTEGER     NOT NULL DEFAULT 0,
    passed_trials    INTEGER     NOT NULL DEFAULT 0,
    success_rate     DOUBLE PRECISION NOT NULL DEFAULT 0,
    pass_at_k        DOUBLE PRECISION NOT NULL DEFAULT 0,
    started_at       TIMESTAMP,
    finished_at      TIMESTAMP,
    duration_ms      BIGINT,
    summary_json     TEXT,
    error_message    TEXT,
    create_time      TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time      TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted          SMALLINT    NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_eval_run_suite_id ON t_eval_run (suite_id);
CREATE INDEX IF NOT EXISTS idx_eval_run_status ON t_eval_run (status);
CREATE INDEX IF NOT EXISTS idx_eval_run_started_at ON t_eval_run (started_at);

COMMENT ON TABLE t_eval_run IS 'RAG 自动化评测运行记录';
COMMENT ON COLUMN t_eval_run.trial_count IS '每个用例执行次数';
COMMENT ON COLUMN t_eval_run.success_rate IS 'trial 级成功率';
COMMENT ON COLUMN t_eval_run.pass_at_k IS 'case 级 Pass@K';
COMMENT ON COLUMN t_eval_run.summary_json IS '汇总报告 JSON';

CREATE TABLE IF NOT EXISTS t_eval_trial (
    id            VARCHAR(64) PRIMARY KEY,
    run_id        VARCHAR(64) NOT NULL,
    suite_id      VARCHAR(64) NOT NULL,
    case_id       VARCHAR(64) NOT NULL,
    trial_index   INTEGER     NOT NULL,
    status        VARCHAR(32) NOT NULL,
    success       SMALLINT    NOT NULL DEFAULT 0,
    trace_id      VARCHAR(64),
    latency_ms    BIGINT,
    started_at    TIMESTAMP,
    finished_at   TIMESTAMP,
    duration_ms   BIGINT,
    snapshot_json TEXT,
    error_message TEXT,
    create_time   TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time   TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted       SMALLINT    NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_eval_trial_run_id ON t_eval_trial (run_id);
CREATE INDEX IF NOT EXISTS idx_eval_trial_case_id ON t_eval_trial (case_id);
CREATE INDEX IF NOT EXISTS idx_eval_trial_trace_id ON t_eval_trial (trace_id);

COMMENT ON TABLE t_eval_trial IS 'RAG 自动化评测 trial 执行记录';
COMMENT ON COLUMN t_eval_trial.success IS '质量评分是否通过 0：失败 1：通过';
COMMENT ON COLUMN t_eval_trial.snapshot_json IS '单次执行结构化快照 JSON';

CREATE TABLE IF NOT EXISTS t_eval_score (
    id           VARCHAR(64) PRIMARY KEY,
    run_id       VARCHAR(64) NOT NULL,
    trial_id     VARCHAR(64) NOT NULL,
    suite_id     VARCHAR(64) NOT NULL,
    case_id      VARCHAR(64) NOT NULL,
    metric_name  VARCHAR(64) NOT NULL,
    score_value  DOUBLE PRECISION NOT NULL DEFAULT 0,
    passed       SMALLINT    NOT NULL DEFAULT 0,
    reason       TEXT,
    detail_json  TEXT,
    create_time  TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time  TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted      SMALLINT    NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_eval_score_run_id ON t_eval_score (run_id);
CREATE INDEX IF NOT EXISTS idx_eval_score_trial_id ON t_eval_score (trial_id);
CREATE INDEX IF NOT EXISTS idx_eval_score_metric_name ON t_eval_score (metric_name);

COMMENT ON TABLE t_eval_score IS 'RAG 自动化评测指标评分';
COMMENT ON COLUMN t_eval_score.metric_name IS '指标名称';
COMMENT ON COLUMN t_eval_score.score_value IS '指标得分';
COMMENT ON COLUMN t_eval_score.passed IS '指标是否通过 0：失败 1：通过';
COMMENT ON COLUMN t_eval_score.detail_json IS '指标明细 JSON';
