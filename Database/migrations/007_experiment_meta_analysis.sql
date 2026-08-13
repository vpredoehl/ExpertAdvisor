CREATE TABLE IF NOT EXISTS experiment_meta_analysis (
    meta_analysis_id bigserial PRIMARY KEY,
    analysis_scope text NOT NULL,
    generated_at timestamptz NOT NULL DEFAULT now(),
    completed_experiments bigint,
    completed_models bigint,
    summary_markdown text,
    recommendations_json jsonb,
    statistics_json jsonb,
    leaderboard_snapshot jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS experiment_meta_analysis_generated_idx
    ON experiment_meta_analysis (generated_at DESC);

CREATE INDEX IF NOT EXISTS experiment_meta_analysis_scope_idx
    ON experiment_meta_analysis (analysis_scope, generated_at DESC);

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment_meta_analysis TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE experiment_meta_analysis_meta_analysis_id_seq TO pqxx;
