CREATE TABLE IF NOT EXISTS experiment_recommendation_scan (
    recommendation_scan_id bigserial PRIMARY KEY,
    status text NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    policy_canonical text NOT NULL,
    policy_hash text NOT NULL,
    policy_version integer NOT NULL CHECK (policy_version > 0),
    symbol_filter text,
    horizon_filter integer CHECK (horizon_filter IS NULL OR horizon_filter > 0),
    source_experiment_filter bigint REFERENCES experiment(experiment_id),
    requested_maximum integer CHECK (requested_maximum IS NULL OR requested_maximum > 0),
    sources_scanned integer NOT NULL DEFAULT 0 CHECK (sources_scanned >= 0),
    sources_eligible integer NOT NULL DEFAULT 0 CHECK (sources_eligible >= 0),
    sources_skipped integer NOT NULL DEFAULT 0 CHECK (sources_skipped >= 0),
    candidates_generated integer NOT NULL DEFAULT 0 CHECK (candidates_generated >= 0),
    candidates_rejected integer NOT NULL DEFAULT 0 CHECK (candidates_rejected >= 0),
    duplicates_existing_experiment integer NOT NULL DEFAULT 0 CHECK (duplicates_existing_experiment >= 0),
    duplicates_terminal_experiment integer NOT NULL DEFAULT 0 CHECK (duplicates_terminal_experiment >= 0),
    duplicates_active_recommendation integer NOT NULL DEFAULT 0 CHECK (duplicates_active_recommendation >= 0),
    duplicates_historical_recommendation integer NOT NULL DEFAULT 0 CHECK (duplicates_historical_recommendation >= 0),
    hash_collisions integer NOT NULL DEFAULT 0 CHECK (hash_collisions >= 0),
    recommendations_created integer NOT NULL DEFAULT 0 CHECK (recommendations_created >= 0),
    recommendations_already_existing integer NOT NULL DEFAULT 0 CHECK (recommendations_already_existing >= 0),
    persistence_errors integer NOT NULL DEFAULT 0 CHECK (persistence_errors >= 0),
    started_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    error_message text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (
        (status = 'running' AND completed_at IS NULL AND error_message IS NULL)
        OR (status = 'completed' AND completed_at IS NOT NULL AND error_message IS NULL)
        OR (status = 'failed' AND completed_at IS NOT NULL AND error_message IS NOT NULL)
    )
);

-- This CREATE handles fresh databases. The ALTER block below upgrades the
-- empty legacy Phase 4 prototype table recorded by historical migration 025
-- without rewriting or guessing any old canonical identities.
CREATE TABLE IF NOT EXISTS experiment_recommendation (
    recommendation_id bigserial PRIMARY KEY,
    recommendation_scan_id bigint NOT NULL REFERENCES experiment_recommendation_scan(recommendation_scan_id),
    status text NOT NULL DEFAULT 'proposed'
        CHECK (status IN ('proposed', 'rejected', 'expired', 'approved')),
    source_experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
    source_model_id bigint,
    source_analysis_id bigint REFERENCES experiment_analysis_result(analysis_id),
    source_symbol text NOT NULL,
    source_prediction_horizon integer NOT NULL,
    source_leader_score double precision NOT NULL,
    source_infer_accuracy double precision NOT NULL,
    source_predicted_neutral_proportion double precision,
    source_evidence_count bigint NOT NULL,
    changed_parameter text NOT NULL,
    source_value_canonical text NOT NULL,
    proposed_value_canonical text NOT NULL,
    absolute_delta double precision NOT NULL,
    relative_delta double precision,
    horizon_delta integer,
    semantic_configuration_canonical text NOT NULL,
    semantic_hash text NOT NULL,
    invocation_configuration_canonical text NOT NULL,
    invocation_hash text NOT NULL,
    policy_canonical text NOT NULL,
    policy_hash text NOT NULL,
    generation_ordinal integer NOT NULL,
    structural_rank integer NOT NULL,
    duplicate_type text NOT NULL DEFAULT 'no_duplicate',
    matched_experiment_id bigint REFERENCES experiment(experiment_id),
    matched_recommendation_id bigint REFERENCES experiment_recommendation(recommendation_id),
    reason text NOT NULL,
    approved_experiment_id bigint REFERENCES experiment(experiment_id),
    rejected_at timestamptz,
    rejected_reason text,
    expired_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- Legacy migration 025 used a candidate-hash/policy-hash uniqueness rule and
-- required placeholder score fields. Step 3 must neither trust a 64-bit hash
-- nor invent scores, so remove that uniqueness constraint and make only the
-- obsolete legacy-only representation nullable.
ALTER TABLE experiment_recommendation
    DROP CONSTRAINT IF EXISTS experiment_recommendation_candidate_policy_uidx;

DO $$
DECLARE
    legacy_column text;
BEGIN
    FOREACH legacy_column IN ARRAY ARRAY[
        'proposed_symbol', 'proposed_prediction_horizon',
        'proposed_configuration', 'proposed_configuration_canonical',
        'changed_parameter_names', 'candidate_hash',
        'recommendation_policy_hash', 'recommendation_policy_canonical',
        'recommendation_score', 'score_components',
        'rank_within_source', 'rank_within_scan'
    ]
    LOOP
        IF EXISTS (
            SELECT 1 FROM information_schema.columns c
            WHERE c.table_schema='public'
              AND c.table_name='experiment_recommendation'
              AND c.column_name=legacy_column
        ) THEN
            EXECUTE format(
                'ALTER TABLE experiment_recommendation ALTER COLUMN %I DROP NOT NULL',
                legacy_column);
        END IF;
    END LOOP;
END $$;

ALTER TABLE experiment_recommendation
    ADD COLUMN IF NOT EXISTS recommendation_scan_id bigint REFERENCES experiment_recommendation_scan(recommendation_scan_id),
    ADD COLUMN IF NOT EXISTS source_predicted_neutral_proportion double precision,
    ADD COLUMN IF NOT EXISTS source_evidence_count bigint,
    ADD COLUMN IF NOT EXISTS changed_parameter text,
    ADD COLUMN IF NOT EXISTS source_value_canonical text,
    ADD COLUMN IF NOT EXISTS proposed_value_canonical text,
    ADD COLUMN IF NOT EXISTS absolute_delta double precision,
    ADD COLUMN IF NOT EXISTS relative_delta double precision,
    ADD COLUMN IF NOT EXISTS horizon_delta integer,
    ADD COLUMN IF NOT EXISTS semantic_configuration_canonical text,
    ADD COLUMN IF NOT EXISTS semantic_hash text,
    ADD COLUMN IF NOT EXISTS invocation_configuration_canonical text,
    ADD COLUMN IF NOT EXISTS invocation_hash text,
    ADD COLUMN IF NOT EXISTS policy_canonical text,
    ADD COLUMN IF NOT EXISTS policy_hash text,
    ADD COLUMN IF NOT EXISTS generation_ordinal integer,
    ADD COLUMN IF NOT EXISTS structural_rank integer,
    ADD COLUMN IF NOT EXISTS duplicate_type text NOT NULL DEFAULT 'no_duplicate',
    ADD COLUMN IF NOT EXISTS matched_experiment_id bigint REFERENCES experiment(experiment_id),
    ADD COLUMN IF NOT EXISTS matched_recommendation_id bigint REFERENCES experiment_recommendation(recommendation_id),
    ADD COLUMN IF NOT EXISTS expired_at timestamptz;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='experiment_recommendation_step3_shape_check') THEN
        ALTER TABLE experiment_recommendation
            ADD CONSTRAINT experiment_recommendation_step3_shape_check
            CHECK (
                recommendation_scan_id IS NULL
                OR (
                    source_analysis_id IS NOT NULL
                    AND source_evidence_count > 0
                    AND changed_parameter IN ('core_lr_mult','head_lr_mult','label_threshold','prediction_horizon')
                    AND source_value_canonical IS NOT NULL
                    AND proposed_value_canonical IS NOT NULL
                    AND absolute_delta >= 0
                    AND semantic_configuration_canonical IS NOT NULL
                    AND semantic_hash IS NOT NULL
                    AND invocation_configuration_canonical IS NOT NULL
                    AND invocation_hash IS NOT NULL
                    AND policy_canonical IS NOT NULL
                    AND policy_hash IS NOT NULL
                    AND generation_ordinal > 0
                    AND structural_rank > 0
                    AND duplicate_type IN ('no_duplicate','existing_experiment','excluded_terminal_experiment','active_recommendation','historical_recommendation')
                )
            ) NOT VALID;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='experiment_recommendation_step3_status_shape_check') THEN
        ALTER TABLE experiment_recommendation
            ADD CONSTRAINT experiment_recommendation_step3_status_shape_check
            CHECK (
                (status <> 'rejected' OR (rejected_at IS NOT NULL AND rejected_reason IS NOT NULL))
                AND (status <> 'expired' OR expired_at IS NOT NULL)
                AND (status <> 'approved' OR approved_experiment_id IS NOT NULL)
            ) NOT VALID;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS experiment_recommendation_scan_status_idx
    ON experiment_recommendation_scan(status, recommendation_scan_id DESC);
CREATE INDEX IF NOT EXISTS experiment_recommendation_scan_policy_hash_idx
    ON experiment_recommendation_scan(policy_hash, recommendation_scan_id DESC);
CREATE INDEX IF NOT EXISTS experiment_recommendation_step3_source_idx
    ON experiment_recommendation(source_experiment_id, recommendation_id DESC);
CREATE INDEX IF NOT EXISTS experiment_recommendation_step3_scan_idx
    ON experiment_recommendation(recommendation_scan_id, structural_rank, recommendation_id);
CREATE INDEX IF NOT EXISTS experiment_recommendation_step3_status_symbol_idx
    ON experiment_recommendation(status, source_symbol, source_prediction_horizon, recommendation_id DESC);
CREATE INDEX IF NOT EXISTS experiment_recommendation_step3_semantic_hash_idx
    ON experiment_recommendation(semantic_hash, policy_hash, recommendation_id DESC);

-- Canonical text, not FNV-1a, is authoritative. NULL excludes unmapped legacy
-- rows; no historical canonical identity is guessed.
CREATE UNIQUE INDEX IF NOT EXISTS experiment_recommendation_active_canonical_uidx
    ON experiment_recommendation(semantic_configuration_canonical, policy_canonical)
    WHERE status IN ('proposed', 'approved')
      AND semantic_configuration_canonical IS NOT NULL
      AND policy_canonical IS NOT NULL;

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment_recommendation_scan TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE experiment_recommendation_scan_recommendation_scan_id_seq TO pqxx;
GRANT SELECT, INSERT, UPDATE, DELETE ON experiment_recommendation TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE experiment_recommendation_recommendation_id_seq TO pqxx;
