ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS continuation_policy_enabled boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS continuation_policy_target_epochs integer,
    ADD COLUMN IF NOT EXISTS continuation_policy_min_evals integer NOT NULL DEFAULT 2,
    ADD COLUMN IF NOT EXISTS continuation_policy_patience integer NOT NULL DEFAULT 2,
    ADD COLUMN IF NOT EXISTS continuation_policy_min_leader_score double precision,
    ADD COLUMN IF NOT EXISTS continuation_policy_min_infer_accuracy double precision,
    ADD COLUMN IF NOT EXISTS continuation_policy_min_improvement double precision,
    ADD COLUMN IF NOT EXISTS continuation_policy_max_degradation double precision,
    ADD COLUMN IF NOT EXISTS continuation_policy_top_n integer,
    ADD COLUMN IF NOT EXISTS continuation_policy_scope text NOT NULL DEFAULT 'symbol_horizon',
    ADD COLUMN IF NOT EXISTS continuation_policy_trend_mode text NOT NULL DEFAULT 'none',
    ADD COLUMN IF NOT EXISTS continuation_policy_source_mode text NOT NULL DEFAULT 'best_checkpoint',
    ADD COLUMN IF NOT EXISTS continuation_policy_include_excluded boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS continuation_candidate_excluded boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS continuation_policy_revision bigint NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS continuation_policy_last_decision text,
    ADD COLUMN IF NOT EXISTS continuation_policy_last_decision_at timestamptz,
    ADD COLUMN IF NOT EXISTS continuation_policy_last_reason text,
    ADD COLUMN IF NOT EXISTS continuation_policy_selected_model_id bigint,
    ADD COLUMN IF NOT EXISTS continuation_policy_queued_experiment_id bigint,
    ADD COLUMN IF NOT EXISTS parent_experiment_id bigint,
    ADD COLUMN IF NOT EXISTS continuation_source_experiment_id bigint,
    ADD COLUMN IF NOT EXISTS continuation_source_model_id bigint,
    ADD COLUMN IF NOT EXISTS continuation_source_epoch integer,
    ADD COLUMN IF NOT EXISTS continuation_generation integer NOT NULL DEFAULT 1;

CREATE TABLE IF NOT EXISTS experiment_continuation_decision (
    continuation_decision_id BIGSERIAL PRIMARY KEY,
    source_experiment_id BIGINT NOT NULL REFERENCES experiment(experiment_id),
    source_model_id BIGINT NOT NULL REFERENCES model(model_id),
    source_analysis_id BIGINT NOT NULL REFERENCES experiment_analysis_result(analysis_id),
    source_checkpoint_eval_id BIGINT REFERENCES experiment_checkpoint_eval(checkpoint_eval_id),
    source_epoch INTEGER NOT NULL,
    target_epochs INTEGER NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    leader_score DOUBLE PRECISION,
    infer_accuracy DOUBLE PRECISION,
    rank_value INTEGER,
    rank_scope TEXT,
    observed_eval_count INTEGER NOT NULL,
    patience_window INTEGER NOT NULL,
    trend_metric TEXT,
    trend_value DOUBLE PRECISION,
    policy_revision BIGINT NOT NULL,
    policy_hash TEXT NOT NULL,
    evidence_watermark TEXT NOT NULL,
    queued_experiment_id BIGINT REFERENCES experiment(experiment_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS continuation_decision_id bigint;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_target_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_target_check
            CHECK (continuation_policy_target_epochs IS NULL OR continuation_policy_target_epochs > 0);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_counts_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_counts_check
            CHECK (continuation_policy_min_evals > 0 AND continuation_policy_patience > 0);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_finite_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_finite_check
            CHECK (
                (continuation_policy_min_leader_score IS NULL OR
                 abs(continuation_policy_min_leader_score) < 'Infinity'::double precision)
                AND
                (continuation_policy_min_infer_accuracy IS NULL OR
                 abs(continuation_policy_min_infer_accuracy) < 'Infinity'::double precision)
                AND
                (continuation_policy_min_improvement IS NULL OR
                 (continuation_policy_min_improvement >= 0 AND
                  continuation_policy_min_improvement < 'Infinity'::double precision))
                AND
                (continuation_policy_max_degradation IS NULL OR
                 (continuation_policy_max_degradation >= 0 AND
                  continuation_policy_max_degradation < 'Infinity'::double precision))
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_top_n_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_top_n_check
            CHECK (continuation_policy_top_n IS NULL OR continuation_policy_top_n > 0);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_scope_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_scope_check
            CHECK (continuation_policy_scope IN ('symbol_horizon', 'horizon', 'global'));
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_trend_mode_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_trend_mode_check
            CHECK (continuation_policy_trend_mode IN ('none', 'non_degrading', 'improving'));
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_source_mode_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_source_mode_check
            CHECK (continuation_policy_source_mode IN ('best_checkpoint', 'latest_checkpoint', 'final_model'));
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_trend_config_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_trend_config_check
            CHECK (
                (continuation_policy_trend_mode = 'none'
                 AND continuation_policy_min_improvement IS NULL
                 AND continuation_policy_max_degradation IS NULL)
                OR
                (continuation_policy_trend_mode = 'non_degrading'
                 AND continuation_policy_max_degradation IS NOT NULL
                 AND continuation_policy_min_improvement IS NULL)
                OR
                (continuation_policy_trend_mode = 'improving'
                 AND continuation_policy_min_improvement IS NOT NULL
                 AND continuation_policy_max_degradation IS NULL)
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_enabled_config_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_enabled_config_check
            CHECK (
                NOT continuation_policy_enabled
                OR (
                    continuation_policy_target_epochs IS NOT NULL
                    AND (
                        continuation_policy_min_leader_score IS NOT NULL
                        OR continuation_policy_min_infer_accuracy IS NOT NULL
                        OR continuation_policy_top_n IS NOT NULL
                        OR continuation_policy_trend_mode <> 'none'
                    )
                )
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_revision_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_revision_check
            CHECK (continuation_policy_revision > 0);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_last_decision_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_last_decision_check
            CHECK (
                continuation_policy_last_decision IS NULL
                OR continuation_policy_last_decision IN (
                    'eligible',
                    'insufficient_evidence',
                    'rejected_threshold',
                    'rejected_rank',
                    'rejected_trend',
                    'already_continued',
                    'continuation_queued',
                    'skipped',
                    'error'
                )
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_lineage_shape_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_lineage_shape_check
            CHECK (
                (parent_experiment_id IS NULL
                 AND continuation_source_experiment_id IS NULL
                 AND continuation_source_model_id IS NULL
                 AND continuation_source_epoch IS NULL
                 AND continuation_decision_id IS NULL)
                OR
                (parent_experiment_id IS NOT NULL
                 AND continuation_source_experiment_id IS NOT NULL
                 AND continuation_source_model_id IS NOT NULL
                 AND continuation_source_epoch IS NOT NULL
                 AND continuation_decision_id IS NOT NULL
                 AND parent_experiment_id = continuation_source_experiment_id
                 AND resume_model_id = continuation_source_model_id
                 AND target_epochs > continuation_source_epoch
                 AND continuation_source_experiment_id <> experiment_id)
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_parent_fkey'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_parent_fkey
            FOREIGN KEY (parent_experiment_id) REFERENCES experiment(experiment_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_source_experiment_fkey'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_source_experiment_fkey
            FOREIGN KEY (continuation_source_experiment_id) REFERENCES experiment(experiment_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_source_model_fkey'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_source_model_fkey
            FOREIGN KEY (continuation_source_model_id) REFERENCES model(model_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_decision_fkey'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_decision_fkey
            FOREIGN KEY (continuation_decision_id)
            REFERENCES experiment_continuation_decision(continuation_decision_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_selected_model_fkey'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_selected_model_fkey
            FOREIGN KEY (continuation_policy_selected_model_id) REFERENCES model(model_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_queued_experiment_fkey'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_queued_experiment_fkey
            FOREIGN KEY (continuation_policy_queued_experiment_id) REFERENCES experiment(experiment_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_decision_value_check'
    ) THEN
        ALTER TABLE experiment_continuation_decision
            ADD CONSTRAINT experiment_continuation_decision_value_check
            CHECK (decision IN (
                'eligible',
                'insufficient_evidence',
                'rejected_threshold',
                'rejected_rank',
                'rejected_trend',
                'already_continued',
                'continuation_queued',
                'skipped',
                'error'
            ));
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_decision_values_check'
    ) THEN
        ALTER TABLE experiment_continuation_decision
            ADD CONSTRAINT experiment_continuation_decision_values_check
            CHECK (
                source_epoch > 0
                AND target_epochs > source_epoch
                AND observed_eval_count > 0
                AND patience_window > 0
                AND policy_revision > 0
                AND policy_hash <> ''
                AND evidence_watermark <> ''
                AND (rank_value IS NULL OR rank_value > 0)
                AND (rank_scope IS NULL OR rank_scope IN ('symbol_horizon', 'horizon', 'global'))
                AND (trend_metric IS NULL OR trend_metric IN ('leader_score', 'infer_accuracy'))
                AND (trend_value IS NULL OR abs(trend_value) < 'Infinity'::double precision)
                AND (queued_experiment_id IS NULL OR queued_experiment_id <> source_experiment_id)
            );
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS experiment_continuation_decision_source_target_uidx
    ON experiment_continuation_decision(source_experiment_id, target_epochs);

CREATE UNIQUE INDEX IF NOT EXISTS experiment_continuation_decision_model_target_uidx
    ON experiment_continuation_decision(source_model_id, target_epochs);

CREATE UNIQUE INDEX IF NOT EXISTS experiment_continuation_decision_queued_uidx
    ON experiment_continuation_decision(queued_experiment_id)
    WHERE queued_experiment_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS experiment_continuation_source_model_target_uidx
    ON experiment(continuation_source_model_id, target_epochs)
    WHERE continuation_source_model_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS experiment_continuation_policy_enabled_idx
    ON experiment(continuation_policy_enabled, status, phase);

CREATE INDEX IF NOT EXISTS experiment_continuation_source_idx
    ON experiment(continuation_source_experiment_id, continuation_source_model_id);

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
GRANT SELECT, INSERT, UPDATE, DELETE ON experiment_continuation_decision TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE experiment_continuation_decision_continuation_decision_id_seq TO pqxx;
