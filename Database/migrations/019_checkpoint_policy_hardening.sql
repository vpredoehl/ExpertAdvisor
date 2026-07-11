DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'experiment_checkpoint_policy_min_leader_score_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_checkpoint_policy_min_leader_score_check
            CHECK (
                checkpoint_policy_min_leader_score IS NULL
                OR (
                    checkpoint_policy_min_leader_score > 0
                    AND checkpoint_policy_min_leader_score < 'Infinity'::double precision
                )
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'experiment_checkpoint_policy_min_infer_accuracy_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_checkpoint_policy_min_infer_accuracy_check
            CHECK (
                checkpoint_policy_min_infer_accuracy IS NULL
                OR (
                    checkpoint_policy_min_infer_accuracy > 0
                    AND checkpoint_policy_min_infer_accuracy < 'Infinity'::double precision
                )
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'experiment_checkpoint_policy_top_n_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_checkpoint_policy_top_n_check
            CHECK (checkpoint_policy_top_n IS NULL OR checkpoint_policy_top_n > 0);
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'experiment_checkpoint_policy_scope_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_checkpoint_policy_scope_check
            CHECK (checkpoint_policy_scope IN ('symbol_horizon', 'horizon', 'global'));
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'experiment_checkpoint_policy_stop_mode_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_checkpoint_policy_stop_mode_check
            CHECK (
                checkpoint_policy_stop_mode IN (
                    'next_checkpoint',
                    'current_checkpoint_if_possible',
                    'mark_pruned_when_not_running'
                )
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'experiment_checkpoint_policy_grace_evals_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_checkpoint_policy_grace_evals_check
            CHECK (checkpoint_policy_grace_evals > 0);
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'experiment_checkpoint_policy_enabled_rules_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_checkpoint_policy_enabled_rules_check
            CHECK (
                NOT checkpoint_policy_enabled
                OR checkpoint_policy_min_leader_score IS NOT NULL
                OR checkpoint_policy_min_infer_accuracy IS NOT NULL
                OR checkpoint_policy_top_n IS NOT NULL
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'experiment_checkpoint_policy_requires_infer_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_checkpoint_policy_requires_infer_check
            CHECK (
                NOT checkpoint_policy_enabled
                OR checkpoint_infer_enabled
                OR opportunistic_checkpoint_infer
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'experiment_checkpoint_decision_value_check'
    ) THEN
        ALTER TABLE experiment_checkpoint_decision
            ADD CONSTRAINT experiment_checkpoint_decision_value_check
            CHECK (decision IN ('continue', 'continue_grace', 'stop_requested', 'skipped'));
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'experiment_checkpoint_decision_rank_scope_check'
    ) THEN
        ALTER TABLE experiment_checkpoint_decision
            ADD CONSTRAINT experiment_checkpoint_decision_rank_scope_check
            CHECK (rank_scope IS NULL OR rank_scope IN ('symbol_horizon', 'horizon', 'global'));
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'experiment_checkpoint_policy_last_decision_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_checkpoint_policy_last_decision_check
            CHECK (
                checkpoint_policy_last_decision IS NULL
                OR checkpoint_policy_last_decision IN (
                    'continue',
                    'continue_grace',
                    'stop_requested',
                    'skipped'
                )
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'experiment_analysis_result_scope_check'
    ) THEN
        ALTER TABLE experiment_analysis_result
            ADD CONSTRAINT experiment_analysis_result_scope_check
            CHECK (analysis_scope IN ('final', 'checkpoint'));
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'experiment_analysis_result_scope_identity_check'
    ) THEN
        ALTER TABLE experiment_analysis_result
            ADD CONSTRAINT experiment_analysis_result_scope_identity_check
            CHECK (
                (analysis_scope = 'final'
                 AND checkpoint_eval_id IS NULL
                 AND parent_experiment_id IS NULL
                 AND checkpoint_epoch IS NULL)
                OR
                (analysis_scope = 'checkpoint'
                 AND checkpoint_eval_id IS NOT NULL
                 AND parent_experiment_id IS NOT NULL
                 AND checkpoint_epoch IS NOT NULL)
            );
    END IF;
END $$;

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
GRANT SELECT, INSERT, UPDATE, DELETE ON experiment_checkpoint_decision TO pqxx;
