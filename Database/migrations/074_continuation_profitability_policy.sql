-- Profitability Phase 2B is explicitly opt-in. NULL policy fields preserve
-- every pre-2B continuation policy and decision behavior.
ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS
        continuation_policy_min_profit_actionable_count bigint,
    ADD COLUMN IF NOT EXISTS
        continuation_policy_min_profit_aggregate_log_return_sum
        double precision,
    ADD COLUMN IF NOT EXISTS
        continuation_policy_min_profit_average_log_return
        double precision;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname =
            'experiment_cont_profit_actionable_count_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT
                experiment_cont_profit_actionable_count_check
            CHECK (
                continuation_policy_min_profit_actionable_count IS NULL
                OR continuation_policy_min_profit_actionable_count > 0
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname =
            'experiment_cont_profit_return_finite_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT
                experiment_cont_profit_return_finite_check
            CHECK (
                (
                    continuation_policy_min_profit_aggregate_log_return_sum
                    IS NULL
                    OR continuation_policy_min_profit_aggregate_log_return_sum
                       NOT IN (
                           'NaN'::double precision,
                           'Infinity'::double precision,
                           '-Infinity'::double precision
                       )
                )
                AND
                (
                    continuation_policy_min_profit_average_log_return
                    IS NULL
                    OR continuation_policy_min_profit_average_log_return
                       NOT IN (
                           'NaN'::double precision,
                           'Infinity'::double precision,
                           '-Infinity'::double precision
                       )
                )
            );
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_last_decision_check'
    ) THEN
        ALTER TABLE experiment
            DROP CONSTRAINT experiment_continuation_policy_last_decision_check;
    END IF;

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
                'rejected_profitability',
                'already_continued',
                'continuation_queued',
                'skipped',
                'error'
            )
        );

    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_decision_value_check'
    ) THEN
        ALTER TABLE experiment_continuation_decision
            DROP CONSTRAINT experiment_continuation_decision_value_check;
    END IF;

    ALTER TABLE experiment_continuation_decision
        ADD CONSTRAINT experiment_continuation_decision_value_check
        CHECK (
            decision IN (
                'eligible',
                'insufficient_evidence',
                'rejected_threshold',
                'rejected_rank',
                'rejected_trend',
                'rejected_profitability',
                'already_continued',
                'continuation_queued',
                'skipped',
                'error'
            )
        );
END $$;

COMMENT ON COLUMN
    experiment.continuation_policy_min_profit_actionable_count IS
    'Optional minimum actionable predictions in the exact selected continuation-source profitability observation.';
COMMENT ON COLUMN
    experiment.continuation_policy_min_profit_aggregate_log_return_sum IS
    'Optional minimum aggregate terminal-horizon directional log-return sum; not portfolio P&L.';
COMMENT ON COLUMN
    experiment.continuation_policy_min_profit_average_log_return IS
    'Optional minimum average terminal-horizon directional log return per actionable prediction; NULL observation averages fail this configured gate.';

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
