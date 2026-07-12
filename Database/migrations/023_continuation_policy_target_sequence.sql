ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS continuation_policy_progression_mode text,
    ADD COLUMN IF NOT EXISTS continuation_policy_target_sequence integer[];

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_progression_mode_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_progression_mode_check
            CHECK (
                continuation_policy_progression_mode IS NULL
                OR continuation_policy_progression_mode IN (
                    'fixed_increment',
                    'target_sequence'
                )
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_progression_shape_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_progression_shape_check
            CHECK (
                (
                    COALESCE(
                        continuation_policy_progression_mode,
                        'fixed_increment'
                    ) = 'fixed_increment'
                    AND continuation_policy_target_sequence IS NULL
                )
                OR
                (
                    continuation_policy_progression_mode = 'target_sequence'
                    AND continuation_policy_target_increment IS NULL
                    AND continuation_policy_target_sequence IS NOT NULL
                    AND cardinality(continuation_policy_target_sequence) > 0
                    AND (
                        continuation_policy_max_target_epochs IS NULL
                        OR continuation_policy_max_target_epochs =
                           continuation_policy_target_sequence[
                               cardinality(continuation_policy_target_sequence)
                           ]
                    )
                )
            );
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_inherit_config_check'
    ) THEN
        ALTER TABLE experiment
            DROP CONSTRAINT experiment_continuation_policy_inherit_config_check;
    END IF;

    ALTER TABLE experiment
        ADD CONSTRAINT experiment_continuation_policy_inherit_config_check
        CHECK (
            NOT continuation_policy_inherit_to_child
            OR
            (
                COALESCE(
                    continuation_policy_progression_mode,
                    'fixed_increment'
                ) = 'fixed_increment'
                AND continuation_policy_target_increment IS NOT NULL
                AND continuation_policy_target_sequence IS NULL
            )
            OR
            (
                continuation_policy_progression_mode = 'target_sequence'
                AND continuation_policy_target_increment IS NULL
                AND continuation_policy_target_sequence IS NOT NULL
                AND cardinality(continuation_policy_target_sequence) > 0
            )
        );
END $$;

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
