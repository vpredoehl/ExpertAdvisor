ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS worker_started_at timestamptz;

CREATE OR REPLACE FUNCTION continuation_target_sequence_is_valid(sequence integer[])
RETURNS boolean
LANGUAGE sql
IMMUTABLE
STRICT
AS $$
    SELECT cardinality(sequence) > 0
       AND NOT EXISTS (
            SELECT 1
            FROM generate_subscripts(sequence, 1) AS position
            WHERE sequence[position] <= 0
               OR (
                    position > array_lower(sequence, 1)
                    AND sequence[position] <= sequence[position - 1]
               )
       );
$$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_progression_shape_check'
    ) THEN
        ALTER TABLE experiment
            DROP CONSTRAINT experiment_continuation_policy_progression_shape_check;
    END IF;

    ALTER TABLE experiment
        ADD CONSTRAINT experiment_continuation_policy_progression_shape_check
        CHECK (
            CASE COALESCE(
                continuation_policy_progression_mode,
                'fixed_increment'
            )
                WHEN 'fixed_increment' THEN
                    continuation_policy_target_sequence IS NULL
                WHEN 'target_sequence' THEN
                    continuation_policy_target_increment IS NULL
                    AND continuation_policy_target_sequence IS NOT NULL
                    AND continuation_target_sequence_is_valid(
                        continuation_policy_target_sequence
                    )
                    AND (
                        continuation_policy_max_target_epochs IS NULL
                        OR continuation_policy_max_target_epochs =
                           continuation_policy_target_sequence[
                               cardinality(continuation_policy_target_sequence)
                           ]
                    )
                ELSE false
            END
        );

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
            CASE COALESCE(
                continuation_policy_progression_mode,
                'fixed_increment'
            )
                WHEN 'fixed_increment' THEN
                    continuation_policy_target_increment IS NOT NULL
                    AND continuation_policy_target_sequence IS NULL
                WHEN 'target_sequence' THEN
                    continuation_policy_target_increment IS NULL
                    AND continuation_policy_target_sequence IS NOT NULL
                    AND continuation_target_sequence_is_valid(
                        continuation_policy_target_sequence
                    )
                ELSE false
            END
        );
END $$;

GRANT EXECUTE ON FUNCTION continuation_target_sequence_is_valid(integer[]) TO pqxx;
GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
