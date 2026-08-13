ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS continuation_policy_max_target_epochs integer,
    ADD COLUMN IF NOT EXISTS continuation_policy_inherited_from_revision bigint,
    ADD COLUMN IF NOT EXISTS continuation_policy_inherited_from_hash text;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_inheritance_status_check'
    ) THEN
        ALTER TABLE experiment
            DROP CONSTRAINT experiment_continuation_policy_inheritance_status_check;
    END IF;

    ALTER TABLE experiment
        ADD CONSTRAINT experiment_continuation_policy_inheritance_status_check
        CHECK (
            continuation_policy_inheritance_status IN (
                'not_requested',
                'valid',
                'max_target_reached'
            )
        );

    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_inheritance_shape_check'
    ) THEN
        ALTER TABLE experiment
            DROP CONSTRAINT experiment_continuation_policy_inheritance_shape_check;
    END IF;

    ALTER TABLE experiment
        ADD CONSTRAINT experiment_continuation_policy_inheritance_shape_check
        CHECK (
            (NOT continuation_policy_inherited
             AND continuation_policy_inherited_from_experiment_id IS NULL
             AND continuation_policy_inheritance_status = 'not_requested')
            OR
            (continuation_policy_inherited
             AND continuation_policy_inherited_from_experiment_id IS NOT NULL
             AND continuation_policy_inheritance_status IN ('valid', 'max_target_reached'))
        );

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_max_target_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_max_target_check
            CHECK (
                continuation_policy_max_target_epochs IS NULL
                OR (
                    continuation_policy_max_target_epochs > 0
                    AND continuation_policy_max_target_epochs >= target_epochs
                    AND (
                        continuation_policy_inheritance_status = 'max_target_reached'
                        OR continuation_policy_max_target_epochs > target_epochs
                    )
                    AND (
                        continuation_policy_target_epochs IS NULL
                        OR continuation_policy_target_epochs <= continuation_policy_max_target_epochs
                    )
                )
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_provenance_revision_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_provenance_revision_check
            CHECK (
                continuation_policy_inherited_from_revision IS NULL
                OR continuation_policy_inherited_from_revision > 0
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_provenance_hash_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_provenance_hash_check
            CHECK (
                continuation_policy_inherited_from_hash IS NULL
                OR continuation_policy_inherited_from_hash <> ''
            );
    END IF;
END $$;

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
