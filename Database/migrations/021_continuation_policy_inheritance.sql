ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS continuation_policy_inherit_to_child boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS continuation_policy_target_increment integer,
    ADD COLUMN IF NOT EXISTS continuation_policy_inherited boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS continuation_policy_inherited_from_experiment_id bigint,
    ADD COLUMN IF NOT EXISTS continuation_policy_inheritance_status text NOT NULL DEFAULT 'not_requested';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_increment_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_increment_check
            CHECK (
                continuation_policy_target_increment IS NULL
                OR continuation_policy_target_increment > 0
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_inherit_config_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_inherit_config_check
            CHECK (
                NOT continuation_policy_inherit_to_child
                OR continuation_policy_target_increment IS NOT NULL
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_inheritance_status_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_inheritance_status_check
            CHECK (
                continuation_policy_inheritance_status IN ('not_requested', 'valid')
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_inheritance_shape_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_inheritance_shape_check
            CHECK (
                (NOT continuation_policy_inherited
                 AND continuation_policy_inherited_from_experiment_id IS NULL
                 AND continuation_policy_inheritance_status = 'not_requested')
                OR
                (continuation_policy_inherited
                 AND continuation_policy_inherited_from_experiment_id IS NOT NULL
                 AND continuation_policy_inheritance_status = 'valid')
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_continuation_policy_inherited_from_fkey'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_continuation_policy_inherited_from_fkey
            FOREIGN KEY (continuation_policy_inherited_from_experiment_id)
            REFERENCES experiment(experiment_id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS experiment_continuation_policy_inherited_from_idx
    ON experiment(continuation_policy_inherited_from_experiment_id)
    WHERE continuation_policy_inherited_from_experiment_id IS NOT NULL;

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
