-- Phase 4A Step 3 hardening.  This migration is intentionally additive: 026
-- remains the creation/legacy-upgrade migration, while these constraints make
-- status metadata symmetric for all newly inserted or updated rows.  NOT VALID
-- avoids guessing or rewriting any legacy Phase 4 prototype history.

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_recommendation_scan_source_filter_positive_check'
    ) THEN
        ALTER TABLE experiment_recommendation_scan
            ADD CONSTRAINT experiment_recommendation_scan_source_filter_positive_check
            CHECK (source_experiment_filter IS NULL OR source_experiment_filter > 0)
            NOT VALID;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_recommendation_scan_failed_error_nonempty_check'
    ) THEN
        ALTER TABLE experiment_recommendation_scan
            ADD CONSTRAINT experiment_recommendation_scan_failed_error_nonempty_check
            CHECK (status <> 'failed' OR btrim(error_message) <> '')
            NOT VALID;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_recommendation_status_metadata_symmetric_check'
    ) THEN
        ALTER TABLE experiment_recommendation
            ADD CONSTRAINT experiment_recommendation_status_metadata_symmetric_check
            CHECK (
                (
                    (status = 'rejected'
                     AND rejected_at IS NOT NULL
                     AND rejected_reason IS NOT NULL
                     AND btrim(rejected_reason) <> '')
                    OR
                    (status <> 'rejected'
                     AND rejected_at IS NULL
                     AND rejected_reason IS NULL)
                )
                AND (
                    (status = 'expired' AND expired_at IS NOT NULL)
                    OR
                    (status <> 'expired' AND expired_at IS NULL)
                )
                AND (
                    (status = 'approved' AND approved_experiment_id IS NOT NULL)
                    OR
                    (status <> 'approved' AND approved_experiment_id IS NULL)
                )
            ) NOT VALID;
    END IF;
END $$;
