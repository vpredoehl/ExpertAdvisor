-- Phase 4B scale hardening: canonical identity text is authoritative evidence,
-- not a B-tree key. Large complete evidence snapshots can exceed PostgreSQL's
-- B-tree tuple limit. Repository transactions serialize an advisory key
-- derived from the complete canonical text, use bounded hash indexes for
-- candidate lookup, and then compare canonical text exactly.

ALTER TABLE experiment_recommendation_evaluation_run
    DROP CONSTRAINT IF EXISTS
        experiment_recommendation_evaluation_run_identity_uidx;

ALTER TABLE experiment_recommendation_evaluation_result
    DROP CONSTRAINT IF EXISTS
        experiment_recommendation_evaluation_result_identity_uidx;

DO $$
BEGIN
    IF to_regclass(current_schema() ||
                   '.experiment_recommendation_ranking_snapshot') IS NOT NULL THEN
        ALTER TABLE experiment_recommendation_ranking_snapshot
            DROP CONSTRAINT IF EXISTS recommendation_ranking_snapshot_identity_uidx;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid=(current_schema() ||
                         '.experiment_recommendation_evaluation_run')::regclass
          AND conname='experiment_recommendation_evaluation_run_hash_size_check')
    THEN
        ALTER TABLE experiment_recommendation_evaluation_run
            ADD CONSTRAINT experiment_recommendation_evaluation_run_hash_size_check
            CHECK (octet_length(evaluation_run_identity_hash) <= 128);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid=(current_schema() ||
                         '.experiment_recommendation_evaluation_result')::regclass
          AND conname='experiment_recommendation_evaluation_result_hash_size_check')
    THEN
        ALTER TABLE experiment_recommendation_evaluation_result
            ADD CONSTRAINT experiment_recommendation_evaluation_result_hash_size_check
            CHECK (octet_length(evaluation_identity_hash) <= 128);
    END IF;
    IF to_regclass(current_schema() ||
                   '.experiment_recommendation_ranking_snapshot') IS NOT NULL THEN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conrelid=(current_schema() ||
                             '.experiment_recommendation_ranking_snapshot')::regclass
              AND conname='recommendation_ranking_snapshot_hash_size_check')
        THEN
            ALTER TABLE experiment_recommendation_ranking_snapshot
                ADD CONSTRAINT recommendation_ranking_snapshot_hash_size_check
                CHECK (octet_length(ranking_snapshot_identity_hash) <= 128);
        END IF;
    END IF;
END $$;

-- The run and snapshot hash indexes already exist from migrations 034 and
-- 035. Reassert them here for independently restored Phase 4B schemas.
CREATE INDEX IF NOT EXISTS experiment_recommendation_evaluation_run_hash_idx
    ON experiment_recommendation_evaluation_run(
        evaluation_run_identity_hash, recommendation_evaluation_run_id DESC);
CREATE INDEX IF NOT EXISTS experiment_recommendation_evaluation_result_hash_idx
    ON experiment_recommendation_evaluation_result(
        recommendation_evaluation_run_id, evaluation_identity_hash,
        recommendation_evaluation_result_id DESC);
DO $$
BEGIN
    IF to_regclass(current_schema() ||
                   '.experiment_recommendation_ranking_snapshot') IS NOT NULL THEN
        EXECUTE 'CREATE INDEX IF NOT EXISTS recommendation_ranking_snapshot_hash_idx '
            || 'ON experiment_recommendation_ranking_snapshot('
            || 'ranking_snapshot_identity_hash,recommendation_ranking_snapshot_id DESC)';
    END IF;
END $$;

COMMENT ON COLUMN experiment_recommendation_evaluation_run.
    evaluation_run_identity_canonical IS
    'Authoritative complete identity. Hash lookup plus exact canonical comparison is collision-safe.';
COMMENT ON COLUMN experiment_recommendation_evaluation_result.
    evaluation_identity_canonical IS
    'Authoritative complete identity. Hash lookup plus exact canonical comparison is collision-safe.';
DO $$
BEGIN
    IF to_regclass(current_schema() ||
                   '.experiment_recommendation_ranking_snapshot') IS NOT NULL THEN
        EXECUTE 'COMMENT ON COLUMN '
            || 'experiment_recommendation_ranking_snapshot.'
            || 'ranking_snapshot_identity_canonical IS '
            || quote_literal('Authoritative complete identity. Hash lookup plus exact canonical comparison is collision-safe.');
    END IF;
END $$;
