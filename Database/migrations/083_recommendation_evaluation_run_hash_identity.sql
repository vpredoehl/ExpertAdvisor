-- evaluation_run_identity_canonical can be hundreds of kilobytes and cannot
-- be safely indexed by PostgreSQL B-tree. Use the persisted tagged identity
-- hash for uniqueness, while application retry validation continues to compare
-- the full canonical identity and fail closed on any hash collision.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM experiment_recommendation_evaluation_run
        GROUP BY evaluation_run_identity_hash
        HAVING COUNT(DISTINCT evaluation_run_identity_canonical) > 1
    ) THEN
        RAISE EXCEPTION
            'evaluation run identity hash collision prevents migration';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid =
              'experiment_recommendation_evaluation_run'::regclass
          AND conname =
              'experiment_recommendation_evaluation_run_identity_uidx'
    ) THEN
        ALTER TABLE experiment_recommendation_evaluation_run
        DROP CONSTRAINT
            experiment_recommendation_evaluation_run_identity_uidx;
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS
    experiment_recommendation_evaluation_run_identity_hash_uidx
ON experiment_recommendation_evaluation_run(
    evaluation_run_identity_hash
);
