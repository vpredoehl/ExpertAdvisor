-- ranking_snapshot_identity_canonical can be very large and is not suitable
-- as the PostgreSQL B-tree uniqueness key. Use the persisted tagged identity
-- hash for uniqueness and retry lookup. Application retry validation continues
-- to compare the complete canonical identity and therefore fails closed on any
-- hash collision or semantic mismatch.

DO $$
BEGIN
    -- Refuse migration if one hash already identifies multiple canonical
    -- ranking snapshot identities.
    IF EXISTS (
        SELECT 1
        FROM experiment_recommendation_ranking_snapshot
        GROUP BY ranking_snapshot_identity_hash
        HAVING COUNT(DISTINCT ranking_snapshot_identity_canonical) > 1
    ) THEN
        RAISE EXCEPTION
            'ranking snapshot identity hash collision prevents migration';
    END IF;

    -- Migration 035 originally defined canonical-text uniqueness. Some
    -- production databases no longer retain that constraint, so removal is
    -- intentionally conditional.
    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid =
              'experiment_recommendation_ranking_snapshot'::regclass
          AND conname =
              'recommendation_ranking_snapshot_identity_uidx'
    ) THEN
        ALTER TABLE experiment_recommendation_ranking_snapshot
        DROP CONSTRAINT recommendation_ranking_snapshot_identity_uidx;
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS
    recommendation_ranking_snapshot_identity_hash_uidx
ON experiment_recommendation_ranking_snapshot(
    ranking_snapshot_identity_hash
);
