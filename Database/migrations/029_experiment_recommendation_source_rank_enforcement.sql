-- Phase 4A Step 4 hardening: source_rank existed only in Step 3 generation
-- memory before migration 028.  Existing recommendation rows may therefore
-- legitimately contain NULL and must not be assigned a guessed rank.
--
-- A CHECK constraint cannot distinguish those legacy NULLs from a new NULL
-- (and NULL satisfies a CHECK expression).  This trigger enforces the
-- positive-rank contract at the write boundary for every new recommendation.
-- It also prevents a persisted positive rank from being cleared or made
-- nonpositive, while allowing unrelated updates to a legacy NULL-rank row.

CREATE OR REPLACE FUNCTION enforce_experiment_recommendation_source_rank()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.recommendation_scan_id IS NOT NULL AND
           (NEW.source_rank IS NULL OR NEW.source_rank <= 0) THEN
            RAISE EXCEPTION 'experiment_recommendation_source_rank_required'
                USING ERRCODE = '23514';
        END IF;
    ELSIF NEW.recommendation_scan_id IS NOT NULL AND
          NEW.source_rank IS DISTINCT FROM OLD.source_rank AND
          (NEW.source_rank IS NULL OR NEW.source_rank <= 0) THEN
        RAISE EXCEPTION 'experiment_recommendation_source_rank_invalid'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_trigger
        WHERE tgname = 'experiment_recommendation_source_rank_enforce_trigger'
          AND tgrelid = 'experiment_recommendation'::regclass
          AND NOT tgisinternal
    ) THEN
        CREATE TRIGGER experiment_recommendation_source_rank_enforce_trigger
        BEFORE INSERT OR UPDATE OF source_rank, recommendation_scan_id
        ON experiment_recommendation
        FOR EACH ROW
        EXECUTE FUNCTION enforce_experiment_recommendation_source_rank();
    END IF;
END $$;

GRANT EXECUTE ON FUNCTION enforce_experiment_recommendation_source_rank() TO pqxx;
