-- Phase 4A Step 4 hardening: recommendation_scan_id is nullable only to
-- preserve unmapped legacy rows upgraded from migration 025. A persisted
-- Step 3 recommendation must retain its originating scan provenance.

CREATE OR REPLACE FUNCTION enforce_experiment_recommendation_source_rank()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        IF OLD.recommendation_scan_id IS NOT NULL AND
           NEW.recommendation_scan_id IS NULL THEN
            RAISE EXCEPTION 'experiment_recommendation_scan_detachment_invalid'
                USING ERRCODE = '23514';
        END IF;

        IF OLD.source_rank > 0 AND
           (NEW.source_rank IS NULL OR NEW.source_rank <= 0) THEN
            RAISE EXCEPTION 'experiment_recommendation_source_rank_invalid'
                USING ERRCODE = '23514';
        END IF;
    END IF;

    IF NEW.recommendation_scan_id IS NOT NULL AND
       (NEW.source_rank IS NULL OR NEW.source_rank <= 0) THEN
        RAISE EXCEPTION 'experiment_recommendation_source_rank_invalid'
            USING ERRCODE = '23514';
    END IF;

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS experiment_recommendation_source_rank_enforce_trigger
    ON experiment_recommendation;

CREATE TRIGGER experiment_recommendation_source_rank_enforce_trigger
BEFORE INSERT OR UPDATE OF source_rank, recommendation_scan_id
ON experiment_recommendation
FOR EACH ROW
EXECUTE FUNCTION enforce_experiment_recommendation_source_rank();

GRANT EXECUTE ON FUNCTION enforce_experiment_recommendation_source_rank() TO pqxx;
