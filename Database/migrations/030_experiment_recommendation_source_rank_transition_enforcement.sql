-- Phase 4A Step 4 hardening: migration 029 did not reject an UPDATE that
-- associated a legacy NULL/NULL recommendation with a scan unless source_rank
-- was also named in the UPDATE. Validate the resulting row whenever either
-- source_rank or recommendation_scan_id changes, without rewriting legacy rows.

CREATE OR REPLACE FUNCTION enforce_experiment_recommendation_source_rank()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.recommendation_scan_id IS NOT NULL AND
       (NEW.source_rank IS NULL OR NEW.source_rank <= 0) THEN
        RAISE EXCEPTION 'experiment_recommendation_source_rank_invalid'
            USING ERRCODE = '23514';
    END IF;

    IF TG_OP = 'UPDATE' AND
       OLD.source_rank > 0 AND
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
