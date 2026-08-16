-- A model's experiment linkage is its sole feature-ablation provenance.
-- Historical rows may receive their first linkage through the established
-- scheduler handoff, but a persisted linkage can never be removed or
-- redirected thereafter.
CREATE OR REPLACE FUNCTION enforce_model_experiment_lineage_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.experiment_id IS NOT NULL AND
       NEW.experiment_id IS DISTINCT FROM OLD.experiment_id THEN
        RAISE EXCEPTION
            'model experiment_id is immutable once linked';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS model_experiment_lineage_immutable_trigger ON model;
CREATE TRIGGER model_experiment_lineage_immutable_trigger
BEFORE UPDATE OF experiment_id ON model
FOR EACH ROW EXECUTE FUNCTION enforce_model_experiment_lineage_immutable();

GRANT EXECUTE ON FUNCTION enforce_model_experiment_lineage_immutable()
    TO pqxx;
