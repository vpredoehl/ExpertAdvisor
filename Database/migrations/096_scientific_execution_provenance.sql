-- Durable scientific execution provenance.  Creation RunProvenance remains on
-- experiment; this migration records the validated worker artifact selected at
-- reservation and binds persisted scientific output to that exact attempt.
BEGIN;

ALTER TABLE experiment_scheduler_worker_attempt
    ADD COLUMN IF NOT EXISTS semantic_layout_version integer,
    ADD COLUMN IF NOT EXISTS model_input_width integer,
    ADD COLUMN IF NOT EXISTS semantic_worker_role text,
    ADD COLUMN IF NOT EXISTS source_commit text,
    ADD COLUMN IF NOT EXISTS executable_sha256 text,
    ADD COLUMN IF NOT EXISTS runtime_identity text,
    ADD COLUMN IF NOT EXISTS canonical_manifest_path text;

ALTER TABLE model
    ADD COLUMN IF NOT EXISTS producer_worker_attempt_id bigint
        REFERENCES experiment_scheduler_worker_attempt(worker_attempt_id)
        ON DELETE RESTRICT;
ALTER TABLE inference_eval_result
    ADD COLUMN IF NOT EXISTS producer_worker_attempt_id bigint
        REFERENCES experiment_scheduler_worker_attempt(worker_attempt_id)
        ON DELETE RESTRICT;

CREATE INDEX IF NOT EXISTS model_producer_worker_attempt_idx
    ON model(producer_worker_attempt_id)
    WHERE producer_worker_attempt_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS inference_eval_result_producer_worker_attempt_idx
    ON inference_eval_result(producer_worker_attempt_id)
    WHERE producer_worker_attempt_id IS NOT NULL;

CREATE OR REPLACE FUNCTION expertadvisor_validate_scientific_producer_attempt()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE a experiment_scheduler_worker_attempt%ROWTYPE;
BEGIN
  IF NEW.producer_worker_attempt_id IS NULL THEN RETURN NEW; END IF;
  SELECT * INTO a FROM experiment_scheduler_worker_attempt
   WHERE worker_attempt_id=NEW.producer_worker_attempt_id;
  IF NOT FOUND OR a.experiment_id IS DISTINCT FROM NEW.experiment_id
     OR a.worker_kind <> 'experiment' OR a.lifecycle_phase <> 'train'
     OR a.capacity_class <> 'train' THEN
    RAISE EXCEPTION 'invalid model producer worker attempt';
  END IF;
  RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS model_scientific_producer_attempt_trigger ON model;
CREATE TRIGGER model_scientific_producer_attempt_trigger
BEFORE INSERT OR UPDATE OF producer_worker_attempt_id,experiment_id ON model
FOR EACH ROW EXECUTE FUNCTION expertadvisor_validate_scientific_producer_attempt();

CREATE OR REPLACE FUNCTION expertadvisor_validate_inference_producer_attempt()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE a experiment_scheduler_worker_attempt%ROWTYPE;
        producer_exists boolean;
        owner_experiment bigint;
        owner_exists boolean;
BEGIN
  IF NEW.producer_worker_attempt_id IS NULL THEN RETURN NEW; END IF;
  SELECT * INTO a FROM experiment_scheduler_worker_attempt
   WHERE worker_attempt_id=NEW.producer_worker_attempt_id;
  producer_exists := FOUND;
  SELECT experiment_id INTO owner_experiment FROM model WHERE model_id=NEW.model_id;
  owner_exists := FOUND;
  IF NOT producer_exists OR NOT owner_exists OR owner_experiment IS NULL
     OR a.experiment_id IS DISTINCT FROM owner_experiment
     OR a.lifecycle_phase <> 'infer' OR a.capacity_class <> 'infer'
     OR (NEW.inference_scope = 'final' AND
         (a.worker_kind <> 'experiment' OR a.checkpoint_eval_id IS NOT NULL))
     OR (NEW.inference_scope = 'checkpoint' AND
         (a.worker_kind <> 'checkpoint_infer' OR NEW.checkpoint_eval_id IS NULL
          OR a.checkpoint_eval_id IS DISTINCT FROM NEW.checkpoint_eval_id))
     OR NEW.inference_scope NOT IN ('final','checkpoint') THEN
    RAISE EXCEPTION 'invalid inference producer worker attempt';
  END IF;
  RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS inference_scientific_producer_attempt_trigger ON inference_eval_result;
CREATE TRIGGER inference_scientific_producer_attempt_trigger
BEFORE INSERT OR UPDATE OF producer_worker_attempt_id,model_id,inference_scope,checkpoint_eval_id ON inference_eval_result
FOR EACH ROW EXECUTE FUNCTION expertadvisor_validate_inference_producer_attempt();

COMMIT;
