-- Checkpoint-policy lifecycle hardening prerequisite. Profitability is
-- intentionally absent from policy and evidence identity.
ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS checkpoint_policy_revision bigint NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS checkpoint_policy_hash text,
    ADD COLUMN IF NOT EXISTS checkpoint_policy_last_decision_id bigint,
    ADD COLUMN IF NOT EXISTS checkpoint_policy_stop_decision_id bigint;

ALTER TABLE experiment_checkpoint_decision
    ADD COLUMN IF NOT EXISTS analysis_id bigint,
    ADD COLUMN IF NOT EXISTS inference_eval_result_id bigint,
    ADD COLUMN IF NOT EXISTS policy_revision bigint,
    ADD COLUMN IF NOT EXISTS policy_hash text,
    ADD COLUMN IF NOT EXISTS evidence_watermark text,
    ADD COLUMN IF NOT EXISTS rank_population_watermark text,
    ADD COLUMN IF NOT EXISTS identity_status text NOT NULL DEFAULT 'legacy',
    ADD COLUMN IF NOT EXISTS superseded_at timestamptz,
    ADD COLUMN IF NOT EXISTS superseded_reason text,
    ADD COLUMN IF NOT EXISTS superseded_by_decision_id bigint,
    ADD COLUMN IF NOT EXISTS stop_request_applied boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS stop_request_applied_at timestamptz,
    ADD COLUMN IF NOT EXISTS stop_action_worker_attempt_id bigint;

DROP INDEX IF EXISTS experiment_checkpoint_decision_eval_uidx;

CREATE UNIQUE INDEX IF NOT EXISTS
    experiment_checkpoint_decision_semantic_uidx
    ON experiment_checkpoint_decision(
        checkpoint_eval_id, policy_revision, policy_hash, evidence_watermark
    )
    WHERE policy_revision IS NOT NULL
      AND policy_hash IS NOT NULL
      AND evidence_watermark IS NOT NULL;

CREATE INDEX IF NOT EXISTS
    experiment_checkpoint_decision_authority_idx
    ON experiment_checkpoint_decision(
        parent_experiment_id, identity_status, checkpoint_epoch,
        checkpoint_decision_id
    );

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_checkpoint_policy_revision_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_checkpoint_policy_revision_check
            CHECK (checkpoint_policy_revision > 0);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'checkpoint_decision_versioned_identity_check'
    ) THEN
        ALTER TABLE experiment_checkpoint_decision
            ADD CONSTRAINT checkpoint_decision_versioned_identity_check
            CHECK (
                (identity_status = 'legacy'
                 AND policy_revision IS NULL
                 AND policy_hash IS NULL
                 AND evidence_watermark IS NULL)
                OR
                (identity_status IN ('active', 'superseded', 'action_applied')
                 AND policy_revision > 0
                 AND length(policy_hash) = 16
                 AND length(evidence_watermark) = 16
                 AND analysis_id IS NOT NULL
                 AND inference_eval_result_id IS NOT NULL)
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'checkpoint_decision_supersession_shape_check'
    ) THEN
        ALTER TABLE experiment_checkpoint_decision
            ADD CONSTRAINT checkpoint_decision_supersession_shape_check
            CHECK (
                (identity_status = 'superseded'
                 AND superseded_at IS NOT NULL
                 AND superseded_reason IS NOT NULL)
                OR
                (identity_status <> 'superseded'
                 AND superseded_at IS NULL
                 AND superseded_reason IS NULL
                 AND superseded_by_decision_id IS NULL)
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'checkpoint_decision_stop_action_shape_check'
    ) THEN
        ALTER TABLE experiment_checkpoint_decision
            ADD CONSTRAINT checkpoint_decision_stop_action_shape_check
            CHECK (
                (stop_request_applied
                 AND identity_status = 'action_applied'
                 AND decision = 'stop_requested'
                 AND requested_stop_epoch IS NOT NULL
                 AND stop_request_applied_at IS NOT NULL
                 AND stop_action_worker_attempt_id IS NOT NULL)
                OR
                (NOT stop_request_applied
                 AND stop_request_applied_at IS NULL
                 AND stop_action_worker_attempt_id IS NULL)
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'checkpoint_decision_analysis_fkey'
    ) THEN
        ALTER TABLE experiment_checkpoint_decision
            ADD CONSTRAINT checkpoint_decision_analysis_fkey
            FOREIGN KEY (analysis_id)
            REFERENCES experiment_analysis_result(analysis_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'checkpoint_decision_inference_result_fkey'
    ) THEN
        ALTER TABLE experiment_checkpoint_decision
            ADD CONSTRAINT checkpoint_decision_inference_result_fkey
            FOREIGN KEY (inference_eval_result_id)
            REFERENCES inference_eval_result(id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'checkpoint_decision_superseded_by_fkey'
    ) THEN
        ALTER TABLE experiment_checkpoint_decision
            ADD CONSTRAINT checkpoint_decision_superseded_by_fkey
            FOREIGN KEY (superseded_by_decision_id)
            REFERENCES experiment_checkpoint_decision(checkpoint_decision_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'checkpoint_decision_stop_attempt_fkey'
    ) THEN
        ALTER TABLE experiment_checkpoint_decision
            ADD CONSTRAINT checkpoint_decision_stop_attempt_fkey
            FOREIGN KEY (stop_action_worker_attempt_id)
            REFERENCES experiment_scheduler_worker_attempt(worker_attempt_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_checkpoint_policy_last_decision_fkey'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_checkpoint_policy_last_decision_fkey
            FOREIGN KEY (checkpoint_policy_last_decision_id)
            REFERENCES experiment_checkpoint_decision(checkpoint_decision_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_checkpoint_policy_stop_decision_fkey'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_checkpoint_policy_stop_decision_fkey
            FOREIGN KEY (checkpoint_policy_stop_decision_id)
            REFERENCES experiment_checkpoint_decision(checkpoint_decision_id);
    END IF;
END $$;

-- Decision meaning is immutable. Lifecycle-only transitions may mark a row
-- superseded or atomically attest that its stop action was applied.
CREATE OR REPLACE FUNCTION expertadvisor_guard_checkpoint_decision_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.checkpoint_eval_id IS DISTINCT FROM NEW.checkpoint_eval_id
       OR OLD.parent_experiment_id IS DISTINCT FROM NEW.parent_experiment_id
       OR OLD.checkpoint_epoch IS DISTINCT FROM NEW.checkpoint_epoch
       OR OLD.checkpoint_model_id IS DISTINCT FROM NEW.checkpoint_model_id
       OR OLD.analysis_id IS DISTINCT FROM NEW.analysis_id
       OR OLD.inference_eval_result_id IS DISTINCT FROM NEW.inference_eval_result_id
       OR OLD.policy_revision IS DISTINCT FROM NEW.policy_revision
       OR OLD.policy_hash IS DISTINCT FROM NEW.policy_hash
       OR OLD.evidence_watermark IS DISTINCT FROM NEW.evidence_watermark
       OR OLD.decision IS DISTINCT FROM NEW.decision
       OR OLD.reason IS DISTINCT FROM NEW.reason
       OR OLD.leader_score IS DISTINCT FROM NEW.leader_score
       OR OLD.infer_accuracy IS DISTINCT FROM NEW.infer_accuracy
       OR OLD.rank_value IS DISTINCT FROM NEW.rank_value
       OR OLD.rank_scope IS DISTINCT FROM NEW.rank_scope
       OR OLD.rank_population_watermark IS DISTINCT FROM NEW.rank_population_watermark
       OR OLD.requested_stop_epoch IS DISTINCT FROM NEW.requested_stop_epoch
       OR OLD.created_at IS DISTINCT FROM NEW.created_at
    THEN
        RAISE EXCEPTION 'checkpoint policy decision meaning is immutable'
            USING ERRCODE = '55000';
    END IF;
    IF OLD.identity_status = 'active'
       AND NEW.identity_status NOT IN ('active', 'superseded', 'action_applied')
    THEN
        RAISE EXCEPTION 'invalid checkpoint policy decision lifecycle transition'
            USING ERRCODE = '55000';
    END IF;
    IF OLD.identity_status IN ('legacy', 'superseded', 'action_applied')
       AND NEW IS DISTINCT FROM OLD
    THEN
        RAISE EXCEPTION 'terminal checkpoint policy decision is immutable'
            USING ERRCODE = '55000';
    END IF;
    IF OLD.stop_request_applied AND NOT NEW.stop_request_applied THEN
        RAISE EXCEPTION 'checkpoint policy stop attribution is terminal'
            USING ERRCODE = '55000';
    END IF;
    IF OLD.identity_status = 'action_applied'
       AND NEW.identity_status <> 'action_applied'
    THEN
        RAISE EXCEPTION 'applied checkpoint policy stop is terminal'
            USING ERRCODE = '55000';
    END IF;
    RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS checkpoint_decision_immutable_guard
    ON experiment_checkpoint_decision;
CREATE TRIGGER checkpoint_decision_immutable_guard
BEFORE UPDATE ON experiment_checkpoint_decision
FOR EACH ROW
EXECUTE FUNCTION expertadvisor_guard_checkpoint_decision_immutable();

COMMENT ON COLUMN experiment.checkpoint_policy_revision IS
    'Monotonic semantic revision; legacy experiments begin at revision 1.';
COMMENT ON COLUMN experiment.checkpoint_policy_hash IS
    'FNV-1a hash of canonical decision-bearing checkpoint-policy configuration; NULL means legacy/unmaterialized identity.';
COMMENT ON COLUMN experiment.checkpoint_policy_stop_decision_id IS
    'Terminal attribution for the exact durable decision that set stop_after_checkpoint_epoch; manual stop controls are separate.';
COMMENT ON COLUMN experiment_checkpoint_decision.identity_status IS
    'legacy rows are unversioned; active rows are observationally authoritative; superseded rows cannot act; action_applied rows are terminal stop attributions.';
COMMENT ON COLUMN experiment_checkpoint_decision.evidence_watermark IS
    'FNV-1a hash of exact checkpoint, analysis, checkpoint-inference, metric, grace-count, rank, and population identity; profitability is excluded.';

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
GRANT SELECT, INSERT, UPDATE, DELETE ON experiment_checkpoint_decision TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE
    experiment_checkpoint_decision_checkpoint_decision_id_seq TO pqxx;
