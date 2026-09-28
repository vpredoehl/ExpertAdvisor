-- Durable declared controlled-experiment family foundation.  This migration
-- records immutable scientific declarations only; it does not plan, approve,
-- materialize, authorize, queue, or execute experiments.
BEGIN;

CREATE TABLE controlled_experiment_family (
    controlled_experiment_family_id bigserial PRIMARY KEY,
    family_key text COLLATE "C" NOT NULL UNIQUE CHECK (btrim(family_key) <> ''),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    created_by text NOT NULL CHECK (btrim(created_by) <> ''),
    retired_at timestamptz,
    retired_reason text,
    CONSTRAINT controlled_experiment_family_retirement_shape_check CHECK (
        (retired_at IS NULL AND retired_reason IS NULL) OR
        (retired_at IS NOT NULL AND btrim(retired_reason) <> ''))
);

CREATE TABLE controlled_experiment_family_version (
    controlled_experiment_family_version_id bigserial PRIMARY KEY,
    controlled_experiment_family_id bigint NOT NULL REFERENCES controlled_experiment_family(controlled_experiment_family_id) ON DELETE RESTRICT,
    version_ordinal integer NOT NULL CHECK (version_ordinal > 0),
    supersedes_family_version_id bigint REFERENCES controlled_experiment_family_version(controlled_experiment_family_version_id) ON DELETE RESTRICT,
    specification_contract_version integer NOT NULL CHECK (specification_contract_version > 0),
    planning_contract_version integer NOT NULL CHECK (planning_contract_version > 0),
    specification_identity_canonical text COLLATE "C" NOT NULL CHECK (btrim(specification_identity_canonical) <> ''),
    specification_identity_hash text NOT NULL CHECK (specification_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    specification_hash_collision_ordinal integer NOT NULL DEFAULT 0 CHECK (specification_hash_collision_ordinal >= 0),
    plan_identity_canonical text COLLATE "C" NOT NULL CHECK (btrim(plan_identity_canonical) <> ''),
    plan_identity_hash text NOT NULL CHECK (plan_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    plan_hash_collision_ordinal integer NOT NULL DEFAULT 0 CHECK (plan_hash_collision_ordinal >= 0),
    expected_cell_count integer NOT NULL CHECK (expected_cell_count > 0),
    expected_member_count integer NOT NULL CHECK (expected_member_count > 0),
    expected_arm_count integer NOT NULL CHECK (expected_arm_count > 0),
    train_start timestamptz NOT NULL,
    train_end timestamptz NOT NULL,
    infer_start timestamptz NOT NULL,
    infer_end timestamptz NOT NULL,
    prediction_threshold numeric NOT NULL,
    target_epochs integer NOT NULL CHECK (target_epochs > 0),
    resume_model_id bigint,
    resume_expand_input_width boolean NOT NULL DEFAULT false,
    model_input_width integer NOT NULL CHECK (model_input_width > 4),
    model_input_semantic_layout_version integer NOT NULL CHECK (model_input_semantic_layout_version > 0),
    feature_warmup_scope text NOT NULL CHECK (feature_warmup_scope IN ('legacy_cold_boundary','full_history_warmup')),
    donchian20_mode text NOT NULL CHECK (donchian20_mode IN ('enabled','zero_ablation')),
    donchian_lookback integer NOT NULL CHECK (donchian_lookback > 0),
    training_contract_version integer NOT NULL CHECK (training_contract_version > 0),
    training_contract_canonical text COLLATE "C" NOT NULL CHECK (btrim(training_contract_canonical) <> ''),
    training_contract_hash text NOT NULL CHECK (training_contract_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    target_generation_contract_canonical text COLLATE "C" NOT NULL CHECK (btrim(target_generation_contract_canonical) <> ''),
    target_generation_contract_hash text NOT NULL CHECK (target_generation_contract_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    training_objective_id text NOT NULL CHECK (btrim(training_objective_id) <> ''),
    training_objective_canonical text COLLATE "C" NOT NULL CHECK (btrim(training_objective_canonical) <> ''),
    training_objective_hash text NOT NULL CHECK (training_objective_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    checkpoint_contract_canonical text COLLATE "C" NOT NULL CHECK (btrim(checkpoint_contract_canonical) <> ''),
    checkpoint_contract_hash text NOT NULL CHECK (checkpoint_contract_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    continuation_mode text NOT NULL CHECK (continuation_mode IN ('prohibited','declared_policy')),
    continuation_policy_canonical text COLLATE "C",
    continuation_policy_hash text,
    economic_calendar_snapshot_id bigint NOT NULL REFERENCES economic_calendar_snapshot(economic_calendar_snapshot_id) ON DELETE RESTRICT,
    economic_calendar_snapshot_hash text NOT NULL CHECK (economic_calendar_snapshot_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    final_evidence_contract_canonical text COLLATE "C" NOT NULL CHECK (btrim(final_evidence_contract_canonical) <> ''),
    final_evidence_contract_hash text NOT NULL CHECK (final_evidence_contract_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    initial_scheduler_priority text NOT NULL CHECK (initial_scheduler_priority IN ('high','normal','low')),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    created_by text NOT NULL CHECK (btrim(created_by) <> ''),
    creation_reason text NOT NULL CHECK (btrim(creation_reason) <> ''),
    rendered_specification_snapshot jsonb NOT NULL CHECK (jsonb_typeof(rendered_specification_snapshot) = 'object'),
    CONSTRAINT controlled_experiment_family_version_counts_check CHECK (expected_member_count = expected_cell_count * expected_arm_count),
    CONSTRAINT controlled_experiment_family_version_ranges_check CHECK (train_end > train_start AND infer_end > infer_start),
    CONSTRAINT controlled_experiment_family_version_fresh_start_check CHECK (resume_model_id IS NULL AND NOT resume_expand_input_width),
    CONSTRAINT controlled_experiment_family_version_continuation_shape_check CHECK (
        (continuation_mode = 'prohibited' AND continuation_policy_canonical IS NULL AND continuation_policy_hash IS NULL) OR
        (continuation_mode = 'declared_policy' AND btrim(continuation_policy_canonical) <> '' AND continuation_policy_hash ~ '^fnv1a64:[0-9a-f]{16}$')),
    CONSTRAINT controlled_experiment_family_version_ordinal_uidx UNIQUE (controlled_experiment_family_id, version_ordinal),
    CONSTRAINT controlled_experiment_family_version_spec_hash_ordinal_uidx UNIQUE (specification_identity_hash, specification_hash_collision_ordinal),
    CONSTRAINT controlled_experiment_family_version_plan_hash_ordinal_uidx UNIQUE (plan_identity_hash, plan_hash_collision_ordinal)
);
CREATE UNIQUE INDEX controlled_experiment_family_version_spec_canonical_uidx ON controlled_experiment_family_version(specification_identity_canonical);
CREATE UNIQUE INDEX controlled_experiment_family_version_plan_canonical_uidx ON controlled_experiment_family_version(plan_identity_canonical);

CREATE TABLE controlled_experiment_family_arm (
    controlled_experiment_family_arm_id bigserial PRIMARY KEY,
    controlled_experiment_family_version_id bigint NOT NULL REFERENCES controlled_experiment_family_version(controlled_experiment_family_version_id) ON DELETE RESTRICT,
    arm_ordinal integer NOT NULL CHECK (arm_ordinal > 0),
    arm_key text COLLATE "C" NOT NULL CHECK (btrim(arm_key) <> ''),
    display_role text NOT NULL CHECK (btrim(display_role) <> ''),
    baseline_arm_key text COLLATE "C",
    comparison_direction text,
    declared_difference_canonical text COLLATE "C" NOT NULL CHECK (btrim(declared_difference_canonical) <> ''),
    declared_difference_hash text NOT NULL CHECK (declared_difference_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    feature_ablation_mask text NOT NULL DEFAULT '',
    feature_ablation_mask_hash text NOT NULL CHECK (feature_ablation_mask_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    arm_override_canonical text COLLATE "C" NOT NULL DEFAULT '',
    arm_override_hash text NOT NULL CHECK (arm_override_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    UNIQUE (controlled_experiment_family_version_id, arm_ordinal),
    UNIQUE (controlled_experiment_family_version_id, arm_key)
);

CREATE TABLE controlled_experiment_family_cell (
    controlled_experiment_family_cell_id bigserial PRIMARY KEY,
    controlled_experiment_family_version_id bigint NOT NULL REFERENCES controlled_experiment_family_version(controlled_experiment_family_version_id) ON DELETE RESTRICT,
    cell_ordinal integer NOT NULL CHECK (cell_ordinal > 0),
    symbol text COLLATE "C" NOT NULL CHECK (symbol = lower(symbol) AND btrim(symbol) <> ''),
    prediction_horizon integer NOT NULL CHECK (prediction_horizon > 0),
    fresh_initialization_seed bigint NOT NULL CHECK (fresh_initialization_seed > 0 AND fresh_initialization_seed <= 4294967295),
    cell_identity_canonical text COLLATE "C" NOT NULL CHECK (btrim(cell_identity_canonical) <> ''),
    cell_identity_hash text NOT NULL CHECK (cell_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    UNIQUE (controlled_experiment_family_version_id, cell_ordinal),
    UNIQUE (controlled_experiment_family_version_id, symbol, prediction_horizon, fresh_initialization_seed),
    UNIQUE (controlled_experiment_family_version_id, cell_identity_hash)
);

CREATE TABLE controlled_experiment_family_member (
    controlled_experiment_family_member_id bigserial PRIMARY KEY,
    controlled_experiment_family_version_id bigint NOT NULL REFERENCES controlled_experiment_family_version(controlled_experiment_family_version_id) ON DELETE RESTRICT,
    controlled_experiment_family_cell_id bigint NOT NULL REFERENCES controlled_experiment_family_cell(controlled_experiment_family_cell_id) ON DELETE RESTRICT,
    controlled_experiment_family_arm_id bigint NOT NULL REFERENCES controlled_experiment_family_arm(controlled_experiment_family_arm_id) ON DELETE RESTRICT,
    member_ordinal integer NOT NULL CHECK (member_ordinal > 0),
    planned_experiment_identity_canonical text COLLATE "C" NOT NULL CHECK (btrim(planned_experiment_identity_canonical) <> ''),
    planned_experiment_identity_hash text NOT NULL CHECK (planned_experiment_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    planned_experiment_hash_collision_ordinal integer NOT NULL DEFAULT 0 CHECK (planned_experiment_hash_collision_ordinal >= 0),
    experiment_id bigint UNIQUE REFERENCES experiment(experiment_id) ON DELETE RESTRICT,
    UNIQUE (controlled_experiment_family_cell_id, controlled_experiment_family_arm_id),
    UNIQUE (controlled_experiment_family_version_id, member_ordinal),
    UNIQUE (controlled_experiment_family_version_id, planned_experiment_identity_hash, planned_experiment_hash_collision_ordinal),
    UNIQUE (controlled_experiment_family_version_id, planned_experiment_identity_canonical)
);

CREATE TABLE controlled_experiment_family_execution_requirement (
    controlled_experiment_family_execution_requirement_id bigserial PRIMARY KEY,
    controlled_experiment_family_version_id bigint NOT NULL REFERENCES controlled_experiment_family_version(controlled_experiment_family_version_id) ON DELETE RESTRICT,
    lifecycle_phase text NOT NULL CHECK (lifecycle_phase IN ('train','infer','analyze')),
    semantic_layout_version integer NOT NULL CHECK (semantic_layout_version > 0),
    model_input_width integer NOT NULL CHECK (model_input_width > 4),
    semantic_worker_role text NOT NULL CHECK (semantic_worker_role IN ('train','infer','analyze')),
    required_capabilities_canonical text COLLATE "C" NOT NULL,
    required_capabilities_hash text NOT NULL CHECK (required_capabilities_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    source_commit text NOT NULL CHECK (source_commit ~ '^[0-9a-f]{40}$'),
    executable_sha256 text NOT NULL CHECK (executable_sha256 ~ '^[0-9a-f]{64}$'),
    runtime_identity text NOT NULL CHECK (runtime_identity ~ '^[0-9a-f]{64}$'),
    canonical_manifest_path text,
    manifest_sha256 text CHECK (manifest_sha256 IS NULL OR manifest_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT cef_execution_requirement_manifest_shape_check CHECK (canonical_manifest_path IS NOT NULL OR manifest_sha256 IS NULL),
    UNIQUE (controlled_experiment_family_version_id, lifecycle_phase)
);

CREATE TABLE controlled_experiment_family_review_event (
    controlled_experiment_family_review_event_id bigserial PRIMARY KEY,
    controlled_experiment_family_version_id bigint NOT NULL REFERENCES controlled_experiment_family_version(controlled_experiment_family_version_id) ON DELETE RESTRICT,
    review_contract_version integer NOT NULL CHECK (review_contract_version > 0),
    specification_identity_canonical text COLLATE "C" NOT NULL,
    specification_identity_hash text NOT NULL CHECK (specification_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    plan_identity_canonical text COLLATE "C" NOT NULL,
    plan_identity_hash text NOT NULL CHECK (plan_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    review_result text NOT NULL CHECK (review_result IN ('reviewed','rejected')),
    reviewer_identity text NOT NULL CHECK (btrim(reviewer_identity) <> ''),
    reason_text text NOT NULL CHECK (btrim(reason_text) <> ''),
    review_identity_canonical text COLLATE "C" NOT NULL,
    review_identity_hash text NOT NULL CHECK (review_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (review_identity_canonical)
);

CREATE TABLE controlled_experiment_family_approval (
    controlled_experiment_family_approval_id bigserial PRIMARY KEY,
    controlled_experiment_family_version_id bigint NOT NULL REFERENCES controlled_experiment_family_version(controlled_experiment_family_version_id) ON DELETE RESTRICT,
    controlled_experiment_family_review_event_id bigint NOT NULL REFERENCES controlled_experiment_family_review_event(controlled_experiment_family_review_event_id) ON DELETE RESTRICT,
    approval_contract_version integer NOT NULL CHECK (approval_contract_version > 0),
    decision text NOT NULL CHECK (decision IN ('approved','rejected')),
    approver_identity text NOT NULL CHECK (btrim(approver_identity) <> ''),
    reason_text text NOT NULL CHECK (btrim(reason_text) <> ''),
    approval_identity_canonical text COLLATE "C" NOT NULL,
    approval_identity_hash text NOT NULL CHECK (approval_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    approval_hash_collision_ordinal integer NOT NULL DEFAULT 0 CHECK (approval_hash_collision_ordinal >= 0),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (approval_identity_canonical),
    UNIQUE (approval_identity_hash, approval_hash_collision_ordinal)
);

CREATE TABLE controlled_experiment_family_materialization (
    controlled_experiment_family_materialization_id bigserial PRIMARY KEY,
    controlled_experiment_family_version_id bigint NOT NULL UNIQUE REFERENCES controlled_experiment_family_version(controlled_experiment_family_version_id) ON DELETE RESTRICT,
    controlled_experiment_family_approval_id bigint NOT NULL REFERENCES controlled_experiment_family_approval(controlled_experiment_family_approval_id) ON DELETE RESTRICT,
    materialization_contract_version integer NOT NULL CHECK (materialization_contract_version > 0),
    materialization_identity_canonical text COLLATE "C" NOT NULL UNIQUE,
    materialization_identity_hash text NOT NULL CHECK (materialization_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    expected_member_count integer NOT NULL CHECK (expected_member_count > 0),
    materialized_by text NOT NULL CHECK (btrim(materialized_by) <> ''),
    materialization_reason text NOT NULL CHECK (btrim(materialization_reason) <> ''),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE controlled_experiment_family_materialization_member (
    controlled_experiment_family_materialization_member_id bigserial PRIMARY KEY,
    controlled_experiment_family_materialization_id bigint NOT NULL REFERENCES controlled_experiment_family_materialization(controlled_experiment_family_materialization_id) ON DELETE RESTRICT,
    controlled_experiment_family_member_id bigint NOT NULL UNIQUE REFERENCES controlled_experiment_family_member(controlled_experiment_family_member_id) ON DELETE RESTRICT,
    experiment_id bigint NOT NULL UNIQUE REFERENCES experiment(experiment_id) ON DELETE RESTRICT,
    member_ordinal integer NOT NULL CHECK (member_ordinal > 0),
    UNIQUE (controlled_experiment_family_materialization_id, member_ordinal)
);

CREATE TABLE controlled_experiment_family_execution_authorization (
    controlled_experiment_family_execution_authorization_id bigserial PRIMARY KEY,
    controlled_experiment_family_version_id bigint NOT NULL REFERENCES controlled_experiment_family_version(controlled_experiment_family_version_id) ON DELETE RESTRICT,
    controlled_experiment_family_materialization_id bigint NOT NULL REFERENCES controlled_experiment_family_materialization(controlled_experiment_family_materialization_id) ON DELETE RESTRICT,
    authorization_scope text NOT NULL CHECK (authorization_scope IN ('family','cell_set','member_set')),
    member_set_identity_canonical text COLLATE "C" NOT NULL,
    member_set_identity_hash text NOT NULL CHECK (member_set_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    authorizer_identity text NOT NULL CHECK (btrim(authorizer_identity) <> ''),
    reason_text text NOT NULL CHECK (btrim(reason_text) <> ''),
    grant_identity_canonical text COLLATE "C" NOT NULL UNIQUE,
    grant_identity_hash text NOT NULL CHECK (grant_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    superseded_by_authorization_id bigint UNIQUE REFERENCES controlled_experiment_family_execution_authorization(controlled_experiment_family_execution_authorization_id) ON DELETE RESTRICT,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE controlled_experiment_family_execution_authorization_member (
    controlled_experiment_family_execution_authorization_id bigint NOT NULL REFERENCES controlled_experiment_family_execution_authorization(controlled_experiment_family_execution_authorization_id) ON DELETE RESTRICT,
    controlled_experiment_family_member_id bigint NOT NULL REFERENCES controlled_experiment_family_member(controlled_experiment_family_member_id) ON DELETE RESTRICT,
    PRIMARY KEY (controlled_experiment_family_execution_authorization_id, controlled_experiment_family_member_id)
);

ALTER TABLE experiment ADD COLUMN IF NOT EXISTS controlled_experiment_family_member_id bigint;
ALTER TABLE experiment ADD CONSTRAINT experiment_controlled_experiment_family_member_fkey FOREIGN KEY (controlled_experiment_family_member_id) REFERENCES controlled_experiment_family_member(controlled_experiment_family_member_id) ON DELETE RESTRICT;
CREATE UNIQUE INDEX experiment_controlled_experiment_family_member_uidx ON experiment(controlled_experiment_family_member_id) WHERE controlled_experiment_family_member_id IS NOT NULL;

CREATE OR REPLACE FUNCTION controlled_experiment_family_immutable_row()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'controlled experiment family provenance is immutable';
END $$;
CREATE TRIGGER cef_version_immutable_trg BEFORE UPDATE OR DELETE ON controlled_experiment_family_version FOR EACH ROW EXECUTE FUNCTION controlled_experiment_family_immutable_row();
CREATE TRIGGER cef_arm_immutable_trg BEFORE UPDATE OR DELETE ON controlled_experiment_family_arm FOR EACH ROW EXECUTE FUNCTION controlled_experiment_family_immutable_row();
CREATE TRIGGER cef_cell_immutable_trg BEFORE UPDATE OR DELETE ON controlled_experiment_family_cell FOR EACH ROW EXECUTE FUNCTION controlled_experiment_family_immutable_row();
CREATE TRIGGER cef_member_immutable_trg BEFORE UPDATE OR DELETE ON controlled_experiment_family_member FOR EACH ROW EXECUTE FUNCTION controlled_experiment_family_immutable_row();
CREATE TRIGGER cef_requirement_immutable_trg BEFORE UPDATE OR DELETE ON controlled_experiment_family_execution_requirement FOR EACH ROW EXECUTE FUNCTION controlled_experiment_family_immutable_row();
CREATE TRIGGER cef_review_immutable_trg BEFORE UPDATE OR DELETE ON controlled_experiment_family_review_event FOR EACH ROW EXECUTE FUNCTION controlled_experiment_family_immutable_row();
CREATE TRIGGER cef_approval_immutable_trg BEFORE UPDATE OR DELETE ON controlled_experiment_family_approval FOR EACH ROW EXECUTE FUNCTION controlled_experiment_family_immutable_row();
CREATE TRIGGER cef_materialization_immutable_trg BEFORE UPDATE OR DELETE ON controlled_experiment_family_materialization FOR EACH ROW EXECUTE FUNCTION controlled_experiment_family_immutable_row();
CREATE TRIGGER cef_materialization_member_immutable_trg BEFORE UPDATE OR DELETE ON controlled_experiment_family_materialization_member FOR EACH ROW EXECUTE FUNCTION controlled_experiment_family_immutable_row();
CREATE TRIGGER cef_authorization_immutable_trg BEFORE UPDATE OR DELETE ON controlled_experiment_family_execution_authorization FOR EACH ROW EXECUTE FUNCTION controlled_experiment_family_immutable_row();
CREATE TRIGGER cef_authorization_member_immutable_trg BEFORE UPDATE OR DELETE ON controlled_experiment_family_execution_authorization_member FOR EACH ROW EXECUTE FUNCTION controlled_experiment_family_immutable_row();

CREATE OR REPLACE FUNCTION controlled_experiment_family_member_version_match()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE cell_version bigint; arm_version bigint;
BEGIN
    SELECT controlled_experiment_family_version_id INTO cell_version FROM controlled_experiment_family_cell WHERE controlled_experiment_family_cell_id=NEW.controlled_experiment_family_cell_id;
    SELECT controlled_experiment_family_version_id INTO arm_version FROM controlled_experiment_family_arm WHERE controlled_experiment_family_arm_id=NEW.controlled_experiment_family_arm_id;
    IF cell_version IS DISTINCT FROM NEW.controlled_experiment_family_version_id OR arm_version IS DISTINCT FROM NEW.controlled_experiment_family_version_id THEN
        RAISE EXCEPTION 'controlled experiment family member version mismatch';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER controlled_experiment_family_member_version_match_trigger BEFORE INSERT ON controlled_experiment_family_member FOR EACH ROW EXECUTE FUNCTION controlled_experiment_family_member_version_match();

CREATE OR REPLACE FUNCTION controlled_experiment_family_version_graph_complete()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    expected_cells integer;
    expected_members integer;
    expected_arms integer;
    actual_cells integer;
    actual_members integer;
    actual_arms integer;
BEGIN
    SELECT expected_cell_count, expected_member_count, expected_arm_count
      INTO expected_cells, expected_members, expected_arms
      FROM controlled_experiment_family_version
     WHERE controlled_experiment_family_version_id = NEW.controlled_experiment_family_version_id;

    SELECT count(*) INTO actual_cells FROM controlled_experiment_family_cell
     WHERE controlled_experiment_family_version_id = NEW.controlled_experiment_family_version_id;
    SELECT count(*) INTO actual_members FROM controlled_experiment_family_member
     WHERE controlled_experiment_family_version_id = NEW.controlled_experiment_family_version_id;
    SELECT count(*) INTO actual_arms FROM controlled_experiment_family_arm
     WHERE controlled_experiment_family_version_id = NEW.controlled_experiment_family_version_id;

    IF actual_cells <> expected_cells OR actual_members <> expected_members OR actual_arms <> expected_arms
       OR EXISTS (
           SELECT 1
             FROM controlled_experiment_family_cell cell
            WHERE cell.controlled_experiment_family_version_id = NEW.controlled_experiment_family_version_id
              AND (SELECT count(*) FROM controlled_experiment_family_member member
                    WHERE member.controlled_experiment_family_cell_id = cell.controlled_experiment_family_cell_id) <> expected_arms) THEN
        RAISE EXCEPTION 'controlled experiment family version graph is incomplete';
    END IF;
    RETURN NEW;
END $$;
CREATE CONSTRAINT TRIGGER cef_version_graph_complete_trg
AFTER INSERT ON controlled_experiment_family_version
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION controlled_experiment_family_version_graph_complete();

CREATE INDEX controlled_experiment_family_version_family_idx ON controlled_experiment_family_version(controlled_experiment_family_id, version_ordinal);
CREATE INDEX controlled_experiment_family_member_version_idx ON controlled_experiment_family_member(controlled_experiment_family_version_id, member_ordinal);
CREATE INDEX controlled_experiment_family_execution_requirement_version_idx ON controlled_experiment_family_execution_requirement(controlled_experiment_family_version_id);

COMMIT;
