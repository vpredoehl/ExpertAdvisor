-- Durable campaign-materialization pause/resume provenance and ownership.

BEGIN;

CREATE TABLE experiment_campaign_materialization_control_operation (
    campaign_materialization_control_operation_id bigserial PRIMARY KEY,
    recommendation_campaign_materialization_id bigint NOT NULL REFERENCES
        experiment_recommendation_campaign_materialization(
            recommendation_campaign_materialization_id) ON DELETE RESTRICT,
    action text NOT NULL CHECK (action IN ('pause', 'resume')),
    invocation_identity text COLLATE "C" NOT NULL UNIQUE CHECK (
        btrim(invocation_identity) <> '' AND
        octet_length(invocation_identity) <= 4096),
    requester_identity text COLLATE "C" NOT NULL CHECK (
        btrim(requester_identity) <> '' AND
        octet_length(requester_identity) <= 200),
    expected_member_count integer NOT NULL CHECK (expected_member_count > 0),
    resolution_status text NOT NULL CHECK (
        resolution_status IN ('resolved', 'failed')),
    resolution_reason text NOT NULL CHECK (
        btrim(resolution_reason) <> '' AND
        octet_length(resolution_reason) <= 1000),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX campaign_materialization_control_operation_materialization_idx
    ON experiment_campaign_materialization_control_operation(
        recommendation_campaign_materialization_id,
        campaign_materialization_control_operation_id);

CREATE TABLE experiment_campaign_materialization_control_member (
    campaign_materialization_control_member_id bigserial PRIMARY KEY,
    campaign_materialization_control_operation_id bigint NOT NULL REFERENCES
        experiment_campaign_materialization_control_operation(
            campaign_materialization_control_operation_id) ON DELETE RESTRICT,
    recommendation_campaign_materialization_member_id bigint NOT NULL REFERENCES
        experiment_recommendation_campaign_materialization_member(
            recommendation_campaign_materialization_member_id)
        ON DELETE RESTRICT,
    member_ordinal integer NOT NULL CHECK (member_ordinal > 0),
    recommendation_conversion_proposal_id bigint NOT NULL CHECK (
        recommendation_conversion_proposal_id > 0),
    recommendation_conversion_execution_id bigint REFERENCES
        experiment_recommendation_conversion_execution(
            recommendation_conversion_execution_id) ON DELETE RESTRICT,
    recommendation_conversion_activation_id bigint REFERENCES
        experiment_recommendation_conversion_activation(
            recommendation_conversion_activation_id) ON DELETE RESTRICT,
    experiment_id bigint REFERENCES experiment(experiment_id)
        ON DELETE RESTRICT,
    source_pause_operation_id bigint REFERENCES
        experiment_campaign_materialization_control_operation(
            campaign_materialization_control_operation_id) ON DELETE RESTRICT,
    pre_status text,
    pre_phase text,
    pre_scheduler_priority text,
    pre_resume_requested boolean,
    worker_attempt_id bigint REFERENCES experiment_scheduler_worker_attempt(
        worker_attempt_id) ON DELETE RESTRICT,
    worker_attempt_lifecycle_state text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT campaign_materialization_control_member_identity_uidx UNIQUE (
        campaign_materialization_control_operation_id,
        campaign_materialization_control_member_id),
    CONSTRAINT campaign_materialization_control_member_operation_uidx UNIQUE (
        campaign_materialization_control_operation_id,
        recommendation_campaign_materialization_member_id),
    CONSTRAINT campaign_materialization_control_member_ordinal_uidx UNIQUE (
        campaign_materialization_control_operation_id, member_ordinal),
    CONSTRAINT campaign_materialization_control_member_snapshot_check CHECK (
        (experiment_id IS NULL AND pre_status IS NULL AND pre_phase IS NULL AND
         pre_scheduler_priority IS NULL AND pre_resume_requested IS NULL AND
         worker_attempt_id IS NULL AND worker_attempt_lifecycle_state IS NULL)
        OR
        (experiment_id IS NOT NULL AND pre_status IS NOT NULL AND
         pre_phase IS NOT NULL AND pre_scheduler_priority IS NOT NULL AND
         pre_resume_requested IS NOT NULL)),
    CONSTRAINT campaign_materialization_control_member_priority_check CHECK (
        pre_scheduler_priority IS NULL OR
        pre_scheduler_priority IN ('high', 'normal', 'low'))
);

CREATE INDEX campaign_materialization_control_member_experiment_idx
    ON experiment_campaign_materialization_control_member(experiment_id)
    WHERE experiment_id IS NOT NULL;

CREATE TABLE experiment_campaign_materialization_control_outcome (
    campaign_materialization_control_outcome_id bigserial PRIMARY KEY,
    campaign_materialization_control_operation_id bigint NOT NULL,
    campaign_materialization_control_member_id bigint NOT NULL UNIQUE,
    outcome_kind text NOT NULL CHECK (outcome_kind IN (
        'changed_by_group_pause', 'already_paused',
        'terminal_non_applicable', 'unresolved_failed', 'identity_failure',
        'released_by_group_resume', 'not_group_owned',
        'resume_predicate_mismatch')),
    changed_by_operation boolean NOT NULL,
    identity_result text,
    resulting_status text,
    resulting_phase text,
    resulting_resume_requested boolean,
    resulting_worker_attempt_id bigint REFERENCES
        experiment_scheduler_worker_attempt(worker_attempt_id)
        ON DELETE RESTRICT,
    reason text NOT NULL CHECK (
        btrim(reason) <> '' AND octet_length(reason) <= 2000),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT campaign_materialization_control_outcome_member_fkey
        FOREIGN KEY (
            campaign_materialization_control_operation_id,
            campaign_materialization_control_member_id)
        REFERENCES experiment_campaign_materialization_control_member(
            campaign_materialization_control_operation_id,
            campaign_materialization_control_member_id) ON DELETE RESTRICT,
    CONSTRAINT campaign_materialization_control_outcome_changed_check CHECK (
        changed_by_operation = (outcome_kind IN (
            'changed_by_group_pause', 'released_by_group_resume')))
);

CREATE TABLE experiment_campaign_materialization_pause_ownership (
    campaign_materialization_pause_ownership_id bigserial PRIMARY KEY,
    pause_operation_id bigint NOT NULL REFERENCES
        experiment_campaign_materialization_control_operation(
            campaign_materialization_control_operation_id) ON DELETE RESTRICT,
    pause_control_member_id bigint NOT NULL UNIQUE,
    recommendation_campaign_materialization_member_id bigint NOT NULL REFERENCES
        experiment_recommendation_campaign_materialization_member(
            recommendation_campaign_materialization_member_id)
        ON DELETE RESTRICT,
    experiment_id bigint NOT NULL REFERENCES experiment(experiment_id)
        ON DELETE RESTRICT,
    ownership_state text NOT NULL CHECK (
        ownership_state IN ('active', 'superseded', 'consumed')),
    acquired_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    superseded_at timestamptz,
    superseded_action text CHECK (
        superseded_action IS NULL OR superseded_action IN ('pause', 'resume')),
    superseded_by text COLLATE "C",
    consumed_at timestamptz,
    consumed_resume_operation_id bigint REFERENCES
        experiment_campaign_materialization_control_operation(
            campaign_materialization_control_operation_id) ON DELETE RESTRICT,
    CONSTRAINT campaign_materialization_pause_ownership_member_uidx UNIQUE (
        pause_operation_id,
        recommendation_campaign_materialization_member_id),
    CONSTRAINT campaign_materialization_pause_ownership_control_member_fkey
        FOREIGN KEY (pause_operation_id, pause_control_member_id)
        REFERENCES experiment_campaign_materialization_control_member(
            campaign_materialization_control_operation_id,
            campaign_materialization_control_member_id) ON DELETE RESTRICT,
    CONSTRAINT campaign_materialization_pause_ownership_state_shape CHECK (
        (ownership_state = 'active' AND superseded_at IS NULL AND
         superseded_action IS NULL AND superseded_by IS NULL AND
         consumed_at IS NULL AND consumed_resume_operation_id IS NULL)
        OR
        (ownership_state = 'superseded' AND superseded_at IS NOT NULL AND
         superseded_action IS NOT NULL AND superseded_by IS NOT NULL AND
         consumed_at IS NULL AND consumed_resume_operation_id IS NULL)
        OR
        (ownership_state = 'consumed' AND superseded_at IS NULL AND
         superseded_action IS NULL AND superseded_by IS NULL AND
         consumed_at IS NOT NULL AND consumed_resume_operation_id IS NOT NULL))
);

CREATE UNIQUE INDEX campaign_materialization_pause_active_experiment_uidx
    ON experiment_campaign_materialization_pause_ownership(experiment_id)
    WHERE ownership_state = 'active';

CREATE INDEX campaign_materialization_pause_member_state_idx
    ON experiment_campaign_materialization_pause_ownership(
        recommendation_campaign_materialization_member_id, ownership_state);

CREATE OR REPLACE FUNCTION
    reject_campaign_materialization_control_evidence_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'campaign materialization control evidence is immutable'
        USING ERRCODE = '55000';
END $$;

CREATE OR REPLACE FUNCTION
    enforce_campaign_materialization_pause_ownership()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' AND NOT EXISTS (
        SELECT 1
        FROM experiment_campaign_materialization_control_operation operation
        JOIN experiment_campaign_materialization_control_member member
          ON member.campaign_materialization_control_operation_id =
             operation.campaign_materialization_control_operation_id
        JOIN experiment_campaign_materialization_control_outcome outcome
          ON outcome.campaign_materialization_control_member_id =
             member.campaign_materialization_control_member_id
        WHERE operation.campaign_materialization_control_operation_id =
              NEW.pause_operation_id
          AND operation.action = 'pause'
          AND member.campaign_materialization_control_member_id =
              NEW.pause_control_member_id
          AND member.recommendation_campaign_materialization_member_id =
              NEW.recommendation_campaign_materialization_member_id
          AND member.experiment_id = NEW.experiment_id
          AND outcome.outcome_kind = 'changed_by_group_pause'
          AND outcome.changed_by_operation
    ) THEN
        RAISE EXCEPTION 'campaign pause ownership provenance mismatch'
            USING ERRCODE = '23514';
    END IF;

    IF TG_OP = 'UPDATE' AND NEW.ownership_state = 'consumed' AND NOT EXISTS (
        SELECT 1
        FROM experiment_campaign_materialization_control_operation operation
        JOIN experiment_campaign_materialization_control_member member
          ON member.campaign_materialization_control_operation_id =
             operation.campaign_materialization_control_operation_id
        JOIN experiment_campaign_materialization_control_outcome outcome
          ON outcome.campaign_materialization_control_member_id =
             member.campaign_materialization_control_member_id
        WHERE operation.campaign_materialization_control_operation_id =
              NEW.consumed_resume_operation_id
          AND operation.action = 'resume'
          AND member.recommendation_campaign_materialization_member_id =
              NEW.recommendation_campaign_materialization_member_id
          AND member.experiment_id = NEW.experiment_id
          AND member.source_pause_operation_id = NEW.pause_operation_id
          AND outcome.outcome_kind = 'released_by_group_resume'
          AND outcome.changed_by_operation
    ) THEN
        RAISE EXCEPTION 'campaign resume ownership consumption mismatch'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END $$;

CREATE TRIGGER campaign_materialization_control_operation_immutable_trigger
BEFORE UPDATE OR DELETE
ON experiment_campaign_materialization_control_operation
FOR EACH ROW EXECUTE FUNCTION
    reject_campaign_materialization_control_evidence_mutation();

CREATE TRIGGER campaign_materialization_control_member_immutable_trigger
BEFORE UPDATE OR DELETE
ON experiment_campaign_materialization_control_member
FOR EACH ROW EXECUTE FUNCTION
    reject_campaign_materialization_control_evidence_mutation();

CREATE TRIGGER campaign_materialization_control_outcome_immutable_trigger
BEFORE UPDATE OR DELETE
ON experiment_campaign_materialization_control_outcome
FOR EACH ROW EXECUTE FUNCTION
    reject_campaign_materialization_control_evidence_mutation();

CREATE TRIGGER campaign_materialization_pause_ownership_provenance_trigger
BEFORE INSERT OR UPDATE
ON experiment_campaign_materialization_pause_ownership
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_materialization_pause_ownership();

COMMENT ON TABLE experiment_campaign_materialization_control_operation IS
    'Immutable campaign-materialization pause/resume operation identity.';
COMMENT ON TABLE experiment_campaign_materialization_control_member IS
    'Immutable exact frozen member and pre-operation lifecycle snapshot.';
COMMENT ON TABLE experiment_campaign_materialization_control_outcome IS
    'Immutable per-member control outcome evidence.';
COMMENT ON TABLE experiment_campaign_materialization_pause_ownership IS
    'Narrow mutable ownership proving which group pause may release an experiment.';

REVOKE ALL PRIVILEGES ON
    experiment_campaign_materialization_control_operation,
    experiment_campaign_materialization_control_member,
    experiment_campaign_materialization_control_outcome,
    experiment_campaign_materialization_pause_ownership FROM PUBLIC;
REVOKE ALL PRIVILEGES ON
    experiment_campaign_materialization_control_operation,
    experiment_campaign_materialization_control_member,
    experiment_campaign_materialization_control_outcome,
    experiment_campaign_materialization_pause_ownership FROM pqxx;

GRANT SELECT, INSERT ON
    experiment_campaign_materialization_control_operation,
    experiment_campaign_materialization_control_member,
    experiment_campaign_materialization_control_outcome,
    experiment_campaign_materialization_pause_ownership TO pqxx;
GRANT UPDATE (
    ownership_state, superseded_at, superseded_action, superseded_by,
    consumed_at, consumed_resume_operation_id)
ON experiment_campaign_materialization_pause_ownership TO pqxx;

GRANT EXECUTE ON FUNCTION
    reject_campaign_materialization_control_evidence_mutation() TO pqxx;
GRANT EXECUTE ON FUNCTION
    enforce_campaign_materialization_pause_ownership() TO pqxx;

DO $$
DECLARE
    sequence_name text;
BEGIN
    FOREACH sequence_name IN ARRAY ARRAY[
        pg_get_serial_sequence(
            'experiment_campaign_materialization_control_operation',
            'campaign_materialization_control_operation_id'),
        pg_get_serial_sequence(
            'experiment_campaign_materialization_control_member',
            'campaign_materialization_control_member_id'),
        pg_get_serial_sequence(
            'experiment_campaign_materialization_control_outcome',
            'campaign_materialization_control_outcome_id'),
        pg_get_serial_sequence(
            'experiment_campaign_materialization_pause_ownership',
            'campaign_materialization_pause_ownership_id')]
    LOOP
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM PUBLIC', sequence_name);
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM pqxx', sequence_name);
        EXECUTE format('GRANT USAGE ON SEQUENCE %s TO pqxx', sequence_name);
    END LOOP;
END $$;

COMMIT;
