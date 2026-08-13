-- Campaign Operations Phase 5 (architectural Phase G): immutable operational
-- completion and read-only audit/status evidence.  This migration does not
-- complete or otherwise mutate experiment lifecycle, interpret scientific
-- results, refund committed units, or enable any scheduler/worker behavior.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles
                   WHERE rolname = 'campaign_operations_completion_writer') THEN
        CREATE ROLE campaign_operations_completion_writer NOLOGIN;
    END IF;
END $$;

ALTER ROLE campaign_operations_completion_writer
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
REVOKE campaign_operations_completion_writer FROM pqxx;

DO $$
BEGIN
    IF pg_has_role(
            'pqxx', 'campaign_operations_completion_writer', 'MEMBER') THEN
        RAISE EXCEPTION
            'pqxx must not inherit Campaign Operations completion authority';
    END IF;
END $$;

-- The boundary state lives on the row used as the campaign mutex.  Reading
-- the locked tuple itself is required here: a trigger statement that waited
-- for a concurrent completion can otherwise retain a pre-wait MVCC snapshot
-- and miss the newly inserted completion event.
ALTER TABLE campaign_operations_campaign
    ADD COLUMN IF NOT EXISTS completion_boundary_closed boolean
    NOT NULL DEFAULT false;

CREATE TABLE IF NOT EXISTS campaign_operations_completion_event (
    completion_event_id bigserial PRIMARY KEY CHECK (
        completion_event_id > 0),
    operational_campaign_id bigint NOT NULL UNIQUE,
    campaign_identity_canonical text COLLATE "C" NOT NULL CHECK (
        campaign_identity_canonical <> '' AND
        octet_length(campaign_identity_canonical) <= 134217728),
    operation_key text COLLATE "C" NOT NULL CHECK (
        operation_key ~ '^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,127}$'),
    administrative_terminal_state text COLLATE "C" NOT NULL CHECK (
        administrative_terminal_state IN (
            'terminal_completed', 'terminal_cancelled', 'terminal_failed')),
    completion_classification text COLLATE "C" NOT NULL CHECK (
        completion_classification IN (
            'operational_request_failed', 'mixed_terminal_outcomes',
            'downstream_failure', 'terminal_partial_completion',
            'all_scope_cancelled', 'all_downstream_completed')),
    budget_ledger_entry_id bigint NOT NULL,
    budget_ledger_version integer NOT NULL CHECK (
        budget_ledger_version > 0),
    budget_resulting_total bigint NOT NULL CHECK (
        budget_resulting_total >= 0),
    budget_ever_reserved bigint NOT NULL CHECK (
        budget_ever_reserved >= 0),
    budget_committed bigint NOT NULL CHECK (budget_committed >= 0),
    budget_released_or_expired bigint NOT NULL CHECK (
        budget_released_or_expired >= 0),
    budget_held bigint NOT NULL CHECK (budget_held = 0),
    budget_unallocated bigint NOT NULL CHECK (budget_unallocated >= 0),
    scope_member_count integer NOT NULL CHECK (scope_member_count > 0),
    completed_member_count integer NOT NULL CHECK (
        completed_member_count >= 0),
    failed_member_count integer NOT NULL CHECK (
        failed_member_count >= 0),
    cancelled_or_never_dispatched_member_count integer NOT NULL CHECK (
        cancelled_or_never_dispatched_member_count >= 0),
    reservation_count integer NOT NULL CHECK (reservation_count >= 0),
    request_count integer NOT NULL CHECK (request_count >= 0),
    binding_count integer NOT NULL CHECK (binding_count >= 0),
    control_owner_count integer NOT NULL CHECK (control_owner_count >= 0),
    cancellation_request_count integer NOT NULL CHECK (
        cancellation_request_count >= 0),
    cancellation_settlement_count integer NOT NULL CHECK (
        cancellation_settlement_count >= 0),
    unresolved_blocking_observation_count integer NOT NULL CHECK (
        unresolved_blocking_observation_count = 0),
    authorization_evidence_canonical text COLLATE "C" NOT NULL CHECK (
        authorization_evidence_canonical <> ''),
    authorization_evidence_hash text COLLATE "C" NOT NULL CHECK (
        authorization_evidence_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    budget_evidence_canonical text COLLATE "C" NOT NULL CHECK (
        budget_evidence_canonical <> ''),
    budget_evidence_hash text COLLATE "C" NOT NULL CHECK (
        budget_evidence_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    reservation_evidence_canonical text COLLATE "C" NOT NULL CHECK (
        reservation_evidence_canonical <> ''),
    reservation_evidence_hash text COLLATE "C" NOT NULL CHECK (
        reservation_evidence_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    request_evidence_canonical text COLLATE "C" NOT NULL CHECK (
        request_evidence_canonical <> ''),
    request_evidence_hash text COLLATE "C" NOT NULL CHECK (
        request_evidence_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    binding_evidence_canonical text COLLATE "C" NOT NULL CHECK (
        binding_evidence_canonical <> ''),
    binding_evidence_hash text COLLATE "C" NOT NULL CHECK (
        binding_evidence_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    lifecycle_evidence_canonical text COLLATE "C" NOT NULL CHECK (
        lifecycle_evidence_canonical <> ''),
    lifecycle_evidence_hash text COLLATE "C" NOT NULL CHECK (
        lifecycle_evidence_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    cancellation_evidence_canonical text COLLATE "C" NOT NULL CHECK (
        cancellation_evidence_canonical <> ''),
    cancellation_evidence_hash text COLLATE "C" NOT NULL CHECK (
        cancellation_evidence_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    reconciliation_evidence_canonical text COLLATE "C" NOT NULL CHECK (
        reconciliation_evidence_canonical <> ''),
    reconciliation_evidence_hash text COLLATE "C" NOT NULL CHECK (
        reconciliation_evidence_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    actor_identity text COLLATE "C" NOT NULL CHECK (
        actor_identity ~ '^[A-Za-z0-9][A-Za-z0-9._@:/+\-]{0,127}$'),
    capability text COLLATE "C" NOT NULL CHECK (
        capability = 'campaign_operations_completion_writer'),
    reason text COLLATE "C" NOT NULL CHECK (
        reason <> '' AND octet_length(reason) <= 4096 AND
        reason ~ E'[^ \t\r\n]' AND
        translate(reason, E'\t\r\n', '') !~ '[[:cntrl:]]'),
    completion_contract_version integer NOT NULL CHECK (
        completion_contract_version = 1),
    completion_identity_canonical text COLLATE "C" NOT NULL CHECK (
        completion_identity_canonical <> '' AND
        octet_length(completion_identity_canonical) <= 134217728),
    completion_identity_hash text COLLATE "C" NOT NULL CHECK (
        completion_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    recorded_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_completion_campaign_fk FOREIGN KEY (
        operational_campaign_id) REFERENCES
        campaign_operations_campaign(operational_campaign_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_completion_budget_fk FOREIGN KEY (
        budget_ledger_entry_id) REFERENCES
        campaign_operations_budget_ledger_entry(budget_ledger_entry_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_completion_member_equation CHECK (
        completed_member_count + failed_member_count +
            cancelled_or_never_dispatched_member_count =
            scope_member_count),
    CONSTRAINT campaign_operations_completion_budget_equation CHECK (
        budget_ever_reserved =
            budget_committed + budget_released_or_expired + budget_held AND
        budget_resulting_total =
            budget_committed + budget_held + budget_unallocated),
    CONSTRAINT campaign_operations_completion_state_classification CHECK (
        (administrative_terminal_state = 'terminal_failed' AND
            completion_classification IN (
                'operational_request_failed', 'mixed_terminal_outcomes',
                'downstream_failure')) OR
        (administrative_terminal_state = 'terminal_cancelled' AND
            completion_classification = 'all_scope_cancelled') OR
        (administrative_terminal_state = 'terminal_completed' AND
            completion_classification IN (
                'terminal_partial_completion',
                'all_downstream_completed')))
);

UPDATE campaign_operations_campaign campaign
   SET completion_boundary_closed = true
 WHERE NOT campaign.completion_boundary_closed
   AND EXISTS (
       SELECT 1 FROM campaign_operations_completion_event completion
       WHERE completion.operational_campaign_id =
             campaign.operational_campaign_id);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM campaign_operations_campaign campaign
        WHERE campaign.completion_boundary_closed <>
              EXISTS (
                  SELECT 1
                  FROM campaign_operations_completion_event completion
                  WHERE completion.operational_campaign_id =
                        campaign.operational_campaign_id)) THEN
        RAISE EXCEPTION
            'campaign operations completion boundary/event mismatch'
            USING ERRCODE = '23514';
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS campaign_operations_completion_hash_idx
    ON campaign_operations_completion_event(completion_identity_hash);
CREATE INDEX IF NOT EXISTS campaign_operations_completion_recorded_idx
    ON campaign_operations_completion_event(recorded_at, completion_event_id);

CREATE TABLE IF NOT EXISTS
campaign_operations_completion_audit_reference_event (
    completion_audit_reference_event_id bigserial PRIMARY KEY CHECK (
        completion_audit_reference_event_id > 0),
    operational_campaign_id bigint NOT NULL UNIQUE,
    completion_event_id bigint NOT NULL UNIQUE,
    actor_identity text COLLATE "C" NOT NULL,
    capability text COLLATE "C" NOT NULL CHECK (
        capability = 'campaign_operations_completion_writer'),
    reason text COLLATE "C" NOT NULL,
    outcome text COLLATE "C" NOT NULL CHECK (outcome = 'recorded'),
    replay_disposition text COLLATE "C" NOT NULL CHECK (
        replay_disposition = 'recorded'),
    diagnostic_code text COLLATE "C" NOT NULL CHECK (
        diagnostic_code = 'all_completion_prerequisites_proven'),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_completion_audit_campaign_fk FOREIGN KEY (
        operational_campaign_id) REFERENCES
        campaign_operations_campaign(operational_campaign_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_completion_audit_event_fk FOREIGN KEY (
        completion_event_id) REFERENCES
        campaign_operations_completion_event(completion_event_id)
        ON DELETE RESTRICT
);

CREATE OR REPLACE FUNCTION campaign_operations_completion_evidence_text(
    target_campaign_id bigint, evidence_kind text)
RETURNS text
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
AS $$
DECLARE result text;
BEGIN
    CASE evidence_kind
    WHEN 'authorization' THEN
        SELECT 'authorization_v1;' || COALESCE(string_agg(format(
            'head_id=%s,version=%s,kind=%s,action=%s,scope=%s,'
            'not_before=%s,expires_at=%s,canonical=%s:%s',
            head.authorization_event_id, head.chain_version,
            head.event_kind, head.action_kind, head.scope_kind,
            to_char(head.not_before AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'),
            COALESCE(to_char(head.expires_at AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'), 'none'),
            octet_length(head.authorization_identity_canonical),
            head.authorization_identity_canonical), ';'
            ORDER BY head.action_kind, head.action_contract_version,
                head.scope_kind, head.scope_contract_version), 'none')
        INTO result
        FROM (
            SELECT DISTINCT ON (
                action_kind, action_contract_version,
                scope_kind, scope_contract_version) *
            FROM campaign_operations_authorization_event
            WHERE operational_campaign_id = target_campaign_id
            ORDER BY action_kind, action_contract_version,
                scope_kind, scope_contract_version,
                chain_version DESC, authorization_event_id DESC) head;
    WHEN 'budget' THEN
        SELECT format(
            'budget_v1;id=%s;version=%s;status=%s;total=%s;'
            'ever_reserved=%s;committed=%s;released_or_expired=%s;'
            'held=%s;unallocated=%s;canonical=%s:%s',
            head.budget_ledger_entry_id, head.ledger_version,
            head.ledger_status, head.resulting_total,
            COALESCE(accounting.ever_reserved, 0),
            COALESCE(accounting.committed, 0),
            COALESCE(accounting.released_or_expired, 0),
            COALESCE(accounting.held, 0),
            head.resulting_total - COALESCE(accounting.committed, 0) -
                COALESCE(accounting.held, 0),
            length(head.budget_identity_canonical),
            head.budget_identity_canonical)
        INTO result
        FROM LATERAL (
            SELECT * FROM campaign_operations_budget_ledger_entry
            WHERE operational_campaign_id = target_campaign_id
            ORDER BY ledger_version DESC LIMIT 1) head
        CROSS JOIN LATERAL (
            SELECT
                COALESCE(sum(amount), 0)::bigint AS ever_reserved,
                COALESCE(sum(amount) FILTER (
                    WHERE reservation_state = 'committed'), 0)::bigint
                    AS committed,
                COALESCE(sum(amount) FILTER (
                    WHERE reservation_state IN ('released', 'expired')), 0)
                    ::bigint AS released_or_expired,
                COALESCE(sum(amount) FILTER (
                    WHERE reservation_state = 'held'), 0)::bigint AS held
            FROM campaign_operations_reservation
            WHERE operational_campaign_id = target_campaign_id) accounting;
    WHEN 'reservation' THEN
        SELECT 'reservations_v1;' || COALESCE(string_agg(format(
            'id=%s,state=%s,version=%s,amount=%s,canonical=%s:%s;events=%s',
            reservation.reservation_id, reservation.reservation_state,
            reservation.state_version, reservation.amount,
            octet_length(reservation.reservation_identity_canonical),
            reservation.reservation_identity_canonical,
            COALESCE((SELECT string_agg(format(
                'event=%s,kind=%s,expected=%s,resulting=%s,canonical=%s:%s',
                event.reservation_event_id, event.transition_kind,
                event.expected_version, event.resulting_version,
                octet_length(event.reservation_event_identity_canonical),
                event.reservation_event_identity_canonical), ','
                ORDER BY event.resulting_version,
                    event.reservation_event_id)
                FROM campaign_operations_reservation_event event
                WHERE event.reservation_id = reservation.reservation_id),
                'none')), ';'
            ORDER BY reservation.reservation_id), 'none')
        INTO result
        FROM campaign_operations_reservation reservation
        WHERE reservation.operational_campaign_id = target_campaign_id;
    WHEN 'request' THEN
        SELECT 'requests_v1;' || COALESCE(string_agg(format(
            'id=%s,state=%s,version=%s,reservation=%s,lease=%s,'
            'canonical=%s:%s;attempts=%s',
            request.operational_request_id, request.request_state,
            request.state_version, request.reservation_id,
            CASE WHEN request.lease_token_hash IS NULL THEN 'none'
                 ELSE 'present' END,
            octet_length(request.request_identity_canonical),
            request.request_identity_canonical,
            COALESCE((SELECT string_agg(format(
                'attempt=%s,ordinal=%s,canonical=%s:%s,outcome=%s',
                attempt.dispatch_attempt_id, attempt.attempt_ordinal,
                octet_length(attempt.attempt_identity_canonical),
                attempt.attempt_identity_canonical,
                CASE WHEN outcome.dispatch_attempt_outcome_id IS NULL
                     THEN 'none'
                     ELSE format('id=%s,result=%s,canonical=%s:%s',
                         outcome.dispatch_attempt_outcome_id,
                         outcome.result_classification,
                         octet_length(outcome.outcome_identity_canonical),
                         outcome.outcome_identity_canonical)
                END), ',' ORDER BY attempt.attempt_ordinal,
                    attempt.dispatch_attempt_id)
                FROM campaign_operations_dispatch_attempt attempt
                LEFT JOIN campaign_operations_dispatch_attempt_outcome outcome
                  ON outcome.dispatch_attempt_id =
                     attempt.dispatch_attempt_id
                WHERE attempt.operational_request_id =
                      request.operational_request_id), 'none')), ';'
            ORDER BY request.operational_request_id), 'none')
        INTO result
        FROM campaign_operations_operational_request request
        WHERE request.operational_campaign_id = target_campaign_id;
    WHEN 'binding' THEN
        SELECT 'bindings_v1;' || COALESCE(string_agg(format(
            'binding=%s,request=%s,member=%s,ordinal=%s,experiment=%s,'
            'canonical=%s:%s,owner=%s',
            binding.request_binding_id, binding.operational_request_id,
            binding.recommendation_campaign_materialization_member_id,
            binding.member_ordinal, binding.experiment_id,
            octet_length(binding.binding_identity_canonical),
            binding.binding_identity_canonical,
            CASE WHEN owner.downstream_control_owner_id IS NULL THEN 'none'
                 ELSE format('id=%s,mode=%s,canonical=%s:%s',
                     owner.downstream_control_owner_id, owner.control_mode,
                     octet_length(owner.owner_identity_canonical),
                     owner.owner_identity_canonical)
            END), ';'
            ORDER BY binding.operational_request_id, binding.member_ordinal),
            'none')
        INTO result
        FROM campaign_operations_request_binding binding
        JOIN campaign_operations_operational_request request
          ON request.operational_request_id =
             binding.operational_request_id
        LEFT JOIN campaign_operations_downstream_control_owner owner
          ON owner.request_binding_id = binding.request_binding_id
        WHERE request.operational_campaign_id = target_campaign_id;
    WHEN 'lifecycle' THEN
        SELECT 'lifecycle_v1;' || COALESCE(string_agg(format(
            'binding=%s,experiment=%s,status=%s,phase=%s,updated=%s',
            binding.request_binding_id, experiment.experiment_id,
            experiment.status, experiment.phase,
            to_char(experiment.updated_at AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')), ';'
            ORDER BY binding.member_ordinal), 'none')
        INTO result
        FROM campaign_operations_request_binding binding
        JOIN campaign_operations_operational_request request
          ON request.operational_request_id =
             binding.operational_request_id
        JOIN experiment
          ON experiment.experiment_id = binding.experiment_id
        WHERE request.operational_campaign_id = target_campaign_id;
    WHEN 'cancellation' THEN
        SELECT 'cancellations_v1;' || COALESCE(string_agg(format(
            'request=%s,target=%s,canonical=%s:%s,settlement=%s',
            cancellation.cancellation_request_id,
            COALESCE(cancellation.operational_request_id::text, 'campaign'),
            octet_length(cancellation.cancellation_identity_canonical),
            cancellation.cancellation_identity_canonical,
            CASE WHEN settlement.cancellation_settlement_id IS NULL
                 THEN 'none'
                 ELSE format('id=%s,disposition=%s,canonical=%s:%s',
                     settlement.cancellation_settlement_id,
                     settlement.disposition,
                     octet_length(settlement.settlement_identity_canonical),
                     settlement.settlement_identity_canonical)
            END), ';'
            ORDER BY cancellation.cancellation_request_id), 'none')
        INTO result
        FROM campaign_operations_cancellation_request cancellation
        LEFT JOIN campaign_operations_cancellation_settlement settlement
          ON settlement.cancellation_request_id =
             cancellation.cancellation_request_id
        WHERE cancellation.operational_campaign_id = target_campaign_id;
    WHEN 'reconciliation' THEN
        SELECT 'reconciliation_v1;' || COALESCE(string_agg(format(
            'observation=%s,reason=%s,evidence=%s:%s,canonical=%s:%s,'
            'resolution=%s',
            observation.reconciliation_observation_id,
            observation.reason_code,
            octet_length(observation.evidence_identity_canonical),
            observation.evidence_identity_canonical,
            octet_length(observation.observation_identity_canonical),
            observation.observation_identity_canonical,
            CASE WHEN resolution.reconciliation_resolution_id IS NULL
                 THEN 'none'
                 ELSE format('id=%s,disposition=%s,transition=%s:%s,'
                     'canonical=%s:%s',
                     resolution.reconciliation_resolution_id,
                     resolution.resolution_disposition,
                     octet_length(
                         resolution.transition_identity_canonical),
                     resolution.transition_identity_canonical,
                     octet_length(
                         resolution.resolution_identity_canonical),
                     resolution.resolution_identity_canonical)
            END), ';'
            ORDER BY observation.reconciliation_observation_id), 'none')
        INTO result
        FROM campaign_operations_reconciliation_observation observation
        LEFT JOIN campaign_operations_reconciliation_resolution resolution
          ON resolution.reconciliation_observation_id =
             observation.reconciliation_observation_id
        WHERE observation.operational_campaign_id = target_campaign_id;
    ELSE
        RAISE EXCEPTION 'unknown completion evidence kind'
            USING ERRCODE = '22023';
    END CASE;
    RETURN COALESCE(result, evidence_kind || '_v1;none');
END;
$$;

CREATE OR REPLACE FUNCTION campaign_operations_completion_blockers(
    target_campaign_id bigint)
RETURNS TABLE (
    blocker_code text,
    blocker_detail text,
    blocker_class text)
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
AS $$
DECLARE scope_count integer;
DECLARE request_total integer;
BEGIN
    SELECT materialization_member_count INTO scope_count
    FROM campaign_operations_campaign
    WHERE operational_campaign_id = target_campaign_id;
    IF NOT FOUND THEN
        RETURN QUERY SELECT 'campaign_not_found'::text,
            target_campaign_id::text, 'inconsistent'::text;
        RETURN;
    END IF;

    SELECT count(*)::integer INTO request_total
    FROM campaign_operations_operational_request
    WHERE operational_campaign_id = target_campaign_id;

    RETURN QUERY
    SELECT 'campaign_paused',
           COALESCE(control.control_event_id::text, 'unknown'), 'unsettled'
    FROM LATERAL (
        SELECT control_event_id, event_kind
        FROM campaign_operations_control_event
        WHERE operational_campaign_id = target_campaign_id
        ORDER BY control_version DESC LIMIT 1) control
    WHERE control.event_kind = 'pause'
    UNION ALL
    SELECT 'authorization_missing', target_campaign_id::text, 'inconsistent'
    WHERE NOT EXISTS (
        SELECT 1 FROM campaign_operations_authorization_event
        WHERE operational_campaign_id = target_campaign_id)
    UNION ALL
    SELECT 'active_authorization_unexhausted',
           auth_head.authorization_event_id::text, 'unsettled'
    FROM (
        SELECT DISTINCT ON (
            action_kind, action_contract_version,
            scope_kind, scope_contract_version) *
        FROM campaign_operations_authorization_event
        WHERE operational_campaign_id = target_campaign_id
        ORDER BY action_kind, action_contract_version,
            scope_kind, scope_contract_version,
            chain_version DESC, authorization_event_id DESC) auth_head
    WHERE auth_head.event_kind = 'granted'
      AND auth_head.not_before <= transaction_timestamp()
      AND (auth_head.expires_at IS NULL OR
           auth_head.expires_at > transaction_timestamp())
      AND NOT EXISTS (
          SELECT 1 FROM campaign_operations_operational_request
          WHERE operational_campaign_id = target_campaign_id
            AND request_state IN (
                'bound', 'permanently_failed', 'cancelled'))
    UNION ALL
    SELECT 'budget_missing', target_campaign_id::text, 'inconsistent'
    WHERE NOT EXISTS (
        SELECT 1 FROM campaign_operations_budget_ledger_entry
        WHERE operational_campaign_id = target_campaign_id)
    UNION ALL
    SELECT 'budget_accounting_inconsistent',
           format('total=%s,ever=%s,committed=%s,released=%s,held=%s',
               head.resulting_total, accounting.ever_reserved,
               accounting.committed, accounting.released_or_expired,
               accounting.held),
           'inconsistent'
    FROM LATERAL (
        SELECT * FROM campaign_operations_budget_ledger_entry
        WHERE operational_campaign_id = target_campaign_id
        ORDER BY ledger_version DESC LIMIT 1) head
    CROSS JOIN LATERAL (
        SELECT
            COALESCE(sum(amount), 0)::bigint AS ever_reserved,
            COALESCE(sum(amount) FILTER (
                WHERE reservation_state = 'committed'), 0)::bigint
                AS committed,
            COALESCE(sum(amount) FILTER (
                WHERE reservation_state IN ('released', 'expired')), 0)::
                bigint AS released_or_expired,
            COALESCE(sum(amount) FILTER (
                WHERE reservation_state = 'held'), 0)::bigint AS held
        FROM campaign_operations_reservation
        WHERE operational_campaign_id = target_campaign_id) accounting
    WHERE accounting.ever_reserved <>
              accounting.committed + accounting.released_or_expired +
              accounting.held
       OR accounting.committed + accounting.held > head.resulting_total
       OR head.resulting_total < 0
    UNION ALL
    SELECT 'request_missing', target_campaign_id::text, 'unsettled'
    WHERE request_total = 0
    UNION ALL
    SELECT 'request_cardinality_conflict', request_total::text, 'inconsistent'
    WHERE request_total > 1
    UNION ALL
    SELECT CASE request_state
               WHEN 'ready' THEN 'request_ready'
               WHEN 'dispatching' THEN 'request_dispatching'
               ELSE 'request_reconciliation_required'
           END,
           operational_request_id::text,
           CASE WHEN request_state = 'reconciliation_required'
                THEN 'reconciliation_required' ELSE 'unsettled' END
    FROM campaign_operations_operational_request
    WHERE operational_campaign_id = target_campaign_id
      AND request_state IN ('ready', 'dispatching', 'reconciliation_required')
    UNION ALL
    SELECT 'active_dispatch_lease', operational_request_id::text, 'unsettled'
    FROM campaign_operations_operational_request
    WHERE operational_campaign_id = target_campaign_id
      AND (lease_token_hash IS NOT NULL OR lease_expires_at IS NOT NULL OR
           dispatcher_identity IS NOT NULL)
    UNION ALL
    SELECT 'reservation_request_cardinality_mismatch',
           format('reservations=%s,requests=%s',
               (SELECT count(*) FROM campaign_operations_reservation
                WHERE operational_campaign_id = target_campaign_id),
               request_total), 'inconsistent'
    WHERE (SELECT count(*) FROM campaign_operations_reservation
           WHERE operational_campaign_id = target_campaign_id) <> request_total
    UNION ALL
    SELECT CASE reservation_state
               WHEN 'held' THEN 'reservation_held'
               ELSE 'reservation_reconciliation_required'
           END,
           reservation_id::text,
           CASE WHEN reservation_state = 'reconciliation_required'
                THEN 'reconciliation_required' ELSE 'unsettled' END
    FROM campaign_operations_reservation
    WHERE operational_campaign_id = target_campaign_id
      AND reservation_state IN ('held', 'reconciliation_required')
    UNION ALL
    SELECT 'reservation_request_state_conflict',
           format('reservation=%s,request=%s',
               reservation.reservation_state, request.request_state),
           'inconsistent'
    FROM campaign_operations_reservation reservation
    JOIN campaign_operations_operational_request request
      ON request.reservation_id = reservation.reservation_id
    WHERE reservation.operational_campaign_id = target_campaign_id
      AND NOT (
          (request.request_state = 'bound' AND
           reservation.reservation_state = 'committed') OR
          (request.request_state IN ('permanently_failed', 'cancelled') AND
           reservation.reservation_state IN ('released', 'expired')))
    UNION ALL
    SELECT 'incomplete_dispatch_attempt', attempt.dispatch_attempt_id::text,
           'reconciliation_required'
    FROM campaign_operations_dispatch_attempt attempt
    JOIN campaign_operations_operational_request request
      ON request.operational_request_id = attempt.operational_request_id
    LEFT JOIN campaign_operations_dispatch_attempt_outcome outcome
      ON outcome.dispatch_attempt_id = attempt.dispatch_attempt_id
    WHERE request.operational_campaign_id = target_campaign_id
      AND outcome.dispatch_attempt_outcome_id IS NULL
    UNION ALL
    SELECT 'ambiguous_dispatch_attempt', outcome.dispatch_attempt_id::text,
           'reconciliation_required'
    FROM campaign_operations_dispatch_attempt_outcome outcome
    JOIN campaign_operations_dispatch_attempt attempt
      ON attempt.dispatch_attempt_id = outcome.dispatch_attempt_id
    JOIN campaign_operations_operational_request request
      ON request.operational_request_id = attempt.operational_request_id
    WHERE request.operational_campaign_id = target_campaign_id
      AND (outcome.result_classification = 'reconciliation_required' OR
           outcome.uncertain_commit_recovery_classification =
               'ambiguous_evidence' OR
           outcome.downstream_evidence_classification IN (
               'partial_phase5_evidence', 'progressed_unbound_evidence',
               'causally_ambiguous'))
    UNION ALL
    SELECT 'binding_cardinality_mismatch',
           format('request=%s,bindings=%s,scope=%s',
               request.operational_request_id,
               (SELECT count(*) FROM campaign_operations_request_binding b
                WHERE b.operational_request_id =
                      request.operational_request_id),
               scope_count), 'inconsistent'
    FROM campaign_operations_operational_request request
    WHERE request.operational_campaign_id = target_campaign_id
      AND request.request_state = 'bound'
      AND (SELECT count(*) FROM campaign_operations_request_binding b
           WHERE b.operational_request_id =
                 request.operational_request_id) <> scope_count
    UNION ALL
    SELECT 'control_owner_cardinality_mismatch',
           format('request=%s,owners=%s,scope=%s',
               request.operational_request_id,
               (SELECT count(*)
                FROM campaign_operations_downstream_control_owner owner
                WHERE owner.operational_request_id =
                      request.operational_request_id),
               scope_count), 'inconsistent'
    FROM campaign_operations_operational_request request
    WHERE request.operational_campaign_id = target_campaign_id
      AND request.request_state = 'bound'
      AND (SELECT count(*)
           FROM campaign_operations_downstream_control_owner owner
           WHERE owner.operational_request_id =
                 request.operational_request_id) <> scope_count
    UNION ALL
    SELECT 'unexpected_unbound_binding', request.operational_request_id::text,
           'inconsistent'
    FROM campaign_operations_operational_request request
    WHERE request.operational_campaign_id = target_campaign_id
      AND request.request_state IN ('permanently_failed', 'cancelled')
      AND EXISTS (
          SELECT 1 FROM campaign_operations_request_binding binding
          WHERE binding.operational_request_id =
                request.operational_request_id)
    UNION ALL
    SELECT 'downstream_lifecycle_nonterminal',
           format('experiment=%s,status=%s,phase=%s',
               experiment.experiment_id, experiment.status,
               experiment.phase), 'unsettled'
    FROM campaign_operations_request_binding binding
    JOIN campaign_operations_operational_request request
      ON request.operational_request_id = binding.operational_request_id
    JOIN experiment
      ON experiment.experiment_id = binding.experiment_id
    WHERE request.operational_campaign_id = target_campaign_id
      AND experiment.status NOT IN ('completed', 'failed', 'cancelled')
    UNION ALL
    SELECT 'cancellation_unsettled',
           cancellation.cancellation_request_id::text, 'unsettled'
    FROM campaign_operations_cancellation_request cancellation
    LEFT JOIN campaign_operations_cancellation_settlement settlement
      ON settlement.cancellation_request_id =
         cancellation.cancellation_request_id
    WHERE cancellation.operational_campaign_id = target_campaign_id
      AND settlement.cancellation_settlement_id IS NULL
    UNION ALL
    SELECT 'cancellation_inconsistent',
           cancellation.cancellation_request_id::text, 'inconsistent'
    FROM campaign_operations_cancellation_request cancellation
    JOIN campaign_operations_cancellation_settlement settlement
      ON settlement.cancellation_request_id =
         cancellation.cancellation_request_id
    WHERE cancellation.operational_campaign_id = target_campaign_id
      AND settlement.disposition = 'inconsistent'
    UNION ALL
    SELECT 'blocking_reconciliation_observation',
           format('observation=%s,reason=%s',
               observation.reconciliation_observation_id,
               observation.reason_code), 'reconciliation_required'
    FROM campaign_operations_reconciliation_observation observation
    LEFT JOIN campaign_operations_reconciliation_resolution resolution
      ON resolution.reconciliation_observation_id =
         observation.reconciliation_observation_id
    WHERE observation.operational_campaign_id = target_campaign_id
      AND resolution.reconciliation_resolution_id IS NULL
      AND observation.reason_code NOT IN (
          'terminal_lifecycle_completion_ready',
          'post_completion_lifecycle_changed');
END;
$$;

CREATE OR REPLACE FUNCTION campaign_operations_completion_classification(
    target_campaign_id bigint)
RETURNS TABLE (
    administrative_terminal_state text,
    completion_classification text,
    scope_member_count integer,
    completed_member_count integer,
    failed_member_count integer,
    cancelled_or_never_dispatched_member_count integer,
    reservation_count integer,
    request_count integer,
    binding_count integer,
    control_owner_count integer,
    cancellation_request_count integer,
    cancellation_settlement_count integer,
    unresolved_blocking_observation_count integer)
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
AS $$
DECLARE request_state_value text;
DECLARE completed_count integer;
DECLARE failed_count integer;
DECLARE cancelled_count integer;
DECLARE scope_count integer;
DECLARE bound_count integer;
BEGIN
    IF EXISTS (
        SELECT 1 FROM campaign_operations_completion_blockers(
            target_campaign_id)) THEN
        RAISE EXCEPTION 'campaign operations completion remains blocked'
            USING ERRCODE = 'P0001';
    END IF;
    SELECT campaign.materialization_member_count, request.request_state
      INTO STRICT scope_count, request_state_value
    FROM campaign_operations_campaign campaign
    JOIN campaign_operations_operational_request request
      ON request.operational_campaign_id =
         campaign.operational_campaign_id
    WHERE campaign.operational_campaign_id = target_campaign_id;
    SELECT
        count(*) FILTER (WHERE experiment.status = 'completed')::integer,
        count(*) FILTER (WHERE experiment.status = 'failed')::integer,
        count(*) FILTER (WHERE experiment.status = 'cancelled')::integer,
        count(*)::integer
      INTO completed_count, failed_count, cancelled_count, bound_count
    FROM campaign_operations_request_binding binding
    JOIN campaign_operations_operational_request request
      ON request.operational_request_id = binding.operational_request_id
    JOIN experiment ON experiment.experiment_id = binding.experiment_id
    WHERE request.operational_campaign_id = target_campaign_id;
    IF request_state_value = 'permanently_failed' THEN
        completion_classification := 'operational_request_failed';
        administrative_terminal_state := 'terminal_failed';
        completed_count := 0;
        failed_count := 0;
        cancelled_count := scope_count;
    ELSIF request_state_value = 'cancelled' THEN
        completion_classification := 'all_scope_cancelled';
        administrative_terminal_state := 'terminal_cancelled';
        completed_count := 0;
        failed_count := 0;
        cancelled_count := scope_count;
    ELSIF failed_count > 0 AND
          (completed_count > 0 OR cancelled_count > 0) THEN
        completion_classification := 'mixed_terminal_outcomes';
        administrative_terminal_state := 'terminal_failed';
    ELSIF failed_count = scope_count THEN
        completion_classification := 'downstream_failure';
        administrative_terminal_state := 'terminal_failed';
    ELSIF failed_count = 0 AND completed_count > 0 AND
          cancelled_count > 0 THEN
        completion_classification := 'terminal_partial_completion';
        administrative_terminal_state := 'terminal_completed';
    ELSIF cancelled_count = scope_count THEN
        completion_classification := 'all_scope_cancelled';
        administrative_terminal_state := 'terminal_cancelled';
    ELSIF completed_count = scope_count THEN
        completion_classification := 'all_downstream_completed';
        administrative_terminal_state := 'terminal_completed';
    ELSE
        RAISE EXCEPTION 'campaign operations classification is ambiguous'
            USING ERRCODE = 'P0001';
    END IF;
    scope_member_count := scope_count;
    completed_member_count := completed_count;
    failed_member_count := failed_count;
    cancelled_or_never_dispatched_member_count := cancelled_count;
    SELECT count(*)::integer INTO reservation_count
    FROM campaign_operations_reservation
    WHERE operational_campaign_id = target_campaign_id;
    SELECT count(*)::integer INTO request_count
    FROM campaign_operations_operational_request
    WHERE operational_campaign_id = target_campaign_id;
    binding_count := bound_count;
    SELECT count(*)::integer INTO control_owner_count
    FROM campaign_operations_downstream_control_owner owner
    JOIN campaign_operations_operational_request request
      ON request.operational_request_id = owner.operational_request_id
    WHERE request.operational_campaign_id = target_campaign_id;
    SELECT count(*)::integer INTO cancellation_request_count
    FROM campaign_operations_cancellation_request
    WHERE operational_campaign_id = target_campaign_id;
    SELECT count(*)::integer INTO cancellation_settlement_count
    FROM campaign_operations_cancellation_settlement settlement
    JOIN campaign_operations_cancellation_request cancellation
      ON cancellation.cancellation_request_id =
         settlement.cancellation_request_id
    WHERE cancellation.operational_campaign_id = target_campaign_id;
    unresolved_blocking_observation_count := 0;
    RETURN NEXT;
END;
$$;

-- The C++ identity contract hashes the UTF-8 bytes of canonical text with
-- unsigned FNV-1a 64-bit arithmetic and a fixed lower-case tagged rendering.
-- numeric arithmetic provides the exact modulo-2^64 behavior PostgreSQL
-- bigint overflow cannot provide.
CREATE OR REPLACE FUNCTION campaign_operations_tagged_fnv1a64(
    canonical_value text)
RETURNS text
LANGUAGE plpgsql
IMMUTABLE
STRICT
SECURITY DEFINER
AS $$
DECLARE canonical_bytes bytea := convert_to(canonical_value, 'UTF8');
DECLARE hash_value numeric := 14695981039346656037;
DECLARE byte_index integer;
DECLARE low_byte integer;
DECLARE high_word bigint;
DECLARE low_word bigint;
BEGIN
    IF octet_length(canonical_bytes) > 0 THEN
        FOR byte_index IN 0..octet_length(canonical_bytes) - 1 LOOP
            low_byte := mod(hash_value, 256)::integer #
                get_byte(canonical_bytes, byte_index);
            hash_value := hash_value - mod(hash_value, 256) + low_byte;
            hash_value := mod(
                hash_value * 1099511628211, 18446744073709551616);
        END LOOP;
    END IF;
    high_word := trunc(hash_value / 4294967296)::bigint;
    low_word := mod(hash_value, 4294967296)::bigint;
    RETURN 'fnv1a64:' || lpad(to_hex(high_word), 8, '0') ||
        lpad(to_hex(low_word), 8, '0');
END;
$$;

CREATE OR REPLACE FUNCTION campaign_operations_completion_identity_valid(
    candidate campaign_operations_completion_event)
RETURNS boolean
LANGUAGE plpgsql
IMMUTABLE
STRICT
SECURITY DEFINER
AS $$
DECLARE expected_canonical text;
BEGIN
    expected_canonical := 'campaign_operations_completion_v1' ||
        ';campaign=' || octet_length(candidate.campaign_identity_canonical) ||
            ':' || candidate.campaign_identity_canonical ||
        ';operation_key=' || octet_length(candidate.operation_key) || ':' ||
            candidate.operation_key ||
        ';terminal_state=' || candidate.administrative_terminal_state ||
        ';classification=' || candidate.completion_classification ||
        ';budget_ledger_entry_id=' || candidate.budget_ledger_entry_id ||
        ';budget_ledger_version=' || candidate.budget_ledger_version ||
        ';budget_resulting_total=' || candidate.budget_resulting_total ||
        ';budget_ever_reserved=' || candidate.budget_ever_reserved ||
        ';budget_committed=' || candidate.budget_committed ||
        ';budget_released_or_expired=' ||
            candidate.budget_released_or_expired ||
        ';budget_held=' || candidate.budget_held ||
        ';budget_unallocated=' || candidate.budget_unallocated ||
        ';scope_member_count=' || candidate.scope_member_count ||
        ';completed_member_count=' || candidate.completed_member_count ||
        ';failed_member_count=' || candidate.failed_member_count ||
        ';cancelled_or_never_dispatched_member_count=' ||
            candidate.cancelled_or_never_dispatched_member_count ||
        ';reservation_count=' || candidate.reservation_count ||
        ';request_count=' || candidate.request_count ||
        ';binding_count=' || candidate.binding_count ||
        ';control_owner_count=' || candidate.control_owner_count ||
        ';cancellation_request_count=' ||
            candidate.cancellation_request_count ||
        ';cancellation_settlement_count=' ||
            candidate.cancellation_settlement_count ||
        ';unresolved_blocking_observation_count=' ||
            candidate.unresolved_blocking_observation_count ||
        ';authorization_evidence=' ||
            octet_length(candidate.authorization_evidence_canonical) || ':' ||
            candidate.authorization_evidence_canonical ||
        ';budget_evidence=' ||
            octet_length(candidate.budget_evidence_canonical) || ':' ||
            candidate.budget_evidence_canonical ||
        ';reservation_evidence=' ||
            octet_length(candidate.reservation_evidence_canonical) || ':' ||
            candidate.reservation_evidence_canonical ||
        ';request_evidence=' ||
            octet_length(candidate.request_evidence_canonical) || ':' ||
            candidate.request_evidence_canonical ||
        ';binding_evidence=' ||
            octet_length(candidate.binding_evidence_canonical) || ':' ||
            candidate.binding_evidence_canonical ||
        ';lifecycle_evidence=' ||
            octet_length(candidate.lifecycle_evidence_canonical) || ':' ||
            candidate.lifecycle_evidence_canonical ||
        ';cancellation_evidence=' ||
            octet_length(candidate.cancellation_evidence_canonical) || ':' ||
            candidate.cancellation_evidence_canonical ||
        ';reconciliation_evidence=' ||
            octet_length(candidate.reconciliation_evidence_canonical) || ':' ||
            candidate.reconciliation_evidence_canonical ||
        ';actor=' || octet_length(candidate.actor_identity) || ':' ||
            candidate.actor_identity ||
        ';capability=campaign_operations_completion_writer' ||
        ';reason=' || octet_length(candidate.reason) || ':' ||
            candidate.reason;

    RETURN candidate.authorization_evidence_hash =
               campaign_operations_tagged_fnv1a64(
                   candidate.authorization_evidence_canonical) AND
           candidate.budget_evidence_hash =
               campaign_operations_tagged_fnv1a64(
                   candidate.budget_evidence_canonical) AND
           candidate.reservation_evidence_hash =
               campaign_operations_tagged_fnv1a64(
                   candidate.reservation_evidence_canonical) AND
           candidate.request_evidence_hash =
               campaign_operations_tagged_fnv1a64(
                   candidate.request_evidence_canonical) AND
           candidate.binding_evidence_hash =
               campaign_operations_tagged_fnv1a64(
                   candidate.binding_evidence_canonical) AND
           candidate.lifecycle_evidence_hash =
               campaign_operations_tagged_fnv1a64(
                   candidate.lifecycle_evidence_canonical) AND
           candidate.cancellation_evidence_hash =
               campaign_operations_tagged_fnv1a64(
                   candidate.cancellation_evidence_canonical) AND
           candidate.reconciliation_evidence_hash =
               campaign_operations_tagged_fnv1a64(
                   candidate.reconciliation_evidence_canonical) AND
           candidate.completion_identity_canonical = expected_canonical AND
           candidate.completion_identity_hash =
               campaign_operations_tagged_fnv1a64(expected_canonical);
END;
$$;

CREATE OR REPLACE FUNCTION enforce_campaign_operations_completion_event()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE classified record;
DECLARE budget record;
DECLARE boundary_closed boolean;
BEGIN
    SELECT completion_boundary_closed INTO boundary_closed
    FROM campaign_operations_campaign
    WHERE operational_campaign_id = NEW.operational_campaign_id
      AND campaign_identity_canonical = NEW.campaign_identity_canonical
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign operations completion campaign mismatch'
            USING ERRCODE = '23514';
    END IF;
    SELECT * INTO STRICT classified
    FROM campaign_operations_completion_classification(
        NEW.operational_campaign_id);
    SELECT
        head.budget_ledger_entry_id, head.ledger_version,
        head.resulting_total,
        COALESCE(sum(reservation.amount), 0)::bigint AS ever_reserved,
        COALESCE(sum(reservation.amount) FILTER (
            WHERE reservation.reservation_state = 'committed'), 0)::bigint
            AS committed,
        COALESCE(sum(reservation.amount) FILTER (
            WHERE reservation.reservation_state IN ('released', 'expired')),
            0)::bigint AS released_or_expired,
        COALESCE(sum(reservation.amount) FILTER (
            WHERE reservation.reservation_state = 'held'), 0)::bigint AS held
      INTO STRICT budget
    FROM LATERAL (
        SELECT * FROM campaign_operations_budget_ledger_entry
        WHERE operational_campaign_id = NEW.operational_campaign_id
        ORDER BY ledger_version DESC LIMIT 1) head
    LEFT JOIN campaign_operations_reservation reservation
      ON reservation.operational_campaign_id =
         NEW.operational_campaign_id
    GROUP BY head.budget_ledger_entry_id, head.ledger_version,
             head.resulting_total;
    IF NEW.administrative_terminal_state <>
            classified.administrative_terminal_state OR
       NEW.completion_classification <>
            classified.completion_classification OR
       NEW.scope_member_count <> classified.scope_member_count OR
       NEW.completed_member_count <>
            classified.completed_member_count OR
       NEW.failed_member_count <> classified.failed_member_count OR
       NEW.cancelled_or_never_dispatched_member_count <>
            classified.cancelled_or_never_dispatched_member_count OR
       NEW.reservation_count <> classified.reservation_count OR
       NEW.request_count <> classified.request_count OR
       NEW.binding_count <> classified.binding_count OR
       NEW.control_owner_count <> classified.control_owner_count OR
       NEW.cancellation_request_count <>
            classified.cancellation_request_count OR
       NEW.cancellation_settlement_count <>
            classified.cancellation_settlement_count OR
       NEW.unresolved_blocking_observation_count <> 0 OR
       NEW.budget_ledger_entry_id <> budget.budget_ledger_entry_id OR
       NEW.budget_ledger_version <> budget.ledger_version OR
       NEW.budget_resulting_total <> budget.resulting_total OR
       NEW.budget_ever_reserved <> budget.ever_reserved OR
       NEW.budget_committed <> budget.committed OR
       NEW.budget_released_or_expired <> budget.released_or_expired OR
       NEW.budget_held <> budget.held OR
       NEW.budget_unallocated <>
            budget.resulting_total - budget.committed - budget.held OR
       NEW.authorization_evidence_canonical <>
            campaign_operations_completion_evidence_text(
                NEW.operational_campaign_id, 'authorization') OR
       NEW.budget_evidence_canonical <>
            campaign_operations_completion_evidence_text(
                NEW.operational_campaign_id, 'budget') OR
       NEW.reservation_evidence_canonical <>
            campaign_operations_completion_evidence_text(
                NEW.operational_campaign_id, 'reservation') OR
       NEW.request_evidence_canonical <>
            campaign_operations_completion_evidence_text(
                NEW.operational_campaign_id, 'request') OR
       NEW.binding_evidence_canonical <>
            campaign_operations_completion_evidence_text(
                NEW.operational_campaign_id, 'binding') OR
       NEW.lifecycle_evidence_canonical <>
            campaign_operations_completion_evidence_text(
                NEW.operational_campaign_id, 'lifecycle') OR
       NEW.cancellation_evidence_canonical <>
            campaign_operations_completion_evidence_text(
                NEW.operational_campaign_id, 'cancellation') OR
       NEW.reconciliation_evidence_canonical <>
            campaign_operations_completion_evidence_text(
                NEW.operational_campaign_id, 'reconciliation') THEN
        RAISE EXCEPTION
            'campaign operations completion evidence changed or mismatched'
            USING ERRCODE = '40001';
    END IF;
    IF NOT campaign_operations_completion_identity_valid(NEW) THEN
        RAISE EXCEPTION
            'campaign operations completion canonical or hash mismatch'
            USING ERRCODE = '23514';
    END IF;
    IF boundary_closed THEN
        RAISE EXCEPTION
            'campaign operations completion already exists'
            USING ERRCODE = '23505';
    END IF;
    UPDATE campaign_operations_campaign
       SET completion_boundary_closed = true
     WHERE operational_campaign_id = NEW.operational_campaign_id;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
enforce_campaign_operations_completion_audit_complete()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    PERFORM 1
    FROM campaign_operations_campaign campaign
    WHERE campaign.operational_campaign_id = NEW.operational_campaign_id
      AND campaign.completion_boundary_closed;
    IF NOT FOUND THEN
        RAISE EXCEPTION
            'campaign operations completion boundary is not closed'
            USING ERRCODE = '23514';
    END IF;
    PERFORM 1
    FROM campaign_operations_completion_audit_reference_event audit
    WHERE audit.operational_campaign_id = NEW.operational_campaign_id
      AND audit.completion_event_id = NEW.completion_event_id
      AND audit.actor_identity = NEW.actor_identity
      AND audit.capability = NEW.capability
      AND audit.reason = NEW.reason
      AND audit.outcome = 'recorded'
      AND audit.replay_disposition = 'recorded'
      AND audit.diagnostic_code =
          'all_completion_prerequisites_proven';
    IF NOT FOUND THEN
        RAISE EXCEPTION
            'campaign operations completion audit reference missing'
            USING ERRCODE = '23514';
    END IF;
    RETURN NULL;
END;
$$;

-- completion_boundary_closed is only a serialization witness for the immutable
-- completion event.  It may transition false -> true only from the nested
-- completion INSERT trigger, and it can never be reopened.  This owner-level
-- guard prevents the mutex tuple from becoming an independent completion fact.
CREATE OR REPLACE FUNCTION
guard_campaign_operations_completion_boundary_mutation()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    IF TG_OP = 'TRUNCATE' THEN
        IF EXISTS (
            SELECT 1 FROM campaign_operations_campaign
            WHERE completion_boundary_closed) THEN
            RAISE EXCEPTION
                'campaign operations completion boundary is immutable'
                USING ERRCODE = '55000';
        END IF;
        RETURN NULL;
    END IF;
    IF TG_OP = 'DELETE' THEN
        IF OLD.completion_boundary_closed THEN
            RAISE EXCEPTION
                'campaign operations completion boundary is immutable'
                USING ERRCODE = '55000';
        END IF;
        RETURN OLD;
    END IF;
    IF OLD.completion_boundary_closed IS NOT DISTINCT FROM
            NEW.completion_boundary_closed THEN
        RETURN NEW;
    END IF;
    IF OLD.completion_boundary_closed OR
       NOT NEW.completion_boundary_closed OR
       pg_trigger_depth() <> 2 THEN
        RAISE EXCEPTION
            'campaign operations completion boundary is immutable'
            USING ERRCODE = '55000';
    END IF;
    RETURN NEW;
END;
$$;

-- The deferred check observes the final transaction state.  Together with the
-- immediate transition guard, it proves that the witness and the one immutable
-- completion fact commit or roll back together in both directions.
CREATE OR REPLACE FUNCTION
enforce_campaign_operations_completion_boundary_consistent()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE completion_exists boolean;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM campaign_operations_completion_event completion
        WHERE completion.operational_campaign_id =
              NEW.operational_campaign_id)
      INTO completion_exists;
    IF NEW.completion_boundary_closed <> completion_exists THEN
        RAISE EXCEPTION
            'campaign operations completion boundary/event mismatch'
            USING ERRCODE = '23514';
    END IF;
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION
reject_campaign_operations_completion_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION
        'campaign operations completion history is immutable'
        USING ERRCODE = '55000';
END;
$$;

CREATE OR REPLACE FUNCTION guard_campaign_operations_completed_campaign()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE target_campaign_id bigint;
DECLARE target_campaign_closed boolean;
BEGIN
    CASE TG_TABLE_NAME
    WHEN 'campaign_operations_reservation_event' THEN
        SELECT reservation.operational_campaign_id
          INTO STRICT target_campaign_id
        FROM campaign_operations_reservation reservation
        WHERE reservation.reservation_id = NEW.reservation_id;
    WHEN 'campaign_operations_dispatch_attempt' THEN
        SELECT request.operational_campaign_id INTO STRICT target_campaign_id
        FROM campaign_operations_operational_request request
        WHERE request.operational_request_id =
              NEW.operational_request_id;
    WHEN 'campaign_operations_dispatch_attempt_outcome' THEN
        SELECT request.operational_campaign_id
          INTO STRICT target_campaign_id
        FROM campaign_operations_dispatch_attempt attempt
        JOIN campaign_operations_operational_request request
          ON request.operational_request_id =
             attempt.operational_request_id
        WHERE attempt.dispatch_attempt_id = NEW.dispatch_attempt_id;
    WHEN 'campaign_operations_request_binding' THEN
        SELECT request.operational_campaign_id INTO STRICT target_campaign_id
        FROM campaign_operations_operational_request request
        WHERE request.operational_request_id =
              NEW.operational_request_id;
    WHEN 'campaign_operations_downstream_control_owner' THEN
        SELECT request.operational_campaign_id INTO STRICT target_campaign_id
        FROM campaign_operations_operational_request request
        WHERE request.operational_request_id =
              NEW.operational_request_id;
    WHEN 'campaign_operations_reservation_commitment' THEN
        SELECT reservation.operational_campaign_id
          INTO STRICT target_campaign_id
        FROM campaign_operations_reservation reservation
        WHERE reservation.reservation_id = NEW.reservation_id;
    WHEN 'campaign_operations_cancellation_settlement' THEN
        SELECT cancellation.operational_campaign_id
          INTO STRICT target_campaign_id
        FROM campaign_operations_cancellation_request cancellation
        WHERE cancellation.cancellation_request_id =
              NEW.cancellation_request_id;
    WHEN 'campaign_operations_reconciliation_resolution' THEN
        SELECT observation.operational_campaign_id
          INTO STRICT target_campaign_id
        FROM campaign_operations_reconciliation_observation observation
        WHERE observation.reconciliation_observation_id =
              NEW.reconciliation_observation_id;
    WHEN 'experiment_lifecycle_cancellation_event' THEN
        SELECT cancellation.operational_campaign_id
          INTO STRICT target_campaign_id
        FROM campaign_operations_cancellation_request cancellation
        WHERE cancellation.cancellation_request_id =
              NEW.cancellation_request_id;
    ELSE
        target_campaign_id := NEW.operational_campaign_id;
    END CASE;
    -- Serialize the terminal boundary itself.  A check without this lock can
    -- miss an uncommitted completion and allow a child insert to resume after
    -- the completion commits.  Owning workflows already take this campaign
    -- lock before reservation/request locks; the reconciliation observation
    -- repository follows the same order after Phase G.
    SELECT completion_boundary_closed INTO target_campaign_closed
    FROM campaign_operations_campaign
    WHERE operational_campaign_id = target_campaign_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign operations completion campaign missing'
            USING ERRCODE = '23503';
    END IF;
    IF target_campaign_closed THEN
        RAISE EXCEPTION
            'campaign operations completed campaign is immutable'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
campaign_operations_lifecycle_cancellation_boundary_completed(
    target_campaign_id bigint)
RETURNS boolean
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
AS $$
DECLARE campaign_completed boolean;
BEGIN
    SELECT completion_boundary_closed INTO campaign_completed
    FROM campaign_operations_campaign
    WHERE operational_campaign_id = target_campaign_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'lifecycle cancellation campaign missing'
            USING ERRCODE = '23503';
    END IF;
    RETURN campaign_completed;
END;
$$;

-- Phase F deliberately invokes lifecycle-owned cancellation only after
-- releasing Campaign Operations locks.  The lifecycle transition therefore
-- acquires the campaign boundary before its experiment row, matching
-- completion's campaign-before-evidence order without introducing an inverse
-- experiment-to-campaign wait.  Ordinary lifecycle retry/requeue does not call
-- this Campaign Operations-requested transition and remains unaffected.
CREATE OR REPLACE FUNCTION apply_experiment_lifecycle_cancellation(
    target_cancellation_request_id bigint,
    target_control_owner_id bigint,
    target_experiment_id bigint,
    expected_status_value text,
    expected_phase_value text,
    actor_value text,
    event_canonical_value text,
    event_hash_value text)
RETURNS experiment_lifecycle_cancellation_event
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    target_campaign_id bigint;
    campaign_completed boolean;
    current_experiment experiment%ROWTYPE;
    disposition_value text;
    resulting_status_value text;
    result experiment_lifecycle_cancellation_event%ROWTYPE;
BEGIN
    SELECT cancellation.operational_campaign_id
      INTO STRICT target_campaign_id
    FROM campaign_operations_cancellation_request cancellation
    JOIN campaign_operations_downstream_control_owner control_owner
      ON control_owner.operational_request_id =
         cancellation.operational_request_id
    WHERE cancellation.cancellation_request_id =
              target_cancellation_request_id
      AND cancellation.actor_identity = actor_value
      AND control_owner.downstream_control_owner_id =
              target_control_owner_id
      AND control_owner.experiment_id = target_experiment_id;

    campaign_completed :=
        campaign_operations_lifecycle_cancellation_boundary_completed(
            target_campaign_id);

    SELECT * INTO result
    FROM experiment_lifecycle_cancellation_event
    WHERE cancellation_request_id = target_cancellation_request_id
      AND downstream_control_owner_id = target_control_owner_id;
    IF result.lifecycle_cancellation_event_id IS NOT NULL THEN
        IF result.event_identity_canonical <> event_canonical_value OR
           result.event_identity_hash <> event_hash_value THEN
            RAISE EXCEPTION 'lifecycle cancellation replay conflicts'
                USING ERRCODE = '23514';
        END IF;
        RETURN result;
    END IF;

    IF campaign_completed THEN
        RAISE EXCEPTION
            'campaign operations completed campaign is immutable'
            USING ERRCODE = '23514';
    END IF;

    SELECT * INTO STRICT current_experiment
    FROM experiment WHERE experiment_id = target_experiment_id FOR UPDATE;
    IF current_experiment.status <> expected_status_value OR
       current_experiment.phase <> expected_phase_value THEN
        RAISE EXCEPTION 'experiment lifecycle changed during cancellation'
            USING ERRCODE = '40001';
    END IF;
    IF current_experiment.status IN ('pending', 'paused') THEN
        disposition_value := 'accepted';
        resulting_status_value := 'cancelled';
        UPDATE experiment
           SET status = 'cancelled', completed_at = transaction_timestamp(),
               updated_at = transaction_timestamp()
         WHERE experiment_id = target_experiment_id;
    ELSIF current_experiment.status IN (
            'completed', 'failed', 'cancelled') THEN
        disposition_value := 'already_terminal';
        resulting_status_value := current_experiment.status;
    ELSIF current_experiment.status = 'running' THEN
        disposition_value := 'running_not_supported';
        resulting_status_value := current_experiment.status;
    ELSE
        RAISE EXCEPTION 'unsupported experiment lifecycle state'
            USING ERRCODE = '23514';
    END IF;
    INSERT INTO experiment_lifecycle_cancellation_event(
        cancellation_request_id,
        cancellation_request_identity_canonical,
        downstream_control_owner_id, experiment_id,
        expected_status, expected_phase, observed_status, observed_phase,
        resulting_status, disposition, actor_identity,
        event_contract_version, event_identity_canonical,
        event_identity_hash)
    SELECT target_cancellation_request_id,
           cancellation_identity_canonical,
           target_control_owner_id, target_experiment_id,
           expected_status_value, expected_phase_value,
           current_experiment.status, current_experiment.phase,
           resulting_status_value, disposition_value, actor_value,
           1, event_canonical_value, event_hash_value
    FROM campaign_operations_cancellation_request
    WHERE cancellation_request_id = target_cancellation_request_id
    RETURNING * INTO result;
    RETURN result;
EXCEPTION WHEN unique_violation THEN
    SELECT * INTO STRICT result
    FROM experiment_lifecycle_cancellation_event
    WHERE cancellation_request_id = target_cancellation_request_id
      AND downstream_control_owner_id = target_control_owner_id;
    IF result.event_identity_canonical <> event_canonical_value OR
       result.event_identity_hash <> event_hash_value THEN
        RAISE EXCEPTION 'lifecycle cancellation replay conflicts'
            USING ERRCODE = '23514';
    END IF;
    RETURN result;
END;
$$;

-- Preserve the accepted Phase F control predicate while adding the Phase G
-- terminal boundary.  Existing repositories consult this function before
-- accepting or dispatching new work; row triggers below provide independent
-- database enforcement for every other authoritative mutation path.
CREATE OR REPLACE FUNCTION
campaign_operations_future_actions_allowed(target_campaign_id bigint)
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
AS $$
    SELECT NOT EXISTS(
        SELECT 1 FROM campaign_operations_cancellation_request
        WHERE operational_campaign_id = target_campaign_id)
    AND NOT COALESCE((
        SELECT event_kind = 'pause'
        FROM campaign_operations_control_event
        WHERE operational_campaign_id = target_campaign_id
        ORDER BY control_version DESC
        LIMIT 1), false)
    AND NOT EXISTS(
        SELECT 1 FROM campaign_operations_completion_event
        WHERE operational_campaign_id = target_campaign_id);
$$;

DO $$
DECLARE completion_schema text := current_schema();
DECLARE function_name text;
BEGIN
    FOREACH function_name IN ARRAY ARRAY[
        'campaign_operations_completion_evidence_text(bigint,text)',
        'campaign_operations_completion_blockers(bigint)',
        'campaign_operations_completion_classification(bigint)',
        'campaign_operations_tagged_fnv1a64(text)',
        'campaign_operations_completion_identity_valid(campaign_operations_completion_event)',
        'enforce_campaign_operations_completion_event()',
        'enforce_campaign_operations_completion_audit_complete()',
        'guard_campaign_operations_completion_boundary_mutation()',
        'enforce_campaign_operations_completion_boundary_consistent()',
        'reject_campaign_operations_completion_mutation()',
        'guard_campaign_operations_completed_campaign()',
        'campaign_operations_lifecycle_cancellation_boundary_completed(bigint)',
        'apply_experiment_lifecycle_cancellation(bigint,bigint,bigint,text,text,text,text,text)',
        'campaign_operations_future_actions_allowed(bigint)']
    LOOP
        EXECUTE format(
            'ALTER FUNCTION %I.%s SET search_path TO pg_catalog, %I, pg_temp',
            completion_schema, function_name, completion_schema);
    END LOOP;
END $$;

DROP TRIGGER IF EXISTS campaign_operations_completion_boundary_update_guard
    ON campaign_operations_campaign;
CREATE TRIGGER campaign_operations_completion_boundary_update_guard
BEFORE UPDATE OF completion_boundary_closed OR DELETE
ON campaign_operations_campaign
FOR EACH ROW EXECUTE FUNCTION
    guard_campaign_operations_completion_boundary_mutation();

DROP TRIGGER IF EXISTS campaign_operations_completion_boundary_truncate_guard
    ON campaign_operations_campaign;
CREATE TRIGGER campaign_operations_completion_boundary_truncate_guard
BEFORE TRUNCATE ON campaign_operations_campaign
FOR EACH STATEMENT EXECUTE FUNCTION
    guard_campaign_operations_completion_boundary_mutation();

DROP TRIGGER IF EXISTS
    campaign_operations_completion_boundary_consistency_trigger
    ON campaign_operations_campaign;
CREATE CONSTRAINT TRIGGER
campaign_operations_completion_boundary_consistency_trigger
AFTER INSERT OR UPDATE ON campaign_operations_campaign
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_completion_boundary_consistent();

DROP TRIGGER IF EXISTS campaign_operations_completion_validate_trigger
    ON campaign_operations_completion_event;
CREATE TRIGGER campaign_operations_completion_validate_trigger
BEFORE INSERT ON campaign_operations_completion_event
FOR EACH ROW EXECUTE FUNCTION enforce_campaign_operations_completion_event();

DROP TRIGGER IF EXISTS campaign_operations_completion_audit_complete_trigger
    ON campaign_operations_completion_event;
CREATE CONSTRAINT TRIGGER
campaign_operations_completion_audit_complete_trigger
AFTER INSERT ON campaign_operations_completion_event
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_completion_audit_complete();

DROP TRIGGER IF EXISTS campaign_operations_completion_immutable_trigger
    ON campaign_operations_completion_event;
CREATE TRIGGER campaign_operations_completion_immutable_trigger
BEFORE UPDATE OR DELETE ON campaign_operations_completion_event
FOR EACH ROW EXECUTE FUNCTION
    reject_campaign_operations_completion_mutation();
DROP TRIGGER IF EXISTS campaign_operations_completion_truncate_trigger
    ON campaign_operations_completion_event;
CREATE TRIGGER campaign_operations_completion_truncate_trigger
BEFORE TRUNCATE ON campaign_operations_completion_event
FOR EACH STATEMENT EXECUTE FUNCTION
    reject_campaign_operations_completion_mutation();
DROP TRIGGER IF EXISTS
    campaign_operations_completion_audit_immutable_trigger
    ON campaign_operations_completion_audit_reference_event;
CREATE TRIGGER campaign_operations_completion_audit_immutable_trigger
BEFORE UPDATE OR DELETE
ON campaign_operations_completion_audit_reference_event
FOR EACH ROW EXECUTE FUNCTION
    reject_campaign_operations_completion_mutation();
DROP TRIGGER IF EXISTS campaign_operations_completion_audit_truncate_trigger
    ON campaign_operations_completion_audit_reference_event;
CREATE TRIGGER campaign_operations_completion_audit_truncate_trigger
BEFORE TRUNCATE ON campaign_operations_completion_audit_reference_event
FOR EACH STATEMENT EXECUTE FUNCTION
    reject_campaign_operations_completion_mutation();

DO $$
DECLARE table_name text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'campaign_operations_governance_provenance_event',
        'campaign_operations_authorization_event',
        'campaign_operations_budget_ledger_entry',
        'campaign_operations_reservation',
        'campaign_operations_operational_request',
        'campaign_operations_reservation_event',
        'campaign_operations_dispatch_attempt',
        'campaign_operations_dispatch_attempt_outcome',
        'campaign_operations_request_binding',
        'campaign_operations_downstream_control_owner',
        'campaign_operations_reservation_commitment',
        'campaign_operations_control_event',
        'campaign_operations_cancellation_request',
        'campaign_operations_cancellation_settlement',
        'campaign_operations_reconciliation_observation',
        'campaign_operations_reconciliation_resolution',
        'campaign_operations_audit_reference_event',
        'campaign_operations_dispatch_audit_reference_event',
        'campaign_operations_control_audit_reference_event',
        'experiment_lifecycle_cancellation_event'
    ]
    LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS %I ON %I',
            table_name || '_completion_gate_trigger', table_name);
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE INSERT ON %I FOR EACH ROW '
            'EXECUTE FUNCTION guard_campaign_operations_completed_campaign()',
            table_name || '_completion_gate_trigger', table_name);
    END LOOP;
END $$;

DROP TRIGGER IF EXISTS campaign_operations_reservation_completion_update_gate
    ON campaign_operations_reservation;
CREATE TRIGGER campaign_operations_reservation_completion_update_gate
BEFORE UPDATE ON campaign_operations_reservation
FOR EACH ROW EXECUTE FUNCTION
    guard_campaign_operations_completed_campaign();

DROP TRIGGER IF EXISTS campaign_operations_request_completion_update_gate
    ON campaign_operations_operational_request;
CREATE TRIGGER campaign_operations_request_completion_update_gate
BEFORE UPDATE ON campaign_operations_operational_request
FOR EACH ROW EXECUTE FUNCTION
    guard_campaign_operations_completed_campaign();

CREATE OR REPLACE VIEW campaign_operations_completion_status_v1 AS
SELECT
    campaign.operational_campaign_id,
    completion.completion_event_id IS NOT NULL AS completion_recorded,
    completion.completion_event_id IS NOT NULL AS logically_archived,
    completion.completion_event_id,
    completion.recorded_at,
    completion.administrative_terminal_state AS recorded_terminal_state,
    completion.completion_classification,
    completion.completion_identity_hash,
    completion.lifecycle_evidence_canonical AS recorded_lifecycle_evidence,
    campaign_operations_completion_evidence_text(
        campaign.operational_campaign_id, 'lifecycle')
        AS current_lifecycle_evidence,
    completion.completion_event_id IS NOT NULL AND
        completion.lifecycle_evidence_canonical <>
            campaign_operations_completion_evidence_text(
                campaign.operational_campaign_id, 'lifecycle')
        AS post_completion_lifecycle_changed,
    (SELECT count(*) FROM campaign_operations_cancellation_request request
     WHERE request.operational_campaign_id =
           campaign.operational_campaign_id) AS cancellation_request_count,
    (SELECT count(*)
     FROM campaign_operations_reconciliation_observation observation
     LEFT JOIN campaign_operations_reconciliation_resolution resolution
       ON resolution.reconciliation_observation_id =
          observation.reconciliation_observation_id
     WHERE observation.operational_campaign_id =
           campaign.operational_campaign_id
       AND resolution.reconciliation_resolution_id IS NULL
       AND observation.reason_code NOT IN (
           'terminal_lifecycle_completion_ready',
           'post_completion_lifecycle_changed'))
        AS unresolved_blocking_observation_count
FROM campaign_operations_campaign campaign
LEFT JOIN campaign_operations_completion_event completion
  ON completion.operational_campaign_id =
     campaign.operational_campaign_id;

REVOKE ALL PRIVILEGES ON campaign_operations_completion_event,
    campaign_operations_completion_audit_reference_event
    FROM PUBLIC, pqxx,
         campaign_operations_campaign_creator,
         campaign_operations_authorizer,
         campaign_operations_budget_administrator,
         campaign_operations_request_acceptor,
         campaign_operations_dispatcher,
         campaign_operations_phase5_transactional,
         campaign_operations_controller,
         campaign_operations_cancellation_coordinator,
         campaign_operations_reconciler,
         campaign_operations_recovery,
         campaign_operations_reader;
REVOKE ALL PRIVILEGES ON campaign_operations_completion_status_v1
    FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    campaign_operations_completion_evidence_text(bigint, text),
    campaign_operations_completion_blockers(bigint),
    campaign_operations_completion_classification(bigint),
    campaign_operations_tagged_fnv1a64(text),
    campaign_operations_completion_identity_valid(
        campaign_operations_completion_event),
    enforce_campaign_operations_completion_event(),
    enforce_campaign_operations_completion_audit_complete(),
    guard_campaign_operations_completion_boundary_mutation(),
    enforce_campaign_operations_completion_boundary_consistent(),
    reject_campaign_operations_completion_mutation(),
    guard_campaign_operations_completed_campaign()
    FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    campaign_operations_lifecycle_cancellation_boundary_completed(bigint)
    FROM PUBLIC, pqxx,
         campaign_operations_campaign_creator,
         campaign_operations_authorizer,
         campaign_operations_budget_administrator,
         campaign_operations_request_acceptor,
         campaign_operations_dispatcher,
         campaign_operations_phase5_transactional,
         campaign_operations_controller,
         campaign_operations_cancellation_coordinator,
         campaign_operations_reconciler,
         campaign_operations_recovery,
         campaign_operations_reader,
         campaign_operations_auditor,
         campaign_operations_completion_writer,
         experiment_lifecycle_cancellation;

GRANT SELECT ON campaign_operations_campaign,
    campaign_operations_authorization_event,
    campaign_operations_budget_ledger_entry,
    campaign_operations_reservation,
    campaign_operations_reservation_event,
    campaign_operations_operational_request,
    campaign_operations_dispatch_attempt,
    campaign_operations_dispatch_attempt_outcome,
    campaign_operations_request_binding,
    campaign_operations_downstream_control_owner,
    campaign_operations_control_event,
    campaign_operations_cancellation_request,
    campaign_operations_cancellation_settlement,
    campaign_operations_reconciliation_observation,
    campaign_operations_reconciliation_resolution,
    campaign_operations_completion_event,
    campaign_operations_completion_audit_reference_event
    TO campaign_operations_completion_writer;
-- Hydrating the immutable operational campaign revalidates its exact
-- materialization binding.  These are the only recommendation-owned rows the
-- completion writer may read; it receives no recommendation mutation right.
GRANT SELECT ON experiment_recommendation_campaign_materialization,
    experiment_recommendation_campaign_materialization_member
    TO campaign_operations_completion_writer, campaign_operations_reader;
-- Lifecycle is a separate, read-only display/evidence dimension.  The
-- SECURITY DEFINER evaluators need only these four columns and receive no
-- experiment mutation privilege.
GRANT SELECT (experiment_id, status, phase, updated_at)
    ON experiment TO campaign_operations_owner;
GRANT INSERT (
    operational_campaign_id, campaign_identity_canonical, operation_key,
    administrative_terminal_state, completion_classification,
    budget_ledger_entry_id, budget_ledger_version, budget_resulting_total,
    budget_ever_reserved, budget_committed, budget_released_or_expired,
    budget_held, budget_unallocated, scope_member_count,
    completed_member_count, failed_member_count,
    cancelled_or_never_dispatched_member_count, reservation_count,
    request_count, binding_count, control_owner_count,
    cancellation_request_count, cancellation_settlement_count,
    unresolved_blocking_observation_count,
    authorization_evidence_canonical, authorization_evidence_hash,
    budget_evidence_canonical,
    budget_evidence_hash, reservation_evidence_canonical,
    reservation_evidence_hash, request_evidence_canonical,
    request_evidence_hash, binding_evidence_canonical,
    binding_evidence_hash, lifecycle_evidence_canonical,
    lifecycle_evidence_hash, cancellation_evidence_canonical,
    cancellation_evidence_hash, reconciliation_evidence_canonical,
    reconciliation_evidence_hash, actor_identity, capability, reason,
    completion_contract_version, completion_identity_canonical,
    completion_identity_hash)
    ON campaign_operations_completion_event
    TO campaign_operations_completion_writer;
GRANT INSERT (
    operational_campaign_id, completion_event_id, actor_identity,
    capability, reason, outcome, replay_disposition, diagnostic_code)
    ON campaign_operations_completion_audit_reference_event
    TO campaign_operations_completion_writer;
GRANT EXECUTE ON FUNCTION
    campaign_operations_completion_evidence_text(bigint, text),
    campaign_operations_completion_blockers(bigint),
    campaign_operations_completion_classification(bigint),
    lock_campaign_operations_authorization_head(bigint, text),
    lock_campaign_operations_budget_head(bigint),
    lock_campaign_operations_campaign(bigint),
    lock_campaign_operations_reservation(bigint),
    lock_campaign_operations_request(bigint)
    TO campaign_operations_completion_writer;
GRANT EXECUTE ON FUNCTION
    campaign_operations_lifecycle_cancellation_boundary_completed(bigint)
    TO experiment_lifecycle_cancellation_owner;

GRANT SELECT ON campaign_operations_completion_event,
    campaign_operations_completion_audit_reference_event,
    campaign_operations_completion_status_v1
    TO campaign_operations_reader, campaign_operations_auditor;
GRANT EXECUTE ON FUNCTION
    campaign_operations_completion_evidence_text(bigint, text),
    campaign_operations_completion_blockers(bigint),
    campaign_operations_completion_classification(bigint)
    TO campaign_operations_reader, campaign_operations_auditor;

DO $$
DECLARE sequence_name regclass;
BEGIN
    FOREACH sequence_name IN ARRAY ARRAY[
        pg_get_serial_sequence(
            'campaign_operations_completion_event',
            'completion_event_id')::regclass,
        pg_get_serial_sequence(
            'campaign_operations_completion_audit_reference_event',
            'completion_audit_reference_event_id')::regclass]
    LOOP
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM PUBLIC, pqxx',
            sequence_name);
        EXECUTE format(
            'GRANT USAGE ON SEQUENCE %s TO '
            'campaign_operations_completion_writer', sequence_name);
    END LOOP;
END $$;

ALTER TABLE campaign_operations_completion_event
    OWNER TO campaign_operations_owner;
ALTER TABLE campaign_operations_completion_audit_reference_event
    OWNER TO campaign_operations_owner;
DO $$
DECLARE sequence_name regclass;
BEGIN
    FOREACH sequence_name IN ARRAY ARRAY[
        pg_get_serial_sequence(
            'campaign_operations_completion_event',
            'completion_event_id')::regclass,
        pg_get_serial_sequence(
            'campaign_operations_completion_audit_reference_event',
            'completion_audit_reference_event_id')::regclass]
    LOOP
        EXECUTE format(
            'ALTER SEQUENCE %s OWNER TO campaign_operations_owner',
            sequence_name);
    END LOOP;
END $$;
ALTER VIEW campaign_operations_completion_status_v1
    OWNER TO campaign_operations_owner;
ALTER FUNCTION campaign_operations_completion_evidence_text(bigint, text)
    OWNER TO campaign_operations_owner;
ALTER FUNCTION campaign_operations_completion_blockers(bigint)
    OWNER TO campaign_operations_owner;
ALTER FUNCTION campaign_operations_completion_classification(bigint)
    OWNER TO campaign_operations_owner;
ALTER FUNCTION campaign_operations_tagged_fnv1a64(text)
    OWNER TO campaign_operations_owner;
ALTER FUNCTION campaign_operations_completion_identity_valid(
    campaign_operations_completion_event)
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_completion_event()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_completion_audit_complete()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION guard_campaign_operations_completion_boundary_mutation()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_completion_boundary_consistent()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION reject_campaign_operations_completion_mutation()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION guard_campaign_operations_completed_campaign()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION
    campaign_operations_lifecycle_cancellation_boundary_completed(bigint)
    OWNER TO campaign_operations_owner;

DO $$
DECLARE completion_schema text;
BEGIN
    SELECT n.nspname INTO STRICT completion_schema
    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE c.oid = 'campaign_operations_completion_event'::regclass;
    EXECUTE format(
        'GRANT USAGE ON SCHEMA %I TO '
        'campaign_operations_completion_writer, '
        'campaign_operations_reader, campaign_operations_auditor',
        completion_schema);
END $$;
