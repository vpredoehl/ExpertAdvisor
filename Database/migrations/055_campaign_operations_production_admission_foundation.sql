-- Campaign Operations Phase H1: production-admission authority and immutable
-- evidence foundation.  This migration is deliberately operationally inert:
-- it grants no LOGIN membership and exposes no enable, disable, acquisition,
-- handoff, Manager, scheduler-mutation, or lifecycle-mutation command.
-- H1_MANIFEST_DIGEST_SHA256: 5e4234e929be2f9e844bd491fc913f1ed451da21a510d67da14dabd50b32f4bf

DO $$
BEGIN
    IF to_regclass('campaign_operations_completion_event') IS NULL OR
       to_regprocedure('campaign_operations_tagged_fnv1a64(text)') IS NULL OR
       to_regclass('experiment_scheduler_protocol') IS NULL THEN
        RAISE EXCEPTION 'H1A011 migration 055 requires migrations 052 and 054'
            USING ERRCODE = '55000';
    END IF;
    IF octet_length(
           'migration-052-scheduler-generation-52-exact-attempt-authority') <>
           61 OR
       octet_length(
           'scheduler-generation-52-exact-attempt-authority-v1') <> 50 THEN
        RAISE EXCEPTION 'H1A009 Phase H scheduler canonical literal length mismatch'
            USING ERRCODE = '55000';
    END IF;
    IF EXISTS (SELECT 1 FROM campaign_operations_operational_request
               WHERE production_dispatch_enabled) OR
       EXISTS (SELECT 1 FROM campaign_operations_dispatch_attempt
               WHERE attempt_contract_version <> 1) THEN
        RAISE EXCEPTION 'H1A011 migration 055 unsupported production upgrade state'
            USING ERRCODE = '55000';
    END IF;
END $$;

DO $$
DECLARE
    boundary_oid oid;
    offending_entry text;
    allowed_relations text[] := ARRAY[
        'campaign_operations_production_transition_context',
        'campaign_operations_production_enablement_event',
        'campaign_operations_production_enablement_audit_reference_event',
        'campaign_operations_request_production_admission',
        'campaign_operations_operational_request',
        'campaign_operations_dispatch_attempt',
        'campaign_operations_dispatch_audit_reference_event',
        'campaign_operations_completion_event',
        'campaign_operations_completion_audit_reference_event',
        'campaign_operations_production_readiness_v1',
        'campaign_operations_production_status_v1'];
    protected_functions text[] := ARRAY[
        'campaign_operations_scheduler_protocol_evidence_snapshot_v1',
        'campaign_operations_scheduler_protocol_evidence_lock_v1',
        'record_campaign_operations_production_enable_v1',
        'record_campaign_operations_production_disable_v1',
        'transition_campaign_operations_request_dispatch_production_v2',
        'campaign_operations_production_context_valid_v1',
        'enforce_campaign_operations_production_context_empty_v1',
        'campaign_operations_production_enable_replay_v1',
        'campaign_operations_production_disable_replay_v1',
        'campaign_operations_production_acquire_replay_v2',
        'campaign_operations_production_enablement_history_valid_v1',
        'campaign_operations_manager_build_canonical_v1',
        'campaign_operations_scheduler_protocol_evidence_canonical_v1',
        'campaign_operations_production_enablement_canonical_v1',
        'campaign_operations_request_production_admission_canonical_v1',
        'campaign_operations_dispatch_attempt_v2_canonical',
        'campaign_operations_has_explicit_role_v1',
        'campaign_operations_isolated_v1_authority_valid',
        'transition_campaign_operations_request_dispatching',
        'guard_campaign_operations_attempt_v1_isolation',
        'validate_campaign_operations_production_enablement_insert',
        'validate_campaign_ops_production_admission_insert',
        'validate_campaign_operations_dispatch_attempt_v2_insert',
        'guard_campaign_operations_production_admission_witness',
        'enforce_campaign_operations_production_admission_consistent',
        'enforce_campaign_ops_enablement_audit_complete',
        'reject_campaign_operations_production_mutation',
        'guard_campaign_operations_production_attempt_mutation',
        'guard_campaign_operations_production_dispatch_audit_mutation',
        'guard_campaign_operations_production_dispatch_audit_insert',
        'guard_campaign_operations_production_post_completion',
        'guard_campaign_operations_completion_v2_evidence',
        'enforce_campaign_operations_request',
        'enforce_campaign_operations_request_acquisition_complete',
        'enforce_campaign_operations_dispatch_acquisition_complete',
        'enforce_campaign_operations_complete_binding',
        'guard_campaign_operations_dispatch_control',
        'enforce_campaign_operations_phase4_request_transition_complete',
        'guard_campaign_operations_completed_campaign',
        'enforce_campaign_operations_completion_event',
        'enforce_campaign_operations_completion_audit_complete',
        'reject_campaign_operations_completion_mutation',
        'campaign_operations_tagged_fnv1a64',
        'lock_campaign_operations_authorization_head',
        'lock_campaign_operations_budget_head',
        'lock_campaign_operations_campaign',
        'lock_campaign_operations_reservation',
        'lock_campaign_operations_request',
        'campaign_operations_h1_deployment_audit_v1'];
    allowed_function_signatures text[] := ARRAY[
        'public.campaign_operations_dispatch_attempt_v2_canonical(campaign_operations_dispatch_attempt)',
        'public.campaign_operations_h1_deployment_audit_v1(text,boolean,boolean)',
        'public.campaign_operations_has_explicit_role_v1(name,name)',
        'public.campaign_operations_isolated_v1_authority_valid()',
        'public.campaign_operations_manager_build_canonical_v1(text,text,text,text)',
        'public.campaign_operations_production_acquire_replay_v2(bigint,integer,text,timestamp with time zone,text,text,text)',
        'public.campaign_operations_production_context_valid_v1(text,bigint,text)',
        'public.campaign_operations_production_disable_replay_v1(text,bigint,text,integer,text,text)',
        'public.campaign_operations_production_enable_replay_v1(text,integer,text,text,text,text,text,text,text,text,text)',
        'public.campaign_operations_production_enablement_history_valid_v1(bigint)',
        'public.campaign_operations_production_enablement_canonical_v1(campaign_operations_production_enablement_event)',
        'public.campaign_operations_request_production_admission_canonical_v1(campaign_operations_request_production_admission)',
        'public.campaign_operations_scheduler_protocol_evidence_canonical_v1(integer,text,timestamp with time zone,text,text,text)',
        'public.campaign_operations_scheduler_protocol_evidence_lock_v1()',
        'public.campaign_operations_scheduler_protocol_evidence_snapshot_v1()',
        'public.campaign_operations_tagged_fnv1a64(text)',
        'public.enforce_campaign_operations_complete_binding()',
        'public.enforce_campaign_operations_completion_audit_complete()',
        'public.enforce_campaign_operations_completion_event()',
        'public.enforce_campaign_operations_dispatch_acquisition_complete()',
        'public.enforce_campaign_operations_phase4_request_transition_complete()',
        'public.enforce_campaign_operations_production_admission_consistent()',
        'public.enforce_campaign_operations_production_context_empty_v1()',
        'public.enforce_campaign_operations_request()',
        'public.enforce_campaign_operations_request_acquisition_complete()',
        'public.enforce_campaign_ops_enablement_audit_complete()',
        'public.guard_campaign_operations_attempt_v1_isolation()',
        'public.guard_campaign_operations_completed_campaign()',
        'public.guard_campaign_operations_completion_v2_evidence()',
        'public.guard_campaign_operations_dispatch_control()',
        'public.guard_campaign_operations_production_admission_witness()',
        'public.guard_campaign_operations_production_attempt_mutation()',
        'public.guard_campaign_operations_production_dispatch_audit_insert()',
        'public.guard_campaign_operations_production_dispatch_audit_mutation()',
        'public.guard_campaign_operations_production_post_completion()',
        'public.lock_campaign_operations_authorization_head(bigint,text)',
        'public.lock_campaign_operations_budget_head(bigint)',
        'public.lock_campaign_operations_campaign(bigint)',
        'public.lock_campaign_operations_request(bigint)',
        'public.lock_campaign_operations_reservation(bigint)',
        'public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)',
        'public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)',
        'public.reject_campaign_operations_completion_mutation()',
        'public.reject_campaign_operations_production_mutation()',
        'public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)',
        'public.transition_campaign_operations_request_dispatching(bigint,integer,text,text)',
        'public.validate_campaign_operations_dispatch_attempt_v2_insert()',
        'public.validate_campaign_operations_production_enablement_insert()',
        'public.validate_campaign_ops_production_admission_insert()'];
    allowed_function_contracts text[] := ARRAY[
        'public.campaign_operations_dispatch_attempt_v2_canonical(campaign_operations_dispatch_attempt)|sql|false|i|u|0|0|NULL',
        'public.campaign_operations_h1_deployment_audit_v1(text,boolean,boolean)|plpgsql|false|v|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_has_explicit_role_v1(name,name)|sql|true|s|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_isolated_v1_authority_valid()|sql|true|s|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_manager_build_canonical_v1(text,text,text,text)|plpgsql|false|i|u|0|0|NULL',
        'public.campaign_operations_production_acquire_replay_v2(bigint,integer,text,timestamp with time zone,text,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_production_context_valid_v1(text,bigint,text)|sql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_production_disable_replay_v1(text,bigint,text,integer,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_production_enable_replay_v1(text,integer,text,text,text,text,text,text,text,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_production_enablement_history_valid_v1(bigint)|plpgsql|true|s|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_production_enablement_canonical_v1(campaign_operations_production_enablement_event)|plpgsql|false|i|u|0|0|NULL',
        'public.campaign_operations_request_production_admission_canonical_v1(campaign_operations_request_production_admission)|sql|false|i|u|0|0|NULL',
        'public.campaign_operations_scheduler_protocol_evidence_canonical_v1(integer,text,timestamp with time zone,text,text,text)|plpgsql|false|i|u|0|0|NULL',
        'public.campaign_operations_scheduler_protocol_evidence_lock_v1()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_scheduler_protocol_evidence_snapshot_v1()|plpgsql|true|s|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_tagged_fnv1a64(text)|plpgsql|true|i|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_complete_binding()|plpgsql|false|v|u|0|0|search_path=pg_catalog, public, pg_temp',
        'public.enforce_campaign_operations_completion_audit_complete()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_completion_event()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_dispatch_acquisition_complete()|plpgsql|false|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_phase4_request_transition_complete()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_production_admission_consistent()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_production_context_empty_v1()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_request()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_request_acquisition_complete()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_ops_enablement_audit_complete()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_attempt_v1_isolation()|plpgsql|false|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_completed_campaign()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_completion_v2_evidence()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_dispatch_control()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_production_admission_witness()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_production_attempt_mutation()|plpgsql|false|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_production_dispatch_audit_insert()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_production_dispatch_audit_mutation()|plpgsql|false|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_production_post_completion()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.lock_campaign_operations_authorization_head(bigint,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.lock_campaign_operations_budget_head(bigint)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.lock_campaign_operations_campaign(bigint)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.lock_campaign_operations_request(bigint)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.lock_campaign_operations_reservation(bigint)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.reject_campaign_operations_completion_mutation()|plpgsql|false|v|u|0|0|search_path=pg_catalog, public, pg_temp',
        'public.reject_campaign_operations_production_mutation()|plpgsql|false|v|u|0|0|NULL',
        'public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.transition_campaign_operations_request_dispatching(bigint,integer,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.validate_campaign_operations_dispatch_attempt_v2_insert()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.validate_campaign_operations_production_enablement_insert()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.validate_campaign_ops_production_admission_insert()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public'];
    protected_function_signatures text[] := ARRAY[
        'public.campaign_operations_dispatch_attempt_v2_canonical(campaign_operations_dispatch_attempt)',
        'public.campaign_operations_h1_deployment_audit_v1(text,boolean,boolean)',
        'public.campaign_operations_has_explicit_role_v1(name,name)',
        'public.campaign_operations_isolated_v1_authority_valid()',
        'public.campaign_operations_manager_build_canonical_v1(text,text,text,text)',
        'public.campaign_operations_production_acquire_replay_v2(bigint,integer,text,timestamp with time zone,text,text,text)',
        'public.campaign_operations_production_context_valid_v1(text,bigint,text)',
        'public.campaign_operations_production_disable_replay_v1(text,bigint,text,integer,text,text)',
        'public.campaign_operations_production_enable_replay_v1(text,integer,text,text,text,text,text,text,text,text,text)',
        'public.campaign_operations_production_enablement_history_valid_v1(bigint)',
        'public.campaign_operations_production_enablement_canonical_v1(campaign_operations_production_enablement_event)',
        'public.campaign_operations_request_production_admission_canonical_v1(campaign_operations_request_production_admission)',
        'public.campaign_operations_scheduler_protocol_evidence_canonical_v1(integer,text,timestamp with time zone,text,text,text)',
        'public.campaign_operations_scheduler_protocol_evidence_lock_v1()',
        'public.campaign_operations_scheduler_protocol_evidence_snapshot_v1()',
        'public.campaign_operations_tagged_fnv1a64(text)',
        'public.enforce_campaign_operations_complete_binding()',
        'public.enforce_campaign_operations_completion_audit_complete()',
        'public.enforce_campaign_operations_completion_event()',
        'public.enforce_campaign_operations_dispatch_acquisition_complete()',
        'public.enforce_campaign_operations_phase4_request_transition_complete()',
        'public.enforce_campaign_operations_production_admission_consistent()',
        'public.enforce_campaign_operations_production_context_empty_v1()',
        'public.enforce_campaign_operations_request()',
        'public.enforce_campaign_operations_request_acquisition_complete()',
        'public.enforce_campaign_ops_enablement_audit_complete()',
        'public.guard_campaign_operations_attempt_v1_isolation()',
        'public.guard_campaign_operations_completed_campaign()',
        'public.guard_campaign_operations_completion_v2_evidence()',
        'public.guard_campaign_operations_dispatch_control()',
        'public.guard_campaign_operations_production_admission_witness()',
        'public.guard_campaign_operations_production_attempt_mutation()',
        'public.guard_campaign_operations_production_dispatch_audit_insert()',
        'public.guard_campaign_operations_production_dispatch_audit_mutation()',
        'public.guard_campaign_operations_production_post_completion()',
        'public.lock_campaign_operations_authorization_head(bigint,text)',
        'public.lock_campaign_operations_budget_head(bigint)',
        'public.lock_campaign_operations_campaign(bigint)',
        'public.lock_campaign_operations_request(bigint)',
        'public.lock_campaign_operations_reservation(bigint)',
        'public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)',
        'public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)',
        'public.reject_campaign_operations_completion_mutation()',
        'public.reject_campaign_operations_production_mutation()',
        'public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)',
        'public.transition_campaign_operations_request_dispatching(bigint,integer,text,text)',
        'public.validate_campaign_operations_dispatch_attempt_v2_insert()',
        'public.validate_campaign_operations_production_enablement_insert()',
        'public.validate_campaign_ops_production_admission_insert()'];
    protected_function_contracts text[] := ARRAY[
        'public.campaign_operations_dispatch_attempt_v2_canonical(campaign_operations_dispatch_attempt)|sql|false|i|u|0|0|NULL',
        'public.campaign_operations_h1_deployment_audit_v1(text,boolean,boolean)|plpgsql|false|v|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_has_explicit_role_v1(name,name)|sql|true|s|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_isolated_v1_authority_valid()|sql|true|s|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_manager_build_canonical_v1(text,text,text,text)|plpgsql|false|i|u|0|0|NULL',
        'public.campaign_operations_production_acquire_replay_v2(bigint,integer,text,timestamp with time zone,text,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_production_context_valid_v1(text,bigint,text)|sql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_production_disable_replay_v1(text,bigint,text,integer,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_production_enable_replay_v1(text,integer,text,text,text,text,text,text,text,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_production_enablement_history_valid_v1(bigint)|plpgsql|true|s|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_production_enablement_canonical_v1(campaign_operations_production_enablement_event)|plpgsql|false|i|u|0|0|NULL',
        'public.campaign_operations_request_production_admission_canonical_v1(campaign_operations_request_production_admission)|sql|false|i|u|0|0|NULL',
        'public.campaign_operations_scheduler_protocol_evidence_canonical_v1(integer,text,timestamp with time zone,text,text,text)|plpgsql|false|i|u|0|0|NULL',
        'public.campaign_operations_scheduler_protocol_evidence_lock_v1()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_scheduler_protocol_evidence_snapshot_v1()|plpgsql|true|s|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_tagged_fnv1a64(text)|plpgsql|true|i|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_complete_binding()|plpgsql|false|v|u|0|0|search_path=pg_catalog, public, pg_temp',
        'public.enforce_campaign_operations_completion_audit_complete()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_completion_event()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_dispatch_acquisition_complete()|plpgsql|false|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_phase4_request_transition_complete()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_production_admission_consistent()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_production_context_empty_v1()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_request()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_request_acquisition_complete()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_ops_enablement_audit_complete()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_attempt_v1_isolation()|plpgsql|false|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_completed_campaign()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_completion_v2_evidence()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_dispatch_control()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_production_admission_witness()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_production_attempt_mutation()|plpgsql|false|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_production_dispatch_audit_insert()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_production_dispatch_audit_mutation()|plpgsql|false|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_production_post_completion()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.lock_campaign_operations_authorization_head(bigint,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.lock_campaign_operations_budget_head(bigint)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.lock_campaign_operations_campaign(bigint)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.lock_campaign_operations_request(bigint)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.lock_campaign_operations_reservation(bigint)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.reject_campaign_operations_completion_mutation()|plpgsql|false|v|u|0|0|search_path=pg_catalog, public, pg_temp',
        'public.reject_campaign_operations_production_mutation()|plpgsql|false|v|u|0|0|NULL',
        'public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.transition_campaign_operations_request_dispatching(bigint,integer,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.validate_campaign_operations_dispatch_attempt_v2_insert()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.validate_campaign_operations_production_enablement_insert()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.validate_campaign_ops_production_admission_insert()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public'];
    protected_function_return_contracts text[] := ARRAY[
        'public.campaign_operations_dispatch_attempt_v2_canonical(campaign_operations_dispatch_attempt)|pg_catalog.text|false|NULL|NULL|NULL',
        'public.campaign_operations_h1_deployment_audit_v1(text,boolean,boolean)|pg_catalog.bool|false|NULL|NULL|NULL',
        'public.campaign_operations_has_explicit_role_v1(name,name)|pg_catalog.bool|false|NULL|NULL|NULL',
        'public.campaign_operations_isolated_v1_authority_valid()|pg_catalog.bool|false|NULL|NULL|NULL',
        'public.campaign_operations_manager_build_canonical_v1(text,text,text,text)|pg_catalog.text|false|NULL|NULL|NULL',
        'public.campaign_operations_production_acquire_replay_v2(bigint,integer,text,timestamp with time zone,text,text,text)|public.campaign_operations_dispatch_attempt|true|NULL|NULL|NULL',
        'public.campaign_operations_production_context_valid_v1(text,bigint,text)|pg_catalog.bool|false|NULL|NULL|NULL',
        'public.campaign_operations_production_disable_replay_v1(text,bigint,text,integer,text,text)|public.campaign_operations_production_enablement_event|true|NULL|NULL|NULL',
        'public.campaign_operations_production_enable_replay_v1(text,integer,text,text,text,text,text,text,text,text,text)|public.campaign_operations_production_enablement_event|true|NULL|NULL|NULL',
        'public.campaign_operations_production_enablement_history_valid_v1(bigint)|pg_catalog.bool|false|NULL|NULL|NULL',
        'public.campaign_operations_production_enablement_canonical_v1(campaign_operations_production_enablement_event)|pg_catalog.text|false|NULL|NULL|NULL',
        'public.campaign_operations_request_production_admission_canonical_v1(campaign_operations_request_production_admission)|pg_catalog.text|false|NULL|NULL|NULL',
        'public.campaign_operations_scheduler_protocol_evidence_canonical_v1(integer,text,timestamp with time zone,text,text,text)|pg_catalog.text|false|NULL|NULL|NULL',
        'public.campaign_operations_scheduler_protocol_evidence_lock_v1()|pg_catalog.record|true|pg_catalog.int4,pg_catalog.text,pg_catalog.text,pg_catalog.text,pg_catalog.text,pg_catalog.text,pg_catalog.bool,pg_catalog.text,pg_catalog.text|t,t,t,t,t,t,t,t,t|required_generation,cutover_state,cutover_completed_at,cutover_completed_by,cutover_executable_path,cutover_process_evidence,evidence_complete,evidence_canonical,evidence_hash',
        'public.campaign_operations_scheduler_protocol_evidence_snapshot_v1()|pg_catalog.record|true|pg_catalog.int4,pg_catalog.text,pg_catalog.text,pg_catalog.text,pg_catalog.text,pg_catalog.text,pg_catalog.bool,pg_catalog.text,pg_catalog.text|t,t,t,t,t,t,t,t,t|required_generation,cutover_state,cutover_completed_at,cutover_completed_by,cutover_executable_path,cutover_process_evidence,evidence_complete,evidence_canonical,evidence_hash',
        'public.campaign_operations_tagged_fnv1a64(text)|pg_catalog.text|false|NULL|NULL|NULL',
        'public.enforce_campaign_operations_complete_binding()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.enforce_campaign_operations_completion_audit_complete()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.enforce_campaign_operations_completion_event()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.enforce_campaign_operations_dispatch_acquisition_complete()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.enforce_campaign_operations_phase4_request_transition_complete()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.enforce_campaign_operations_production_admission_consistent()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.enforce_campaign_operations_production_context_empty_v1()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.enforce_campaign_operations_request()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.enforce_campaign_operations_request_acquisition_complete()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.enforce_campaign_ops_enablement_audit_complete()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.guard_campaign_operations_attempt_v1_isolation()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.guard_campaign_operations_completed_campaign()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.guard_campaign_operations_completion_v2_evidence()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.guard_campaign_operations_dispatch_control()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.guard_campaign_operations_production_admission_witness()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.guard_campaign_operations_production_attempt_mutation()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.guard_campaign_operations_production_dispatch_audit_insert()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.guard_campaign_operations_production_dispatch_audit_mutation()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.guard_campaign_operations_production_post_completion()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.lock_campaign_operations_authorization_head(bigint,text)|public.campaign_operations_authorization_event|false|NULL|NULL|NULL',
        'public.lock_campaign_operations_budget_head(bigint)|public.campaign_operations_budget_ledger_entry|false|NULL|NULL|NULL',
        'public.lock_campaign_operations_campaign(bigint)|pg_catalog.void|false|NULL|NULL|NULL',
        'public.lock_campaign_operations_request(bigint)|public.campaign_operations_operational_request|false|NULL|NULL|NULL',
        'public.lock_campaign_operations_reservation(bigint)|public.campaign_operations_reservation|false|NULL|NULL|NULL',
        'public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)|public.campaign_operations_production_enablement_event|false|NULL|NULL|NULL',
        'public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)|public.campaign_operations_production_enablement_event|false|NULL|NULL|NULL',
        'public.reject_campaign_operations_completion_mutation()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.reject_campaign_operations_production_mutation()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)|public.campaign_operations_dispatch_attempt|false|NULL|NULL|NULL',
        'public.transition_campaign_operations_request_dispatching(bigint,integer,text,text)|public.campaign_operations_operational_request|false|NULL|NULL|NULL',
        'public.validate_campaign_operations_dispatch_attempt_v2_insert()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.validate_campaign_operations_production_enablement_insert()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.validate_campaign_ops_production_admission_insert()|pg_catalog.trigger|false|NULL|NULL|NULL'];
    protected_function_identity_argument_contracts text[] := ARRAY[
        'public.campaign_operations_dispatch_attempt_v2_canonical(campaign_operations_dispatch_attempt)|candidate campaign_operations_dispatch_attempt',
        'public.campaign_operations_h1_deployment_audit_v1(text,boolean,boolean)|expected_migration_checksum text, require_migration_ledger boolean, require_historical_bytes boolean',
        'public.campaign_operations_has_explicit_role_v1(name,name)|principal_name name, target_role_name name',
        'public.campaign_operations_manager_build_canonical_v1(text,text,text,text)|manager_service_contract_value text, source_commit_value text, compiler_contract_value text, executable_sha256_value text',
        'public.campaign_operations_production_acquire_replay_v2(bigint,integer,text,timestamp with time zone,text,text,text)|target_request_id bigint, expected_version_value integer, lease_digest_value text, lease_expires_at_value timestamp with time zone, operation_key_value text, requesting_actor_value text, approved_build_contract_canonical_value text',
        'public.campaign_operations_production_context_valid_v1(text,bigint,text)|transition_kind_value text, target_request_id bigint, operation_key_value text',
        'public.campaign_operations_production_disable_replay_v1(text,bigint,text,integer,text,text)|operation_key_value text, predecessor_event_id_value bigint, predecessor_event_canonical_value text, expected_prior_version_value integer, disabling_actor_value text, reason_value text',
        'public.campaign_operations_production_enable_replay_v1(text,integer,text,text,text,text,text,text,text,text,text)|operation_key_value text, expected_prior_version_value integer, scheduler_evidence_canonical_value text, independent_verification_reference_value text, authorizing_actor_value text, manager_service_contract_value text, approved_build_contract_canonical_value text, approved_build_source_commit_value text, approved_build_compiler_contract_value text, approved_build_executable_sha256_value text, reason_value text',
        'public.campaign_operations_production_enablement_history_valid_v1(bigint)|start_event_id bigint',
        'public.campaign_operations_production_enablement_canonical_v1(campaign_operations_production_enablement_event)|candidate campaign_operations_production_enablement_event',
        'public.campaign_operations_request_production_admission_canonical_v1(campaign_operations_request_production_admission)|candidate campaign_operations_request_production_admission',
        'public.campaign_operations_scheduler_protocol_evidence_canonical_v1(integer,text,timestamp with time zone,text,text,text)|required_generation_value integer, cutover_state_value text, cutover_completed_at_value timestamp with time zone, cutover_completed_by_value text, cutover_executable_path_value text, cutover_process_evidence_value text',
        'public.campaign_operations_tagged_fnv1a64(text)|canonical_value text',
        'public.lock_campaign_operations_authorization_head(bigint,text)|target_campaign_id bigint, target_action_kind text',
        'public.lock_campaign_operations_budget_head(bigint)|target_campaign_id bigint',
        'public.lock_campaign_operations_campaign(bigint)|target_operational_campaign_id bigint',
        'public.lock_campaign_operations_request(bigint)|target_request_id bigint',
        'public.lock_campaign_operations_reservation(bigint)|target_reservation_id bigint',
        'public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)|operation_key_value text, predecessor_event_id_value bigint, predecessor_event_canonical_value text, expected_prior_version_value integer, disabling_actor_value text, reason_value text',
        'public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)|operation_key_value text, expected_prior_version_value integer, scheduler_evidence_canonical_value text, independent_verification_reference_value text, authorizing_actor_value text, manager_service_contract_value text, approved_build_contract_canonical_value text, approved_build_source_commit_value text, approved_build_compiler_contract_value text, approved_build_executable_sha256_value text, reason_value text',
        'public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)|target_request_id bigint, expected_version_value integer, lease_digest_value text, lease_expires_at_value timestamp with time zone, operation_key_value text, requesting_actor_value text, approved_build_contract_canonical_value text',
        'public.transition_campaign_operations_request_dispatching(bigint,integer,text,text)|target_request_id bigint, expected_version_value integer, lease_digest_value text, dispatcher_value text'];
    protected_function_final_extra_acl_contracts text[] := ARRAY[
        'public.campaign_operations_scheduler_protocol_evidence_lock_v1()|campaign_operations_scheduler_protocol_evidence_reader',
        'public.campaign_operations_scheduler_protocol_evidence_snapshot_v1()|campaign_operations_owner',
        'public.campaign_operations_scheduler_protocol_evidence_snapshot_v1()|campaign_operations_scheduler_protocol_evidence_reader',
        'public.campaign_operations_tagged_fnv1a64(text)|campaign_operations_owner',
        'public.lock_campaign_operations_authorization_head(bigint,text)|campaign_operations_completion_writer',
        'public.lock_campaign_operations_authorization_head(bigint,text)|campaign_operations_owner',
        'public.lock_campaign_operations_budget_head(bigint)|campaign_operations_cancellation_coordinator',
        'public.lock_campaign_operations_budget_head(bigint)|campaign_operations_completion_writer',
        'public.lock_campaign_operations_budget_head(bigint)|campaign_operations_owner',
        'public.lock_campaign_operations_campaign(bigint)|campaign_operations_cancellation_coordinator',
        'public.lock_campaign_operations_campaign(bigint)|campaign_operations_completion_writer',
        'public.lock_campaign_operations_campaign(bigint)|campaign_operations_controller',
        'public.lock_campaign_operations_campaign(bigint)|campaign_operations_owner',
        'public.lock_campaign_operations_campaign(bigint)|campaign_operations_reconciler',
        'public.lock_campaign_operations_campaign(bigint)|campaign_operations_recovery',
        'public.lock_campaign_operations_request(bigint)|campaign_operations_cancellation_coordinator',
        'public.lock_campaign_operations_request(bigint)|campaign_operations_completion_writer',
        'public.lock_campaign_operations_request(bigint)|campaign_operations_owner',
        'public.lock_campaign_operations_request(bigint)|campaign_operations_reconciler',
        'public.lock_campaign_operations_request(bigint)|campaign_operations_recovery',
        'public.lock_campaign_operations_reservation(bigint)|campaign_operations_cancellation_coordinator',
        'public.lock_campaign_operations_reservation(bigint)|campaign_operations_completion_writer',
        'public.lock_campaign_operations_reservation(bigint)|campaign_operations_owner',
        'public.lock_campaign_operations_reservation(bigint)|campaign_operations_recovery'];
    protected_function_legacy_public_acl_signatures text[] := ARRAY[
        'public.enforce_campaign_operations_complete_binding()',
        'public.enforce_campaign_operations_dispatch_acquisition_complete()',
        'public.enforce_campaign_operations_request()',
        'public.enforce_campaign_operations_request_acquisition_complete()',
        'public.lock_campaign_operations_authorization_head(bigint,text)',
        'public.lock_campaign_operations_budget_head(bigint)',
        'public.lock_campaign_operations_campaign(bigint)',
        'public.lock_campaign_operations_request(bigint)',
        'public.lock_campaign_operations_reservation(bigint)',
        'public.transition_campaign_operations_request_dispatching(bigint,integer,text,text)'];
    protected_function_legacy_null_acl_signatures text[] := ARRAY[
        'public.enforce_campaign_operations_complete_binding()',
        'public.enforce_campaign_operations_dispatch_acquisition_complete()',
        'public.enforce_campaign_operations_request()',
        'public.enforce_campaign_operations_request_acquisition_complete()',
        'public.transition_campaign_operations_request_dispatching(bigint,integer,text,text)'];
    -- Some deployed pre-H1 databases already carry explicit hardened ACLs
    -- instead of the historical PUBLIC-capable predecessor representation.
    -- These signatures and grants describe only exact accepted predecessor
    -- state; they are not part of the final H1 ACL contract.
    protected_function_hardened_predecessor_acl_signatures text[] := ARRAY[
        'public.lock_campaign_operations_authorization_head(bigint,text)',
        'public.lock_campaign_operations_budget_head(bigint)',
        'public.lock_campaign_operations_campaign(bigint)',
        'public.lock_campaign_operations_request(bigint)',
        'public.lock_campaign_operations_reservation(bigint)',
        'public.transition_campaign_operations_request_dispatching(bigint,integer,text,text)'];
    protected_function_hardened_predecessor_extra_acl_contracts text[] := ARRAY[
        'public.lock_campaign_operations_authorization_head(bigint,text)|campaign_operations_dispatcher',
        'public.lock_campaign_operations_authorization_head(bigint,text)|campaign_operations_phase5_transactional',
        'public.lock_campaign_operations_budget_head(bigint)|campaign_operations_dispatcher',
        'public.lock_campaign_operations_budget_head(bigint)|campaign_operations_phase5_transactional',
        'public.lock_campaign_operations_campaign(bigint)|campaign_operations_budget_administrator',
        'public.lock_campaign_operations_campaign(bigint)|campaign_operations_dispatcher',
        'public.lock_campaign_operations_campaign(bigint)|campaign_operations_phase5_transactional',
        'public.lock_campaign_operations_campaign(bigint)|campaign_operations_request_acceptor',
        'public.lock_campaign_operations_request(bigint)|campaign_operations_dispatcher',
        'public.lock_campaign_operations_request(bigint)|campaign_operations_phase5_transactional',
        'public.lock_campaign_operations_reservation(bigint)|campaign_operations_dispatcher',
        'public.lock_campaign_operations_reservation(bigint)|campaign_operations_phase5_transactional',
        'public.transition_campaign_operations_request_dispatching(bigint,integer,text,text)|campaign_operations_dispatcher'];
    new_h1_functions text[] := ARRAY[
        'campaign_operations_dispatch_attempt_v2_canonical',
        'campaign_operations_h1_deployment_audit_v1',
        'campaign_operations_has_explicit_role_v1',
        'campaign_operations_isolated_v1_authority_valid',
        'campaign_operations_manager_build_canonical_v1',
        'campaign_operations_production_acquire_replay_v2',
        'campaign_operations_production_context_valid_v1',
        'campaign_operations_production_disable_replay_v1',
        'campaign_operations_production_enable_replay_v1',
        'campaign_operations_production_enablement_history_valid_v1',
        'campaign_operations_production_enablement_canonical_v1',
        'campaign_operations_request_production_admission_canonical_v1',
        'campaign_operations_scheduler_protocol_evidence_canonical_v1',
        'campaign_operations_scheduler_protocol_evidence_lock_v1',
        'campaign_operations_scheduler_protocol_evidence_snapshot_v1',
        'enforce_campaign_operations_production_admission_consistent',
        'enforce_campaign_operations_production_context_empty_v1',
        'enforce_campaign_ops_enablement_audit_complete',
        'guard_campaign_operations_attempt_v1_isolation',
        'guard_campaign_operations_completion_v2_evidence',
        'guard_campaign_operations_production_admission_witness',
        'guard_campaign_operations_production_attempt_mutation',
        'guard_campaign_operations_production_dispatch_audit_insert',
        'guard_campaign_operations_production_dispatch_audit_mutation',
        'guard_campaign_operations_production_post_completion',
        'record_campaign_operations_production_disable_v1',
        'record_campaign_operations_production_enable_v1',
        'reject_campaign_operations_production_mutation',
        'transition_campaign_operations_request_dispatch_production_v2',
        'validate_campaign_operations_dispatch_attempt_v2_insert',
        'validate_campaign_operations_production_enablement_insert',
        'validate_campaign_ops_production_admission_insert'];
    allowed_trigger_entries text[] := ARRAY[
        'public.campaign_operations_audit_reference_event.campaign_operations_audit_reference_event_completion_gate_trigg',
        'public.campaign_operations_authorization_event.campaign_operations_authorization_event_completion_gate_trigger',
        'public.campaign_operations_budget_ledger_entry.campaign_operations_budget_ledger_entry_completion_gate_trigger',
        'public.campaign_operations_cancellation_request.campaign_operations_cancellation_request_completion_gate_trigge',
        'public.campaign_operations_cancellation_settlement.campaign_operations_cancellation_settlement_completion_gate_tri',
        'public.campaign_operations_completion_audit_reference_event.campaign_operations_completion_audit_immutable_trigger',
        'public.campaign_operations_completion_audit_reference_event.campaign_operations_completion_audit_truncate_trigger',
        'public.campaign_operations_completion_event.campaign_operations_completion_audit_complete_trigger',
        'public.campaign_operations_completion_event.campaign_operations_completion_immutable_trigger',
        'public.campaign_operations_completion_event.campaign_operations_completion_truncate_trigger',
        'public.campaign_operations_completion_event.campaign_operations_completion_v2_evidence_gate',
        'public.campaign_operations_completion_event.campaign_operations_completion_validate_trigger',
        'public.campaign_operations_control_audit_reference_event.campaign_operations_control_audit_reference_event_completion_ga',
        'public.campaign_operations_control_event.campaign_operations_control_event_completion_gate_trigger',
        'public.campaign_operations_dispatch_attempt.campaign_operations_dispatch_attempt_completion_gate_trigger',
        'public.campaign_operations_dispatch_attempt.campaign_operations_dispatch_attempt_v2_validate',
        'public.campaign_operations_dispatch_attempt.campaign_operations_production_attempt_consistency',
        'public.campaign_operations_dispatch_attempt.campaign_ops_attempt_v1_isolation',
        'public.campaign_operations_dispatch_attempt.campaign_ops_attempt_v2_immutable_row',
        'public.campaign_operations_dispatch_attempt.campaign_ops_attempt_v2_immutable_truncate',
        'public.campaign_operations_dispatch_attempt_outcome.campaign_operations_dispatch_attempt_outcome_completion_gate_tr',
        'public.campaign_operations_dispatch_audit_reference_event.campaign_operations_dispatch_audit_reference_event_completion_g',
        'public.campaign_operations_dispatch_audit_reference_event.campaign_operations_production_audit_consistency',
        'public.campaign_operations_dispatch_audit_reference_event.campaign_operations_production_audit_insert_guard',
        'public.campaign_operations_dispatch_audit_reference_event.campaign_ops_production_audit_immutable_row',
        'public.campaign_operations_dispatch_audit_reference_event.campaign_ops_production_audit_immutable_truncate',
        'public.campaign_operations_downstream_control_owner.campaign_operations_complete_owner_trigger',
        'public.campaign_operations_downstream_control_owner.campaign_operations_downstream_control_owner_completion_gate_tr',
        'public.campaign_operations_governance_provenance_event.campaign_operations_governance_provenance_event_completion_gate',
        'public.campaign_operations_operational_request.campaign_operations_complete_request_trigger',
        'public.campaign_operations_operational_request.campaign_operations_dispatch_acquisition_complete_trigger',
        'public.campaign_operations_operational_request.campaign_operations_dispatch_control_gate_trigger',
        'public.campaign_operations_operational_request.campaign_operations_operational_request_completion_gate_trigger',
        'public.campaign_operations_operational_request.campaign_operations_phase4_request_transition_complete_trigger',
        'public.campaign_operations_operational_request.campaign_operations_production_request_consistency',
        'public.campaign_operations_operational_request.campaign_operations_production_witness_truncate_guard',
        'public.campaign_operations_operational_request.campaign_operations_production_witness_update_guard',
        'public.campaign_operations_operational_request.campaign_operations_request_acquisition_complete_trigger',
        'public.campaign_operations_operational_request.campaign_operations_request_completion_update_gate',
        'public.campaign_operations_operational_request.campaign_operations_request_trigger',
        'public.campaign_operations_production_enablement_audit_reference_event.campaign_operations_enablement_audit_complete',
        'public.campaign_operations_production_enablement_audit_reference_event.campaign_ops_enablement_audit_immutable_row',
        'public.campaign_operations_production_enablement_audit_reference_event.campaign_ops_enablement_audit_immutable_truncate',
        'public.campaign_operations_production_enablement_event.campaign_operations_enablement_audit_event_complete',
        'public.campaign_operations_production_enablement_event.campaign_operations_production_enablement_validate',
        'public.campaign_operations_production_enablement_event.campaign_ops_enablement_immutable_row',
        'public.campaign_operations_production_enablement_event.campaign_ops_enablement_immutable_truncate',
        'public.campaign_operations_production_transition_context.campaign_operations_production_context_empty_v1',
        'public.campaign_operations_reconciliation_observation.campaign_operations_reconciliation_observation_completion_gate_',
        'public.campaign_operations_reconciliation_resolution.campaign_operations_reconciliation_resolution_completion_gate_t',
        'public.campaign_operations_request_binding.campaign_operations_complete_binding_trigger',
        'public.campaign_operations_request_binding.campaign_operations_request_binding_completion_gate_trigger',
        'public.campaign_operations_request_production_admission.campaign_operations_production_admission_consistency',
        'public.campaign_operations_request_production_admission.campaign_operations_production_admission_post_completion',
        'public.campaign_operations_request_production_admission.campaign_operations_production_admission_validate',
        'public.campaign_operations_request_production_admission.campaign_ops_admission_immutable_row',
        'public.campaign_operations_request_production_admission.campaign_ops_admission_immutable_truncate',
        'public.campaign_operations_reservation.campaign_operations_complete_reservation_trigger',
        'public.campaign_operations_reservation.campaign_operations_reservation_completion_gate_trigger',
        'public.campaign_operations_reservation.campaign_operations_reservation_completion_update_gate',
        'public.campaign_operations_reservation_commitment.campaign_operations_reservation_commitment_completion_gate_trig',
        'public.campaign_operations_reservation_event.campaign_operations_reservation_event_completion_gate_trigger',
        'public.experiment_lifecycle_cancellation_event.experiment_lifecycle_cancellation_event_completion_gate_trigger'];
BEGIN
    SELECT role.oid INTO boundary_oid
    FROM pg_catalog.pg_roles role
    WHERE role.rolname = 'campaign_operations_h1_boundary_authority';

    -- New H1 names may be absent or may be the exact restored sealed objects;
    -- an ordinary pre-existing owner is never replaced silently.
    IF (boundary_oid IS NOT NULL AND (EXISTS (
            SELECT 1 FROM pg_catalog.pg_proc function_row
            WHERE function_row.proowner = boundary_oid
              AND function_row.proname = ANY (protected_functions)) OR EXISTS (
            SELECT 1 FROM pg_catalog.pg_class relation
            WHERE relation.relowner = boundary_oid
              AND relation.relname = ANY (allowed_relations))) AND
        ((SELECT count(*)
        FROM pg_catalog.pg_trigger trigger_row
        JOIN pg_catalog.pg_proc trigger_function
          ON trigger_function.oid = trigger_row.tgfoid
        WHERE NOT trigger_row.tgisinternal
          AND trigger_function.proowner = boundary_oid) <>
       pg_catalog.cardinality(allowed_trigger_entries) OR EXISTS (
        SELECT 1
        FROM pg_catalog.pg_trigger trigger_row
        JOIN pg_catalog.pg_proc trigger_function
          ON trigger_function.oid = trigger_row.tgfoid
        JOIN pg_catalog.pg_class trigger_relation
          ON trigger_relation.oid = trigger_row.tgrelid
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = trigger_relation.relnamespace
        WHERE NOT trigger_row.tgisinternal
          AND trigger_function.proowner = boundary_oid
          AND (namespace.nspname || '.' || trigger_relation.relname || '.' ||
               trigger_row.tgname) <> ALL (allowed_trigger_entries)))) OR EXISTS (
        SELECT 1
        FROM pg_catalog.pg_class relation
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = relation.relnamespace
        WHERE namespace.nspname = 'public'
          AND relation.relname = ANY (allowed_relations[1:4])
          AND (boundary_oid IS NULL OR relation.relowner <> boundary_oid)) THEN
        RAISE EXCEPTION 'H1A004 pre-existing protected relation owner mismatch'
            USING ERRCODE = '42501';
    END IF;

    IF boundary_oid IS NOT NULL THEN
        IF EXISTS (
            SELECT 1
            FROM pg_catalog.pg_proc function_row
            JOIN pg_catalog.pg_namespace namespace
              ON namespace.oid = function_row.pronamespace
            WHERE function_row.proowner = boundary_oid
              AND namespace.nspname NOT LIKE 'pg\_%' ESCAPE '\'
              AND namespace.nspname NOT IN ('information_schema', 'public')) THEN
            RAISE EXCEPTION
                'H1A005 alternate-schema boundary-owned function or wrapper'
                USING ERRCODE = '42501';
        END IF;
        IF EXISTS (
            SELECT 1
            FROM pg_catalog.pg_namespace namespace
            WHERE namespace.nspowner = boundary_oid) THEN
            RAISE EXCEPTION 'H1A004 unexpected boundary-owned schema'
                USING ERRCODE = '42501';
        END IF;

        IF EXISTS (
            SELECT 1
            FROM pg_catalog.pg_class relation
            JOIN pg_catalog.pg_namespace namespace
              ON namespace.oid = relation.relnamespace
            LEFT JOIN pg_catalog.pg_index index_row
              ON index_row.indexrelid = relation.oid
            LEFT JOIN pg_catalog.pg_class indexed_relation
              ON indexed_relation.oid = index_row.indrelid
            WHERE relation.relowner = boundary_oid
              AND namespace.nspname NOT LIKE 'pg\_%' ESCAPE '\'
              AND namespace.nspname <> 'information_schema'
              AND NOT (
                namespace.nspname = 'public' AND
                (relation.relname = ANY (allowed_relations) OR
                 coalesce(indexed_relation.relname = ANY (allowed_relations),
                          false) OR
                 (relation.relkind = 'S' AND EXISTS (
                    SELECT 1
                    FROM pg_catalog.pg_depend dependency
                    JOIN pg_catalog.pg_class table_row
                      ON table_row.oid = dependency.refobjid
                    WHERE dependency.classid = 'pg_class'::regclass
                      AND dependency.objid = relation.oid
                      AND dependency.refclassid = 'pg_class'::regclass
                      AND dependency.deptype IN ('a', 'i')
                      AND table_row.relname = ANY (allowed_relations)))))) THEN
            RAISE EXCEPTION 'H1A004 unexpected boundary-owned relation'
                USING ERRCODE = '42501';
        END IF;

        IF EXISTS (
            SELECT 1
            FROM pg_catalog.pg_proc function_row
            JOIN pg_catalog.pg_namespace namespace
              ON namespace.oid = function_row.pronamespace
            WHERE function_row.proowner = boundary_oid
              AND NOT (namespace.nspname = 'public' AND
                       function_row.proname = ANY (protected_functions))) THEN
            RAISE EXCEPTION 'H1A004 unexpected boundary-owned function'
                USING ERRCODE = '42501';
        END IF;

        IF EXISTS (
            SELECT 1
            FROM pg_catalog.pg_type type_row
            JOIN pg_catalog.pg_namespace namespace
              ON namespace.oid = type_row.typnamespace
        WHERE type_row.typowner = boundary_oid
          AND namespace.nspname NOT LIKE 'pg\_%' ESCAPE '\'
          AND namespace.nspname <> 'information_schema'
          AND NOT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_type coupled_type
            JOIN pg_catalog.pg_class relation
              ON relation.oid = coupled_type.typrelid
            JOIN pg_catalog.pg_namespace relation_namespace
              ON relation_namespace.oid = relation.relnamespace
            WHERE coupled_type.oid IN (type_row.oid, type_row.typelem)
              AND relation_namespace.nspname = 'public'
              AND relation.relname = ANY (allowed_relations))) THEN
            RAISE EXCEPTION 'H1A004 unexpected boundary-owned type or domain'
                USING ERRCODE = '42501';
        END IF;

        IF EXISTS (SELECT 1 FROM pg_catalog.pg_operator
                   WHERE oprowner = boundary_oid) OR
           EXISTS (SELECT 1 FROM pg_catalog.pg_event_trigger
                   WHERE evtowner = boundary_oid) OR
           EXISTS (SELECT 1 FROM pg_catalog.pg_largeobject_metadata
                   WHERE lomowner = boundary_oid) OR
           EXISTS (SELECT 1 FROM pg_catalog.pg_publication
                   WHERE pubowner = boundary_oid) OR
           EXISTS (SELECT 1 FROM pg_catalog.pg_subscription
                   WHERE subowner = boundary_oid) THEN
            RAISE EXCEPTION 'H1A004 unexpected boundary-owned catalog object'
                USING ERRCODE = '42501';
        END IF;
    END IF;

    -- Protected names are unique across every non-system schema.  This catches
    -- alternate-schema same names, overloads, default-argument variants,
    -- procedures, and aggregates before CREATE OR REPLACE could hide them.
    IF EXISTS (
        SELECT function_row.proname
        FROM pg_catalog.pg_proc function_row
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = function_row.pronamespace
        WHERE function_row.proname = ANY (protected_functions)
          AND namespace.nspname NOT LIKE 'pg\_%' ESCAPE '\'
          AND namespace.nspname <> 'information_schema'
        GROUP BY function_row.proname
        HAVING count(*) > 1 OR bool_or(namespace.nspname <> 'public')) THEN
        RAISE EXCEPTION 'H1A005 protected function alternate schema or overload'
            USING ERRCODE = '42501';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_proc function_row
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = function_row.pronamespace
        WHERE function_row.proname = ANY (protected_functions)
          AND namespace.nspname NOT LIKE 'pg\_%' ESCAPE '\'
          AND namespace.nspname <> 'information_schema'
          AND pg_catalog.format('%I.%s', namespace.nspname,
                function_row.oid::pg_catalog.regprocedure::text) <>
              ALL (protected_function_signatures)) THEN
        RAISE EXCEPTION 'H1A005 protected function signature mismatch'
            USING ERRCODE = '42501';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_proc function_row
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = function_row.pronamespace
        WHERE function_row.proname = ANY (protected_functions)
          AND namespace.nspname NOT LIKE 'pg\_%' ESCAPE '\'
          AND namespace.nspname <> 'information_schema'
          AND (function_row.pronargdefaults <> 0 OR
               function_row.provariadic <> 0)) THEN
        RAISE EXCEPTION
            'H1A005 protected function default or variadic mismatch'
            USING ERRCODE = '42501';
    END IF;

    SELECT pg_catalog.format('%I.%s', namespace.nspname,
             function_row.oid::pg_catalog.regprocedure::text)
      INTO offending_entry
        FROM pg_catalog.pg_proc function_row
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = function_row.pronamespace
        WHERE namespace.nspname = 'public'
          AND function_row.proname = ANY (new_h1_functions)
          AND (boundary_oid IS NULL OR function_row.proowner <> boundary_oid)
        ORDER BY 1
        LIMIT 1;
    IF offending_entry IS NOT NULL THEN
        RAISE EXCEPTION
            'H1A004 incompatible pre-existing protected function owner: %',
            offending_entry
            USING ERRCODE = '42501';
    END IF;

    SELECT pg_catalog.format('%I.%s', namespace.nspname,
             function_row.oid::pg_catalog.regprocedure::text)
      INTO offending_entry
        FROM pg_catalog.pg_proc function_row
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = function_row.pronamespace
        JOIN pg_catalog.pg_roles owner_role
          ON owner_role.oid = function_row.proowner
        WHERE namespace.nspname = 'public'
          AND function_row.proname = ANY (protected_functions)
          AND function_row.proname <> ALL (new_h1_functions)
          AND owner_role.rolname NOT IN (
              'campaign_operations_owner',
              'campaign_operations_h1_boundary_authority')
        ORDER BY 1
        LIMIT 1;
    IF offending_entry IS NOT NULL THEN
        RAISE EXCEPTION
            'H1A004 incompatible pre-existing protected function owner: %',
            offending_entry
            USING ERRCODE = '42501';
    END IF;

    -- Every protected entry is compared with the same frozen catalog tuple
    -- before any CREATE OR REPLACE or ownership mutation.  The sole legacy
    -- allowance is the already-accepted safe pg_temp suffix on a pre-H1
    -- function owned by campaign_operations_owner; migration 055 pins the
    -- functions it replaces while transferring authority to the sealed owner.
    SELECT pg_catalog.format('%I.%s', namespace.nspname,
             function_row.oid::pg_catalog.regprocedure::text) || '|' ||
           function_row.prokind::text || '|' || language.lanname || '|' ||
           function_row.prosecdef::text || '|' ||
           function_row.provolatile::text || '|' ||
           function_row.proparallel::text || '|' ||
           function_row.proleakproof::text || '|' ||
           function_row.pronargdefaults::text || '|' ||
           function_row.provariadic::text || '|' ||
           coalesce(pg_catalog.array_to_string(function_row.proconfig, ','),
                    'NULL')
      INTO offending_entry
        FROM pg_catalog.pg_proc function_row
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = function_row.pronamespace
        JOIN pg_catalog.pg_language language
          ON language.oid = function_row.prolang
        JOIN pg_catalog.pg_roles owner_role
          ON owner_role.oid = function_row.proowner
        WHERE namespace.nspname = 'public'
          AND function_row.proname = ANY (protected_functions)
          AND (function_row.prokind <> 'f' OR
               function_row.proleakproof OR
               NOT EXISTS (
              SELECT 1
              FROM pg_catalog.unnest(protected_function_contracts)
                   expected(contract)
              WHERE pg_catalog.split_part(expected.contract, '|', 1) =
                    pg_catalog.format('%I.%s', namespace.nspname,
                      function_row.oid::pg_catalog.regprocedure::text)
                AND pg_catalog.split_part(expected.contract, '|', 2) =
                    language.lanname
                AND pg_catalog.split_part(expected.contract, '|', 3)::boolean =
                    function_row.prosecdef
                AND pg_catalog.split_part(expected.contract, '|', 4) =
                    function_row.provolatile::text
                AND pg_catalog.split_part(expected.contract, '|', 5) =
                    function_row.proparallel::text
                AND pg_catalog.split_part(expected.contract, '|', 6)::integer =
                    function_row.pronargdefaults
                AND pg_catalog.split_part(expected.contract, '|', 7)::oid =
                    function_row.provariadic
                AND (
                  pg_catalog.split_part(expected.contract, '|', 8) =
                    coalesce(pg_catalog.array_to_string(
                      function_row.proconfig, ','), 'NULL')
                  OR (owner_role.rolname = 'campaign_operations_owner'
                      AND function_row.proconfig =
                        ARRAY['search_path=pg_catalog, public, pg_temp']::text[])
                )))
        ORDER BY 1
        LIMIT 1;
    IF offending_entry IS NOT NULL THEN
        RAISE EXCEPTION
            'H1A008 protected function preflight catalog mismatch: %',
            offending_entry
            USING ERRCODE = '55000';
    END IF;

    -- Return identity is catalog identity, not merely CREATE OR REPLACE
    -- compatibility.  proallargtypes/proargmodes and the named OUT positions
    -- keep OUT, INOUT, and TABLE declarations inside this same preflight.
    SELECT pg_catalog.format('%I.%s', namespace.nspname,
             function_row.oid::pg_catalog.regprocedure::text)
      INTO offending_entry
        FROM pg_catalog.pg_proc function_row
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = function_row.pronamespace
        JOIN pg_catalog.pg_type return_type
          ON return_type.oid = function_row.prorettype
        JOIN pg_catalog.pg_namespace return_namespace
          ON return_namespace.oid = return_type.typnamespace
        WHERE namespace.nspname = 'public'
          AND function_row.proname = ANY (protected_functions)
          AND NOT EXISTS (
              SELECT 1
              FROM pg_catalog.unnest(protected_function_return_contracts)
                   expected(contract)
              WHERE pg_catalog.split_part(expected.contract, '|', 1) =
                    pg_catalog.format('%I.%s', namespace.nspname,
                      function_row.oid::pg_catalog.regprocedure::text)
                AND pg_catalog.pg_get_function_identity_arguments(
                      function_row.oid) = coalesce((
                    SELECT pg_catalog.split_part(identity.contract, '|', 2)
                    FROM pg_catalog.unnest(
                           protected_function_identity_argument_contracts)
                         identity(contract)
                    WHERE pg_catalog.split_part(identity.contract, '|', 1) =
                          pg_catalog.format('%I.%s', namespace.nspname,
                            function_row.oid::pg_catalog.regprocedure::text)), '')
                AND pg_catalog.split_part(expected.contract, '|', 2) =
                    pg_catalog.format('%I.%I', return_namespace.nspname,
                      return_type.typname)
                AND pg_catalog.split_part(expected.contract, '|', 3)::boolean =
                    function_row.proretset
                AND pg_catalog.split_part(expected.contract, '|', 4) =
                    CASE WHEN function_row.proallargtypes IS NULL THEN 'NULL'
                    ELSE (SELECT pg_catalog.string_agg(
                              pg_catalog.format('%I.%I', argument_namespace.nspname,
                                argument_type.typname), ',' ORDER BY argument.ordinality)
                          FROM pg_catalog.unnest(function_row.proallargtypes)
                               WITH ORDINALITY argument(type_oid, ordinality)
                          JOIN pg_catalog.pg_type argument_type
                            ON argument_type.oid = argument.type_oid
                          JOIN pg_catalog.pg_namespace argument_namespace
                            ON argument_namespace.oid = argument_type.typnamespace)
                    END
                AND pg_catalog.split_part(expected.contract, '|', 5) =
                    coalesce(pg_catalog.array_to_string(
                      function_row.proargmodes, ','), 'NULL')
                AND pg_catalog.split_part(expected.contract, '|', 6) =
                    CASE WHEN function_row.proargmodes IS NULL THEN 'NULL'
                    ELSE (SELECT pg_catalog.string_agg(
                              function_row.proargnames[argument.ordinality], ','
                              ORDER BY argument.ordinality)
                          FROM pg_catalog.unnest(function_row.proargmodes)
                               WITH ORDINALITY argument(mode, ordinality)
                          WHERE argument.mode IN ('o', 'b', 't'))
                    END)
        ORDER BY 1
        LIMIT 1;
    IF offending_entry IS NOT NULL THEN
        RAISE EXCEPTION
            'H1A008 protected function preflight return contract mismatch: %',
            offending_entry USING ERRCODE = '55000';
    END IF;

    -- A named grantee is part of the ACL identity, not an optional catalog
    -- lookup.  Only contracts for protected functions already present in the
    -- pre-H1 catalog apply here; new H1 functions and roles are created later.
    -- Once a protected function exists, however, every named grantee must also
    -- exist before any migration mutation can occur.
    SELECT pg_catalog.format('role=%s object=%s stage=preflight',
             pg_catalog.split_part(expected.contract, '|', 2),
             pg_catalog.split_part(expected.contract, '|', 1))
      INTO offending_entry
        FROM pg_catalog.unnest(
               protected_function_final_extra_acl_contracts)
             expected(contract)
        LEFT JOIN pg_catalog.pg_roles grantee_role
          ON grantee_role.rolname =
             pg_catalog.split_part(expected.contract, '|', 2)
        WHERE grantee_role.oid IS NULL
          AND pg_catalog.to_regprocedure(
                pg_catalog.split_part(expected.contract, '|', 1)) IS NOT NULL
        ORDER BY pg_catalog.split_part(expected.contract, '|', 2),
                 pg_catalog.split_part(expected.contract, '|', 1)
        LIMIT 1;
    IF offending_entry IS NOT NULL THEN
        RAISE EXCEPTION
            'H1A006 missing expected protected function ACL grantee role: %',
            offending_entry USING ERRCODE = '42501';
    END IF;

    -- Compare the complete expanded ACL before any REVOKE/GRANT.  NULL ACLs
    -- are expanded with PostgreSQL's acldefault('f', owner) semantics; catalog
    -- origin is separately exact for both the accepted 054 and final phases.
    SELECT pg_catalog.format('%I.%s', namespace.nspname,
             function_row.oid::pg_catalog.regprocedure::text)
      INTO offending_entry
        FROM pg_catalog.pg_proc function_row
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = function_row.pronamespace
        JOIN pg_catalog.pg_roles owner_role
          ON owner_role.oid = function_row.proowner
        WHERE namespace.nspname = 'public'
          AND function_row.proname = ANY (protected_functions)
          AND (
            (owner_role.rolname = 'campaign_operations_h1_boundary_authority'
             AND function_row.proacl IS NULL) OR
            -- Historical NULL/default ACL functions may also arrive in an
            -- explicitly hardened predecessor form.  Non-legacy protected
            -- functions owned by campaign_operations_owner still require
            -- explicit ACL origin.
            (owner_role.rolname = 'campaign_operations_owner' AND
             NOT (pg_catalog.format('%I.%s', namespace.nspname,
                    function_row.oid::pg_catalog.regprocedure::text) =
                      ANY (protected_function_legacy_null_acl_signatures))
             AND function_row.proacl IS NULL) OR
            EXISTS (
              WITH actual_acl(grantee, privilege_type, is_grantable) AS (
                SELECT acl.grantee, acl.privilege_type, acl.is_grantable
                FROM pg_catalog.aclexplode(coalesce(
                  function_row.proacl,
                  pg_catalog.acldefault('f', function_row.proowner))) acl
              ),
              expected_acl(grantee, privilege_type, is_grantable) AS (
                SELECT function_row.proowner, 'EXECUTE'::text, false
                UNION
                SELECT grantee_role.oid, 'EXECUTE'::text, false
                FROM pg_catalog.unnest(
                       protected_function_final_extra_acl_contracts)
                     expected(contract)
                LEFT JOIN pg_catalog.pg_roles grantee_role
                  ON grantee_role.rolname =
                     pg_catalog.split_part(expected.contract, '|', 2)
                WHERE pg_catalog.split_part(expected.contract, '|', 1) =
                      pg_catalog.format('%I.%s', namespace.nspname,
                        function_row.oid::pg_catalog.regprocedure::text)
                  AND (grantee_role.oid IS NULL OR owner_role.rolname =
                         'campaign_operations_h1_boundary_authority' OR
                       grantee_role.oid <> function_row.proowner)
                UNION
                -- Exact grants that are valid only for an explicitly
                -- hardened pre-H1 predecessor.  They intentionally do not
                -- alter protected_function_final_extra_acl_contracts.
                SELECT grantee_role.oid, 'EXECUTE'::text, false
                FROM pg_catalog.unnest(
                       protected_function_hardened_predecessor_extra_acl_contracts)
                     expected(contract)
                LEFT JOIN pg_catalog.pg_roles grantee_role
                  ON grantee_role.rolname =
                     pg_catalog.split_part(expected.contract, '|', 2)
                WHERE owner_role.rolname = 'campaign_operations_owner'
                  AND function_row.proacl IS NOT NULL
                  AND (
                    pg_catalog.format('%I.%s', namespace.nspname,
                      function_row.oid::pg_catalog.regprocedure::text) =
                      'public.transition_campaign_operations_request_dispatching(bigint,integer,text,text)'
                    OR NOT EXISTS (
                      SELECT 1
                      FROM pg_catalog.aclexplode(function_row.proacl)
                        predecessor_acl
                      WHERE predecessor_acl.grantee = 0))
                  AND pg_catalog.split_part(expected.contract, '|', 1) =
                      pg_catalog.format('%I.%s', namespace.nspname,
                        function_row.oid::pg_catalog.regprocedure::text)
                UNION
                -- PUBLIC EXECUTE is valid only for the historical NULL ACL
                -- representation.  Explicitly hardened predecessors must
                -- match the owner/per-function contract without PUBLIC.
                SELECT 0::oid, 'EXECUTE'::text, false
                WHERE owner_role.rolname = 'campaign_operations_owner'
                  AND pg_catalog.format('%I.%s', namespace.nspname,
                        function_row.oid::pg_catalog.regprocedure::text) =
                      ANY (protected_function_legacy_public_acl_signatures)
                  -- Historical NULL/default-ACL signatures may be explicitly
                  -- owner-hardened. Exact hardened-predecessor signatures may
                  -- instead retain their listed predecessor-only grants.
                  AND (
                    (NOT (
                      pg_catalog.format('%I.%s', namespace.nspname,
                        function_row.oid::pg_catalog.regprocedure::text) =
                      ANY (protected_function_legacy_null_acl_signatures)
                    )
                    AND NOT (
                      pg_catalog.format('%I.%s', namespace.nspname,
                        function_row.oid::pg_catalog.regprocedure::text) =
                      ANY (protected_function_hardened_predecessor_acl_signatures)
                    ))
                    OR function_row.proacl IS NULL
                    OR EXISTS (
                      SELECT 1
                      FROM pg_catalog.aclexplode(function_row.proacl)
                        predecessor_acl
                      WHERE predecessor_acl.grantee = 0
                        AND predecessor_acl.privilege_type = 'EXECUTE'
                        AND NOT predecessor_acl.is_grantable)
                  )
              )
              SELECT 1 FROM (
                (SELECT * FROM actual_acl EXCEPT SELECT * FROM expected_acl)
                UNION ALL
                (SELECT * FROM expected_acl EXCEPT SELECT * FROM actual_acl)
              ) difference
            ))
        ORDER BY 1
        LIMIT 1;
    IF offending_entry IS NOT NULL THEN
        RAISE EXCEPTION
            'H1A006 protected function preflight ACL mismatch: %',
            offending_entry USING ERRCODE = '42501';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_proc wrapper
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = wrapper.pronamespace
        WHERE namespace.nspname NOT LIKE 'pg\_%' ESCAPE '\'
          AND namespace.nspname NOT IN ('information_schema', 'public')
          AND (wrapper.prosecdef OR
               pg_catalog.has_function_privilege(0, wrapper.oid, 'EXECUTE'))
          AND wrapper.prosrc ~
              '(record_campaign_operations_production_|transition_campaign_operations_request_dispatch_production_v2|campaign_operations_production_transition_context)') THEN
        RAISE EXCEPTION 'H1A005 alternate-schema protected wrapper'
            USING ERRCODE = '42501';
    END IF;
END $$;

DO $$
DECLARE
    role_name text;
    role_oid oid;
    role_is_boundary boolean;
BEGIN
    -- ADR-0019B fail-closed role preflight.  A pre-existing role is inspected
    -- completely before any migration mutation; it is never normalized.
    FOREACH role_name IN ARRAY ARRAY[
        'campaign_operations_h1_boundary_authority',
        'campaign_operations_production_enabler',
        'campaign_operations_production_disabler',
        'campaign_operations_production_dispatcher',
        'campaign_operations_production_phase5_transactional',
        'campaign_operations_production_reader',
        'campaign_operations_scheduler_protocol_evidence_owner',
        'campaign_operations_scheduler_protocol_evidence_reader']
    LOOP
        role_is_boundary :=
            role_name = 'campaign_operations_h1_boundary_authority';
        SELECT role.oid INTO role_oid
        FROM pg_catalog.pg_authid role
        WHERE role.rolname = role_name;

        IF role_oid IS NOT NULL AND NOT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_authid role
            WHERE role.oid = role_oid
              AND NOT role.rolcanlogin
              AND role.rolsuper = role_is_boundary
              AND role.rolinherit
              AND NOT role.rolcreatedb
              AND NOT role.rolcreaterole
              AND NOT role.rolreplication
              AND NOT role.rolbypassrls
              AND role.rolconnlimit = -1
              AND role.rolpassword IS NULL
              AND role.rolvaliduntil IS NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM pg_catalog.pg_db_role_setting setting
                  WHERE setting.setrole = role.oid)) THEN
            RAISE EXCEPTION
                'H1A002 role identity mismatch: %', role_name
                USING ERRCODE = '42501';
        END IF;
    END LOOP;

    -- Any direct edge involving an H1 role necessarily creates a recursive
    -- path into or out of the otherwise empty graph, including ADMIN OPTION,
    -- NOLOGIN intermediate chains, and inherited privilege paths.
    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_auth_members membership
        JOIN pg_catalog.pg_roles granted
          ON granted.oid = membership.roleid
        JOIN pg_catalog.pg_roles member_role
          ON member_role.oid = membership.member
        WHERE granted.rolname = ANY (ARRAY[
            'campaign_operations_h1_boundary_authority',
            'campaign_operations_production_enabler',
            'campaign_operations_production_disabler',
            'campaign_operations_production_dispatcher',
            'campaign_operations_production_phase5_transactional',
            'campaign_operations_production_reader',
            'campaign_operations_scheduler_protocol_evidence_owner',
            'campaign_operations_scheduler_protocol_evidence_reader'])
           OR member_role.rolname = ANY (ARRAY[
            'campaign_operations_h1_boundary_authority',
            'campaign_operations_production_enabler',
            'campaign_operations_production_disabler',
            'campaign_operations_production_dispatcher',
            'campaign_operations_production_phase5_transactional',
            'campaign_operations_production_reader',
            'campaign_operations_scheduler_protocol_evidence_owner',
            'campaign_operations_scheduler_protocol_evidence_reader'])) THEN
        RAISE EXCEPTION 'H1A003 prohibited H1 role-graph edge'
            USING ERRCODE = '42501';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles
                   WHERE rolname =
                       'campaign_operations_h1_boundary_authority') THEN
        CREATE ROLE campaign_operations_h1_boundary_authority
            NOLOGIN SUPERUSER INHERIT NOCREATEDB NOCREATEROLE NOREPLICATION
            NOBYPASSRLS CONNECTION LIMIT -1 PASSWORD NULL;
    END IF;

    FOREACH role_name IN ARRAY ARRAY[
        'campaign_operations_production_enabler',
        'campaign_operations_production_disabler',
        'campaign_operations_production_dispatcher',
        'campaign_operations_production_phase5_transactional',
        'campaign_operations_production_reader',
        'campaign_operations_scheduler_protocol_evidence_owner',
        'campaign_operations_scheduler_protocol_evidence_reader']
    LOOP
        IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles
                       WHERE rolname = role_name) THEN
            EXECUTE format(
                'CREATE ROLE %I NOLOGIN NOSUPERUSER INHERIT NOCREATEDB '
                'NOCREATEROLE NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1 '
                'PASSWORD NULL', role_name);
        END IF;
    END LOOP;
END $$;

DO $$
BEGIN
    IF EXISTS (
        WITH actual_acl(owner_oid, namespace_oid, object_type, grantee_oid,
                        privilege_type, is_grantable) AS (
            SELECT default_acl.defaclrole, default_acl.defaclnamespace,
                   default_acl.defaclobjtype, acl.grantee,
                   acl.privilege_type, acl.is_grantable
            FROM pg_catalog.pg_default_acl default_acl
            JOIN pg_catalog.pg_roles owner_role
              ON owner_role.oid = default_acl.defaclrole,
            LATERAL pg_catalog.aclexplode(default_acl.defaclacl) acl
            WHERE owner_role.rolname IN (
                'campaign_operations_h1_boundary_authority',
                'campaign_operations_owner',
                'campaign_operations_scheduler_protocol_evidence_owner')
              AND acl.grantee <> default_acl.defaclrole
        ),
        predecessor_acl(owner_oid, namespace_oid, object_type, grantee_oid,
                        privilege_type, is_grantable) AS (
            SELECT owner_role.oid, namespace.oid, predecessor.object_type,
                   pqxx_role.oid, predecessor.privilege_type, false
            FROM pg_catalog.pg_roles owner_role
            JOIN pg_catalog.pg_roles pqxx_role ON pqxx_role.rolname = 'pqxx'
            JOIN pg_catalog.pg_namespace namespace ON namespace.nspname = 'public'
            CROSS JOIN (VALUES ('r'::"char", 'SELECT'::text),
                               ('S'::"char", 'SELECT'::text),
                               ('S'::"char", 'USAGE'::text))
                       predecessor(object_type, privilege_type)
            WHERE owner_role.rolname = 'campaign_operations_owner'
        )
        SELECT 1
        WHERE EXISTS (SELECT 1 FROM actual_acl)
          AND EXISTS (
              (SELECT * FROM actual_acl EXCEPT SELECT * FROM predecessor_acl)
              UNION ALL
              (SELECT * FROM predecessor_acl EXCEPT SELECT * FROM actual_acl)
          )) THEN
        RAISE EXCEPTION 'H1A007 unsafe pre-existing default ACL'
            USING ERRCODE = '42501';
    END IF;
END $$;

-- This row is never committed.  The sealed fixed-transition owner inserts
-- it while a fixed SECURITY DEFINER transition is executing and removes it
-- before returning.  Statement failure rolls the insertion back and transaction
-- end therefore cannot leave caller-authoritative context behind.
CREATE TABLE IF NOT EXISTS campaign_operations_production_transition_context (
    backend_pid integer PRIMARY KEY,
    transaction_id bigint NOT NULL,
    transition_kind text COLLATE "C" NOT NULL CHECK (
        transition_kind IN ('enable', 'disable', 'dispatch_v2')),
    operational_request_id bigint,
    operation_key text COLLATE "C" NOT NULL CHECK (
        operation_key ~ '^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,127}$'),
    CHECK ((transition_kind = 'dispatch_v2' AND
            operational_request_id IS NOT NULL) OR
           (transition_kind IN ('enable', 'disable') AND
            operational_request_id IS NULL))
);
ALTER TABLE campaign_operations_production_transition_context
    OWNER TO campaign_operations_h1_boundary_authority;
REVOKE ALL PRIVILEGES ON campaign_operations_production_transition_context
    FROM PUBLIC, pqxx, campaign_operations_owner,
         campaign_operations_production_enabler,
         campaign_operations_production_disabler,
         campaign_operations_production_dispatcher,
         campaign_operations_production_phase5_transactional,
         campaign_operations_production_reader;

CREATE OR REPLACE FUNCTION
enforce_campaign_operations_production_context_empty_v1()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM public.campaign_operations_production_transition_context) THEN
        RAISE EXCEPTION
          'production transition context must be empty at transaction end'
          USING ERRCODE = '23514',
                CONSTRAINT =
                  'campaign_operations_production_context_empty_v1';
    END IF;
    RETURN NULL;
END;
$$;

DROP TRIGGER IF EXISTS campaign_operations_production_context_empty_v1
    ON campaign_operations_production_transition_context;
CREATE CONSTRAINT TRIGGER campaign_operations_production_context_empty_v1
AFTER INSERT OR UPDATE ON campaign_operations_production_transition_context
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_production_context_empty_v1();

CREATE TABLE IF NOT EXISTS
campaign_operations_production_enablement_event (
    production_enablement_event_id bigserial PRIMARY KEY CHECK (
        production_enablement_event_id > 0),
    event_kind text COLLATE "C" NOT NULL CHECK (
        event_kind IN ('enable', 'disable')),
    operation_key text COLLATE "C" NOT NULL UNIQUE CHECK (
        operation_key ~ '^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,127}$'),
    predecessor_event_id bigint,
    predecessor_event_canonical text COLLATE "C" NOT NULL,
    expected_prior_version integer NOT NULL CHECK (
        expected_prior_version >= 0),
    resulting_version integer NOT NULL UNIQUE CHECK (
        resulting_version = expected_prior_version + 1),
    scheduler_protocol_evidence_canonical text COLLATE "C",
    scheduler_protocol_evidence_hash text COLLATE "C",
    scheduler_required_generation integer,
    scheduler_cutover_state text COLLATE "C",
    scheduler_cutover_completed_at timestamptz,
    scheduler_cutover_completed_by text COLLATE "C",
    scheduler_cutover_executable_path text COLLATE "C",
    scheduler_cutover_process_evidence text COLLATE "C",
    independent_verification_reference text COLLATE "C",
    actor_identity text COLLATE "C" NOT NULL CHECK (
        actor_identity ~ '^[A-Za-z0-9][A-Za-z0-9._@:/+\-]{0,127}$'),
    capability text COLLATE "C" NOT NULL CHECK (capability IN (
        'campaign_operations_production_enabler',
        'campaign_operations_production_disabler')),
    manager_service_contract text COLLATE "C",
    approved_build_contract_canonical text COLLATE "C",
    approved_build_contract_hash text COLLATE "C",
    approved_build_source_commit text COLLATE "C",
    approved_build_compiler_contract text COLLATE "C",
    approved_build_executable_sha256 text COLLATE "C",
    reason text COLLATE "C" NOT NULL CHECK (
        reason <> '' AND octet_length(reason) <= 4096 AND
        reason ~ E'[^ \t\r\n]' AND
        translate(reason, E'\t\r\n', '') !~ '[[:cntrl:]]'),
    enablement_contract_version integer NOT NULL CHECK (
        enablement_contract_version = 1),
    enablement_identity_canonical text COLLATE "C" NOT NULL CHECK (
        enablement_identity_canonical <> '' AND
        octet_length(enablement_identity_canonical) <= 134217728),
    enablement_identity_hash text COLLATE "C" NOT NULL CHECK (
        enablement_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    recorded_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_production_enablement_predecessor_fk
        FOREIGN KEY (predecessor_event_id) REFERENCES
        campaign_operations_production_enablement_event(
            production_enablement_event_id) ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_production_enablement_shape CHECK (
        (event_kind = 'enable' AND
         capability = 'campaign_operations_production_enabler' AND
         scheduler_protocol_evidence_canonical IS NOT NULL AND
         scheduler_protocol_evidence_hash IS NOT NULL AND
         scheduler_protocol_evidence_hash ~ '^fnv1a64:[0-9a-f]{16}$' AND
         scheduler_required_generation IS NOT NULL AND
         scheduler_required_generation = 52 AND
         scheduler_cutover_state IS NOT NULL AND
         scheduler_cutover_state = 'complete' AND
         scheduler_cutover_completed_at IS NOT NULL AND
         scheduler_cutover_completed_by IS NOT NULL AND
         scheduler_cutover_executable_path IS NOT NULL AND
         scheduler_cutover_process_evidence IS NOT NULL AND
         independent_verification_reference IS NOT NULL AND
         independent_verification_reference <> '' AND
         manager_service_contract IS NOT NULL AND
         manager_service_contract =
            'campaign-operations-production-dispatch-and-manager-run-once-v1' AND
         approved_build_contract_canonical IS NOT NULL AND
         approved_build_contract_hash IS NOT NULL AND
         approved_build_contract_hash ~ '^fnv1a64:[0-9a-f]{16}$' AND
         approved_build_source_commit IS NOT NULL AND
         approved_build_source_commit ~ '^[0-9a-f]{40}$' AND
         approved_build_compiler_contract IS NOT NULL AND
         approved_build_executable_sha256 IS NOT NULL AND
         approved_build_executable_sha256 ~ '^sha256:[0-9a-f]{64}$') OR
        (event_kind = 'disable' AND
         capability = 'campaign_operations_production_disabler' AND
         predecessor_event_id IS NOT NULL AND
         scheduler_protocol_evidence_canonical IS NULL AND
         scheduler_protocol_evidence_hash IS NULL AND
         scheduler_required_generation IS NULL AND
         scheduler_cutover_state IS NULL AND
         scheduler_cutover_completed_at IS NULL AND
         scheduler_cutover_completed_by IS NULL AND
         scheduler_cutover_executable_path IS NULL AND
         scheduler_cutover_process_evidence IS NULL AND
         independent_verification_reference IS NULL AND
         manager_service_contract IS NULL AND
         approved_build_contract_canonical IS NULL AND
         approved_build_contract_hash IS NULL AND
         approved_build_source_commit IS NULL AND
         approved_build_compiler_contract IS NULL AND
         approved_build_executable_sha256 IS NULL))
);

ALTER TABLE campaign_operations_production_enablement_event
    ADD COLUMN IF NOT EXISTS scheduler_required_generation integer,
    ADD COLUMN IF NOT EXISTS scheduler_cutover_state text COLLATE "C",
    ADD COLUMN IF NOT EXISTS scheduler_cutover_completed_at timestamptz,
    ADD COLUMN IF NOT EXISTS scheduler_cutover_completed_by text COLLATE "C",
    ADD COLUMN IF NOT EXISTS scheduler_cutover_executable_path text COLLATE "C",
    ADD COLUMN IF NOT EXISTS scheduler_cutover_process_evidence text COLLATE "C",
    ADD COLUMN IF NOT EXISTS approved_build_source_commit text COLLATE "C",
    ADD COLUMN IF NOT EXISTS approved_build_compiler_contract text COLLATE "C",
    ADD COLUMN IF NOT EXISTS approved_build_executable_sha256 text COLLATE "C";
ALTER TABLE campaign_operations_production_enablement_event
    DROP CONSTRAINT IF EXISTS campaign_operations_production_enablement_shape,
    ADD CONSTRAINT campaign_operations_production_enablement_shape CHECK (
        (event_kind = 'enable' AND
         capability = 'campaign_operations_production_enabler' AND
         scheduler_protocol_evidence_canonical IS NOT NULL AND
         scheduler_protocol_evidence_hash IS NOT NULL AND
         scheduler_protocol_evidence_hash ~ '^fnv1a64:[0-9a-f]{16}$' AND
         scheduler_required_generation IS NOT NULL AND
         scheduler_required_generation = 52 AND
         scheduler_cutover_state IS NOT NULL AND
         scheduler_cutover_state = 'complete' AND
         scheduler_cutover_completed_at IS NOT NULL AND
         scheduler_cutover_completed_by IS NOT NULL AND
         scheduler_cutover_executable_path IS NOT NULL AND
         scheduler_cutover_process_evidence IS NOT NULL AND
         independent_verification_reference IS NOT NULL AND
         independent_verification_reference <> '' AND
         manager_service_contract IS NOT NULL AND
         manager_service_contract =
            'campaign-operations-production-dispatch-and-manager-run-once-v1' AND
         approved_build_contract_canonical IS NOT NULL AND
         approved_build_contract_hash IS NOT NULL AND
         approved_build_contract_hash ~ '^fnv1a64:[0-9a-f]{16}$' AND
         approved_build_source_commit IS NOT NULL AND
         approved_build_source_commit ~ '^[0-9a-f]{40}$' AND
         approved_build_compiler_contract IS NOT NULL AND
         approved_build_executable_sha256 IS NOT NULL AND
         approved_build_executable_sha256 ~ '^sha256:[0-9a-f]{64}$') OR
        (event_kind = 'disable' AND
         capability = 'campaign_operations_production_disabler' AND
         predecessor_event_id IS NOT NULL AND
         scheduler_protocol_evidence_canonical IS NULL AND
         scheduler_protocol_evidence_hash IS NULL AND
         scheduler_required_generation IS NULL AND
         scheduler_cutover_state IS NULL AND
         scheduler_cutover_completed_at IS NULL AND
         scheduler_cutover_completed_by IS NULL AND
         scheduler_cutover_executable_path IS NULL AND
         scheduler_cutover_process_evidence IS NULL AND
         independent_verification_reference IS NULL AND
         manager_service_contract IS NULL AND
         approved_build_contract_canonical IS NULL AND
         approved_build_contract_hash IS NULL AND
         approved_build_source_commit IS NULL AND
         approved_build_compiler_contract IS NULL AND
         approved_build_executable_sha256 IS NULL));

CREATE UNIQUE INDEX IF NOT EXISTS
campaign_operations_production_enablement_predecessor_uidx
    ON campaign_operations_production_enablement_event(predecessor_event_id)
    WHERE predecessor_event_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS
campaign_operations_production_enablement_hash_idx
    ON campaign_operations_production_enablement_event(
        enablement_identity_hash);

CREATE TABLE IF NOT EXISTS
campaign_operations_production_enablement_audit_reference_event (
    production_enablement_audit_reference_event_id bigserial PRIMARY KEY
        CHECK (production_enablement_audit_reference_event_id > 0),
    production_enablement_event_id bigint NOT NULL UNIQUE,
    operation_key text COLLATE "C" NOT NULL UNIQUE,
    event_kind text COLLATE "C" NOT NULL CHECK (
        event_kind IN ('enable', 'disable')),
    actor_identity text COLLATE "C" NOT NULL,
    capability text COLLATE "C" NOT NULL CHECK (capability IN (
        'campaign_operations_production_enabler',
        'campaign_operations_production_disabler')),
    reason text COLLATE "C" NOT NULL,
    outcome text COLLATE "C" NOT NULL CHECK (outcome = 'recorded'),
    replay_disposition text COLLATE "C" NOT NULL CHECK (
        replay_disposition = 'new_operation'),
    diagnostic_code text COLLATE "C" NOT NULL CHECK (
        diagnostic_code = 'immutable_enablement_event_recorded'),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_production_enablement_audit_event_fk
        FOREIGN KEY (production_enablement_event_id) REFERENCES
        campaign_operations_production_enablement_event(
            production_enablement_event_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS
campaign_operations_request_production_admission (
    request_production_admission_id bigserial PRIMARY KEY CHECK (
        request_production_admission_id > 0),
    operational_request_id bigint NOT NULL UNIQUE,
    request_identity_canonical text COLLATE "C" NOT NULL CHECK (
        request_identity_canonical <> ''),
    expected_request_version integer NOT NULL CHECK (
        expected_request_version > 0),
    dispatch_operation_key text COLLATE "C" NOT NULL CHECK (
        dispatch_operation_key ~
            '^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,127}$'),
    production_enablement_event_id bigint NOT NULL,
    enable_event_canonical text COLLATE "C" NOT NULL CHECK (
        enable_event_canonical <> ''),
    requesting_actor text COLLATE "C" NOT NULL CHECK (
        requesting_actor ~ '^[A-Za-z0-9][A-Za-z0-9._@:/+\-]{0,127}$'),
    original_executing_service_principal text COLLATE "C" NOT NULL CHECK (
        original_executing_service_principal ~
            '^[A-Za-z0-9][A-Za-z0-9._@:/+\-]{0,127}$'),
    approved_build_contract_canonical text COLLATE "C" NOT NULL CHECK (
        approved_build_contract_canonical <> ''),
    approved_build_contract_hash text COLLATE "C" NOT NULL CHECK (
        approved_build_contract_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    capability text COLLATE "C" NOT NULL CHECK (
        capability = 'campaign_operations_production_dispatcher'),
    admission_contract_version integer NOT NULL CHECK (
        admission_contract_version = 1),
    admission_identity_canonical text COLLATE "C" NOT NULL CHECK (
        admission_identity_canonical <> '' AND
        octet_length(admission_identity_canonical) <= 134217728),
    admission_identity_hash text COLLATE "C" NOT NULL CHECK (
        admission_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    admitted_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_request_production_admission_request_fk
        FOREIGN KEY (operational_request_id) REFERENCES
        campaign_operations_operational_request(operational_request_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_request_production_admission_enable_fk
        FOREIGN KEY (production_enablement_event_id) REFERENCES
        campaign_operations_production_enablement_event(
            production_enablement_event_id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS campaign_operations_production_admission_hash_idx
    ON campaign_operations_request_production_admission(
        admission_identity_hash);

ALTER TABLE campaign_operations_dispatch_attempt
    ADD COLUMN IF NOT EXISTS request_production_admission_id bigint,
    ADD COLUMN IF NOT EXISTS request_production_admission_canonical
        text COLLATE "C",
    ADD COLUMN IF NOT EXISTS request_production_admission_hash
        text COLLATE "C",
    ADD COLUMN IF NOT EXISTS production_enablement_event_id bigint,
    ADD COLUMN IF NOT EXISTS production_enablement_event_canonical
        text COLLATE "C",
    ADD COLUMN IF NOT EXISTS production_enablement_event_hash text COLLATE "C",
    ADD COLUMN IF NOT EXISTS operation_key text COLLATE "C",
    ADD COLUMN IF NOT EXISTS requesting_actor text COLLATE "C",
    ADD COLUMN IF NOT EXISTS original_executing_service_principal
        text COLLATE "C",
    ADD COLUMN IF NOT EXISTS approved_build_contract_canonical
        text COLLATE "C",
    ADD COLUMN IF NOT EXISTS approved_build_contract_hash text COLLATE "C",
    ADD COLUMN IF NOT EXISTS production_capability text COLLATE "C";

DO $$
DECLARE constraint_name text;
BEGIN
    FOR constraint_name IN
        SELECT conname FROM pg_constraint
        WHERE conrelid = 'campaign_operations_dispatch_attempt'::regclass
          AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE
              '%attempt_contract_version = 1%'
    LOOP
        EXECUTE format(
            'ALTER TABLE campaign_operations_dispatch_attempt '
            'DROP CONSTRAINT %I', constraint_name);
    END LOOP;
END $$;

ALTER TABLE campaign_operations_dispatch_attempt
    DROP CONSTRAINT IF EXISTS
        campaign_operations_dispatch_attempt_contract_version_check,
    DROP CONSTRAINT IF EXISTS
        campaign_operations_dispatch_attempt_v1_v2_shape,
    ADD CONSTRAINT campaign_operations_dispatch_attempt_contract_version_check
        CHECK (attempt_contract_version IN (1, 2)),
    ADD CONSTRAINT campaign_operations_dispatch_attempt_v1_v2_shape CHECK (
        (attempt_contract_version = 1 AND
         request_production_admission_id IS NULL AND
         request_production_admission_canonical IS NULL AND
         request_production_admission_hash IS NULL AND
         production_enablement_event_id IS NULL AND
         production_enablement_event_canonical IS NULL AND
         production_enablement_event_hash IS NULL AND
         operation_key IS NULL AND requesting_actor IS NULL AND
         original_executing_service_principal IS NULL AND
         approved_build_contract_canonical IS NULL AND
         approved_build_contract_hash IS NULL AND
         production_capability IS NULL) OR
        (attempt_contract_version = 2 AND
         request_production_admission_id IS NOT NULL AND
         request_production_admission_canonical IS NOT NULL AND
         request_production_admission_hash IS NOT NULL AND
         request_production_admission_hash ~ '^fnv1a64:[0-9a-f]{16}$' AND
         production_enablement_event_id IS NOT NULL AND
         production_enablement_event_canonical IS NOT NULL AND
         production_enablement_event_hash IS NOT NULL AND
         production_enablement_event_hash ~ '^fnv1a64:[0-9a-f]{16}$' AND
         operation_key IS NOT NULL AND
         operation_key ~ '^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,127}$' AND
         requesting_actor IS NOT NULL AND
         requesting_actor ~ '^[A-Za-z0-9][A-Za-z0-9._@:/+\-]{0,127}$' AND
         dispatcher_identity = requesting_actor AND
         original_executing_service_principal IS NOT NULL AND
         original_executing_service_principal ~
            '^[A-Za-z0-9][A-Za-z0-9._@:/+\-]{0,127}$' AND
         approved_build_contract_canonical IS NOT NULL AND
         approved_build_contract_hash IS NOT NULL AND
         approved_build_contract_hash ~ '^fnv1a64:[0-9a-f]{16}$' AND
         production_capability IS NOT NULL AND
         production_capability =
            'campaign_operations_production_dispatcher'));

ALTER TABLE campaign_operations_dispatch_attempt
    DROP CONSTRAINT IF EXISTS
        campaign_operations_dispatch_attempt_admission_fk,
    DROP CONSTRAINT IF EXISTS
        campaign_operations_dispatch_attempt_enablement_fk,
    ADD CONSTRAINT campaign_operations_dispatch_attempt_admission_fk
        FOREIGN KEY (request_production_admission_id) REFERENCES
        campaign_operations_request_production_admission(
            request_production_admission_id) ON DELETE RESTRICT,
    ADD CONSTRAINT campaign_operations_dispatch_attempt_enablement_fk
        FOREIGN KEY (production_enablement_event_id) REFERENCES
        campaign_operations_production_enablement_event(
            production_enablement_event_id) ON DELETE RESTRICT;

CREATE UNIQUE INDEX IF NOT EXISTS
campaign_operations_dispatch_attempt_v2_operation_uidx
    ON campaign_operations_dispatch_attempt(
        operational_request_id, operation_key)
    WHERE attempt_contract_version = 2;

ALTER TABLE campaign_operations_dispatch_audit_reference_event
    ADD COLUMN IF NOT EXISTS request_production_admission_id bigint,
    ADD COLUMN IF NOT EXISTS production_enablement_event_id bigint;

ALTER TABLE campaign_operations_dispatch_audit_reference_event
    DROP CONSTRAINT IF EXISTS
        campaign_operations_dispatch_audit_admission_fk,
    DROP CONSTRAINT IF EXISTS
        campaign_operations_dispatch_audit_enablement_fk,
    ADD CONSTRAINT campaign_operations_dispatch_audit_admission_fk
        FOREIGN KEY (request_production_admission_id) REFERENCES
        campaign_operations_request_production_admission(
            request_production_admission_id) ON DELETE RESTRICT,
    ADD CONSTRAINT campaign_operations_dispatch_audit_enablement_fk
        FOREIGN KEY (production_enablement_event_id) REFERENCES
        campaign_operations_production_enablement_event(
            production_enablement_event_id) ON DELETE RESTRICT;

DO $$
DECLARE constraint_name text;
BEGIN
    FOR constraint_name IN
        SELECT conname FROM pg_constraint
        WHERE conrelid =
              'campaign_operations_dispatch_audit_reference_event'::regclass
          AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE '%capability%'
    LOOP
        EXECUTE format(
            'ALTER TABLE campaign_operations_dispatch_audit_reference_event '
            'DROP CONSTRAINT %I', constraint_name);
    END LOOP;
END $$;

ALTER TABLE campaign_operations_dispatch_audit_reference_event
    ADD CONSTRAINT campaign_operations_dispatch_audit_capability_check CHECK (
        capability IN ('campaign_operations_dispatcher',
                       'campaign_operations_phase5_transactional',
                       'campaign_operations_cancellation_coordinator',
                       'campaign_operations_recovery',
                       'campaign_operations_production_dispatcher',
                       'campaign_operations_production_phase5_transactional')),
    ADD CONSTRAINT campaign_operations_dispatch_audit_cause_shape CHECK (
        (cause_kind = 'dispatch_lease_acquired' AND
         dispatch_attempt_outcome_id IS NULL AND
         capability IN ('campaign_operations_dispatcher',
                        'campaign_operations_production_dispatcher') AND
         outcome = 'recorded' AND
         resulting_version = prior_version + 1) OR
        (cause_kind IN ('dispatch_handoff_completed',
                        'dispatch_handoff_failed') AND
         dispatch_attempt_outcome_id IS NOT NULL AND
         capability IN ('campaign_operations_phase5_transactional',
             'campaign_operations_production_phase5_transactional')) OR
        (cause_kind = 'dispatch_lease_recovered' AND
         dispatch_attempt_outcome_id IS NOT NULL AND
         capability IN ('campaign_operations_cancellation_coordinator',
                        'campaign_operations_recovery') AND
         outcome = 'recorded' AND
         resulting_version = prior_version + 1)),
    ADD CONSTRAINT campaign_operations_dispatch_audit_production_shape CHECK (
        (request_production_admission_id IS NULL AND
         production_enablement_event_id IS NULL AND
         capability IN ('campaign_operations_dispatcher',
                        'campaign_operations_phase5_transactional',
                        'campaign_operations_cancellation_coordinator',
                        'campaign_operations_recovery')) OR
        (request_production_admission_id IS NOT NULL AND
         production_enablement_event_id IS NOT NULL AND
         capability IN ('campaign_operations_production_dispatcher',
             'campaign_operations_production_phase5_transactional')));

-- Replace only migration 047's false-only check.  Existing false rows and all
-- existing V1 identities remain byte-for-byte untouched.
DO $$
DECLARE constraint_name text;
BEGIN
    FOR constraint_name IN
        SELECT conname FROM pg_constraint
        WHERE conrelid =
              'campaign_operations_operational_request'::regclass
          AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE
              '%production_dispatch_enabled = false%'
    LOOP
        EXECUTE format(
            'ALTER TABLE campaign_operations_operational_request '
            'DROP CONSTRAINT %I', constraint_name);
    END LOOP;
END $$;

-- Superuser privilege is not role membership for the Phase H authority graph.
-- Walk only explicit direct/inherited pg_auth_members edges so readiness and
-- the retained isolated-test adapter report the actual deployment graph.
CREATE OR REPLACE FUNCTION campaign_operations_has_explicit_role_v1(
    principal_name name, target_role_name name)
RETURNS boolean
LANGUAGE sql
STABLE STRICT
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    WITH RECURSIVE memberships(role_oid) AS (
        SELECT oid FROM pg_catalog.pg_roles WHERE rolname = principal_name
        UNION
        SELECT edge.roleid
        FROM pg_catalog.pg_auth_members edge
        JOIN memberships inherited ON inherited.role_oid = edge.member
    )
    SELECT EXISTS (
        SELECT 1
        FROM memberships
        JOIN pg_catalog.pg_roles role_row
          ON role_row.oid = memberships.role_oid
        WHERE role_row.rolname = target_role_name);
$$;

CREATE OR REPLACE FUNCTION
campaign_operations_isolated_v1_authority_valid()
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT
      octet_length(convert_to(current_database(), 'UTF8')) >
        octet_length(convert_to(
          'expertadvisor_campaign_operations_phase3_test_', 'UTF8')) AND
      substring(convert_to(current_database(), 'UTF8') FROM 1 FOR
        octet_length(convert_to(
          'expertadvisor_campaign_operations_phase3_test_', 'UTF8'))) =
        convert_to('expertadvisor_campaign_operations_phase3_test_', 'UTF8') AND
      public.campaign_operations_has_explicit_role_v1(
        session_user::name, 'campaign_operations_dispatcher'::name) AND
      NOT EXISTS (
        SELECT 1
        FROM unnest(ARRAY[
          'campaign_operations_production_enabler',
          'campaign_operations_production_disabler',
          'campaign_operations_production_dispatcher',
          'campaign_operations_production_phase5_transactional',
          'campaign_operations_production_reader',
          'campaign_operations_h1_boundary_authority',
          'campaign_operations_scheduler_protocol_evidence_owner',
          'campaign_operations_scheduler_protocol_evidence_reader'])
          AS production_role(role_name)
        WHERE public.campaign_operations_has_explicit_role_v1(
          session_user::name, production_role.role_name::name)) AND
      NOT EXISTS (
        SELECT 1
        FROM public.campaign_operations_production_enablement_event);
$$;

-- Migration 048's Attempt V1 transition remains the only isolated-test
-- transition.  H1 hardens its pre-existing authority boundary; the fixed
-- production transitions created below remain ungranted to every H1 caller.
CREATE OR REPLACE FUNCTION transition_campaign_operations_request_dispatching(
    target_request_id bigint, expected_version_value integer,
    lease_digest_value text, dispatcher_value text)
RETURNS campaign_operations_operational_request
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE changed public.campaign_operations_operational_request%ROWTYPE;
BEGIN
    IF NOT public.campaign_operations_isolated_v1_authority_valid() THEN
        RAISE EXCEPTION
          'campaign operations Attempt V1 isolated authority required'
          USING ERRCODE = '42501';
    END IF;
    UPDATE public.campaign_operations_operational_request
       SET request_state = 'dispatching',
           state_version = state_version + 1,
           lease_token_hash = lease_digest_value,
           lease_expires_at = transaction_timestamp() + interval '5 minutes',
           dispatcher_identity = dispatcher_value,
           updated_at = transaction_timestamp()
     WHERE operational_request_id = target_request_id
       AND request_state = 'ready'
       AND state_version = expected_version_value
       AND lease_token_hash IS NULL
       AND lease_expires_at IS NULL
       AND dispatcher_identity IS NULL
       AND production_dispatch_enabled = false
       AND NOT EXISTS (
         SELECT 1
         FROM public.campaign_operations_production_enablement_event)
     RETURNING * INTO changed;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign operations lease compare-and-set lost'
            USING ERRCODE = 'P0001';
    END IF;
    RETURN changed;
END;
$$;

CREATE OR REPLACE FUNCTION guard_campaign_operations_attempt_v1_isolation()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF NEW.attempt_contract_version = 1 AND (
       NOT public.campaign_operations_isolated_v1_authority_valid() OR
       NOT EXISTS (
         SELECT 1
         FROM public.campaign_operations_operational_request request
         WHERE request.operational_request_id = NEW.operational_request_id
           AND request.production_dispatch_enabled = false)) THEN
        RAISE EXCEPTION
          'campaign operations Attempt V1 isolated authority required'
          USING ERRCODE = '42501';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS campaign_ops_attempt_v1_isolation
    ON campaign_operations_dispatch_attempt;
CREATE TRIGGER campaign_ops_attempt_v1_isolation
BEFORE INSERT ON campaign_operations_dispatch_attempt
FOR EACH ROW EXECUTE FUNCTION
    guard_campaign_operations_attempt_v1_isolation();

CREATE OR REPLACE FUNCTION campaign_operations_manager_build_canonical_v1(
    manager_service_contract_value text, source_commit_value text,
    compiler_contract_value text, executable_sha256_value text)
RETURNS text
LANGUAGE plpgsql
IMMUTABLE STRICT
AS $$
BEGIN
    IF manager_service_contract_value <>
           'campaign-operations-production-dispatch-and-manager-run-once-v1' OR
       source_commit_value !~ '^[0-9a-f]{40}$' OR
       compiler_contract_value = '' OR
       executable_sha256_value !~ '^sha256:[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'invalid Campaign Manager build contract'
            USING ERRCODE = '22023';
    END IF;
    RETURN 'campaign_operations_manager_build_v1' ||
        ';manager_service_contract=' ||
        octet_length(manager_service_contract_value) || ':' ||
        manager_service_contract_value ||
        ';source_commit=' || source_commit_value ||
        ';source_tree_state=clean' ||
        ';build_configuration=Release' ||
        ';compiler_contract=' || octet_length(compiler_contract_value) || ':' ||
        compiler_contract_value ||
        ';executable_sha256=' || executable_sha256_value ||
        ';build_contract_version=1';
END;
$$;

CREATE OR REPLACE FUNCTION
campaign_operations_scheduler_protocol_evidence_canonical_v1(
    required_generation_value integer, cutover_state_value text,
    cutover_completed_at_value timestamptz,
    cutover_completed_by_value text, cutover_executable_path_value text,
    cutover_process_evidence_value text)
RETURNS text
LANGUAGE plpgsql
IMMUTABLE STRICT
AS $$
DECLARE timestamp_value text;
BEGIN
    IF required_generation_value <> 52 OR
       cutover_state_value <> 'complete' OR
       cutover_completed_by_value = '' OR
       cutover_executable_path_value = '' OR
       cutover_process_evidence_value = '' THEN
        RAISE EXCEPTION 'invalid scheduler generation-52 protocol evidence'
            USING ERRCODE = '22023';
    END IF;
    timestamp_value := to_char(
        cutover_completed_at_value AT TIME ZONE 'UTC',
        'YYYY-MM-DD"T"HH24:MI:SS.US"Z"');
    RETURN 'campaign_operations_scheduler_protocol_evidence_v1' ||
        ';required_generation=52' ||
        ';cutover_state=complete' ||
        ';migration_contract=61:migration-052-scheduler-generation-52-exact-attempt-authority' ||
        ';protocol_contract=50:scheduler-generation-52-exact-attempt-authority-v1' ||
        ';cutover_completed_at=' || octet_length(timestamp_value) || ':' ||
            timestamp_value ||
        ';cutover_completed_by=' ||
            octet_length(cutover_completed_by_value) || ':' ||
            cutover_completed_by_value ||
        ';cutover_executable_path=' ||
            octet_length(cutover_executable_path_value) || ':' ||
            cutover_executable_path_value ||
        ';cutover_process_evidence=' ||
            octet_length(cutover_process_evidence_value) || ':' ||
            cutover_process_evidence_value;
END;
$$;

CREATE OR REPLACE FUNCTION
campaign_operations_production_enablement_canonical_v1(
    candidate campaign_operations_production_enablement_event)
RETURNS text
LANGUAGE plpgsql
IMMUTABLE STRICT
AS $$
BEGIN
    IF candidate.event_kind = 'enable' THEN
        RETURN 'campaign_operations_production_enable_event_v1' ||
            ';operation_key=' || octet_length(candidate.operation_key) || ':' ||
                candidate.operation_key ||
            ';predecessor_event_id=' || coalesce(
                candidate.predecessor_event_id::text, 'none') ||
            ';predecessor_event_canonical=' ||
                octet_length(candidate.predecessor_event_canonical) || ':' ||
                candidate.predecessor_event_canonical ||
            ';expected_prior_version=' || candidate.expected_prior_version ||
            ';resulting_version=' || candidate.resulting_version ||
            ';scheduler_protocol_evidence=' ||
                octet_length(candidate.scheduler_protocol_evidence_canonical) ||
                ':' || candidate.scheduler_protocol_evidence_canonical ||
            ';independent_verification_reference=' ||
                octet_length(candidate.independent_verification_reference) ||
                ':' || candidate.independent_verification_reference ||
            ';authorizing_actor=' || octet_length(candidate.actor_identity) ||
                ':' || candidate.actor_identity ||
            ';capability=campaign_operations_production_enabler' ||
            ';manager_service_contract=' ||
                octet_length(candidate.manager_service_contract) || ':' ||
                candidate.manager_service_contract ||
            ';approved_build_contract=' ||
                octet_length(candidate.approved_build_contract_canonical) ||
                ':' || candidate.approved_build_contract_canonical ||
            ';reason=' || octet_length(candidate.reason) || ':' ||
                candidate.reason ||
            ';enablement_contract_version=1';
    END IF;
    RETURN 'campaign_operations_production_disable_event_v1' ||
        ';operation_key=' || octet_length(candidate.operation_key) || ':' ||
            candidate.operation_key ||
        ';predecessor_event_id=' || candidate.predecessor_event_id ||
        ';predecessor_event_canonical=' ||
            octet_length(candidate.predecessor_event_canonical) || ':' ||
            candidate.predecessor_event_canonical ||
        ';expected_prior_version=' || candidate.expected_prior_version ||
        ';resulting_version=' || candidate.resulting_version ||
        ';disabling_actor=' || octet_length(candidate.actor_identity) || ':' ||
            candidate.actor_identity ||
        ';capability=campaign_operations_production_disabler' ||
        ';reason=' || octet_length(candidate.reason) || ':' ||
            candidate.reason ||
        ';enablement_contract_version=1';
END;
$$;

CREATE OR REPLACE FUNCTION
campaign_operations_request_production_admission_canonical_v1(
    candidate campaign_operations_request_production_admission)
RETURNS text
LANGUAGE sql
IMMUTABLE STRICT
AS $$
SELECT 'campaign_operations_request_production_admission_v1' ||
    ';operational_request_id=' || candidate.operational_request_id ||
    ';request_identity_canonical=' ||
        octet_length(candidate.request_identity_canonical) || ':' ||
        candidate.request_identity_canonical ||
    ';expected_request_version=' || candidate.expected_request_version ||
    ';dispatch_operation_key=' ||
        octet_length(candidate.dispatch_operation_key) || ':' ||
        candidate.dispatch_operation_key ||
    ';enable_event_id=' || candidate.production_enablement_event_id ||
    ';enable_event_canonical=' ||
        octet_length(candidate.enable_event_canonical) || ':' ||
        candidate.enable_event_canonical ||
    ';requesting_actor=' || octet_length(candidate.requesting_actor) || ':' ||
        candidate.requesting_actor ||
    ';original_executing_service_principal=' ||
        octet_length(candidate.original_executing_service_principal) || ':' ||
        candidate.original_executing_service_principal ||
    ';approved_build_contract=' ||
        octet_length(candidate.approved_build_contract_canonical) || ':' ||
        candidate.approved_build_contract_canonical ||
    ';capability=campaign_operations_production_dispatcher' ||
    ';admission_contract_version=1'
$$;

CREATE OR REPLACE FUNCTION
campaign_operations_dispatch_attempt_v2_canonical(
    candidate campaign_operations_dispatch_attempt)
RETURNS text
LANGUAGE sql
IMMUTABLE STRICT
AS $$
SELECT 'campaign_operations_dispatch_attempt_v2' ||
    ';operational_request_id=' || candidate.operational_request_id ||
    ';request_identity_canonical=' ||
        octet_length(candidate.request_identity_canonical) || ':' ||
        candidate.request_identity_canonical ||
    ';request_production_admission_canonical=' ||
        octet_length(candidate.request_production_admission_canonical) || ':' ||
        candidate.request_production_admission_canonical ||
    ';enable_event_id=' || candidate.production_enablement_event_id ||
    ';enable_event_canonical=' ||
        octet_length(candidate.production_enablement_event_canonical) || ':' ||
        candidate.production_enablement_event_canonical ||
    ';operation_key=' || octet_length(candidate.operation_key) || ':' ||
        candidate.operation_key ||
    ';attempt_ordinal=' || candidate.attempt_ordinal ||
    ';expected_request_version=' || candidate.expected_request_version ||
    ';resulting_request_version=' || candidate.resulting_request_version ||
    ';lease_token_digest=' || octet_length(candidate.lease_token_digest) || ':' ||
        candidate.lease_token_digest ||
    ';lease_expires_at=' || octet_length(to_char(
        candidate.lease_expires_at AT TIME ZONE 'UTC',
        'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')) || ':' || to_char(
        candidate.lease_expires_at AT TIME ZONE 'UTC',
        'YYYY-MM-DD"T"HH24:MI:SS.US"Z"') ||
    ';requesting_actor=' || octet_length(candidate.requesting_actor) || ':' ||
        candidate.requesting_actor ||
    ';original_executing_service_principal=' ||
        octet_length(candidate.original_executing_service_principal) || ':' ||
        candidate.original_executing_service_principal ||
    ';approved_build_contract=' ||
        octet_length(candidate.approved_build_contract_canonical) || ':' ||
        candidate.approved_build_contract_canonical ||
    ';capability=campaign_operations_production_dispatcher' ||
    ';attempt_contract_version=2'
$$;

CREATE OR REPLACE FUNCTION campaign_operations_production_context_valid_v1(
    transition_kind_value text, target_request_id bigint,
    operation_key_value text)
RETURNS boolean
LANGUAGE sql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT EXISTS (
        SELECT 1
        FROM public.campaign_operations_production_transition_context context
        WHERE context.backend_pid = pg_catalog.pg_backend_pid()
          AND context.transaction_id = pg_catalog.txid_current()
          AND context.transition_kind = transition_kind_value
          AND context.operational_request_id IS NOT DISTINCT FROM
              target_request_id
          AND (operation_key_value IS NULL OR
               context.operation_key = operation_key_value))
$$;

CREATE OR REPLACE FUNCTION
validate_campaign_operations_production_enablement_insert()
RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE predecessor
    public.campaign_operations_production_enablement_event%ROWTYPE;
DECLARE expected_canonical text;
BEGIN
    IF NOT public.campaign_operations_production_context_valid_v1(
         NEW.event_kind, NULL, NEW.operation_key) THEN
        RAISE EXCEPTION
          'fixed production enablement transition context required'
          USING ERRCODE = '42501';
    END IF;
    expected_canonical :=
        public.campaign_operations_production_enablement_canonical_v1(NEW);
    IF NEW.enablement_identity_canonical <> expected_canonical OR
       NEW.enablement_identity_hash <>
               public.campaign_operations_tagged_fnv1a64(expected_canonical) OR
       (NEW.event_kind = 'enable' AND (
           NEW.scheduler_protocol_evidence_hash <>
               public.campaign_operations_tagged_fnv1a64(
                   NEW.scheduler_protocol_evidence_canonical) OR
           NEW.scheduler_protocol_evidence_canonical <>
               public.campaign_operations_scheduler_protocol_evidence_canonical_v1(
                   NEW.scheduler_required_generation,
                   NEW.scheduler_cutover_state,
                   NEW.scheduler_cutover_completed_at,
                   NEW.scheduler_cutover_completed_by,
                   NEW.scheduler_cutover_executable_path,
                   NEW.scheduler_cutover_process_evidence) OR
           NEW.approved_build_contract_hash <>
               public.campaign_operations_tagged_fnv1a64(
                   NEW.approved_build_contract_canonical) OR
           NEW.approved_build_contract_canonical <>
               public.campaign_operations_manager_build_canonical_v1(
                   NEW.manager_service_contract,
                   NEW.approved_build_source_commit,
                   NEW.approved_build_compiler_contract,
                   NEW.approved_build_executable_sha256))) THEN
        RAISE EXCEPTION 'production enablement canonical mismatch'
            USING ERRCODE = '23514';
    END IF;
    IF NEW.predecessor_event_id IS NULL THEN
        IF NEW.event_kind <> 'enable' OR NEW.expected_prior_version <> 0 OR
           NEW.resulting_version <> 1 OR
           NEW.predecessor_event_canonical <> '' OR
           EXISTS (SELECT 1 FROM
               public.campaign_operations_production_enablement_event) THEN
            RAISE EXCEPTION 'production enablement genesis mismatch'
                USING ERRCODE = '23514';
        END IF;
    ELSE
        SELECT * INTO STRICT predecessor
        FROM public.campaign_operations_production_enablement_event
        WHERE production_enablement_event_id = NEW.predecessor_event_id;
        IF predecessor.enablement_identity_canonical <>
               NEW.predecessor_event_canonical OR
           predecessor.resulting_version <> NEW.expected_prior_version OR
           predecessor.event_kind = NEW.event_kind OR
           EXISTS (SELECT 1
                   FROM public.campaign_operations_production_enablement_event later
                   WHERE later.resulting_version >
                         predecessor.resulting_version) THEN
            RAISE EXCEPTION 'production enablement predecessor mismatch'
                USING ERRCODE = '23514';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
validate_campaign_ops_production_admission_insert()
RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE expected_canonical text;
BEGIN
    IF NOT public.campaign_operations_production_context_valid_v1(
         'dispatch_v2', NEW.operational_request_id,
         NEW.dispatch_operation_key) THEN
        RAISE EXCEPTION
          'fixed production acquisition transition context required'
          USING ERRCODE = '42501';
    END IF;
    expected_canonical :=
        public.campaign_operations_request_production_admission_canonical_v1(
            NEW);
    IF NEW.admission_identity_canonical <> expected_canonical OR
       NEW.admission_identity_hash <>
           public.campaign_operations_tagged_fnv1a64(expected_canonical) OR
       NEW.approved_build_contract_hash <>
           public.campaign_operations_tagged_fnv1a64(
               NEW.approved_build_contract_canonical) OR
       NOT EXISTS (
           SELECT 1
           FROM public.campaign_operations_operational_request request
           JOIN public.campaign_operations_production_enablement_event enablement
             ON enablement.production_enablement_event_id =
                NEW.production_enablement_event_id
           WHERE request.operational_request_id = NEW.operational_request_id
             AND request.request_identity_canonical =
                 NEW.request_identity_canonical
             AND request.state_version = NEW.expected_request_version
             AND NOT request.production_dispatch_enabled
             AND enablement.event_kind = 'enable'
             AND enablement.enablement_identity_canonical =
                 NEW.enable_event_canonical
             AND enablement.approved_build_contract_canonical =
                 NEW.approved_build_contract_canonical) THEN
        RAISE EXCEPTION 'request production admission canonical mismatch'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
validate_campaign_operations_dispatch_attempt_v2_insert()
RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE expected_canonical text;
BEGIN
    IF NEW.attempt_contract_version = 1 THEN RETURN NEW; END IF;
    IF NOT public.campaign_operations_production_context_valid_v1(
         'dispatch_v2', NEW.operational_request_id, NEW.operation_key) THEN
        RAISE EXCEPTION
          'fixed production acquisition transition context required'
          USING ERRCODE = '42501';
    END IF;
    expected_canonical :=
        public.campaign_operations_dispatch_attempt_v2_canonical(NEW);
    IF NEW.attempt_identity_canonical <> expected_canonical OR
       NEW.attempt_identity_hash <>
           public.campaign_operations_tagged_fnv1a64(expected_canonical) OR
       NEW.request_production_admission_hash <>
           public.campaign_operations_tagged_fnv1a64(
               NEW.request_production_admission_canonical) OR
       NEW.production_enablement_event_hash <>
           public.campaign_operations_tagged_fnv1a64(
               NEW.production_enablement_event_canonical) OR
       NEW.approved_build_contract_hash <>
           public.campaign_operations_tagged_fnv1a64(
               NEW.approved_build_contract_canonical) OR
       NOT EXISTS (
           SELECT 1
           FROM public.campaign_operations_request_production_admission admission
           WHERE admission.request_production_admission_id =
                 NEW.request_production_admission_id
             AND admission.operational_request_id =
                 NEW.operational_request_id
             AND admission.request_identity_canonical =
                 NEW.request_identity_canonical
             AND admission.admission_identity_canonical =
                 NEW.request_production_admission_canonical
             AND admission.admission_identity_hash =
                 NEW.request_production_admission_hash) THEN
        RAISE EXCEPTION 'production dispatch Attempt V2 canonical mismatch'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
guard_campaign_operations_production_admission_witness()
RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF TG_OP = 'TRUNCATE' THEN
        IF EXISTS (SELECT 1 FROM public.campaign_operations_operational_request
                   WHERE production_dispatch_enabled) THEN
            RAISE EXCEPTION 'production admission witness is immutable'
                USING ERRCODE = '55000';
        END IF;
        RETURN NULL;
    END IF;
    IF TG_OP = 'DELETE' THEN
        IF OLD.production_dispatch_enabled THEN
            RAISE EXCEPTION 'production admission witness is immutable'
                USING ERRCODE = '55000';
        END IF;
        RETURN OLD;
    END IF;
    IF OLD.production_dispatch_enabled IS NOT DISTINCT FROM
       NEW.production_dispatch_enabled THEN RETURN NEW; END IF;
    IF OLD.production_dispatch_enabled OR
       NOT NEW.production_dispatch_enabled OR
       NOT public.campaign_operations_production_context_valid_v1(
           'dispatch_v2', NEW.operational_request_id,
           NULL) THEN
        RAISE EXCEPTION 'production admission witness is immutable'
            USING ERRCODE = '55000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
enforce_campaign_operations_production_admission_consistent()
RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE target_request_id bigint;
DECLARE witness_value boolean;
DECLARE admission_count integer;
DECLARE matching_attempt_count integer;
DECLARE matching_audit_count integer;
BEGIN
    target_request_id := CASE TG_TABLE_NAME
        WHEN 'campaign_operations_operational_request' THEN
            NEW.operational_request_id
        WHEN 'campaign_operations_request_production_admission' THEN
            NEW.operational_request_id
        WHEN 'campaign_operations_dispatch_attempt' THEN
            NEW.operational_request_id
        ELSE NEW.operational_request_id END;
    SELECT production_dispatch_enabled INTO STRICT witness_value
    FROM public.campaign_operations_operational_request
    WHERE operational_request_id = target_request_id;
    SELECT count(*)::integer INTO admission_count
    FROM public.campaign_operations_request_production_admission
    WHERE operational_request_id = target_request_id;
    IF witness_value <> (admission_count = 1) THEN
        RAISE EXCEPTION 'production admission witness/evidence mismatch'
            USING ERRCODE = '23514';
    END IF;
    IF admission_count = 1 THEN
        SELECT count(*)::integer INTO matching_attempt_count
        FROM public.campaign_operations_request_production_admission admission
        JOIN public.campaign_operations_dispatch_attempt attempt
          ON attempt.request_production_admission_id =
             admission.request_production_admission_id
         AND attempt.attempt_contract_version = 2
         AND attempt.operational_request_id = admission.operational_request_id
         AND attempt.request_identity_canonical =
             admission.request_identity_canonical
         AND attempt.expected_request_version =
             admission.expected_request_version
         AND attempt.operation_key = admission.dispatch_operation_key
         AND attempt.production_enablement_event_id =
             admission.production_enablement_event_id
         AND attempt.request_production_admission_canonical =
             admission.admission_identity_canonical
        WHERE admission.operational_request_id = target_request_id;
        SELECT count(*)::integer INTO matching_audit_count
        FROM public.campaign_operations_request_production_admission admission
        JOIN public.campaign_operations_dispatch_attempt attempt
          ON attempt.request_production_admission_id =
             admission.request_production_admission_id
         AND attempt.attempt_contract_version = 2
        JOIN public.campaign_operations_dispatch_audit_reference_event audit
          ON audit.dispatch_attempt_id = attempt.dispatch_attempt_id
         AND audit.request_production_admission_id =
             admission.request_production_admission_id
         AND audit.production_enablement_event_id =
             admission.production_enablement_event_id
         AND audit.cause_kind = 'dispatch_lease_acquired'
         AND audit.capability =
             'campaign_operations_production_dispatcher'
         AND audit.actor_identity = admission.requesting_actor
         AND audit.prior_version = admission.expected_request_version
         AND audit.resulting_version =
             admission.expected_request_version + 1
        WHERE admission.operational_request_id = target_request_id;
        IF matching_attempt_count <> 1 OR matching_audit_count <> 1 THEN
            RAISE EXCEPTION 'production admission evidence incomplete'
                USING ERRCODE = '23514';
        END IF;
    END IF;
    RETURN NULL;
END;
$$;

-- Extend migration 048's deferred acquisition proof without changing its V1
-- branch.  Production dispatch requires the complete V2/admission/audit chain.
CREATE OR REPLACE FUNCTION
enforce_campaign_operations_dispatch_acquisition_complete()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF NEW.request_state <> 'dispatching' THEN RETURN NEW; END IF;
    IF OLD.request_state <> 'ready' OR
       NEW.state_version <> OLD.state_version + 1 OR
       NEW.lease_token_hash IS NULL OR NEW.lease_expires_at IS NULL OR
       NEW.dispatcher_identity IS NULL OR
       NEW.operational_campaign_id <> OLD.operational_campaign_id OR
       NEW.reservation_id <> OLD.reservation_id OR
       NEW.request_identity_canonical <> OLD.request_identity_canonical OR
       NOT EXISTS (
         SELECT 1
         FROM public.campaign_operations_dispatch_attempt attempt
         JOIN public.campaign_operations_dispatch_audit_reference_event audit
           ON audit.dispatch_attempt_id = attempt.dispatch_attempt_id
         LEFT JOIN public.campaign_operations_request_production_admission admission
           ON admission.request_production_admission_id =
              attempt.request_production_admission_id
         WHERE attempt.operational_request_id = NEW.operational_request_id
           AND attempt.request_identity_canonical =
               NEW.request_identity_canonical
           AND attempt.expected_request_version = OLD.state_version
           AND attempt.resulting_request_version = NEW.state_version
           AND attempt.lease_token_digest = NEW.lease_token_hash
           AND attempt.lease_expires_at = NEW.lease_expires_at
           AND attempt.dispatcher_identity = NEW.dispatcher_identity
           AND audit.operational_campaign_id = NEW.operational_campaign_id
           AND audit.operational_request_id = NEW.operational_request_id
           AND audit.cause_kind = 'dispatch_lease_acquired'
           AND audit.actor_identity = NEW.dispatcher_identity
           AND audit.prior_version = OLD.state_version
           AND audit.resulting_version = NEW.state_version
           AND ((NOT NEW.production_dispatch_enabled AND
                 attempt.attempt_contract_version = 1 AND
                 admission.request_production_admission_id IS NULL AND
                 audit.request_production_admission_id IS NULL AND
                 audit.production_enablement_event_id IS NULL) OR
                (NEW.production_dispatch_enabled AND
                 attempt.attempt_contract_version = 2 AND
                 admission.operational_request_id =
                     NEW.operational_request_id AND
                 audit.request_production_admission_id =
                     admission.request_production_admission_id AND
                 audit.production_enablement_event_id =
                     attempt.production_enablement_event_id))) THEN
        RAISE EXCEPTION
          'campaign operations dispatch acquisition evidence incomplete'
          USING ERRCODE = '23514',
                CONSTRAINT =
                  'campaign_operations_dispatch_acquisition_complete';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
enforce_campaign_ops_enablement_audit_complete()
RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE target_event_id bigint;
BEGIN
    target_event_id := CASE TG_TABLE_NAME
        WHEN 'campaign_operations_production_enablement_event' THEN
            NEW.production_enablement_event_id
        ELSE NEW.production_enablement_event_id END;
    IF NOT EXISTS (
        SELECT 1
        FROM public.campaign_operations_production_enablement_event event
        JOIN public.campaign_operations_production_enablement_audit_reference_event audit
          ON audit.production_enablement_event_id =
             event.production_enablement_event_id
         AND audit.operation_key = event.operation_key
         AND audit.event_kind = event.event_kind
         AND audit.actor_identity = event.actor_identity
         AND audit.capability = event.capability
         AND audit.reason = event.reason
        WHERE event.production_enablement_event_id = target_event_id) THEN
        RAISE EXCEPTION 'production enablement audit incomplete'
            USING ERRCODE = '23514';
    END IF;
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION reject_campaign_operations_production_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'Campaign Operations production evidence is immutable'
        USING ERRCODE = '55000';
END;
$$;

CREATE OR REPLACE FUNCTION
guard_campaign_operations_production_attempt_mutation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF TG_OP = 'TRUNCATE' THEN
        IF EXISTS (
          SELECT 1 FROM public.campaign_operations_dispatch_attempt
          WHERE attempt_contract_version = 2) THEN
            RAISE EXCEPTION
              'Campaign Operations production attempt evidence is immutable'
              USING ERRCODE = '55000';
        END IF;
        RETURN NULL;
    END IF;
    IF OLD.attempt_contract_version = 2 OR
       (TG_OP = 'UPDATE' AND NEW.attempt_contract_version = 2) THEN
        RAISE EXCEPTION
          'Campaign Operations production attempt evidence is immutable'
          USING ERRCODE = '55000';
    END IF;
    RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
END;
$$;

CREATE OR REPLACE FUNCTION
guard_campaign_operations_production_dispatch_audit_mutation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF TG_OP = 'TRUNCATE' THEN
        IF EXISTS (
          SELECT 1
          FROM public.campaign_operations_dispatch_audit_reference_event
          WHERE request_production_admission_id IS NOT NULL) THEN
            RAISE EXCEPTION
              'Campaign Operations production dispatch audit is immutable'
              USING ERRCODE = '55000';
        END IF;
        RETURN NULL;
    END IF;
    IF OLD.request_production_admission_id IS NOT NULL OR
       (TG_OP = 'UPDATE' AND
        NEW.request_production_admission_id IS NOT NULL) THEN
        RAISE EXCEPTION
          'Campaign Operations production dispatch audit is immutable'
          USING ERRCODE = '55000';
    END IF;
    RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
END;
$$;

CREATE OR REPLACE FUNCTION
guard_campaign_operations_production_dispatch_audit_insert()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE operation_key_value text;
BEGIN
    IF NEW.request_production_admission_id IS NULL THEN RETURN NEW; END IF;
    SELECT attempt.operation_key INTO STRICT operation_key_value
    FROM public.campaign_operations_dispatch_attempt attempt
    WHERE attempt.dispatch_attempt_id = NEW.dispatch_attempt_id
      AND attempt.operational_request_id = NEW.operational_request_id
      AND attempt.request_production_admission_id =
          NEW.request_production_admission_id
      AND attempt.production_enablement_event_id =
          NEW.production_enablement_event_id
      AND attempt.attempt_contract_version = 2;
    IF NOT public.campaign_operations_production_context_valid_v1(
         'dispatch_v2', NEW.operational_request_id, operation_key_value) THEN
        RAISE EXCEPTION
          'fixed production acquisition transition context required'
          USING ERRCODE = '42501';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
guard_campaign_operations_production_post_completion()
RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE target_request_id bigint;
DECLARE target_campaign_id bigint;
DECLARE boundary_closed boolean;
BEGIN
    target_request_id := CASE TG_TABLE_NAME
        WHEN 'campaign_operations_request_production_admission' THEN
            NEW.operational_request_id
        WHEN 'campaign_operations_dispatch_attempt' THEN
            NEW.operational_request_id
        ELSE NEW.operational_request_id END;
    SELECT request.operational_campaign_id, campaign.completion_boundary_closed
      INTO STRICT target_campaign_id, boundary_closed
    FROM public.campaign_operations_operational_request request
    JOIN public.campaign_operations_campaign campaign
      ON campaign.operational_campaign_id = request.operational_campaign_id
    WHERE request.operational_request_id = target_request_id
    FOR UPDATE OF campaign;
    IF boundary_closed THEN
        RAISE EXCEPTION 'completed campaign rejects production evidence'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
guard_campaign_operations_completion_v2_evidence()
RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM public.campaign_operations_operational_request request
        JOIN public.campaign_operations_dispatch_attempt attempt
          ON attempt.operational_request_id = request.operational_request_id
         AND attempt.attempt_contract_version = 2
        LEFT JOIN public.campaign_operations_request_production_admission admission
          ON admission.request_production_admission_id =
             attempt.request_production_admission_id
        LEFT JOIN public.campaign_operations_dispatch_audit_reference_event audit
          ON audit.dispatch_attempt_id = attempt.dispatch_attempt_id
         AND audit.cause_kind = 'dispatch_lease_acquired'
        WHERE request.operational_campaign_id = NEW.operational_campaign_id
          AND (NOT request.production_dispatch_enabled OR
               admission.request_production_admission_id IS NULL OR
               audit.dispatch_audit_reference_event_id IS NULL)) THEN
        RAISE EXCEPTION 'completion V1 nested Attempt V2 evidence incomplete'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS campaign_operations_production_enablement_validate
    ON campaign_operations_production_enablement_event;
CREATE TRIGGER campaign_operations_production_enablement_validate
BEFORE INSERT ON campaign_operations_production_enablement_event
FOR EACH ROW EXECUTE FUNCTION
    validate_campaign_operations_production_enablement_insert();
DROP TRIGGER IF EXISTS campaign_operations_production_admission_validate
    ON campaign_operations_request_production_admission;
CREATE TRIGGER campaign_operations_production_admission_validate
BEFORE INSERT ON campaign_operations_request_production_admission
FOR EACH ROW EXECUTE FUNCTION
    validate_campaign_ops_production_admission_insert();
DROP TRIGGER IF EXISTS campaign_operations_production_admission_post_completion
    ON campaign_operations_request_production_admission;
CREATE TRIGGER campaign_operations_production_admission_post_completion
BEFORE INSERT ON campaign_operations_request_production_admission
FOR EACH ROW EXECUTE FUNCTION
    guard_campaign_operations_production_post_completion();
DROP TRIGGER IF EXISTS campaign_operations_dispatch_attempt_v2_validate
    ON campaign_operations_dispatch_attempt;
CREATE TRIGGER campaign_operations_dispatch_attempt_v2_validate
BEFORE INSERT ON campaign_operations_dispatch_attempt
FOR EACH ROW EXECUTE FUNCTION
    validate_campaign_operations_dispatch_attempt_v2_insert();
DROP TRIGGER IF EXISTS campaign_operations_production_audit_insert_guard
    ON campaign_operations_dispatch_audit_reference_event;
CREATE TRIGGER campaign_operations_production_audit_insert_guard
BEFORE INSERT ON campaign_operations_dispatch_audit_reference_event
FOR EACH ROW EXECUTE FUNCTION
    guard_campaign_operations_production_dispatch_audit_insert();

DROP TRIGGER IF EXISTS campaign_operations_production_witness_update_guard
    ON campaign_operations_operational_request;
CREATE TRIGGER campaign_operations_production_witness_update_guard
BEFORE UPDATE OF production_dispatch_enabled OR DELETE
ON campaign_operations_operational_request
FOR EACH ROW EXECUTE FUNCTION
    guard_campaign_operations_production_admission_witness();
DROP TRIGGER IF EXISTS campaign_operations_production_witness_truncate_guard
    ON campaign_operations_operational_request;
CREATE TRIGGER campaign_operations_production_witness_truncate_guard
BEFORE TRUNCATE ON campaign_operations_operational_request
FOR EACH STATEMENT EXECUTE FUNCTION
    guard_campaign_operations_production_admission_witness();

DROP TRIGGER IF EXISTS campaign_operations_production_request_consistency
    ON campaign_operations_operational_request;
CREATE CONSTRAINT TRIGGER campaign_operations_production_request_consistency
AFTER INSERT OR UPDATE OF production_dispatch_enabled
ON campaign_operations_operational_request
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_production_admission_consistent();
DROP TRIGGER IF EXISTS campaign_operations_production_admission_consistency
    ON campaign_operations_request_production_admission;
CREATE CONSTRAINT TRIGGER
campaign_operations_production_admission_consistency
AFTER INSERT ON campaign_operations_request_production_admission
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_production_admission_consistent();
DROP TRIGGER IF EXISTS campaign_operations_production_attempt_consistency
    ON campaign_operations_dispatch_attempt;
CREATE CONSTRAINT TRIGGER campaign_operations_production_attempt_consistency
AFTER INSERT ON campaign_operations_dispatch_attempt
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
WHEN (NEW.attempt_contract_version = 2)
EXECUTE FUNCTION enforce_campaign_operations_production_admission_consistent();
DROP TRIGGER IF EXISTS campaign_operations_production_audit_consistency
    ON campaign_operations_dispatch_audit_reference_event;
CREATE CONSTRAINT TRIGGER campaign_operations_production_audit_consistency
AFTER INSERT ON campaign_operations_dispatch_audit_reference_event
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
WHEN (NEW.request_production_admission_id IS NOT NULL)
EXECUTE FUNCTION enforce_campaign_operations_production_admission_consistent();

DROP TRIGGER IF EXISTS campaign_operations_enablement_audit_event_complete
    ON campaign_operations_production_enablement_event;
CREATE CONSTRAINT TRIGGER campaign_operations_enablement_audit_event_complete
AFTER INSERT ON campaign_operations_production_enablement_event
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_ops_enablement_audit_complete();
DROP TRIGGER IF EXISTS campaign_operations_enablement_audit_complete
    ON campaign_operations_production_enablement_audit_reference_event;
CREATE CONSTRAINT TRIGGER campaign_operations_enablement_audit_complete
AFTER INSERT ON
    campaign_operations_production_enablement_audit_reference_event
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_ops_enablement_audit_complete();

DROP TRIGGER IF EXISTS campaign_ops_enablement_immutable_row
    ON campaign_operations_production_enablement_event;
CREATE TRIGGER campaign_ops_enablement_immutable_row
BEFORE UPDATE OR DELETE ON
    campaign_operations_production_enablement_event
FOR EACH ROW EXECUTE FUNCTION
    reject_campaign_operations_production_mutation();
DROP TRIGGER IF EXISTS campaign_ops_enablement_immutable_truncate
    ON campaign_operations_production_enablement_event;
CREATE TRIGGER campaign_ops_enablement_immutable_truncate
BEFORE TRUNCATE ON campaign_operations_production_enablement_event
FOR EACH STATEMENT EXECUTE FUNCTION
    reject_campaign_operations_production_mutation();
DROP TRIGGER IF EXISTS campaign_ops_enablement_audit_immutable_row
    ON campaign_operations_production_enablement_audit_reference_event;
CREATE TRIGGER campaign_ops_enablement_audit_immutable_row
BEFORE UPDATE OR DELETE ON
    campaign_operations_production_enablement_audit_reference_event
FOR EACH ROW EXECUTE FUNCTION
    reject_campaign_operations_production_mutation();
DROP TRIGGER IF EXISTS campaign_ops_enablement_audit_immutable_truncate
    ON campaign_operations_production_enablement_audit_reference_event;
CREATE TRIGGER campaign_ops_enablement_audit_immutable_truncate
BEFORE TRUNCATE ON
    campaign_operations_production_enablement_audit_reference_event
FOR EACH STATEMENT EXECUTE FUNCTION
    reject_campaign_operations_production_mutation();
DROP TRIGGER IF EXISTS campaign_ops_admission_immutable_row
    ON campaign_operations_request_production_admission;
CREATE TRIGGER campaign_ops_admission_immutable_row
BEFORE UPDATE OR DELETE ON
    campaign_operations_request_production_admission
FOR EACH ROW EXECUTE FUNCTION
    reject_campaign_operations_production_mutation();
DROP TRIGGER IF EXISTS campaign_ops_admission_immutable_truncate
    ON campaign_operations_request_production_admission;
CREATE TRIGGER campaign_ops_admission_immutable_truncate
BEFORE TRUNCATE ON campaign_operations_request_production_admission
FOR EACH STATEMENT EXECUTE FUNCTION
    reject_campaign_operations_production_mutation();

DROP TRIGGER IF EXISTS campaign_ops_attempt_v2_immutable_row
    ON campaign_operations_dispatch_attempt;
CREATE TRIGGER campaign_ops_attempt_v2_immutable_row
BEFORE UPDATE OR DELETE ON campaign_operations_dispatch_attempt
FOR EACH ROW EXECUTE FUNCTION
    guard_campaign_operations_production_attempt_mutation();
DROP TRIGGER IF EXISTS campaign_ops_attempt_v2_immutable_truncate
    ON campaign_operations_dispatch_attempt;
CREATE TRIGGER campaign_ops_attempt_v2_immutable_truncate
BEFORE TRUNCATE ON campaign_operations_dispatch_attempt
FOR EACH STATEMENT EXECUTE FUNCTION
    guard_campaign_operations_production_attempt_mutation();

DROP TRIGGER IF EXISTS campaign_ops_production_audit_immutable_row
    ON campaign_operations_dispatch_audit_reference_event;
CREATE TRIGGER campaign_ops_production_audit_immutable_row
BEFORE UPDATE OR DELETE ON
    campaign_operations_dispatch_audit_reference_event
FOR EACH ROW EXECUTE FUNCTION
    guard_campaign_operations_production_dispatch_audit_mutation();
DROP TRIGGER IF EXISTS campaign_ops_production_audit_immutable_truncate
    ON campaign_operations_dispatch_audit_reference_event;
CREATE TRIGGER campaign_ops_production_audit_immutable_truncate
BEFORE TRUNCATE ON campaign_operations_dispatch_audit_reference_event
FOR EACH STATEMENT EXECUTE FUNCTION
    guard_campaign_operations_production_dispatch_audit_mutation();

DROP TRIGGER IF EXISTS campaign_operations_completion_v2_evidence_gate
    ON campaign_operations_completion_event;
CREATE TRIGGER campaign_operations_completion_v2_evidence_gate
BEFORE INSERT ON campaign_operations_completion_event
FOR EACH ROW EXECUTE FUNCTION
    guard_campaign_operations_completion_v2_evidence();

CREATE OR REPLACE FUNCTION
campaign_operations_scheduler_protocol_evidence_snapshot_v1()
RETURNS TABLE(required_generation integer, cutover_state text,
    cutover_completed_at text, cutover_completed_by text,
    cutover_executable_path text, cutover_process_evidence text,
    evidence_complete boolean, evidence_canonical text, evidence_hash text)
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE generation_value integer;
DECLARE state_value text;
DECLARE completed_at_value timestamptz;
DECLARE completed_by_value text;
DECLARE executable_path_value text;
DECLARE process_evidence_value text;
BEGIN
    SELECT protocol.required_generation, protocol.cutover_state,
           protocol.cutover_completed_at, protocol.cutover_completed_by,
           protocol.cutover_executable_path,
           protocol.cutover_process_evidence
      INTO generation_value, state_value, completed_at_value,
           completed_by_value, executable_path_value, process_evidence_value
    FROM public.experiment_scheduler_protocol protocol
    WHERE protocol.singleton;
    required_generation := generation_value;
    cutover_state := state_value;
    cutover_completed_at := CASE WHEN completed_at_value IS NULL THEN NULL
        ELSE to_char(completed_at_value AT TIME ZONE 'UTC',
            'YYYY-MM-DD"T"HH24:MI:SS.US"Z"') END;
    cutover_completed_by := completed_by_value;
    cutover_executable_path := executable_path_value;
    cutover_process_evidence := process_evidence_value;
    evidence_complete := coalesce(generation_value = 52 AND
        state_value = 'complete' AND completed_at_value IS NOT NULL AND
        completed_by_value IS NOT NULL AND
        executable_path_value IS NOT NULL AND
        process_evidence_value IS NOT NULL, false);
    IF evidence_complete THEN
        evidence_canonical :=
            public.campaign_operations_scheduler_protocol_evidence_canonical_v1(
                generation_value, state_value, completed_at_value,
                completed_by_value, executable_path_value,
                process_evidence_value);
        evidence_hash :=
            public.campaign_operations_tagged_fnv1a64(evidence_canonical);
    END IF;
    RETURN NEXT;
END;
$$;

CREATE OR REPLACE FUNCTION
campaign_operations_scheduler_protocol_evidence_lock_v1()
RETURNS TABLE(required_generation integer, cutover_state text,
    cutover_completed_at text, cutover_completed_by text,
    cutover_executable_path text, cutover_process_evidence text,
    evidence_complete boolean, evidence_canonical text, evidence_hash text)
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE generation_value integer;
DECLARE state_value text;
DECLARE completed_at_value timestamptz;
DECLARE completed_by_value text;
DECLARE executable_path_value text;
DECLARE process_evidence_value text;
BEGIN
    SELECT protocol.required_generation, protocol.cutover_state,
           protocol.cutover_completed_at, protocol.cutover_completed_by,
           protocol.cutover_executable_path,
           protocol.cutover_process_evidence
      INTO generation_value, state_value, completed_at_value,
           completed_by_value, executable_path_value, process_evidence_value
    FROM public.experiment_scheduler_protocol protocol
    WHERE protocol.singleton FOR SHARE;
    required_generation := generation_value;
    cutover_state := state_value;
    cutover_completed_at := CASE WHEN completed_at_value IS NULL THEN NULL
        ELSE to_char(completed_at_value AT TIME ZONE 'UTC',
            'YYYY-MM-DD"T"HH24:MI:SS.US"Z"') END;
    cutover_completed_by := completed_by_value;
    cutover_executable_path := executable_path_value;
    cutover_process_evidence := process_evidence_value;
    evidence_complete := coalesce(generation_value = 52 AND
        state_value = 'complete' AND completed_at_value IS NOT NULL AND
        completed_by_value IS NOT NULL AND
        executable_path_value IS NOT NULL AND
        process_evidence_value IS NOT NULL, false);
    IF evidence_complete THEN
        evidence_canonical :=
            public.campaign_operations_scheduler_protocol_evidence_canonical_v1(
                generation_value, state_value, completed_at_value,
                completed_by_value, executable_path_value,
                process_evidence_value);
        evidence_hash :=
            public.campaign_operations_tagged_fnv1a64(evidence_canonical);
    END IF;
    RETURN NEXT;
END;
$$;

-- This is the sole replay-time proof for an immutable enablement lineage.  It
-- deliberately walks from the referenced event all the way to a valid genesis
-- node; callers must not infer integrity from only an immediate predecessor.
CREATE OR REPLACE FUNCTION
campaign_operations_production_enablement_history_valid_v1(
    start_event_id bigint)
RETURNS boolean
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE event public.campaign_operations_production_enablement_event%ROWTYPE;
DECLARE predecessor
    public.campaign_operations_production_enablement_event%ROWTYPE;
DECLARE current_event_id bigint := start_event_id;
DECLARE expected_canonical text;
DECLARE expected_scheduler_canonical text;
DECLARE expected_build_canonical text;
DECLARE audit_total_count integer;
DECLARE audit_match_count integer;
DECLARE operation_key_count integer;
DECLARE resulting_version_count integer;
DECLARE successor_count integer;
DECLARE visited_event_ids bigint[] := ARRAY[]::bigint[];
DECLARE remaining_depth integer := 1024;
BEGIN
    IF current_event_id IS NULL OR current_event_id <= 0 THEN
        RETURN false;
    END IF;
    LOOP
        IF remaining_depth <= 0 OR current_event_id = ANY(visited_event_ids)
        THEN
            RETURN false;
        END IF;
        remaining_depth := remaining_depth - 1;
        visited_event_ids := array_append(visited_event_ids, current_event_id);
        SELECT candidate.* INTO event
        FROM public.campaign_operations_production_enablement_event candidate
        WHERE candidate.production_enablement_event_id = current_event_id;
        IF NOT FOUND THEN
            RETURN false;
        END IF;
        SELECT count(*)::integer INTO operation_key_count
        FROM public.campaign_operations_production_enablement_event candidate
        WHERE candidate.operation_key = event.operation_key;
        SELECT count(*)::integer INTO resulting_version_count
        FROM public.campaign_operations_production_enablement_event candidate
        WHERE candidate.resulting_version = event.resulting_version;
        SELECT count(*)::integer INTO successor_count
        FROM public.campaign_operations_production_enablement_event candidate
        WHERE candidate.predecessor_event_id =
              event.production_enablement_event_id;
        SELECT count(*)::integer INTO audit_total_count
        FROM public.campaign_operations_production_enablement_audit_reference_event
             audit
        WHERE audit.production_enablement_event_id =
              event.production_enablement_event_id;
        SELECT count(*)::integer INTO audit_match_count
        FROM public.campaign_operations_production_enablement_audit_reference_event
             audit
        WHERE audit.production_enablement_event_id =
              event.production_enablement_event_id
          AND audit.operation_key = event.operation_key
          AND audit.event_kind = event.event_kind
          AND audit.actor_identity = event.actor_identity
          AND audit.capability = event.capability
          AND audit.reason = event.reason
          AND audit.outcome = 'recorded'
          AND audit.replay_disposition = 'new_operation'
          AND audit.diagnostic_code =
              'immutable_enablement_event_recorded';
        IF event.production_enablement_event_id IS NULL OR
           event.production_enablement_event_id <= 0 OR
           event.recorded_at IS NULL OR event.operation_key IS NULL OR
           event.operation_key !~ '^[A-Za-z0-9][A-Za-z0-9._:/\\-]{0,127}$' OR
           event.actor_identity IS NULL OR
           event.actor_identity !~ '^[A-Za-z0-9][A-Za-z0-9._@:/+\\-]{0,127}$' OR
           event.reason IS NULL OR event.reason = '' OR
           octet_length(event.reason) > 4096 OR
           event.reason !~ E'[^ \\t\\r\\n]' OR
           translate(event.reason, E'\\t\\r\\n', '') ~ '[[:cntrl:]]' OR
           event.enablement_contract_version IS DISTINCT FROM 1 OR
           event.expected_prior_version IS NULL OR
           event.expected_prior_version < 0 OR
           event.resulting_version IS DISTINCT FROM
               event.expected_prior_version::bigint + 1 OR
           event.enablement_identity_canonical IS NULL OR
           event.enablement_identity_canonical = '' OR
           event.enablement_identity_hash IS NULL OR
           event.enablement_identity_hash !~ '^fnv1a64:[0-9a-f]{16}$' OR
           operation_key_count <> 1 OR resulting_version_count <> 1 OR
           successor_count > 1 OR audit_total_count <> 1 OR
           audit_match_count <> 1 THEN
            RETURN false;
        END IF;
        BEGIN
            expected_canonical :=
              public.campaign_operations_production_enablement_canonical_v1(
                event);
            IF event.event_kind = 'enable' THEN
                expected_scheduler_canonical :=
                  public.campaign_operations_scheduler_protocol_evidence_canonical_v1(
                    event.scheduler_required_generation,
                    event.scheduler_cutover_state,
                    event.scheduler_cutover_completed_at,
                    event.scheduler_cutover_completed_by,
                    event.scheduler_cutover_executable_path,
                    event.scheduler_cutover_process_evidence);
                expected_build_canonical :=
                  public.campaign_operations_manager_build_canonical_v1(
                    event.manager_service_contract,
                    event.approved_build_source_commit,
                    event.approved_build_compiler_contract,
                    event.approved_build_executable_sha256);
            END IF;
        EXCEPTION WHEN OTHERS THEN
            RETURN false;
        END;
        IF event.enablement_identity_canonical IS DISTINCT FROM
               expected_canonical OR
           event.enablement_identity_hash IS DISTINCT FROM
               public.campaign_operations_tagged_fnv1a64(expected_canonical)
        THEN
            RETURN false;
        END IF;
        IF event.event_kind = 'enable' THEN
            IF event.capability IS DISTINCT FROM
                   'campaign_operations_production_enabler' OR
               event.scheduler_protocol_evidence_canonical IS DISTINCT FROM
                   expected_scheduler_canonical OR
               event.scheduler_protocol_evidence_hash IS DISTINCT FROM
                   public.campaign_operations_tagged_fnv1a64(
                     expected_scheduler_canonical) OR
               event.approved_build_contract_canonical IS DISTINCT FROM
                   expected_build_canonical OR
               event.approved_build_contract_hash IS DISTINCT FROM
                   public.campaign_operations_tagged_fnv1a64(
                     expected_build_canonical) OR
               event.independent_verification_reference IS NULL OR
               event.independent_verification_reference = '' THEN
                RETURN false;
            END IF;
        ELSIF event.event_kind = 'disable' THEN
            IF event.capability IS DISTINCT FROM
                   'campaign_operations_production_disabler' OR
               event.scheduler_protocol_evidence_canonical IS NOT NULL OR
               event.scheduler_protocol_evidence_hash IS NOT NULL OR
               event.scheduler_required_generation IS NOT NULL OR
               event.scheduler_cutover_state IS NOT NULL OR
               event.scheduler_cutover_completed_at IS NOT NULL OR
               event.scheduler_cutover_completed_by IS NOT NULL OR
               event.scheduler_cutover_executable_path IS NOT NULL OR
               event.scheduler_cutover_process_evidence IS NOT NULL OR
               event.independent_verification_reference IS NOT NULL OR
               event.manager_service_contract IS NOT NULL OR
               event.approved_build_contract_canonical IS NOT NULL OR
               event.approved_build_contract_hash IS NOT NULL OR
               event.approved_build_source_commit IS NOT NULL OR
               event.approved_build_compiler_contract IS NOT NULL OR
               event.approved_build_executable_sha256 IS NOT NULL THEN
                RETURN false;
            END IF;
        ELSE
            RETURN false;
        END IF;
        IF event.predecessor_event_id IS NULL THEN
            RETURN event.event_kind = 'enable' AND
                event.expected_prior_version = 0 AND
                event.resulting_version = 1 AND
                event.predecessor_event_canonical = '';
        END IF;
        SELECT candidate.* INTO predecessor
        FROM public.campaign_operations_production_enablement_event candidate
        WHERE candidate.production_enablement_event_id =
              event.predecessor_event_id;
        IF NOT FOUND OR event.predecessor_event_id <= 0 OR
           predecessor.event_kind IS NOT DISTINCT FROM event.event_kind OR
           predecessor.resulting_version IS DISTINCT FROM
               event.expected_prior_version OR
           predecessor.enablement_identity_canonical IS DISTINCT FROM
               event.predecessor_event_canonical THEN
            RETURN false;
        END IF;
        current_event_id := event.predecessor_event_id;
    END LOOP;
END;
$$;

-- Replay readers validate complete immutable evidence without consulting the
-- mutable enablement head or request state.  They return zero rows only when
-- the exact durable operation is absent.
CREATE OR REPLACE FUNCTION campaign_operations_production_enable_replay_v1(
    operation_key_value text, expected_prior_version_value integer,
    scheduler_evidence_canonical_value text,
    independent_verification_reference_value text,
    authorizing_actor_value text, manager_service_contract_value text,
    approved_build_contract_canonical_value text,
    approved_build_source_commit_value text,
    approved_build_compiler_contract_value text,
    approved_build_executable_sha256_value text, reason_value text)
RETURNS SETOF public.campaign_operations_production_enablement_event
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE event public.campaign_operations_production_enablement_event%ROWTYPE;
DECLARE expected_canonical text;
DECLARE expected_scheduler_canonical text;
DECLARE expected_build_canonical text;
DECLARE audit_count integer;
BEGIN
    SELECT candidate.* INTO event
    FROM public.campaign_operations_production_enablement_event candidate
    WHERE candidate.operation_key = operation_key_value;
    IF NOT FOUND THEN RETURN; END IF;
    IF NOT public.campaign_operations_production_enablement_history_valid_v1(
             event.production_enablement_event_id) THEN
        RAISE EXCEPTION 'production enable replay evidence corrupt'
            USING ERRCODE = '23514';
    END IF;
    BEGIN
        expected_canonical :=
          public.campaign_operations_production_enablement_canonical_v1(event);
        expected_scheduler_canonical :=
          public.campaign_operations_scheduler_protocol_evidence_canonical_v1(
            event.scheduler_required_generation, event.scheduler_cutover_state,
            event.scheduler_cutover_completed_at,
            event.scheduler_cutover_completed_by,
            event.scheduler_cutover_executable_path,
            event.scheduler_cutover_process_evidence);
        expected_build_canonical :=
          public.campaign_operations_manager_build_canonical_v1(
            event.manager_service_contract,
            event.approved_build_source_commit,
            event.approved_build_compiler_contract,
            event.approved_build_executable_sha256);
    EXCEPTION WHEN OTHERS THEN
        RAISE EXCEPTION 'production enable replay evidence corrupt'
            USING ERRCODE = '23514';
    END;
    SELECT count(*)::integer INTO audit_count
    FROM public.campaign_operations_production_enablement_audit_reference_event audit
    WHERE audit.production_enablement_event_id =
              event.production_enablement_event_id
      AND audit.operation_key = event.operation_key
      AND audit.event_kind = event.event_kind
      AND audit.actor_identity = event.actor_identity
      AND audit.capability = event.capability
      AND audit.reason = event.reason
      AND audit.outcome = 'recorded'
      AND audit.replay_disposition = 'new_operation'
      AND audit.diagnostic_code = 'immutable_enablement_event_recorded';
    IF event.event_kind <> 'enable' OR
       event.capability <> 'campaign_operations_production_enabler' OR
       event.enablement_contract_version <> 1 OR audit_count <> 1 OR
       event.enablement_identity_canonical <> expected_canonical OR
       event.enablement_identity_hash <>
         public.campaign_operations_tagged_fnv1a64(expected_canonical) OR
       event.scheduler_protocol_evidence_canonical <>
         expected_scheduler_canonical OR
       event.scheduler_protocol_evidence_hash <>
         public.campaign_operations_tagged_fnv1a64(
           expected_scheduler_canonical) OR
       event.approved_build_contract_canonical <> expected_build_canonical OR
       event.approved_build_contract_hash <>
         public.campaign_operations_tagged_fnv1a64(expected_build_canonical)
    THEN
        RAISE EXCEPTION 'production enable replay evidence corrupt'
            USING ERRCODE = '23514';
    END IF;
    IF event.expected_prior_version IS DISTINCT FROM
           expected_prior_version_value OR
       event.scheduler_protocol_evidence_canonical IS DISTINCT FROM
           scheduler_evidence_canonical_value OR
       event.independent_verification_reference IS DISTINCT FROM
           independent_verification_reference_value OR
       event.actor_identity IS DISTINCT FROM authorizing_actor_value OR
       event.manager_service_contract IS DISTINCT FROM
           manager_service_contract_value OR
       event.approved_build_contract_canonical IS DISTINCT FROM
           approved_build_contract_canonical_value OR
       event.approved_build_source_commit IS DISTINCT FROM
           approved_build_source_commit_value OR
       event.approved_build_compiler_contract IS DISTINCT FROM
           approved_build_compiler_contract_value OR
       event.approved_build_executable_sha256 IS DISTINCT FROM
           approved_build_executable_sha256_value OR
       event.reason IS DISTINCT FROM reason_value THEN
        RAISE EXCEPTION 'production enable conflicting replay'
            USING ERRCODE = '23505';
    END IF;
    RETURN NEXT event;
END;
$$;

CREATE OR REPLACE FUNCTION campaign_operations_production_disable_replay_v1(
    operation_key_value text, predecessor_event_id_value bigint,
    predecessor_event_canonical_value text,
    expected_prior_version_value integer, disabling_actor_value text,
    reason_value text)
RETURNS SETOF public.campaign_operations_production_enablement_event
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE event public.campaign_operations_production_enablement_event%ROWTYPE;
DECLARE expected_canonical text;
DECLARE audit_count integer;
BEGIN
    SELECT candidate.* INTO event
    FROM public.campaign_operations_production_enablement_event candidate
    WHERE candidate.operation_key = operation_key_value;
    IF NOT FOUND THEN RETURN; END IF;
    IF NOT public.campaign_operations_production_enablement_history_valid_v1(
             event.production_enablement_event_id) THEN
        RAISE EXCEPTION 'production disable replay evidence corrupt'
            USING ERRCODE = '23514';
    END IF;
    BEGIN
        expected_canonical :=
          public.campaign_operations_production_enablement_canonical_v1(event);
    EXCEPTION WHEN OTHERS THEN
        RAISE EXCEPTION 'production disable replay evidence corrupt'
            USING ERRCODE = '23514';
    END;
    SELECT count(*)::integer INTO audit_count
    FROM public.campaign_operations_production_enablement_audit_reference_event audit
    WHERE audit.production_enablement_event_id =
              event.production_enablement_event_id
      AND audit.operation_key = event.operation_key
      AND audit.event_kind = event.event_kind
      AND audit.actor_identity = event.actor_identity
      AND audit.capability = event.capability
      AND audit.reason = event.reason
      AND audit.outcome = 'recorded'
      AND audit.replay_disposition = 'new_operation'
      AND audit.diagnostic_code = 'immutable_enablement_event_recorded';
    IF event.event_kind <> 'disable' OR
       event.capability <> 'campaign_operations_production_disabler' OR
       event.enablement_contract_version <> 1 OR audit_count <> 1 OR
       event.enablement_identity_canonical <> expected_canonical OR
       event.enablement_identity_hash <>
         public.campaign_operations_tagged_fnv1a64(expected_canonical) THEN
        RAISE EXCEPTION 'production disable replay evidence corrupt'
            USING ERRCODE = '23514';
    END IF;
    IF event.predecessor_event_id IS DISTINCT FROM
           predecessor_event_id_value OR
       event.predecessor_event_canonical IS DISTINCT FROM
           predecessor_event_canonical_value OR
       event.expected_prior_version IS DISTINCT FROM
           expected_prior_version_value OR
       event.actor_identity IS DISTINCT FROM disabling_actor_value OR
       event.reason IS DISTINCT FROM reason_value THEN
        RAISE EXCEPTION 'production disable conflicting replay'
            USING ERRCODE = '23505';
    END IF;
    RETURN NEXT event;
END;
$$;

CREATE OR REPLACE FUNCTION campaign_operations_production_acquire_replay_v2(
    target_request_id bigint, expected_version_value integer,
    lease_digest_value text, lease_expires_at_value timestamptz,
    operation_key_value text, requesting_actor_value text,
    approved_build_contract_canonical_value text)
RETURNS SETOF public.campaign_operations_dispatch_attempt
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE attempt public.campaign_operations_dispatch_attempt%ROWTYPE;
DECLARE admission
    public.campaign_operations_request_production_admission%ROWTYPE;
DECLARE enablement
    public.campaign_operations_production_enablement_event%ROWTYPE;
DECLARE first_attempt public.campaign_operations_dispatch_attempt%ROWTYPE;
DECLARE first_enablement
    public.campaign_operations_production_enablement_event%ROWTYPE;
DECLARE request public.campaign_operations_operational_request%ROWTYPE;
DECLARE expected_attempt_canonical text;
DECLARE expected_first_attempt_canonical text;
DECLARE expected_admission_canonical text;
DECLARE expected_enablement_canonical text;
DECLARE audit_count integer;
DECLARE first_attempt_count integer;
DECLARE first_audit_count integer;
DECLARE first_audit_total_count integer;
BEGIN
    SELECT candidate.* INTO attempt
    FROM public.campaign_operations_dispatch_attempt candidate
    WHERE candidate.operational_request_id = target_request_id
      AND candidate.operation_key = operation_key_value
      AND candidate.attempt_contract_version = 2;
    IF NOT FOUND THEN RETURN; END IF;
    SELECT candidate.* INTO admission
    FROM public.campaign_operations_request_production_admission candidate
    WHERE candidate.request_production_admission_id =
          attempt.request_production_admission_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'production acquisition replay evidence corrupt'
            USING ERRCODE = '23514';
    END IF;
    SELECT candidate.* INTO enablement
    FROM public.campaign_operations_production_enablement_event candidate
    WHERE candidate.production_enablement_event_id =
          attempt.production_enablement_event_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'production acquisition replay evidence corrupt'
            USING ERRCODE = '23514';
    END IF;
    SELECT candidate.* INTO request
    FROM public.campaign_operations_operational_request candidate
    WHERE candidate.operational_request_id = target_request_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'production acquisition replay evidence corrupt'
            USING ERRCODE = '23514';
    END IF;
    SELECT count(*)::integer INTO first_attempt_count
    FROM public.campaign_operations_dispatch_attempt candidate
    WHERE candidate.request_production_admission_id =
              admission.request_production_admission_id
      AND candidate.attempt_ordinal = 1;
    IF first_attempt_count <> 1 THEN
        RAISE EXCEPTION 'production acquisition replay evidence corrupt'
            USING ERRCODE = '23514';
    END IF;
    SELECT candidate.* INTO STRICT first_attempt
    FROM public.campaign_operations_dispatch_attempt candidate
    WHERE candidate.request_production_admission_id =
              admission.request_production_admission_id
      AND candidate.attempt_ordinal = 1;
    SELECT candidate.* INTO first_enablement
    FROM public.campaign_operations_production_enablement_event candidate
    WHERE candidate.production_enablement_event_id =
          first_attempt.production_enablement_event_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'production acquisition replay evidence corrupt'
            USING ERRCODE = '23514';
    END IF;
    IF NOT public.campaign_operations_production_enablement_history_valid_v1(
             first_enablement.production_enablement_event_id) THEN
        RAISE EXCEPTION 'production acquisition replay evidence corrupt'
            USING ERRCODE = '23514';
    END IF;
    BEGIN
        expected_attempt_canonical :=
          public.campaign_operations_dispatch_attempt_v2_canonical(attempt);
        expected_first_attempt_canonical :=
          public.campaign_operations_dispatch_attempt_v2_canonical(
            first_attempt);
        expected_admission_canonical :=
          public.campaign_operations_request_production_admission_canonical_v1(
            admission);
        expected_enablement_canonical :=
          public.campaign_operations_production_enablement_canonical_v1(
            enablement);
    EXCEPTION WHEN OTHERS THEN
        RAISE EXCEPTION 'production acquisition replay evidence corrupt'
            USING ERRCODE = '23514';
    END;
    SELECT count(*)::integer INTO audit_count
    FROM public.campaign_operations_dispatch_audit_reference_event audit
    WHERE audit.operational_campaign_id = request.operational_campaign_id
      AND audit.operational_request_id = target_request_id
      AND audit.dispatch_attempt_id = attempt.dispatch_attempt_id
      AND audit.dispatch_attempt_outcome_id IS NULL
      AND audit.cause_kind = 'dispatch_lease_acquired'
      AND audit.actor_identity = attempt.requesting_actor
      AND audit.capability = 'campaign_operations_production_dispatcher'
      AND audit.prior_version = attempt.expected_request_version
      AND audit.resulting_version = attempt.resulting_request_version
      AND audit.outcome = 'recorded'
      AND audit.replay_disposition = 'new_operation'
      AND audit.diagnostic_code = 'dispatch_lease_acquired'
      AND audit.request_production_admission_id =
          admission.request_production_admission_id
      AND audit.production_enablement_event_id =
          enablement.production_enablement_event_id;
    SELECT count(*)::integer INTO first_audit_total_count
    FROM public.campaign_operations_dispatch_audit_reference_event first_audit
    WHERE first_audit.dispatch_attempt_id = first_attempt.dispatch_attempt_id
      AND first_audit.cause_kind = 'dispatch_lease_acquired';
    SELECT count(*)::integer INTO first_audit_count
    FROM public.campaign_operations_dispatch_audit_reference_event first_audit
    WHERE first_audit.operational_campaign_id =
              request.operational_campaign_id
      AND first_audit.operational_request_id = target_request_id
      AND first_audit.dispatch_attempt_id = first_attempt.dispatch_attempt_id
      AND first_audit.dispatch_attempt_outcome_id IS NULL
      AND first_audit.cause_kind = 'dispatch_lease_acquired'
      AND first_audit.actor_identity = first_attempt.requesting_actor
      AND first_audit.capability =
          'campaign_operations_production_dispatcher'
      AND first_audit.prior_version =
          first_attempt.expected_request_version
      AND first_audit.resulting_version =
          first_attempt.resulting_request_version
      AND first_audit.outcome = 'recorded'
      AND first_audit.replay_disposition = 'new_operation'
      AND first_audit.diagnostic_code = 'dispatch_lease_acquired'
      AND first_audit.request_production_admission_id =
          admission.request_production_admission_id
      AND first_audit.production_enablement_event_id =
          first_enablement.production_enablement_event_id;
    IF audit_count <> 1 OR first_audit_total_count <> 1 OR
       first_audit_count <> 1 OR
       first_attempt.dispatch_attempt_id IS NULL OR
       first_attempt.operational_request_id IS NULL OR
       first_attempt.request_identity_canonical IS NULL OR
       first_attempt.attempt_ordinal IS NULL OR
       first_attempt.expected_request_version IS NULL OR
       first_attempt.resulting_request_version IS NULL OR
       first_attempt.lease_token_digest IS NULL OR
       first_attempt.lease_expires_at IS NULL OR
       first_attempt.dispatcher_identity IS NULL OR
       first_attempt.attempt_contract_version IS NULL OR
       first_attempt.attempt_identity_canonical IS NULL OR
       first_attempt.attempt_identity_hash IS NULL OR
       first_attempt.request_production_admission_id IS NULL OR
       first_attempt.request_production_admission_canonical IS NULL OR
       first_attempt.request_production_admission_hash IS NULL OR
       first_attempt.production_enablement_event_id IS NULL OR
       first_attempt.production_enablement_event_canonical IS NULL OR
       first_attempt.production_enablement_event_hash IS NULL OR
       first_attempt.operation_key IS NULL OR
       first_attempt.requesting_actor IS NULL OR
       first_attempt.original_executing_service_principal IS NULL OR
       first_attempt.approved_build_contract_canonical IS NULL OR
       first_attempt.approved_build_contract_hash IS NULL OR
       first_attempt.production_capability IS NULL OR
       expected_first_attempt_canonical IS NULL OR
       attempt.attempt_contract_version <> 2 OR
       attempt.production_capability <>
         'campaign_operations_production_dispatcher' OR
       attempt.resulting_request_version <>
         attempt.expected_request_version + 1 OR
       attempt.dispatcher_identity <> attempt.requesting_actor OR
       attempt.request_identity_canonical <>
         request.request_identity_canonical OR
       attempt.attempt_identity_canonical <> expected_attempt_canonical OR
       attempt.attempt_identity_hash <>
         public.campaign_operations_tagged_fnv1a64(
           expected_attempt_canonical) OR
       admission.operational_request_id <> target_request_id OR
       admission.request_identity_canonical <>
         attempt.request_identity_canonical OR
       admission.capability <> 'campaign_operations_production_dispatcher' OR
       admission.admission_contract_version <> 1 OR
       admission.admission_identity_canonical <> expected_admission_canonical OR
       admission.admission_identity_hash <>
         public.campaign_operations_tagged_fnv1a64(
           expected_admission_canonical) OR
       attempt.request_production_admission_canonical <>
         admission.admission_identity_canonical OR
       attempt.request_production_admission_hash <>
         admission.admission_identity_hash OR
       attempt.production_enablement_event_id <>
         admission.production_enablement_event_id OR
       enablement.event_kind <> 'enable' OR
       enablement.enablement_contract_version <> 1 OR
       enablement.enablement_identity_canonical <>
         expected_enablement_canonical OR
       enablement.enablement_identity_hash <>
         public.campaign_operations_tagged_fnv1a64(
           expected_enablement_canonical) OR
       attempt.production_enablement_event_canonical <>
         enablement.enablement_identity_canonical OR
       attempt.production_enablement_event_hash <>
         enablement.enablement_identity_hash OR
       attempt.approved_build_contract_canonical <>
         enablement.approved_build_contract_canonical OR
       attempt.approved_build_contract_hash <>
         enablement.approved_build_contract_hash OR
       first_attempt.operational_request_id <> target_request_id OR
       first_attempt.request_identity_canonical <>
         request.request_identity_canonical OR
       first_attempt.attempt_contract_version <> 2 OR
       first_attempt.production_capability <>
         'campaign_operations_production_dispatcher' OR
       first_attempt.dispatcher_identity <> first_attempt.requesting_actor OR
       first_attempt.attempt_ordinal <> 1 OR
       first_attempt.expected_request_version <>
         admission.expected_request_version OR
       first_attempt.resulting_request_version <>
         first_attempt.expected_request_version + 1 OR
       first_attempt.operation_key <> admission.dispatch_operation_key OR
       first_attempt.requesting_actor <> admission.requesting_actor OR
       first_attempt.original_executing_service_principal <>
         admission.original_executing_service_principal OR
       first_attempt.request_production_admission_id <>
         admission.request_production_admission_id OR
       first_attempt.request_production_admission_canonical <>
         admission.admission_identity_canonical OR
       first_attempt.request_production_admission_hash <>
         admission.admission_identity_hash OR
       first_attempt.production_enablement_event_id <>
         admission.production_enablement_event_id OR
       first_attempt.production_enablement_event_canonical <>
         first_enablement.enablement_identity_canonical OR
       first_attempt.production_enablement_event_hash <>
         first_enablement.enablement_identity_hash OR
       first_attempt.approved_build_contract_canonical <>
         admission.approved_build_contract_canonical OR
       first_attempt.approved_build_contract_hash <>
         admission.approved_build_contract_hash OR
       first_attempt.request_production_admission_hash <>
         public.campaign_operations_tagged_fnv1a64(
           first_attempt.request_production_admission_canonical) OR
       first_attempt.production_enablement_event_hash <>
         public.campaign_operations_tagged_fnv1a64(
           first_attempt.production_enablement_event_canonical) OR
       first_attempt.approved_build_contract_hash <>
         public.campaign_operations_tagged_fnv1a64(
           first_attempt.approved_build_contract_canonical) OR
       first_attempt.attempt_identity_canonical <>
         expected_first_attempt_canonical OR
       first_attempt.attempt_identity_hash <>
         public.campaign_operations_tagged_fnv1a64(
           expected_first_attempt_canonical) OR
       admission.production_enablement_event_id <>
         first_enablement.production_enablement_event_id OR
       admission.enable_event_canonical <>
         first_enablement.enablement_identity_canonical OR
       admission.approved_build_contract_canonical <>
         first_enablement.approved_build_contract_canonical OR
       admission.approved_build_contract_hash <>
         first_enablement.approved_build_contract_hash OR
       admission.approved_build_contract_hash <>
         public.campaign_operations_tagged_fnv1a64(
           admission.approved_build_contract_canonical) THEN
        RAISE EXCEPTION 'production acquisition replay evidence corrupt'
            USING ERRCODE = '23514';
    END IF;
    IF attempt.expected_request_version IS DISTINCT FROM
           expected_version_value OR
       attempt.lease_token_digest IS DISTINCT FROM lease_digest_value OR
       attempt.lease_expires_at IS DISTINCT FROM lease_expires_at_value OR
       attempt.requesting_actor IS DISTINCT FROM requesting_actor_value THEN
        RAISE EXCEPTION 'production acquisition conflicting replay'
            USING ERRCODE = '23505';
    END IF;
    RETURN NEXT attempt;
END;
$$;

CREATE OR REPLACE FUNCTION record_campaign_operations_production_enable_v1(
    operation_key_value text, expected_prior_version_value integer,
    scheduler_evidence_canonical_value text,
    independent_verification_reference_value text,
    authorizing_actor_value text, manager_service_contract_value text,
    approved_build_contract_canonical_value text,
    approved_build_source_commit_value text,
    approved_build_compiler_contract_value text,
    approved_build_executable_sha256_value text, reason_value text)
RETURNS campaign_operations_production_enablement_event
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE scheduler_record record;
DECLARE predecessor
    public.campaign_operations_production_enablement_event%ROWTYPE;
DECLARE event public.campaign_operations_production_enablement_event%ROWTYPE;
DECLARE build_canonical_value text;
BEGIN
    IF operation_key_value IS NULL OR octet_length(operation_key_value) > 128 OR
       operation_key_value !~ '^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,127}$' THEN
        RAISE EXCEPTION 'invalid production operation key'
            USING ERRCODE = '22023';
    END IF;
    build_canonical_value :=
        public.campaign_operations_manager_build_canonical_v1(
            manager_service_contract_value,
            approved_build_source_commit_value,
            approved_build_compiler_contract_value,
            approved_build_executable_sha256_value);
    IF build_canonical_value <> approved_build_contract_canonical_value THEN
        RAISE EXCEPTION 'production enable approved build mismatch'
            USING ERRCODE = '23514';
    END IF;
    SELECT candidate.* INTO event
    FROM public.campaign_operations_production_enable_replay_v1(
        operation_key_value, expected_prior_version_value,
        scheduler_evidence_canonical_value,
        independent_verification_reference_value,
        authorizing_actor_value, manager_service_contract_value,
        approved_build_contract_canonical_value,
        approved_build_source_commit_value,
        approved_build_compiler_contract_value,
        approved_build_executable_sha256_value, reason_value) candidate;
    IF FOUND THEN RETURN event; END IF;
    SELECT * INTO STRICT scheduler_record
    FROM public.campaign_operations_scheduler_protocol_evidence_lock_v1();
    IF NOT scheduler_record.evidence_complete OR
       scheduler_record.evidence_canonical <>
           scheduler_evidence_canonical_value THEN
        RAISE EXCEPTION 'production enable scheduler evidence mismatch'
            USING ERRCODE = '23514';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(19055, 1);
    SELECT candidate.* INTO event
    FROM public.campaign_operations_production_enable_replay_v1(
        operation_key_value, expected_prior_version_value,
        scheduler_evidence_canonical_value,
        independent_verification_reference_value,
        authorizing_actor_value, manager_service_contract_value,
        approved_build_contract_canonical_value,
        approved_build_source_commit_value,
        approved_build_compiler_contract_value,
        approved_build_executable_sha256_value, reason_value) candidate;
    IF FOUND THEN RETURN event; END IF;
    SELECT * INTO predecessor
    FROM public.campaign_operations_production_enablement_event candidate
    ORDER BY candidate.resulting_version DESC LIMIT 1;
    IF expected_prior_version_value = 0 THEN
        IF predecessor.production_enablement_event_id IS NOT NULL THEN
            RAISE EXCEPTION 'production enable predecessor mismatch'
                USING ERRCODE = '40001';
        END IF;
    ELSIF predecessor.production_enablement_event_id IS NULL OR
          predecessor.resulting_version <> expected_prior_version_value OR
          predecessor.event_kind <> 'disable' THEN
        RAISE EXCEPTION 'production enable predecessor mismatch'
            USING ERRCODE = '40001';
    END IF;
    INSERT INTO public.campaign_operations_production_transition_context(
        backend_pid, transaction_id, transition_kind,
        operational_request_id, operation_key)
    VALUES(pg_catalog.pg_backend_pid(), pg_catalog.txid_current(), 'enable',
           NULL, operation_key_value);

    event.production_enablement_event_id := pg_catalog.nextval(
        pg_catalog.pg_get_serial_sequence(
          'public.campaign_operations_production_enablement_event',
          'production_enablement_event_id'));
    event.event_kind := 'enable';
    event.operation_key := operation_key_value;
    event.predecessor_event_id := predecessor.production_enablement_event_id;
    event.predecessor_event_canonical := coalesce(
        predecessor.enablement_identity_canonical, '');
    event.expected_prior_version := expected_prior_version_value;
    event.resulting_version := expected_prior_version_value + 1;
    event.scheduler_protocol_evidence_canonical :=
        scheduler_record.evidence_canonical;
    event.scheduler_protocol_evidence_hash := scheduler_record.evidence_hash;
    event.scheduler_required_generation := scheduler_record.required_generation;
    event.scheduler_cutover_state := scheduler_record.cutover_state;
    event.scheduler_cutover_completed_at :=
        scheduler_record.cutover_completed_at::timestamptz;
    event.scheduler_cutover_completed_by :=
        scheduler_record.cutover_completed_by;
    event.scheduler_cutover_executable_path :=
        scheduler_record.cutover_executable_path;
    event.scheduler_cutover_process_evidence :=
        scheduler_record.cutover_process_evidence;
    event.independent_verification_reference :=
        independent_verification_reference_value;
    event.actor_identity := authorizing_actor_value;
    event.capability := 'campaign_operations_production_enabler';
    event.manager_service_contract := manager_service_contract_value;
    event.approved_build_contract_canonical := build_canonical_value;
    event.approved_build_contract_hash :=
        public.campaign_operations_tagged_fnv1a64(build_canonical_value);
    event.approved_build_source_commit := approved_build_source_commit_value;
    event.approved_build_compiler_contract :=
        approved_build_compiler_contract_value;
    event.approved_build_executable_sha256 :=
        approved_build_executable_sha256_value;
    event.reason := reason_value;
    event.enablement_contract_version := 1;
    event.recorded_at := pg_catalog.transaction_timestamp();
    event.enablement_identity_canonical :=
        public.campaign_operations_production_enablement_canonical_v1(event);
    event.enablement_identity_hash :=
        public.campaign_operations_tagged_fnv1a64(
            event.enablement_identity_canonical);
    INSERT INTO public.campaign_operations_production_enablement_event
        SELECT event.*;
    INSERT INTO
      public.campaign_operations_production_enablement_audit_reference_event(
        production_enablement_event_id, operation_key, event_kind,
        actor_identity, capability, reason, outcome, replay_disposition,
        diagnostic_code)
    VALUES(event.production_enablement_event_id, event.operation_key,
        event.event_kind, event.actor_identity, event.capability, event.reason,
        'recorded', 'new_operation', 'immutable_enablement_event_recorded');
    DELETE FROM public.campaign_operations_production_transition_context context
    WHERE context.backend_pid = pg_catalog.pg_backend_pid()
      AND context.transaction_id = pg_catalog.txid_current();
    RETURN event;
END;
$$;

CREATE OR REPLACE FUNCTION record_campaign_operations_production_disable_v1(
    operation_key_value text, predecessor_event_id_value bigint,
    predecessor_event_canonical_value text,
    expected_prior_version_value integer, disabling_actor_value text,
    reason_value text)
RETURNS campaign_operations_production_enablement_event
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE predecessor
    public.campaign_operations_production_enablement_event%ROWTYPE;
DECLARE event public.campaign_operations_production_enablement_event%ROWTYPE;
BEGIN
    IF operation_key_value IS NULL OR octet_length(operation_key_value) > 128 OR
       operation_key_value !~ '^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,127}$' THEN
        RAISE EXCEPTION 'invalid production operation key'
            USING ERRCODE = '22023';
    END IF;
    SELECT candidate.* INTO event
    FROM public.campaign_operations_production_disable_replay_v1(
        operation_key_value, predecessor_event_id_value,
        predecessor_event_canonical_value, expected_prior_version_value,
        disabling_actor_value, reason_value) candidate;
    IF FOUND THEN RETURN event; END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(19055, 1);
    SELECT candidate.* INTO event
    FROM public.campaign_operations_production_disable_replay_v1(
        operation_key_value, predecessor_event_id_value,
        predecessor_event_canonical_value, expected_prior_version_value,
        disabling_actor_value, reason_value) candidate;
    IF FOUND THEN RETURN event; END IF;
    SELECT * INTO STRICT predecessor
    FROM public.campaign_operations_production_enablement_event candidate
    ORDER BY candidate.resulting_version DESC LIMIT 1;
    IF predecessor.production_enablement_event_id <>
           predecessor_event_id_value OR
       predecessor.enablement_identity_canonical <>
           predecessor_event_canonical_value OR
       predecessor.resulting_version <> expected_prior_version_value OR
       predecessor.event_kind <> 'enable' THEN
        RAISE EXCEPTION 'production disable predecessor mismatch'
            USING ERRCODE = '40001';
    END IF;
    INSERT INTO public.campaign_operations_production_transition_context(
        backend_pid, transaction_id, transition_kind,
        operational_request_id, operation_key)
    VALUES(pg_catalog.pg_backend_pid(), pg_catalog.txid_current(), 'disable',
           NULL, operation_key_value);
    event.production_enablement_event_id := pg_catalog.nextval(
        pg_catalog.pg_get_serial_sequence(
          'public.campaign_operations_production_enablement_event',
          'production_enablement_event_id'));
    event.event_kind := 'disable';
    event.operation_key := operation_key_value;
    event.predecessor_event_id := predecessor_event_id_value;
    event.predecessor_event_canonical := predecessor_event_canonical_value;
    event.expected_prior_version := expected_prior_version_value;
    event.resulting_version := expected_prior_version_value + 1;
    event.actor_identity := disabling_actor_value;
    event.capability := 'campaign_operations_production_disabler';
    event.reason := reason_value;
    event.enablement_contract_version := 1;
    event.recorded_at := pg_catalog.transaction_timestamp();
    event.enablement_identity_canonical :=
        public.campaign_operations_production_enablement_canonical_v1(event);
    event.enablement_identity_hash :=
        public.campaign_operations_tagged_fnv1a64(
            event.enablement_identity_canonical);
    INSERT INTO public.campaign_operations_production_enablement_event
        SELECT event.*;
    INSERT INTO
      public.campaign_operations_production_enablement_audit_reference_event(
        production_enablement_event_id, operation_key, event_kind,
        actor_identity, capability, reason, outcome, replay_disposition,
        diagnostic_code)
    VALUES(event.production_enablement_event_id, event.operation_key,
        event.event_kind, event.actor_identity, event.capability, event.reason,
        'recorded', 'new_operation', 'immutable_enablement_event_recorded');
    DELETE FROM public.campaign_operations_production_transition_context context
    WHERE context.backend_pid = pg_catalog.pg_backend_pid()
      AND context.transaction_id = pg_catalog.txid_current();
    RETURN event;
END;
$$;

CREATE OR REPLACE FUNCTION
transition_campaign_operations_request_dispatch_production_v2(
    target_request_id bigint, expected_version_value integer,
    lease_digest_value text, lease_expires_at_value timestamptz,
    operation_key_value text, requesting_actor_value text,
    approved_build_contract_canonical_value text)
RETURNS campaign_operations_dispatch_attempt
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE scheduler_record record;
DECLARE head public.campaign_operations_production_enablement_event%ROWTYPE;
DECLARE request_candidate
    public.campaign_operations_operational_request%ROWTYPE;
DECLARE request public.campaign_operations_operational_request%ROWTYPE;
DECLARE admission
    public.campaign_operations_request_production_admission%ROWTYPE;
DECLARE attempt public.campaign_operations_dispatch_attempt%ROWTYPE;
DECLARE next_ordinal integer;
BEGIN
    IF operation_key_value IS NULL OR octet_length(operation_key_value) > 128 OR
       operation_key_value !~ '^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,127}$' THEN
        RAISE EXCEPTION 'invalid production operation key'
            USING ERRCODE = '22023';
    END IF;
    SELECT candidate.* INTO attempt
    FROM public.campaign_operations_production_acquire_replay_v2(
        target_request_id, expected_version_value, lease_digest_value,
        lease_expires_at_value, operation_key_value, requesting_actor_value,
        approved_build_contract_canonical_value) candidate;
    IF FOUND THEN RETURN attempt; END IF;
    SELECT * INTO STRICT scheduler_record
    FROM public.campaign_operations_scheduler_protocol_evidence_lock_v1();
    IF NOT scheduler_record.evidence_complete THEN
        RAISE EXCEPTION 'production dispatch scheduler evidence incomplete'
            USING ERRCODE = '23514';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock_shared(19055, 1);
    SELECT * INTO STRICT head
    FROM public.campaign_operations_production_enablement_event candidate
    ORDER BY candidate.resulting_version DESC LIMIT 1;
    IF head.event_kind <> 'enable' OR
       head.scheduler_protocol_evidence_canonical <>
           scheduler_record.evidence_canonical OR
       head.approved_build_contract_canonical <>
           approved_build_contract_canonical_value THEN
        RAISE EXCEPTION 'production dispatch enablement is ineffective'
            USING ERRCODE = '23514';
    END IF;
    -- Optimistic identity read only; no tuple lock is taken before the
    -- authorization, budget, and campaign/completion boundaries.
    SELECT * INTO STRICT request_candidate
    FROM public.campaign_operations_operational_request candidate
    WHERE candidate.operational_request_id = target_request_id;

    PERFORM public.lock_campaign_operations_authorization_head(
        request_candidate.operational_campaign_id,
        request_candidate.action_kind);
    PERFORM public.lock_campaign_operations_budget_head(
        request_candidate.operational_campaign_id);
    PERFORM public.lock_campaign_operations_campaign(
        request_candidate.operational_campaign_id);
    PERFORM public.lock_campaign_operations_reservation(
        request_candidate.reservation_id);
    SELECT locked.* INTO STRICT request
    FROM public.lock_campaign_operations_request(target_request_id) locked;
    IF request.operational_campaign_id <>
           request_candidate.operational_campaign_id OR
       request.reservation_id <> request_candidate.reservation_id OR
       request.action_kind <> request_candidate.action_kind THEN
        RAISE EXCEPTION 'production dispatch request identity changed'
            USING ERRCODE = '40001';
    END IF;

    SELECT candidate.* INTO attempt
    FROM public.campaign_operations_production_acquire_replay_v2(
        target_request_id, expected_version_value, lease_digest_value,
        lease_expires_at_value, operation_key_value, requesting_actor_value,
        approved_build_contract_canonical_value) candidate;
    IF FOUND THEN RETURN attempt; END IF;
    IF request.request_state <> 'ready' OR
       request.state_version <> expected_version_value OR
       request.lease_token_hash IS NOT NULL OR
       request.lease_expires_at IS NOT NULL OR
       request.dispatcher_identity IS NOT NULL OR
       lease_expires_at_value <= pg_catalog.transaction_timestamp() OR
       request.production_dispatch_enabled IS DISTINCT FROM EXISTS (
         SELECT 1
         FROM public.campaign_operations_request_production_admission existing
         WHERE existing.operational_request_id = target_request_id) THEN
        RAISE EXCEPTION 'production dispatch request compare-and-set lost'
            USING ERRCODE = '40001';
    END IF;
    SELECT coalesce(max(existing.attempt_ordinal), 0) + 1
      INTO next_ordinal
    FROM public.campaign_operations_dispatch_attempt existing
    WHERE existing.operational_request_id = target_request_id;
    INSERT INTO public.campaign_operations_production_transition_context(
        backend_pid, transaction_id, transition_kind,
        operational_request_id, operation_key)
    VALUES(pg_catalog.pg_backend_pid(), pg_catalog.txid_current(),
           'dispatch_v2', target_request_id, operation_key_value);

    SELECT existing.* INTO admission
    FROM public.campaign_operations_request_production_admission existing
    WHERE existing.operational_request_id = target_request_id;
    IF NOT FOUND THEN
        admission.request_production_admission_id := pg_catalog.nextval(
            pg_catalog.pg_get_serial_sequence(
              'public.campaign_operations_request_production_admission',
              'request_production_admission_id'));
        admission.operational_request_id := target_request_id;
        admission.request_identity_canonical :=
            request.request_identity_canonical;
        admission.expected_request_version := expected_version_value;
        admission.dispatch_operation_key := operation_key_value;
        admission.production_enablement_event_id :=
            head.production_enablement_event_id;
        admission.enable_event_canonical := head.enablement_identity_canonical;
        admission.requesting_actor := requesting_actor_value;
        admission.original_executing_service_principal := session_user::text;
        admission.approved_build_contract_canonical :=
            approved_build_contract_canonical_value;
        admission.approved_build_contract_hash :=
            head.approved_build_contract_hash;
        admission.capability := 'campaign_operations_production_dispatcher';
        admission.admission_contract_version := 1;
        admission.admitted_at := pg_catalog.transaction_timestamp();
        admission.admission_identity_canonical :=
            public.campaign_operations_request_production_admission_canonical_v1(
                admission);
        admission.admission_identity_hash :=
            public.campaign_operations_tagged_fnv1a64(
                admission.admission_identity_canonical);
        INSERT INTO public.campaign_operations_request_production_admission
            SELECT admission.*;
    ELSIF admission.request_identity_canonical <>
              request.request_identity_canonical OR
          admission.admission_identity_canonical <>
              public.campaign_operations_request_production_admission_canonical_v1(
                  admission) OR
          admission.admission_identity_hash <>
              public.campaign_operations_tagged_fnv1a64(
                  admission.admission_identity_canonical) THEN
        RAISE EXCEPTION 'request production admission evidence corrupt'
            USING ERRCODE = '23514';
    END IF;

    UPDATE public.campaign_operations_operational_request candidate
       SET request_state = 'dispatching',
           state_version = candidate.state_version + 1,
           lease_token_hash = lease_digest_value,
           lease_expires_at = lease_expires_at_value,
           dispatcher_identity = requesting_actor_value,
           production_dispatch_enabled = true,
           updated_at = pg_catalog.transaction_timestamp()
     WHERE candidate.operational_request_id = target_request_id
       AND candidate.request_state = 'ready'
       AND candidate.state_version = expected_version_value;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'production dispatch request compare-and-set lost'
            USING ERRCODE = '40001';
    END IF;

    attempt.dispatch_attempt_id := pg_catalog.nextval(
        pg_catalog.pg_get_serial_sequence(
          'public.campaign_operations_dispatch_attempt',
          'dispatch_attempt_id'));
    attempt.operational_request_id := target_request_id;
    attempt.request_identity_canonical := request.request_identity_canonical;
    attempt.attempt_ordinal := next_ordinal;
    attempt.expected_request_version := expected_version_value;
    attempt.resulting_request_version := expected_version_value + 1;
    attempt.lease_token_digest := lease_digest_value;
    attempt.lease_expires_at := lease_expires_at_value;
    attempt.dispatcher_identity := requesting_actor_value;
    attempt.attempt_contract_version := 2;
    attempt.acquired_at := pg_catalog.transaction_timestamp();
    attempt.request_production_admission_id :=
        admission.request_production_admission_id;
    attempt.request_production_admission_canonical :=
        admission.admission_identity_canonical;
    attempt.request_production_admission_hash :=
        admission.admission_identity_hash;
    attempt.production_enablement_event_id :=
        head.production_enablement_event_id;
    attempt.production_enablement_event_canonical :=
        head.enablement_identity_canonical;
    attempt.production_enablement_event_hash := head.enablement_identity_hash;
    attempt.operation_key := operation_key_value;
    attempt.requesting_actor := requesting_actor_value;
    attempt.original_executing_service_principal := session_user::text;
    attempt.approved_build_contract_canonical :=
        approved_build_contract_canonical_value;
    attempt.approved_build_contract_hash := head.approved_build_contract_hash;
    attempt.production_capability :=
        'campaign_operations_production_dispatcher';
    attempt.attempt_identity_canonical :=
        public.campaign_operations_dispatch_attempt_v2_canonical(attempt);
    attempt.attempt_identity_hash :=
        public.campaign_operations_tagged_fnv1a64(
            attempt.attempt_identity_canonical);
    INSERT INTO public.campaign_operations_dispatch_attempt SELECT attempt.*;
    INSERT INTO public.campaign_operations_dispatch_audit_reference_event(
        operational_campaign_id, operational_request_id, dispatch_attempt_id,
        dispatch_attempt_outcome_id, cause_kind, actor_identity, capability,
        prior_version, resulting_version, outcome, replay_disposition,
        diagnostic_code, request_production_admission_id,
        production_enablement_event_id)
    VALUES(request.operational_campaign_id, target_request_id,
        attempt.dispatch_attempt_id, NULL, 'dispatch_lease_acquired',
        requesting_actor_value, 'campaign_operations_production_dispatcher',
        expected_version_value, expected_version_value + 1, 'recorded',
        'new_operation', 'dispatch_lease_acquired',
        admission.request_production_admission_id,
        head.production_enablement_event_id);
    DELETE FROM public.campaign_operations_production_transition_context context
    WHERE context.backend_pid = pg_catalog.pg_backend_pid()
      AND context.transaction_id = pg_catalog.txid_current();
    RETURN attempt;
END;
$$;

-- This is deployment evidence, not a renderer default.  It resides with the
-- authoritative scheduler protocol row and lets a readable wrong deployment
-- marker remain observable to the read-only readiness query.
ALTER TABLE public.experiment_scheduler_protocol
    ADD COLUMN IF NOT EXISTS scheduler_evidence_contract_version integer
        NOT NULL DEFAULT 1 CHECK (scheduler_evidence_contract_version = 1);

CREATE OR REPLACE VIEW campaign_operations_production_readiness_v1 AS
WITH head AS (
    SELECT * FROM campaign_operations_production_enablement_event
    ORDER BY resulting_version DESC LIMIT 1
), scheduler AS (
    SELECT * FROM campaign_operations_scheduler_protocol_evidence_snapshot_v1()
), scheduler_contract AS (
    -- The protocol row is the persisted scheduler authority.  Its typed
    -- contract marker is deliberately separate from the generation/cutover
    -- evidence so readiness can display a readable contradiction.
    SELECT protocol.scheduler_evidence_contract_version
    FROM public.experiment_scheduler_protocol protocol
    WHERE protocol.singleton
), completion_rows AS (
    SELECT string_agg(DISTINCT completion.completion_contract_version::text,
                      '|' ORDER BY completion.completion_contract_version::text)
        AS observed_version
    FROM campaign_operations_completion_event completion
), completion_contract AS (
    -- A deployment can correctly have no completed campaign.  In that case
    -- the deployed Completion V1 column contract is the actual proof-version
    -- evidence; it is not replaced with a readiness literal.
    SELECT substring(pg_get_constraintdef(constraint_row.oid) FROM
               'completion_contract_version = ([0-9]+)') AS observed_version
    FROM pg_catalog.pg_constraint constraint_row
    WHERE constraint_row.conrelid =
              'public.campaign_operations_completion_event'::regclass
      AND constraint_row.contype = 'c'
      AND pg_get_constraintdef(constraint_row.oid) LIKE
              '%completion_contract_version = %'
), role_state AS (
    SELECT
      campaign_operations_has_explicit_role_v1(session_user::name,
        'campaign_operations_production_reader'::name) AS reader_member,
      campaign_operations_has_explicit_role_v1(session_user::name,
        'campaign_operations_production_dispatcher'::name)
        AS dispatcher_member,
      campaign_operations_has_explicit_role_v1(session_user::name,
        'campaign_operations_production_phase5_transactional'::name)
        AS phase5_transactional_member,
      campaign_operations_has_explicit_role_v1(session_user::name,
        'campaign_operations_scheduler_protocol_evidence_reader'::name)
        AS scheduler_evidence_reader_member,
      campaign_operations_has_explicit_role_v1(session_user::name,
        'campaign_operations_production_enabler'::name) AS enabler_member,
      campaign_operations_has_explicit_role_v1(session_user::name,
        'campaign_operations_production_disabler'::name) AS disabler_member,
      campaign_operations_has_explicit_role_v1(session_user::name,
        'campaign_operations_dispatcher'::name)
        AS prohibited_test_dispatcher_member,
      campaign_operations_has_explicit_role_v1(session_user::name,
        'campaign_operations_phase5_transactional'::name)
        AS prohibited_test_phase5_member
)
SELECT
    '055'::text AS migration_version,
    (SELECT filename FROM schema_migrations WHERE version = '055')
        AS migration_filename,
    (SELECT checksum FROM schema_migrations WHERE version = '055')
        AS migration_checksum,
    scheduler_contract.scheduler_evidence_contract_version::text
        AS scheduler_evidence_contract_version,
    (SELECT string_agg(DISTINCT CASE
          WHEN event.approved_build_contract_canonical ~
               ';build_contract_version=[0-9]+$'
            THEN substring(event.approved_build_contract_canonical FROM
                 ';build_contract_version=([0-9]+)$')
          ELSE 'invalid'
        END, '|' ORDER BY CASE
          WHEN event.approved_build_contract_canonical ~
               ';build_contract_version=[0-9]+$'
            THEN substring(event.approved_build_contract_canonical FROM
                 ';build_contract_version=([0-9]+)$')
          ELSE 'invalid'
        END)
       FROM campaign_operations_production_enablement_event event
       WHERE event.event_kind = 'enable') AS manager_build_contract_version,
    (SELECT string_agg(DISTINCT event.enablement_contract_version::text, '|'
                       ORDER BY event.enablement_contract_version::text)
       FROM campaign_operations_production_enablement_event event)
        AS enablement_contract_version,
    (SELECT string_agg(DISTINCT admission.admission_contract_version::text,
                       '|' ORDER BY admission.admission_contract_version::text)
       FROM campaign_operations_request_production_admission admission)
        AS admission_contract_version,
    (SELECT string_agg(DISTINCT attempt.attempt_contract_version::text,
                       '|' ORDER BY attempt.attempt_contract_version::text)
       FROM campaign_operations_dispatch_attempt attempt
       WHERE attempt.request_production_admission_id IS NOT NULL OR
             attempt.production_enablement_event_id IS NOT NULL)
        AS production_attempt_contract_version,
    scheduler.required_generation AS scheduler_required_generation,
    scheduler.cutover_state AS scheduler_cutover_state,
    scheduler.evidence_complete AS scheduler_evidence_complete,
    scheduler.evidence_canonical AS scheduler_evidence_canonical,
    scheduler.evidence_hash AS scheduler_evidence_hash,
    head.production_enablement_event_id,
    head.event_kind AS enablement_kind,
    head.resulting_version AS enablement_version,
    head.enablement_identity_canonical AS enablement_canonical,
    head.enablement_identity_hash AS enablement_hash,
    head.independent_verification_reference,
    head.manager_service_contract,
    head.approved_build_contract_canonical,
    session_user::text AS session_principal,
    current_user::text AS current_principal,
    role_state.reader_member,
    role_state.dispatcher_member,
    role_state.phase5_transactional_member,
    role_state.scheduler_evidence_reader_member,
    role_state.enabler_member,
    role_state.disabler_member,
    role_state.prohibited_test_dispatcher_member,
    role_state.prohibited_test_phase5_member,
    (head.event_kind = 'enable' AND scheduler.evidence_complete AND
     head.scheduler_protocol_evidence_canonical =
         scheduler.evidence_canonical) AS enablement_effective,
    ((SELECT count(*) FROM campaign_operations_operational_request request
       WHERE request.request_state = 'ready' AND
             request.production_dispatch_enabled))::bigint
        AS ready_admitted_request_count,
    ((SELECT count(*) FROM campaign_operations_operational_request request
       WHERE request.request_state = 'ready' AND
             NOT request.production_dispatch_enabled))::bigint
        AS ready_unadmitted_request_count,
    ((SELECT count(*) FROM campaign_operations_dispatch_attempt attempt
       JOIN campaign_operations_operational_request request
         ON request.operational_request_id =
              attempt.operational_request_id
        AND request.request_state = 'dispatching'
        AND request.state_version = attempt.resulting_request_version
        AND request.lease_token_hash = attempt.lease_token_digest
        AND request.lease_expires_at = attempt.lease_expires_at
       WHERE attempt.attempt_contract_version = 2 AND
             attempt.lease_expires_at > transaction_timestamp() AND
             attempt.production_enablement_event_id IS NOT DISTINCT FROM
                 head.production_enablement_event_id))::bigint
        AS active_current_event_lease_count,
    ((SELECT count(*) FROM campaign_operations_dispatch_attempt attempt
       JOIN campaign_operations_operational_request request
         ON request.operational_request_id =
              attempt.operational_request_id
        AND request.request_state = 'dispatching'
        AND request.state_version = attempt.resulting_request_version
        AND request.lease_token_hash = attempt.lease_token_digest
        AND request.lease_expires_at = attempt.lease_expires_at
       WHERE attempt.attempt_contract_version = 2 AND
             attempt.production_enablement_event_id IS DISTINCT FROM
                 head.production_enablement_event_id))::bigint
        AS old_event_blocked_lease_count,
    ((SELECT count(*) FROM campaign_operations_reconciliation_observation o
       WHERE NOT EXISTS (
         SELECT 1 FROM campaign_operations_reconciliation_resolution r
         WHERE r.reconciliation_observation_id =
               o.reconciliation_observation_id)))::bigint
        AS reconciliation_required_count,
    coalesce(completion_rows.observed_version,
             completion_contract.observed_version)
        AS completion_nested_v2_proof_version,
    NOT EXISTS (
        SELECT 1 FROM campaign_operations_dispatch_attempt attempt
        LEFT JOIN campaign_operations_request_production_admission admission
          ON admission.request_production_admission_id =
             attempt.request_production_admission_id
        WHERE attempt.attempt_contract_version = 2 AND
              admission.request_production_admission_id IS NULL)
        AS completion_nested_v2_proof_valid
FROM scheduler CROSS JOIN role_state LEFT JOIN head ON true
LEFT JOIN scheduler_contract ON true
LEFT JOIN completion_rows ON true
LEFT JOIN completion_contract ON true;

CREATE OR REPLACE VIEW campaign_operations_production_status_v1 AS
SELECT request.operational_request_id,
       request.operational_campaign_id,
       request.request_state,
       request.state_version,
       request.production_dispatch_enabled,
       admission.request_production_admission_id,
       admission.admission_identity_canonical,
       admission.admission_identity_hash,
       attempt.dispatch_attempt_id,
       attempt.operation_key,
       attempt.attempt_ordinal,
       attempt.production_enablement_event_id,
       attempt.lease_expires_at,
       attempt.lease_expires_at <= transaction_timestamp()
          AS lease_expired,
       request.request_state = 'dispatching' AND
          request.lease_expires_at <= transaction_timestamp() AND
          NOT EXISTS (
            SELECT 1
            FROM campaign_operations_request_binding binding
            WHERE binding.operational_request_id =
                  request.operational_request_id) AND
          NOT EXISTS (
            SELECT 1
            FROM experiment_recommendation_campaign_materialization_member member
            JOIN experiment_recommendation_conversion_execution execution
              ON execution.recommendation_conversion_proposal_id =
                 member.recommendation_conversion_proposal_id
            WHERE member.recommendation_campaign_materialization_id =
                  request.recommendation_campaign_materialization_id) AND
          NOT EXISTS (
            SELECT 1
            FROM campaign_operations_dispatch_attempt recovery_attempt
            JOIN campaign_operations_dispatch_attempt_outcome outcome
              ON outcome.dispatch_attempt_id =
                 recovery_attempt.dispatch_attempt_id
            WHERE recovery_attempt.operational_request_id =
                  request.operational_request_id
              AND recovery_attempt.attempt_ordinal = (
                SELECT max(latest.attempt_ordinal)
                FROM campaign_operations_dispatch_attempt latest
                WHERE latest.operational_request_id =
                      request.operational_request_id))
          AS phase_f_recovery_eligible,
       observation.reconciliation_observation_id IS NOT NULL
          AS reconciliation_required
FROM campaign_operations_operational_request request
LEFT JOIN campaign_operations_request_production_admission admission
  ON admission.operational_request_id = request.operational_request_id
LEFT JOIN LATERAL (
    SELECT candidate.*
    FROM campaign_operations_dispatch_attempt candidate
    WHERE candidate.operational_request_id = request.operational_request_id
      AND candidate.attempt_contract_version = 2
    ORDER BY candidate.attempt_ordinal DESC,
             candidate.dispatch_attempt_id DESC LIMIT 1) attempt ON true
LEFT JOIN LATERAL (
    SELECT candidate.reconciliation_observation_id
    FROM campaign_operations_reconciliation_observation candidate
    WHERE candidate.operational_request_id = request.operational_request_id
      AND NOT EXISTS (
        SELECT 1 FROM campaign_operations_reconciliation_resolution resolution
        WHERE resolution.reconciliation_observation_id =
              candidate.reconciliation_observation_id)
    ORDER BY candidate.reconciliation_observation_id DESC LIMIT 1
    ) observation ON true;

-- H1 privilege surface.  The mutation capabilities intentionally receive no
-- table or transition-function privilege until their separately accepted
-- workflows exist.  The scheduler-evidence owner receives only exact columns.
REVOKE ALL PRIVILEGES ON
    campaign_operations_production_enablement_event,
    campaign_operations_production_enablement_audit_reference_event,
    campaign_operations_request_production_admission
FROM PUBLIC, pqxx,
    campaign_operations_production_enabler,
    campaign_operations_production_disabler,
    campaign_operations_production_dispatcher,
    campaign_operations_production_phase5_transactional,
    campaign_operations_production_reader;

REVOKE ALL PRIVILEGES ON public.experiment_scheduler_protocol
FROM campaign_operations_scheduler_protocol_evidence_owner;
/* The sealed owner executes these helpers.  The legacy-named ordinary role is
   retained only as a frozen capability name and owns no H1 object. */
/*
GRANT SELECT (singleton, required_generation, cutover_state,
    cutover_completed_at, cutover_completed_by, cutover_executable_path,
    cutover_process_evidence)
ON experiment_scheduler_protocol
TO campaign_operations_scheduler_protocol_evidence_owner;
*/
GRANT SELECT ON schema_migrations TO campaign_operations_owner;

REVOKE ALL PRIVILEGES ON FUNCTION
    campaign_operations_scheduler_protocol_evidence_snapshot_v1(),
    campaign_operations_scheduler_protocol_evidence_lock_v1(),
    campaign_operations_production_context_valid_v1(text, bigint, text),
    enforce_campaign_operations_production_context_empty_v1(),
    campaign_operations_production_enable_replay_v1(
        text, integer, text, text, text, text, text, text, text, text, text),
    campaign_operations_production_disable_replay_v1(
        text, bigint, text, integer, text, text),
    campaign_operations_production_acquire_replay_v2(
        bigint, integer, text, timestamptz, text, text, text),
    record_campaign_operations_production_enable_v1(
        text, integer, text, text, text, text, text, text, text, text, text),
    record_campaign_operations_production_disable_v1(
        text, bigint, text, integer, text, text),
    transition_campaign_operations_request_dispatch_production_v2(
        bigint, integer, text, timestamptz, text, text, text)
FROM PUBLIC, pqxx, campaign_operations_owner,
    campaign_operations_production_enabler,
    campaign_operations_production_disabler,
    campaign_operations_production_dispatcher,
    campaign_operations_production_phase5_transactional,
    campaign_operations_production_reader,
    campaign_operations_scheduler_protocol_evidence_reader;
GRANT EXECUTE ON FUNCTION
    campaign_operations_scheduler_protocol_evidence_snapshot_v1(),
    campaign_operations_scheduler_protocol_evidence_lock_v1()
TO campaign_operations_scheduler_protocol_evidence_reader;
GRANT EXECUTE ON FUNCTION
    campaign_operations_scheduler_protocol_evidence_snapshot_v1()
TO campaign_operations_owner;
GRANT SELECT ON
    campaign_operations_production_enablement_event,
    campaign_operations_production_enablement_audit_reference_event,
    campaign_operations_request_production_admission,
    campaign_operations_operational_request,
    campaign_operations_dispatch_attempt,
    campaign_operations_dispatch_audit_reference_event,
    campaign_operations_completion_event,
    campaign_operations_completion_audit_reference_event
TO campaign_operations_owner;
GRANT UPDATE (
    request_state, state_version, lease_token_hash, lease_expires_at,
    dispatcher_identity, updated_at)
ON campaign_operations_operational_request
TO campaign_operations_owner;

GRANT SELECT ON
    campaign_operations_production_enablement_event,
    campaign_operations_production_enablement_audit_reference_event,
    campaign_operations_request_production_admission,
    campaign_operations_production_readiness_v1,
    campaign_operations_production_status_v1
TO campaign_operations_production_reader;

DO $$
DECLARE sequence_name regclass;
BEGIN
    FOREACH sequence_name IN ARRAY ARRAY[
        pg_get_serial_sequence(
          'campaign_operations_production_enablement_event',
          'production_enablement_event_id')::regclass,
        pg_get_serial_sequence(
          'campaign_operations_production_enablement_audit_reference_event',
          'production_enablement_audit_reference_event_id')::regclass,
        pg_get_serial_sequence(
          'campaign_operations_request_production_admission',
          'request_production_admission_id')::regclass]
    LOOP
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM PUBLIC, pqxx',
            sequence_name);
    END LOOP;
END $$;

ALTER TABLE campaign_operations_production_enablement_event
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER TABLE campaign_operations_production_enablement_audit_reference_event
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER TABLE campaign_operations_request_production_admission
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER TABLE campaign_operations_operational_request
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER TABLE campaign_operations_dispatch_attempt
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER TABLE campaign_operations_dispatch_audit_reference_event
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER TABLE campaign_operations_completion_event
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER TABLE campaign_operations_completion_audit_reference_event
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER VIEW campaign_operations_production_readiness_v1
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER VIEW campaign_operations_production_status_v1
    OWNER TO campaign_operations_h1_boundary_authority;

ALTER FUNCTION campaign_operations_scheduler_protocol_evidence_snapshot_v1()
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER FUNCTION campaign_operations_scheduler_protocol_evidence_lock_v1()
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER FUNCTION record_campaign_operations_production_enable_v1(
    text, integer, text, text, text, text, text, text, text, text, text)
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER FUNCTION record_campaign_operations_production_disable_v1(
    text, bigint, text, integer, text, text)
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER FUNCTION transition_campaign_operations_request_dispatch_production_v2(
    bigint, integer, text, timestamptz, text, text, text)
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER FUNCTION campaign_operations_production_context_valid_v1(
    text, bigint, text)
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER FUNCTION enforce_campaign_operations_production_context_empty_v1()
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER FUNCTION campaign_operations_production_enable_replay_v1(
    text, integer, text, text, text, text, text, text, text, text, text)
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER FUNCTION campaign_operations_production_disable_replay_v1(
    text, bigint, text, integer, text, text)
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER FUNCTION campaign_operations_production_acquire_replay_v2(
    bigint, integer, text, timestamptz, text, text, text)
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER FUNCTION campaign_operations_production_enablement_history_valid_v1(
    bigint) OWNER TO campaign_operations_h1_boundary_authority;

DO $$
DECLARE function_name regprocedure;
BEGIN
    FOREACH function_name IN ARRAY ARRAY[
      'campaign_operations_manager_build_canonical_v1(text,text,text,text)'::regprocedure,
      'campaign_operations_scheduler_protocol_evidence_canonical_v1(integer,text,timestamp with time zone,text,text,text)'::regprocedure,
      'campaign_operations_production_enablement_history_valid_v1(bigint)'::regprocedure,
      'campaign_operations_production_enablement_canonical_v1(campaign_operations_production_enablement_event)'::regprocedure,
      'campaign_operations_request_production_admission_canonical_v1(campaign_operations_request_production_admission)'::regprocedure,
      'campaign_operations_dispatch_attempt_v2_canonical(campaign_operations_dispatch_attempt)'::regprocedure,
      'campaign_operations_has_explicit_role_v1(name,name)'::regprocedure,
      'campaign_operations_isolated_v1_authority_valid()'::regprocedure,
      'guard_campaign_operations_attempt_v1_isolation()'::regprocedure,
      'validate_campaign_operations_production_enablement_insert()'::regprocedure,
      'validate_campaign_ops_production_admission_insert()'::regprocedure,
      'validate_campaign_operations_dispatch_attempt_v2_insert()'::regprocedure,
      'guard_campaign_operations_production_admission_witness()'::regprocedure,
      'enforce_campaign_operations_production_admission_consistent()'::regprocedure,
      'enforce_campaign_ops_enablement_audit_complete()'::regprocedure,
      'reject_campaign_operations_production_mutation()'::regprocedure,
      'guard_campaign_operations_production_attempt_mutation()'::regprocedure,
      'guard_campaign_operations_production_dispatch_audit_mutation()'::regprocedure,
      'guard_campaign_operations_production_dispatch_audit_insert()'::regprocedure,
      'guard_campaign_operations_production_post_completion()'::regprocedure,
      'guard_campaign_operations_completion_v2_evidence()'::regprocedure,
      -- Exact pre-H1 trigger functions on the protected relations.  These are
      -- sealed because their ordinary owner could otherwise replace a guard;
      -- unrelated cancellation/lifecycle/recovery functions are not moved.
      'enforce_campaign_operations_request()'::regprocedure,
      'enforce_campaign_operations_request_acquisition_complete()'::regprocedure,
      'enforce_campaign_operations_dispatch_acquisition_complete()'::regprocedure,
      'enforce_campaign_operations_complete_binding()'::regprocedure,
      'guard_campaign_operations_dispatch_control()'::regprocedure,
      'enforce_campaign_operations_phase4_request_transition_complete()'::regprocedure,
      'guard_campaign_operations_completed_campaign()'::regprocedure,
      'enforce_campaign_operations_completion_event()'::regprocedure,
      'enforce_campaign_operations_completion_audit_complete()'::regprocedure,
      'reject_campaign_operations_completion_mutation()'::regprocedure,
      -- The schema-054 isolated Attempt V1 fixture remains supported but does
      -- not receive context or production-transition authority.
      'transition_campaign_operations_request_dispatching(bigint,integer,text,text)'::regprocedure]
    LOOP
        EXECUTE format('ALTER FUNCTION %s OWNER TO '
            'campaign_operations_h1_boundary_authority',
            function_name);
        EXECUTE format('REVOKE ALL PRIVILEGES ON FUNCTION %s FROM PUBLIC, pqxx',
            function_name);

        -- The schema-054 isolated Attempt V1 transition may arrive with the
        -- explicitly hardened predecessor-only dispatcher capability accepted
        -- by the H1A006 preflight.  That capability is not part of the final
        -- H1 contract, so remove it while sealing the function under the H1
        -- boundary authority.
        IF function_name =
           'transition_campaign_operations_request_dispatching(bigint,integer,text,text)'::regprocedure
        THEN
            EXECUTE format(
                'REVOKE ALL PRIVILEGES ON FUNCTION %s '
                'FROM campaign_operations_dispatcher',
                function_name);
        END IF;

        IF (SELECT function_row.prosecdef
            FROM pg_catalog.pg_proc function_row
            WHERE function_row.oid = function_name) THEN
            EXECUTE format('ALTER FUNCTION %s SET search_path TO '
                'pg_catalog, public', function_name);
        END IF;
    END LOOP;
END $$;

ALTER FUNCTION campaign_operations_tagged_fnv1a64(text)
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER FUNCTION lock_campaign_operations_authorization_head(bigint, text)
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER FUNCTION lock_campaign_operations_budget_head(bigint)
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER FUNCTION lock_campaign_operations_campaign(bigint)
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER FUNCTION lock_campaign_operations_reservation(bigint)
    OWNER TO campaign_operations_h1_boundary_authority;
ALTER FUNCTION lock_campaign_operations_request(bigint)
    OWNER TO campaign_operations_h1_boundary_authority;

-- Literal H1 authority-function ACL/search-path manifest.  This list is
-- intentionally duplicated in the read-only audit so missing and extra
-- signatures fail independently; no name-pattern mutation is permitted.
DO $$
DECLARE function_name regprocedure;
BEGIN
    FOREACH function_name IN ARRAY ARRAY[
      'campaign_operations_scheduler_protocol_evidence_snapshot_v1()'::regprocedure,
      'campaign_operations_scheduler_protocol_evidence_lock_v1()'::regprocedure,
      'record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)'::regprocedure,
      'record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)'::regprocedure,
      'transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)'::regprocedure,
      'campaign_operations_production_context_valid_v1(text,bigint,text)'::regprocedure,
      'enforce_campaign_operations_production_context_empty_v1()'::regprocedure,
      'campaign_operations_production_enable_replay_v1(text,integer,text,text,text,text,text,text,text,text,text)'::regprocedure,
      'campaign_operations_production_disable_replay_v1(text,bigint,text,integer,text,text)'::regprocedure,
      'campaign_operations_production_acquire_replay_v2(bigint,integer,text,timestamp with time zone,text,text,text)'::regprocedure,
      'campaign_operations_production_enablement_history_valid_v1(bigint)'::regprocedure,
      'campaign_operations_tagged_fnv1a64(text)'::regprocedure,
      'lock_campaign_operations_authorization_head(bigint,text)'::regprocedure,
      'lock_campaign_operations_budget_head(bigint)'::regprocedure,
      'lock_campaign_operations_campaign(bigint)'::regprocedure,
      'lock_campaign_operations_reservation(bigint)'::regprocedure,
      'lock_campaign_operations_request(bigint)'::regprocedure]
    LOOP
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON FUNCTION %s FROM PUBLIC, pqxx',
            function_name);

        -- Production predecessors may contain exact historical execution
        -- capabilities that H1A006 accepts only long enough to normalize the
        -- object.  Remove those predecessor-only grants before final H1 audit.
        IF function_name =
           'lock_campaign_operations_authorization_head(bigint,text)'::regprocedure
        THEN
            EXECUTE format(
                'REVOKE ALL PRIVILEGES ON FUNCTION %s FROM '
                'campaign_operations_dispatcher, '
                'campaign_operations_phase5_transactional',
                function_name);
        ELSIF function_name =
              'lock_campaign_operations_budget_head(bigint)'::regprocedure
        THEN
            EXECUTE format(
                'REVOKE ALL PRIVILEGES ON FUNCTION %s FROM '
                'campaign_operations_dispatcher, '
                'campaign_operations_phase5_transactional',
                function_name);
        ELSIF function_name =
              'lock_campaign_operations_campaign(bigint)'::regprocedure
        THEN
            EXECUTE format(
                'REVOKE ALL PRIVILEGES ON FUNCTION %s FROM '
                'campaign_operations_budget_administrator, '
                'campaign_operations_dispatcher, '
                'campaign_operations_phase5_transactional, '
                'campaign_operations_request_acceptor',
                function_name);
        ELSIF function_name =
              'lock_campaign_operations_request(bigint)'::regprocedure
        THEN
            EXECUTE format(
                'REVOKE ALL PRIVILEGES ON FUNCTION %s FROM '
                'campaign_operations_dispatcher, '
                'campaign_operations_phase5_transactional',
                function_name);
        ELSIF function_name =
              'lock_campaign_operations_reservation(bigint)'::regprocedure
        THEN
            EXECUTE format(
                'REVOKE ALL PRIVILEGES ON FUNCTION %s FROM '
                'campaign_operations_dispatcher, '
                'campaign_operations_phase5_transactional',
                function_name);
        END IF;

        IF (SELECT function_row.prosecdef
            FROM pg_catalog.pg_proc function_row
            WHERE function_row.oid = function_name) THEN
            EXECUTE format('ALTER FUNCTION %s SET search_path TO '
                'pg_catalog, public', function_name);
        END IF;
    END LOOP;
END $$;

-- Preserve accepted Phase A-G function composition after sealing these exact
-- helpers.  The former owner receives EXECUTE only; it receives no ownership,
-- grant option, context access, or H1 fixed-transition execution.
GRANT EXECUTE ON FUNCTION
    campaign_operations_tagged_fnv1a64(text),
    lock_campaign_operations_authorization_head(bigint, text),
    lock_campaign_operations_budget_head(bigint),
    lock_campaign_operations_campaign(bigint),
    lock_campaign_operations_reservation(bigint),
    lock_campaign_operations_request(bigint)
TO campaign_operations_owner;

DO $$
DECLARE sequence_name regclass;
BEGIN
    FOREACH sequence_name IN ARRAY ARRAY[
        pg_catalog.pg_get_serial_sequence(
          'public.campaign_operations_production_enablement_event',
          'production_enablement_event_id')::regclass,
        pg_catalog.pg_get_serial_sequence(
          'public.campaign_operations_production_enablement_audit_reference_event',
          'production_enablement_audit_reference_event_id')::regclass,
        pg_catalog.pg_get_serial_sequence(
          'public.campaign_operations_request_production_admission',
          'request_production_admission_id')::regclass,
        pg_catalog.pg_get_serial_sequence(
          'public.campaign_operations_operational_request',
          'operational_request_id')::regclass,
        pg_catalog.pg_get_serial_sequence(
          'public.campaign_operations_dispatch_attempt',
          'dispatch_attempt_id')::regclass,
        pg_catalog.pg_get_serial_sequence(
          'public.campaign_operations_dispatch_audit_reference_event',
          'dispatch_audit_reference_event_id')::regclass,
        pg_catalog.pg_get_serial_sequence(
          'public.campaign_operations_completion_event',
          'completion_event_id')::regclass,
        pg_catalog.pg_get_serial_sequence(
          'public.campaign_operations_completion_audit_reference_event',
          'completion_audit_reference_event_id')::regclass]
    LOOP
        IF sequence_name IS NOT NULL THEN
            EXECUTE format('ALTER SEQUENCE %s OWNER TO '
                'campaign_operations_h1_boundary_authority', sequence_name);
            EXECUTE format('REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM PUBLIC',
                sequence_name);
            -- Canonicalize the catalog representation used by supported
            -- pg_dump/pg_restore.  Ownership still confers full authority;
            -- the explicit tuple matrix retains only acldefault('S') USAGE.
            EXECUTE format('REVOKE SELECT, UPDATE ON SEQUENCE %s FROM '
                'campaign_operations_h1_boundary_authority', sequence_name);
        END IF;
    END LOOP;
END $$;

ALTER DEFAULT PRIVILEGES FOR ROLE
    campaign_operations_h1_boundary_authority
    REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE
    campaign_operations_h1_boundary_authority
    REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE
    campaign_operations_h1_boundary_authority
    REVOKE ALL ON SEQUENCES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE
    campaign_operations_h1_boundary_authority
    REVOKE USAGE ON TYPES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE
    campaign_operations_h1_boundary_authority
    REVOKE USAGE ON SCHEMAS FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE
    campaign_operations_h1_boundary_authority IN SCHEMA public
    REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE
    campaign_operations_h1_boundary_authority IN SCHEMA public
    REVOKE ALL ON SEQUENCES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE
    campaign_operations_h1_boundary_authority IN SCHEMA public
    REVOKE USAGE ON TYPES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE campaign_operations_owner
    REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE campaign_operations_owner
    REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE campaign_operations_owner
    REVOKE ALL ON SEQUENCES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE campaign_operations_owner
    REVOKE USAGE ON TYPES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE campaign_operations_owner
    REVOKE USAGE ON SCHEMAS FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE campaign_operations_owner IN SCHEMA public
    REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE campaign_operations_owner IN SCHEMA public
    REVOKE ALL ON SEQUENCES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE campaign_operations_owner IN SCHEMA public
    REVOKE USAGE ON TYPES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE campaign_operations_owner IN SCHEMA public
    REVOKE SELECT ON TABLES FROM pqxx;
ALTER DEFAULT PRIVILEGES FOR ROLE campaign_operations_owner IN SCHEMA public
    REVOKE SELECT, USAGE ON SEQUENCES FROM pqxx;
ALTER DEFAULT PRIVILEGES FOR ROLE
    campaign_operations_scheduler_protocol_evidence_owner
    REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE
    campaign_operations_scheduler_protocol_evidence_owner
    REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE
    campaign_operations_scheduler_protocol_evidence_owner
    REVOKE ALL ON SEQUENCES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE
    campaign_operations_scheduler_protocol_evidence_owner
    REVOKE USAGE ON TYPES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE
    campaign_operations_scheduler_protocol_evidence_owner
    REVOKE USAGE ON SCHEMAS FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE
    campaign_operations_scheduler_protocol_evidence_owner IN SCHEMA public
    REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE
    campaign_operations_scheduler_protocol_evidence_owner IN SCHEMA public
    REVOKE ALL ON SEQUENCES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE
    campaign_operations_scheduler_protocol_evidence_owner IN SCHEMA public
    REVOKE USAGE ON TYPES FROM PUBLIC;

CREATE OR REPLACE FUNCTION campaign_operations_h1_deployment_audit_v1(
    expected_migration_checksum text,
    require_migration_ledger boolean,
    require_historical_bytes boolean)
RETURNS boolean
LANGUAGE plpgsql
VOLATILE
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
DECLARE
    role_name text;
    role_oid oid;
    role_is_boundary boolean;
    boundary_oid oid;
    offending_entry text;
    allowed_relations text[] := ARRAY[
        'campaign_operations_production_transition_context',
        'campaign_operations_production_enablement_event',
        'campaign_operations_production_enablement_audit_reference_event',
        'campaign_operations_request_production_admission',
        'campaign_operations_operational_request',
        'campaign_operations_dispatch_attempt',
        'campaign_operations_dispatch_audit_reference_event',
        'campaign_operations_completion_event',
        'campaign_operations_completion_audit_reference_event',
        'campaign_operations_production_readiness_v1',
        'campaign_operations_production_status_v1'];
    allowed_functions text[] := ARRAY[
        'campaign_operations_scheduler_protocol_evidence_snapshot_v1',
        'campaign_operations_scheduler_protocol_evidence_lock_v1',
        'record_campaign_operations_production_enable_v1',
        'record_campaign_operations_production_disable_v1',
        'transition_campaign_operations_request_dispatch_production_v2',
        'campaign_operations_production_context_valid_v1',
        'enforce_campaign_operations_production_context_empty_v1',
        'campaign_operations_production_enable_replay_v1',
        'campaign_operations_production_disable_replay_v1',
        'campaign_operations_production_acquire_replay_v2',
        'campaign_operations_production_enablement_history_valid_v1',
        'campaign_operations_manager_build_canonical_v1',
        'campaign_operations_scheduler_protocol_evidence_canonical_v1',
        'campaign_operations_production_enablement_canonical_v1',
        'campaign_operations_request_production_admission_canonical_v1',
        'campaign_operations_dispatch_attempt_v2_canonical',
        'campaign_operations_has_explicit_role_v1',
        'campaign_operations_isolated_v1_authority_valid',
        'transition_campaign_operations_request_dispatching',
        'guard_campaign_operations_attempt_v1_isolation',
        'validate_campaign_operations_production_enablement_insert',
        'validate_campaign_ops_production_admission_insert',
        'validate_campaign_operations_dispatch_attempt_v2_insert',
        'guard_campaign_operations_production_admission_witness',
        'enforce_campaign_operations_production_admission_consistent',
        'enforce_campaign_ops_enablement_audit_complete',
        'reject_campaign_operations_production_mutation',
        'guard_campaign_operations_production_attempt_mutation',
        'guard_campaign_operations_production_dispatch_audit_mutation',
        'guard_campaign_operations_production_dispatch_audit_insert',
        'guard_campaign_operations_production_post_completion',
        'guard_campaign_operations_completion_v2_evidence',
        'enforce_campaign_operations_request',
        'enforce_campaign_operations_request_acquisition_complete',
        'enforce_campaign_operations_dispatch_acquisition_complete',
        'enforce_campaign_operations_complete_binding',
        'guard_campaign_operations_dispatch_control',
        'enforce_campaign_operations_phase4_request_transition_complete',
        'guard_campaign_operations_completed_campaign',
        'enforce_campaign_operations_completion_event',
        'enforce_campaign_operations_completion_audit_complete',
        'reject_campaign_operations_completion_mutation',
        'campaign_operations_tagged_fnv1a64',
        'lock_campaign_operations_authorization_head',
        'lock_campaign_operations_budget_head',
        'lock_campaign_operations_campaign',
        'lock_campaign_operations_reservation',
        'lock_campaign_operations_request',
        'campaign_operations_h1_deployment_audit_v1'];
    allowed_function_signatures text[] := ARRAY[
        'public.campaign_operations_dispatch_attempt_v2_canonical(campaign_operations_dispatch_attempt)',
        'public.campaign_operations_h1_deployment_audit_v1(text,boolean,boolean)',
        'public.campaign_operations_has_explicit_role_v1(name,name)',
        'public.campaign_operations_isolated_v1_authority_valid()',
        'public.campaign_operations_manager_build_canonical_v1(text,text,text,text)',
        'public.campaign_operations_production_acquire_replay_v2(bigint,integer,text,timestamp with time zone,text,text,text)',
        'public.campaign_operations_production_context_valid_v1(text,bigint,text)',
        'public.campaign_operations_production_disable_replay_v1(text,bigint,text,integer,text,text)',
        'public.campaign_operations_production_enable_replay_v1(text,integer,text,text,text,text,text,text,text,text,text)',
        'public.campaign_operations_production_enablement_history_valid_v1(bigint)',
        'public.campaign_operations_production_enablement_canonical_v1(campaign_operations_production_enablement_event)',
        'public.campaign_operations_request_production_admission_canonical_v1(campaign_operations_request_production_admission)',
        'public.campaign_operations_scheduler_protocol_evidence_canonical_v1(integer,text,timestamp with time zone,text,text,text)',
        'public.campaign_operations_scheduler_protocol_evidence_lock_v1()',
        'public.campaign_operations_scheduler_protocol_evidence_snapshot_v1()',
        'public.campaign_operations_tagged_fnv1a64(text)',
        'public.enforce_campaign_operations_complete_binding()',
        'public.enforce_campaign_operations_completion_audit_complete()',
        'public.enforce_campaign_operations_completion_event()',
        'public.enforce_campaign_operations_dispatch_acquisition_complete()',
        'public.enforce_campaign_operations_phase4_request_transition_complete()',
        'public.enforce_campaign_operations_production_admission_consistent()',
        'public.enforce_campaign_operations_production_context_empty_v1()',
        'public.enforce_campaign_operations_request()',
        'public.enforce_campaign_operations_request_acquisition_complete()',
        'public.enforce_campaign_ops_enablement_audit_complete()',
        'public.guard_campaign_operations_attempt_v1_isolation()',
        'public.guard_campaign_operations_completed_campaign()',
        'public.guard_campaign_operations_completion_v2_evidence()',
        'public.guard_campaign_operations_dispatch_control()',
        'public.guard_campaign_operations_production_admission_witness()',
        'public.guard_campaign_operations_production_attempt_mutation()',
        'public.guard_campaign_operations_production_dispatch_audit_insert()',
        'public.guard_campaign_operations_production_dispatch_audit_mutation()',
        'public.guard_campaign_operations_production_post_completion()',
        'public.lock_campaign_operations_authorization_head(bigint,text)',
        'public.lock_campaign_operations_budget_head(bigint)',
        'public.lock_campaign_operations_campaign(bigint)',
        'public.lock_campaign_operations_request(bigint)',
        'public.lock_campaign_operations_reservation(bigint)',
        'public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)',
        'public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)',
        'public.reject_campaign_operations_completion_mutation()',
        'public.reject_campaign_operations_production_mutation()',
        'public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)',
        'public.transition_campaign_operations_request_dispatching(bigint,integer,text,text)',
        'public.validate_campaign_operations_dispatch_attempt_v2_insert()',
        'public.validate_campaign_operations_production_enablement_insert()',
        'public.validate_campaign_ops_production_admission_insert()'];
    allowed_function_contracts text[] := ARRAY[
        'public.campaign_operations_dispatch_attempt_v2_canonical(campaign_operations_dispatch_attempt)|sql|false|i|u|0|0|NULL',
        'public.campaign_operations_h1_deployment_audit_v1(text,boolean,boolean)|plpgsql|false|v|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_has_explicit_role_v1(name,name)|sql|true|s|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_isolated_v1_authority_valid()|sql|true|s|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_manager_build_canonical_v1(text,text,text,text)|plpgsql|false|i|u|0|0|NULL',
        'public.campaign_operations_production_acquire_replay_v2(bigint,integer,text,timestamp with time zone,text,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_production_context_valid_v1(text,bigint,text)|sql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_production_disable_replay_v1(text,bigint,text,integer,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_production_enable_replay_v1(text,integer,text,text,text,text,text,text,text,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_production_enablement_history_valid_v1(bigint)|plpgsql|true|s|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_production_enablement_canonical_v1(campaign_operations_production_enablement_event)|plpgsql|false|i|u|0|0|NULL',
        'public.campaign_operations_request_production_admission_canonical_v1(campaign_operations_request_production_admission)|sql|false|i|u|0|0|NULL',
        'public.campaign_operations_scheduler_protocol_evidence_canonical_v1(integer,text,timestamp with time zone,text,text,text)|plpgsql|false|i|u|0|0|NULL',
        'public.campaign_operations_scheduler_protocol_evidence_lock_v1()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_scheduler_protocol_evidence_snapshot_v1()|plpgsql|true|s|u|0|0|search_path=pg_catalog, public',
        'public.campaign_operations_tagged_fnv1a64(text)|plpgsql|true|i|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_complete_binding()|plpgsql|false|v|u|0|0|search_path=pg_catalog, public, pg_temp',
        'public.enforce_campaign_operations_completion_audit_complete()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_completion_event()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_dispatch_acquisition_complete()|plpgsql|false|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_phase4_request_transition_complete()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_production_admission_consistent()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_production_context_empty_v1()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_request()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_operations_request_acquisition_complete()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.enforce_campaign_ops_enablement_audit_complete()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_attempt_v1_isolation()|plpgsql|false|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_completed_campaign()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_completion_v2_evidence()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_dispatch_control()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_production_admission_witness()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_production_attempt_mutation()|plpgsql|false|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_production_dispatch_audit_insert()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_production_dispatch_audit_mutation()|plpgsql|false|v|u|0|0|search_path=pg_catalog, public',
        'public.guard_campaign_operations_production_post_completion()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.lock_campaign_operations_authorization_head(bigint,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.lock_campaign_operations_budget_head(bigint)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.lock_campaign_operations_campaign(bigint)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.lock_campaign_operations_request(bigint)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.lock_campaign_operations_reservation(bigint)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.reject_campaign_operations_completion_mutation()|plpgsql|false|v|u|0|0|search_path=pg_catalog, public, pg_temp',
        'public.reject_campaign_operations_production_mutation()|plpgsql|false|v|u|0|0|NULL',
        'public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.transition_campaign_operations_request_dispatching(bigint,integer,text,text)|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.validate_campaign_operations_dispatch_attempt_v2_insert()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.validate_campaign_operations_production_enablement_insert()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public',
        'public.validate_campaign_ops_production_admission_insert()|plpgsql|true|v|u|0|0|search_path=pg_catalog, public'];
    allowed_function_return_contracts text[] := ARRAY[
        'public.campaign_operations_dispatch_attempt_v2_canonical(campaign_operations_dispatch_attempt)|pg_catalog.text|false|NULL|NULL|NULL',
        'public.campaign_operations_h1_deployment_audit_v1(text,boolean,boolean)|pg_catalog.bool|false|NULL|NULL|NULL',
        'public.campaign_operations_has_explicit_role_v1(name,name)|pg_catalog.bool|false|NULL|NULL|NULL',
        'public.campaign_operations_isolated_v1_authority_valid()|pg_catalog.bool|false|NULL|NULL|NULL',
        'public.campaign_operations_manager_build_canonical_v1(text,text,text,text)|pg_catalog.text|false|NULL|NULL|NULL',
        'public.campaign_operations_production_acquire_replay_v2(bigint,integer,text,timestamp with time zone,text,text,text)|public.campaign_operations_dispatch_attempt|true|NULL|NULL|NULL',
        'public.campaign_operations_production_context_valid_v1(text,bigint,text)|pg_catalog.bool|false|NULL|NULL|NULL',
        'public.campaign_operations_production_disable_replay_v1(text,bigint,text,integer,text,text)|public.campaign_operations_production_enablement_event|true|NULL|NULL|NULL',
        'public.campaign_operations_production_enable_replay_v1(text,integer,text,text,text,text,text,text,text,text,text)|public.campaign_operations_production_enablement_event|true|NULL|NULL|NULL',
        'public.campaign_operations_production_enablement_history_valid_v1(bigint)|pg_catalog.bool|false|NULL|NULL|NULL',
        'public.campaign_operations_production_enablement_canonical_v1(campaign_operations_production_enablement_event)|pg_catalog.text|false|NULL|NULL|NULL',
        'public.campaign_operations_request_production_admission_canonical_v1(campaign_operations_request_production_admission)|pg_catalog.text|false|NULL|NULL|NULL',
        'public.campaign_operations_scheduler_protocol_evidence_canonical_v1(integer,text,timestamp with time zone,text,text,text)|pg_catalog.text|false|NULL|NULL|NULL',
        'public.campaign_operations_scheduler_protocol_evidence_lock_v1()|pg_catalog.record|true|pg_catalog.int4,pg_catalog.text,pg_catalog.text,pg_catalog.text,pg_catalog.text,pg_catalog.text,pg_catalog.bool,pg_catalog.text,pg_catalog.text|t,t,t,t,t,t,t,t,t|required_generation,cutover_state,cutover_completed_at,cutover_completed_by,cutover_executable_path,cutover_process_evidence,evidence_complete,evidence_canonical,evidence_hash',
        'public.campaign_operations_scheduler_protocol_evidence_snapshot_v1()|pg_catalog.record|true|pg_catalog.int4,pg_catalog.text,pg_catalog.text,pg_catalog.text,pg_catalog.text,pg_catalog.text,pg_catalog.bool,pg_catalog.text,pg_catalog.text|t,t,t,t,t,t,t,t,t|required_generation,cutover_state,cutover_completed_at,cutover_completed_by,cutover_executable_path,cutover_process_evidence,evidence_complete,evidence_canonical,evidence_hash',
        'public.campaign_operations_tagged_fnv1a64(text)|pg_catalog.text|false|NULL|NULL|NULL',
        'public.enforce_campaign_operations_complete_binding()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.enforce_campaign_operations_completion_audit_complete()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.enforce_campaign_operations_completion_event()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.enforce_campaign_operations_dispatch_acquisition_complete()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.enforce_campaign_operations_phase4_request_transition_complete()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.enforce_campaign_operations_production_admission_consistent()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.enforce_campaign_operations_production_context_empty_v1()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.enforce_campaign_operations_request()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.enforce_campaign_operations_request_acquisition_complete()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.enforce_campaign_ops_enablement_audit_complete()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.guard_campaign_operations_attempt_v1_isolation()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.guard_campaign_operations_completed_campaign()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.guard_campaign_operations_completion_v2_evidence()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.guard_campaign_operations_dispatch_control()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.guard_campaign_operations_production_admission_witness()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.guard_campaign_operations_production_attempt_mutation()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.guard_campaign_operations_production_dispatch_audit_insert()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.guard_campaign_operations_production_dispatch_audit_mutation()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.guard_campaign_operations_production_post_completion()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.lock_campaign_operations_authorization_head(bigint,text)|public.campaign_operations_authorization_event|false|NULL|NULL|NULL',
        'public.lock_campaign_operations_budget_head(bigint)|public.campaign_operations_budget_ledger_entry|false|NULL|NULL|NULL',
        'public.lock_campaign_operations_campaign(bigint)|pg_catalog.void|false|NULL|NULL|NULL',
        'public.lock_campaign_operations_request(bigint)|public.campaign_operations_operational_request|false|NULL|NULL|NULL',
        'public.lock_campaign_operations_reservation(bigint)|public.campaign_operations_reservation|false|NULL|NULL|NULL',
        'public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)|public.campaign_operations_production_enablement_event|false|NULL|NULL|NULL',
        'public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)|public.campaign_operations_production_enablement_event|false|NULL|NULL|NULL',
        'public.reject_campaign_operations_completion_mutation()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.reject_campaign_operations_production_mutation()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)|public.campaign_operations_dispatch_attempt|false|NULL|NULL|NULL',
        'public.transition_campaign_operations_request_dispatching(bigint,integer,text,text)|public.campaign_operations_operational_request|false|NULL|NULL|NULL',
        'public.validate_campaign_operations_dispatch_attempt_v2_insert()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.validate_campaign_operations_production_enablement_insert()|pg_catalog.trigger|false|NULL|NULL|NULL',
        'public.validate_campaign_ops_production_admission_insert()|pg_catalog.trigger|false|NULL|NULL|NULL'];
    allowed_function_identity_argument_contracts text[] := ARRAY[
        'public.campaign_operations_dispatch_attempt_v2_canonical(campaign_operations_dispatch_attempt)|candidate campaign_operations_dispatch_attempt',
        'public.campaign_operations_h1_deployment_audit_v1(text,boolean,boolean)|expected_migration_checksum text, require_migration_ledger boolean, require_historical_bytes boolean',
        'public.campaign_operations_has_explicit_role_v1(name,name)|principal_name name, target_role_name name',
        'public.campaign_operations_manager_build_canonical_v1(text,text,text,text)|manager_service_contract_value text, source_commit_value text, compiler_contract_value text, executable_sha256_value text',
        'public.campaign_operations_production_acquire_replay_v2(bigint,integer,text,timestamp with time zone,text,text,text)|target_request_id bigint, expected_version_value integer, lease_digest_value text, lease_expires_at_value timestamp with time zone, operation_key_value text, requesting_actor_value text, approved_build_contract_canonical_value text',
        'public.campaign_operations_production_context_valid_v1(text,bigint,text)|transition_kind_value text, target_request_id bigint, operation_key_value text',
        'public.campaign_operations_production_disable_replay_v1(text,bigint,text,integer,text,text)|operation_key_value text, predecessor_event_id_value bigint, predecessor_event_canonical_value text, expected_prior_version_value integer, disabling_actor_value text, reason_value text',
        'public.campaign_operations_production_enable_replay_v1(text,integer,text,text,text,text,text,text,text,text,text)|operation_key_value text, expected_prior_version_value integer, scheduler_evidence_canonical_value text, independent_verification_reference_value text, authorizing_actor_value text, manager_service_contract_value text, approved_build_contract_canonical_value text, approved_build_source_commit_value text, approved_build_compiler_contract_value text, approved_build_executable_sha256_value text, reason_value text',
        'public.campaign_operations_production_enablement_history_valid_v1(bigint)|start_event_id bigint',
        'public.campaign_operations_production_enablement_canonical_v1(campaign_operations_production_enablement_event)|candidate campaign_operations_production_enablement_event',
        'public.campaign_operations_request_production_admission_canonical_v1(campaign_operations_request_production_admission)|candidate campaign_operations_request_production_admission',
        'public.campaign_operations_scheduler_protocol_evidence_canonical_v1(integer,text,timestamp with time zone,text,text,text)|required_generation_value integer, cutover_state_value text, cutover_completed_at_value timestamp with time zone, cutover_completed_by_value text, cutover_executable_path_value text, cutover_process_evidence_value text',
        'public.campaign_operations_tagged_fnv1a64(text)|canonical_value text',
        'public.lock_campaign_operations_authorization_head(bigint,text)|target_campaign_id bigint, target_action_kind text',
        'public.lock_campaign_operations_budget_head(bigint)|target_campaign_id bigint',
        'public.lock_campaign_operations_campaign(bigint)|target_operational_campaign_id bigint',
        'public.lock_campaign_operations_request(bigint)|target_request_id bigint',
        'public.lock_campaign_operations_reservation(bigint)|target_reservation_id bigint',
        'public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)|operation_key_value text, predecessor_event_id_value bigint, predecessor_event_canonical_value text, expected_prior_version_value integer, disabling_actor_value text, reason_value text',
        'public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)|operation_key_value text, expected_prior_version_value integer, scheduler_evidence_canonical_value text, independent_verification_reference_value text, authorizing_actor_value text, manager_service_contract_value text, approved_build_contract_canonical_value text, approved_build_source_commit_value text, approved_build_compiler_contract_value text, approved_build_executable_sha256_value text, reason_value text',
        'public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)|target_request_id bigint, expected_version_value integer, lease_digest_value text, lease_expires_at_value timestamp with time zone, operation_key_value text, requesting_actor_value text, approved_build_contract_canonical_value text',
        'public.transition_campaign_operations_request_dispatching(bigint,integer,text,text)|target_request_id bigint, expected_version_value integer, lease_digest_value text, dispatcher_value text'];
    allowed_function_final_extra_acl_contracts text[] := ARRAY[
        'public.campaign_operations_scheduler_protocol_evidence_lock_v1()|campaign_operations_scheduler_protocol_evidence_reader',
        'public.campaign_operations_scheduler_protocol_evidence_snapshot_v1()|campaign_operations_owner',
        'public.campaign_operations_scheduler_protocol_evidence_snapshot_v1()|campaign_operations_scheduler_protocol_evidence_reader',
        'public.campaign_operations_tagged_fnv1a64(text)|campaign_operations_owner',
        'public.lock_campaign_operations_authorization_head(bigint,text)|campaign_operations_completion_writer',
        'public.lock_campaign_operations_authorization_head(bigint,text)|campaign_operations_owner',
        'public.lock_campaign_operations_budget_head(bigint)|campaign_operations_cancellation_coordinator',
        'public.lock_campaign_operations_budget_head(bigint)|campaign_operations_completion_writer',
        'public.lock_campaign_operations_budget_head(bigint)|campaign_operations_owner',
        'public.lock_campaign_operations_campaign(bigint)|campaign_operations_cancellation_coordinator',
        'public.lock_campaign_operations_campaign(bigint)|campaign_operations_completion_writer',
        'public.lock_campaign_operations_campaign(bigint)|campaign_operations_controller',
        'public.lock_campaign_operations_campaign(bigint)|campaign_operations_owner',
        'public.lock_campaign_operations_campaign(bigint)|campaign_operations_reconciler',
        'public.lock_campaign_operations_campaign(bigint)|campaign_operations_recovery',
        'public.lock_campaign_operations_request(bigint)|campaign_operations_cancellation_coordinator',
        'public.lock_campaign_operations_request(bigint)|campaign_operations_completion_writer',
        'public.lock_campaign_operations_request(bigint)|campaign_operations_owner',
        'public.lock_campaign_operations_request(bigint)|campaign_operations_reconciler',
        'public.lock_campaign_operations_request(bigint)|campaign_operations_recovery',
        'public.lock_campaign_operations_reservation(bigint)|campaign_operations_cancellation_coordinator',
        'public.lock_campaign_operations_reservation(bigint)|campaign_operations_completion_writer',
        'public.lock_campaign_operations_reservation(bigint)|campaign_operations_owner',
        'public.lock_campaign_operations_reservation(bigint)|campaign_operations_recovery'];
    allowed_trigger_entries text[] := ARRAY[
        'public.campaign_operations_audit_reference_event.campaign_operations_audit_reference_event_completion_gate_trigg',
        'public.campaign_operations_authorization_event.campaign_operations_authorization_event_completion_gate_trigger',
        'public.campaign_operations_budget_ledger_entry.campaign_operations_budget_ledger_entry_completion_gate_trigger',
        'public.campaign_operations_cancellation_request.campaign_operations_cancellation_request_completion_gate_trigge',
        'public.campaign_operations_cancellation_settlement.campaign_operations_cancellation_settlement_completion_gate_tri',
        'public.campaign_operations_completion_audit_reference_event.campaign_operations_completion_audit_immutable_trigger',
        'public.campaign_operations_completion_audit_reference_event.campaign_operations_completion_audit_truncate_trigger',
        'public.campaign_operations_completion_event.campaign_operations_completion_audit_complete_trigger',
        'public.campaign_operations_completion_event.campaign_operations_completion_immutable_trigger',
        'public.campaign_operations_completion_event.campaign_operations_completion_truncate_trigger',
        'public.campaign_operations_completion_event.campaign_operations_completion_v2_evidence_gate',
        'public.campaign_operations_completion_event.campaign_operations_completion_validate_trigger',
        'public.campaign_operations_control_audit_reference_event.campaign_operations_control_audit_reference_event_completion_ga',
        'public.campaign_operations_control_event.campaign_operations_control_event_completion_gate_trigger',
        'public.campaign_operations_dispatch_attempt.campaign_operations_dispatch_attempt_completion_gate_trigger',
        'public.campaign_operations_dispatch_attempt.campaign_operations_dispatch_attempt_v2_validate',
        'public.campaign_operations_dispatch_attempt.campaign_operations_production_attempt_consistency',
        'public.campaign_operations_dispatch_attempt.campaign_ops_attempt_v1_isolation',
        'public.campaign_operations_dispatch_attempt.campaign_ops_attempt_v2_immutable_row',
        'public.campaign_operations_dispatch_attempt.campaign_ops_attempt_v2_immutable_truncate',
        'public.campaign_operations_dispatch_attempt_outcome.campaign_operations_dispatch_attempt_outcome_completion_gate_tr',
        'public.campaign_operations_dispatch_audit_reference_event.campaign_operations_dispatch_audit_reference_event_completion_g',
        'public.campaign_operations_dispatch_audit_reference_event.campaign_operations_production_audit_consistency',
        'public.campaign_operations_dispatch_audit_reference_event.campaign_operations_production_audit_insert_guard',
        'public.campaign_operations_dispatch_audit_reference_event.campaign_ops_production_audit_immutable_row',
        'public.campaign_operations_dispatch_audit_reference_event.campaign_ops_production_audit_immutable_truncate',
        'public.campaign_operations_downstream_control_owner.campaign_operations_complete_owner_trigger',
        'public.campaign_operations_downstream_control_owner.campaign_operations_downstream_control_owner_completion_gate_tr',
        'public.campaign_operations_governance_provenance_event.campaign_operations_governance_provenance_event_completion_gate',
        'public.campaign_operations_operational_request.campaign_operations_complete_request_trigger',
        'public.campaign_operations_operational_request.campaign_operations_dispatch_acquisition_complete_trigger',
        'public.campaign_operations_operational_request.campaign_operations_dispatch_control_gate_trigger',
        'public.campaign_operations_operational_request.campaign_operations_operational_request_completion_gate_trigger',
        'public.campaign_operations_operational_request.campaign_operations_phase4_request_transition_complete_trigger',
        'public.campaign_operations_operational_request.campaign_operations_production_request_consistency',
        'public.campaign_operations_operational_request.campaign_operations_production_witness_truncate_guard',
        'public.campaign_operations_operational_request.campaign_operations_production_witness_update_guard',
        'public.campaign_operations_operational_request.campaign_operations_request_acquisition_complete_trigger',
        'public.campaign_operations_operational_request.campaign_operations_request_completion_update_gate',
        'public.campaign_operations_operational_request.campaign_operations_request_trigger',
        'public.campaign_operations_production_enablement_audit_reference_event.campaign_operations_enablement_audit_complete',
        'public.campaign_operations_production_enablement_audit_reference_event.campaign_ops_enablement_audit_immutable_row',
        'public.campaign_operations_production_enablement_audit_reference_event.campaign_ops_enablement_audit_immutable_truncate',
        'public.campaign_operations_production_enablement_event.campaign_operations_enablement_audit_event_complete',
        'public.campaign_operations_production_enablement_event.campaign_operations_production_enablement_validate',
        'public.campaign_operations_production_enablement_event.campaign_ops_enablement_immutable_row',
        'public.campaign_operations_production_enablement_event.campaign_ops_enablement_immutable_truncate',
        'public.campaign_operations_production_transition_context.campaign_operations_production_context_empty_v1',
        'public.campaign_operations_reconciliation_observation.campaign_operations_reconciliation_observation_completion_gate_',
        'public.campaign_operations_reconciliation_resolution.campaign_operations_reconciliation_resolution_completion_gate_t',
        'public.campaign_operations_request_binding.campaign_operations_complete_binding_trigger',
        'public.campaign_operations_request_binding.campaign_operations_request_binding_completion_gate_trigger',
        'public.campaign_operations_request_production_admission.campaign_operations_production_admission_consistency',
        'public.campaign_operations_request_production_admission.campaign_operations_production_admission_post_completion',
        'public.campaign_operations_request_production_admission.campaign_operations_production_admission_validate',
        'public.campaign_operations_request_production_admission.campaign_ops_admission_immutable_row',
        'public.campaign_operations_request_production_admission.campaign_ops_admission_immutable_truncate',
        'public.campaign_operations_reservation.campaign_operations_complete_reservation_trigger',
        'public.campaign_operations_reservation.campaign_operations_reservation_completion_gate_trigger',
        'public.campaign_operations_reservation.campaign_operations_reservation_completion_update_gate',
        'public.campaign_operations_reservation_commitment.campaign_operations_reservation_commitment_completion_gate_trig',
        'public.campaign_operations_reservation_event.campaign_operations_reservation_event_completion_gate_trigger',
        'public.experiment_lifecycle_cancellation_event.experiment_lifecycle_cancellation_event_completion_gate_trigger'];
BEGIN
    IF current_setting('transaction_read_only') <> 'on' AND
       current_setting('default_transaction_read_only') <> 'on' THEN
        -- The function itself performs no write.  The command wrapper starts a
        -- READ ONLY transaction; direct superuser invocation remains useful to
        -- migration 055 before the ledger exists.
        NULL;
    END IF;

    FOREACH role_name IN ARRAY ARRAY[
        'campaign_operations_h1_boundary_authority',
        'campaign_operations_production_enabler',
        'campaign_operations_production_disabler',
        'campaign_operations_production_dispatcher',
        'campaign_operations_production_phase5_transactional',
        'campaign_operations_production_reader',
        'campaign_operations_scheduler_protocol_evidence_owner',
        'campaign_operations_scheduler_protocol_evidence_reader']
    LOOP
        role_is_boundary :=
            role_name = 'campaign_operations_h1_boundary_authority';
        SELECT role.oid INTO role_oid
        FROM pg_catalog.pg_authid role
        WHERE role.rolname = role_name;
        IF role_oid IS NULL THEN
            RAISE EXCEPTION 'H1A001 required H1 role missing: %', role_name
                USING ERRCODE = '42501';
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM pg_catalog.pg_authid role
            WHERE role.oid = role_oid
              AND NOT role.rolcanlogin
              AND role.rolsuper = role_is_boundary
              AND role.rolinherit
              AND NOT role.rolcreatedb
              AND NOT role.rolcreaterole
              AND NOT role.rolreplication
              AND NOT role.rolbypassrls
              AND role.rolconnlimit = -1
              AND role.rolpassword IS NULL
              AND role.rolvaliduntil IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM pg_catalog.pg_db_role_setting setting
                  WHERE setting.setrole = role.oid)) THEN
            RAISE EXCEPTION 'H1A002 role identity mismatch: %', role_name
                USING ERRCODE = '42501';
        END IF;
    END LOOP;

    SELECT oid INTO STRICT boundary_oid
    FROM pg_catalog.pg_roles
    WHERE rolname = 'campaign_operations_h1_boundary_authority';

    -- Disposable assurance objects may declare the ACL origin that the
    -- production audit must verify.  These annotations cannot make an object
    -- admissible: every such object remains outside the exact ownership
    -- inventory and therefore fails closed.  The early check exists only so
    -- both origin directions and every catalog ACL column receive the stable,
    -- exact H1A006 diagnostic before the broader H1A004 inventory rejection.
    SELECT contract.object_identity INTO offending_entry
    FROM (
      SELECT namespace.nspname AS object_identity,
             pg_catalog.obj_description(namespace.oid, 'pg_namespace') AS policy,
             namespace.nspacl IS NULL AS acl_is_null
      FROM pg_catalog.pg_namespace namespace
      UNION ALL
      SELECT pg_catalog.format('%I.%I', namespace.nspname, relation.relname),
             pg_catalog.obj_description(relation.oid, 'pg_class'),
             relation.relacl IS NULL
      FROM pg_catalog.pg_class relation
      JOIN pg_catalog.pg_namespace namespace
        ON namespace.oid = relation.relnamespace
      WHERE relation.relkind IN ('r', 'p', 'v', 'm', 'f', 'S')
      UNION ALL
      SELECT pg_catalog.format('%I.%s', namespace.nspname,
                               function_row.oid::pg_catalog.regprocedure::text),
             pg_catalog.obj_description(function_row.oid, 'pg_proc'),
             function_row.proacl IS NULL
      FROM pg_catalog.pg_proc function_row
      JOIN pg_catalog.pg_namespace namespace
        ON namespace.oid = function_row.pronamespace
      UNION ALL
      SELECT pg_catalog.format('%I.%I', namespace.nspname, type_row.typname),
             pg_catalog.obj_description(type_row.oid, 'pg_type'),
             type_row.typacl IS NULL
      FROM pg_catalog.pg_type type_row
      JOIN pg_catalog.pg_namespace namespace
        ON namespace.oid = type_row.typnamespace
      UNION ALL
      SELECT pg_catalog.format('%I.%I.%I', namespace.nspname,
                               relation.relname, attribute.attname),
             pg_catalog.col_description(attribute.attrelid, attribute.attnum),
             attribute.attacl IS NULL
      FROM pg_catalog.pg_attribute attribute
      JOIN pg_catalog.pg_class relation ON relation.oid = attribute.attrelid
      JOIN pg_catalog.pg_namespace namespace
        ON namespace.oid = relation.relnamespace
      WHERE attribute.attnum > 0 AND NOT attribute.attisdropped
    ) contract
    WHERE contract.policy IN ('h1-acl-origin:explicit', 'h1-acl-origin:null')
      AND contract.acl_is_null <>
          (contract.policy = 'h1-acl-origin:null')
    ORDER BY contract.object_identity
    LIMIT 1;
    IF offending_entry IS NOT NULL THEN
        RAISE EXCEPTION
            'H1A006 exact ACL origin mismatch object=% stage=database-audit',
            offending_entry USING ERRCODE = '42501';
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_catalog.pg_auth_members membership
        JOIN pg_catalog.pg_roles granted ON granted.oid = membership.roleid
        JOIN pg_catalog.pg_roles member_role ON member_role.oid = membership.member
        WHERE granted.rolname = ANY (ARRAY[
            'campaign_operations_h1_boundary_authority',
            'campaign_operations_production_enabler',
            'campaign_operations_production_disabler',
            'campaign_operations_production_dispatcher',
            'campaign_operations_production_phase5_transactional',
            'campaign_operations_production_reader',
            'campaign_operations_scheduler_protocol_evidence_owner',
            'campaign_operations_scheduler_protocol_evidence_reader'])
           OR member_role.rolname = ANY (ARRAY[
            'campaign_operations_h1_boundary_authority',
            'campaign_operations_production_enabler',
            'campaign_operations_production_disabler',
            'campaign_operations_production_dispatcher',
            'campaign_operations_production_phase5_transactional',
            'campaign_operations_production_reader',
            'campaign_operations_scheduler_protocol_evidence_owner',
            'campaign_operations_scheduler_protocol_evidence_reader'])) THEN
        RAISE EXCEPTION 'H1A003 prohibited H1 role-graph edge'
            USING ERRCODE = '42501';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_catalog.pg_namespace
               WHERE nspowner = boundary_oid) THEN
        RAISE EXCEPTION 'H1A004 unexpected boundary-owned schema'
            USING ERRCODE = '42501';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_type type_row
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = type_row.typnamespace
        WHERE type_row.typowner = boundary_oid
          AND namespace.nspname NOT LIKE 'pg\_%' ESCAPE '\'
          AND namespace.nspname <> 'information_schema'
          AND NOT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_type coupled_type
            JOIN pg_catalog.pg_class relation
              ON relation.oid = coupled_type.typrelid
            JOIN pg_catalog.pg_namespace relation_namespace
              ON relation_namespace.oid = relation.relnamespace
            WHERE coupled_type.oid IN (type_row.oid, type_row.typelem)
              AND relation_namespace.nspname = 'public'
              AND relation.relname = ANY (allowed_relations))) OR
       EXISTS (SELECT 1 FROM pg_catalog.pg_operator
              WHERE oprowner = boundary_oid) OR
       EXISTS (SELECT 1 FROM pg_catalog.pg_event_trigger
              WHERE evtowner = boundary_oid) OR
       EXISTS (SELECT 1 FROM pg_catalog.pg_largeobject_metadata
              WHERE lomowner = boundary_oid) OR
       EXISTS (SELECT 1 FROM pg_catalog.pg_publication
              WHERE pubowner = boundary_oid) OR
       EXISTS (SELECT 1 FROM pg_catalog.pg_subscription
              WHERE subowner = boundary_oid) THEN
        RAISE EXCEPTION 'H1A004 unexpected boundary-owned catalog object'
            USING ERRCODE = '42501';
    END IF;

    SELECT offender.entry INTO offending_entry
    FROM (
      SELECT 'operator ' || namespace.nspname || '.' || operator.oprname ||
             ' -> ' || operator.oprcode::pg_catalog.regprocedure::text AS entry
      FROM pg_catalog.pg_operator operator
      JOIN pg_catalog.pg_namespace namespace
        ON namespace.oid = operator.oprnamespace
      JOIN pg_catalog.pg_proc implementation
        ON implementation.oid = operator.oprcode
      WHERE implementation.proowner = boundary_oid
      UNION ALL
      SELECT 'rule ' || namespace.nspname || '.' || relation.relname || '.' ||
             rule.rulename || ' -> ' ||
             protected_function.oid::pg_catalog.regprocedure::text
      FROM pg_catalog.pg_rewrite rule
      JOIN pg_catalog.pg_class relation ON relation.oid = rule.ev_class
      JOIN pg_catalog.pg_namespace namespace
        ON namespace.oid = relation.relnamespace
      JOIN pg_catalog.pg_depend dependency
        ON dependency.classid = 'pg_rewrite'::pg_catalog.regclass
       AND dependency.objid = rule.oid
       AND dependency.refclassid = 'pg_proc'::pg_catalog.regclass
      JOIN pg_catalog.pg_proc protected_function
        ON protected_function.oid = dependency.refobjid
      WHERE protected_function.proowner = boundary_oid
        AND NOT (rule.rulename = '_RETURN' AND namespace.nspname = 'public'
                 AND relation.relname = ANY (ARRAY[
                   'campaign_operations_production_readiness_v1',
                   'campaign_operations_production_status_v1']))
      ORDER BY 1 LIMIT 1
    ) offender;
    IF offending_entry IS NOT NULL THEN
        RAISE EXCEPTION 'H1A005 protected rule or operator entry point'
            USING ERRCODE = '42501', DETAIL = offending_entry;
    END IF;

    IF (SELECT count(*)
        FROM pg_catalog.pg_trigger trigger_row
        JOIN pg_catalog.pg_proc trigger_function
          ON trigger_function.oid = trigger_row.tgfoid
        WHERE NOT trigger_row.tgisinternal
          AND trigger_function.proowner = boundary_oid) <>
       pg_catalog.cardinality(allowed_trigger_entries) OR EXISTS (
        SELECT 1
        FROM pg_catalog.pg_trigger trigger_row
        JOIN pg_catalog.pg_proc trigger_function
          ON trigger_function.oid = trigger_row.tgfoid
        JOIN pg_catalog.pg_class trigger_relation
          ON trigger_relation.oid = trigger_row.tgrelid
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = trigger_relation.relnamespace
        WHERE NOT trigger_row.tgisinternal
          AND trigger_function.proowner = boundary_oid
          AND (namespace.nspname || '.' || trigger_relation.relname || '.' ||
               trigger_row.tgname) <> ALL (allowed_trigger_entries)) OR
       EXISTS (
        SELECT 1
        FROM pg_catalog.pg_cast cast_row
        JOIN pg_catalog.pg_proc cast_function
          ON cast_function.oid = cast_row.castfunc
        WHERE cast_function.proowner = boundary_oid) OR
       EXISTS (
        SELECT 1
        FROM pg_catalog.pg_depend dependency
        JOIN pg_catalog.pg_proc protected_function
          ON protected_function.oid = dependency.refobjid
        JOIN pg_catalog.pg_proc dependent_function
          ON dependency.classid = 'pg_proc'::regclass
         AND dependent_function.oid = dependency.objid
        WHERE dependency.refclassid = 'pg_proc'::regclass
          AND protected_function.proowner = boundary_oid
          AND dependent_function.proowner <> boundary_oid) THEN
        SELECT pg_catalog.string_agg(entry, ', ' ORDER BY entry)
          INTO offending_entry
        FROM (
          SELECT 'trigger ' || namespace.nspname || '.' ||
                 trigger_relation.relname || '.' || trigger_row.tgname AS entry
          FROM pg_catalog.pg_trigger trigger_row
          JOIN pg_catalog.pg_proc trigger_function
            ON trigger_function.oid = trigger_row.tgfoid
          JOIN pg_catalog.pg_class trigger_relation
            ON trigger_relation.oid = trigger_row.tgrelid
          JOIN pg_catalog.pg_namespace namespace
            ON namespace.oid = trigger_relation.relnamespace
          WHERE NOT trigger_row.tgisinternal
            AND trigger_function.proowner = boundary_oid
            AND (namespace.nspname || '.' || trigger_relation.relname || '.' ||
                 trigger_row.tgname) <> ALL (allowed_trigger_entries)
          UNION ALL
          SELECT 'cast ' || cast_row.oid::text
          FROM pg_catalog.pg_cast cast_row
          JOIN pg_catalog.pg_proc cast_function
            ON cast_function.oid = cast_row.castfunc
          WHERE cast_function.proowner = boundary_oid
          UNION ALL
          SELECT 'dependency ' || dependent_namespace.nspname || '.' ||
                 dependent_function.oid::pg_catalog.regprocedure::text ||
                 ' -> ' || protected_namespace.nspname || '.' ||
                 protected_function.oid::pg_catalog.regprocedure::text
          FROM pg_catalog.pg_depend dependency
          JOIN pg_catalog.pg_proc protected_function
            ON protected_function.oid = dependency.refobjid
          JOIN pg_catalog.pg_namespace protected_namespace
            ON protected_namespace.oid = protected_function.pronamespace
          JOIN pg_catalog.pg_proc dependent_function
            ON dependency.classid = 'pg_proc'::regclass
           AND dependent_function.oid = dependency.objid
          JOIN pg_catalog.pg_namespace dependent_namespace
            ON dependent_namespace.oid = dependent_function.pronamespace
          WHERE dependency.refclassid = 'pg_proc'::regclass
            AND protected_function.proowner = boundary_oid
            AND dependent_function.proowner <> boundary_oid
        ) offender;
        RAISE EXCEPTION 'H1A005 protected alternate dependency or entry point'
            USING ERRCODE = '42501',
                  DETAIL = coalesce(offending_entry, 'unknown');
    END IF;

    IF (SELECT count(*)
        FROM pg_catalog.pg_class relation
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = relation.relnamespace
        WHERE relation.relowner = boundary_oid
          AND namespace.nspname = 'public'
          AND relation.relkind IN ('r', 'p', 'v', 'm', 'f')) <>
       pg_catalog.cardinality(allowed_relations) OR EXISTS (
        SELECT 1
        FROM pg_catalog.pg_class relation
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = relation.relnamespace
        WHERE relation.relowner = boundary_oid
          AND relation.relkind IN ('r', 'p', 'v', 'm', 'f')
          AND NOT (namespace.nspname = 'public' AND
                   relation.relname = ANY (allowed_relations))) THEN
        RAISE EXCEPTION 'H1A004 exact boundary relation inventory mismatch'
            USING ERRCODE = '42501';
    END IF;

    IF (SELECT count(*)
        FROM pg_catalog.pg_class sequence_row
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = sequence_row.relnamespace
        WHERE sequence_row.relowner = boundary_oid
          AND sequence_row.relkind = 'S'
          AND namespace.nspname = 'public') <> 8 OR EXISTS (
        SELECT 1
        FROM pg_catalog.pg_class sequence_row
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = sequence_row.relnamespace
        WHERE sequence_row.relowner = boundary_oid
          AND sequence_row.relkind = 'S'
          AND NOT (namespace.nspname = 'public' AND EXISTS (
            SELECT 1 FROM pg_catalog.pg_depend dependency
            JOIN pg_catalog.pg_class table_row
              ON table_row.oid = dependency.refobjid
            WHERE dependency.classid = 'pg_class'::regclass
              AND dependency.objid = sequence_row.oid
              AND dependency.refclassid = 'pg_class'::regclass
              AND dependency.deptype IN ('a', 'i')
              AND table_row.relname = ANY (allowed_relations)))) THEN
        RAISE EXCEPTION 'H1A004 exact boundary sequence inventory mismatch'
            USING ERRCODE = '42501';
    END IF;

    IF (SELECT count(*)
        FROM pg_catalog.pg_proc function_row
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = function_row.pronamespace
        WHERE function_row.proowner = boundary_oid
          AND namespace.nspname = 'public'
          AND pg_catalog.format('%I.%s', namespace.nspname,
                function_row.oid::pg_catalog.regprocedure::text) =
              ANY (allowed_function_signatures)) <>
       pg_catalog.cardinality(allowed_function_signatures) OR EXISTS (
        SELECT 1
        FROM pg_catalog.pg_proc function_row
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = function_row.pronamespace
        WHERE function_row.proowner = boundary_oid
          AND pg_catalog.format('%I.%s', namespace.nspname,
                function_row.oid::pg_catalog.regprocedure::text) <>
              ALL (allowed_function_signatures)) THEN
        RAISE EXCEPTION 'H1A004 exact boundary function inventory mismatch'
            USING ERRCODE = '42501';
    END IF;

    -- ADR-0019B freezes the catalog origin as well as the expanded ACL
    -- tuples.  An explicit owner-only ACL is deliberately not equivalent to
    -- a NULL ACL whose acldefault() expansion happens to be owner-only.
    SELECT candidate.object_identity INTO offending_entry
    FROM (
        SELECT pg_catalog.format('%I.%I', namespace.nspname,
                   relation.relname) AS object_identity
        FROM pg_catalog.pg_class relation
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = relation.relnamespace
        WHERE relation.relowner = boundary_oid
          AND relation.relkind IN ('r', 'p', 'v', 'm', 'f', 'S')
          AND relation.relacl IS NULL
        UNION ALL
        SELECT pg_catalog.format('%I.%s', namespace.nspname,
                   function_row.oid::regprocedure::text)
        FROM pg_catalog.pg_proc function_row
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = function_row.pronamespace
        WHERE function_row.proowner = boundary_oid
          AND function_row.proacl IS NULL
        UNION ALL
        SELECT 'public'
        FROM pg_catalog.pg_namespace namespace
        WHERE namespace.nspname = 'public'
          AND namespace.nspacl IS NULL
        UNION ALL
        SELECT pg_catalog.format('%I.%I', namespace.nspname, type_row.typname)
        FROM pg_catalog.pg_type type_row
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = type_row.typnamespace
        WHERE type_row.typowner = boundary_oid
          AND namespace.nspname = 'public'
          AND type_row.typacl IS NOT NULL
        UNION ALL
        SELECT pg_catalog.format('%I.%I', namespace.nspname, relation.relname)
        FROM pg_catalog.pg_class relation
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = relation.relnamespace
        WHERE namespace.nspname = 'public'
          AND relation.relname IN
              ('schema_migrations', 'experiment_scheduler_protocol')
          AND relation.relacl IS NULL
    ) candidate
    ORDER BY candidate.object_identity
    LIMIT 1;
    IF offending_entry IS NOT NULL THEN
        RAISE EXCEPTION
            'H1A006 exact ACL origin mismatch object=% stage=database-audit',
            offending_entry USING ERRCODE = '42501';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_proc function_row
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = function_row.pronamespace,
        LATERAL pg_catalog.aclexplode(coalesce(
            function_row.proacl,
            pg_catalog.acldefault('f', function_row.proowner))) acl
        WHERE function_row.proowner = boundary_oid
          AND (acl.grantee = 0 OR acl.is_grantable)) THEN
        RAISE EXCEPTION 'H1A006 boundary function PUBLIC/grant-option ACL'
            USING ERRCODE = '42501';
    END IF;

    -- Fail closed before expanding expected ACL tuples.  A missing role must
    -- remain visible as a contract failure rather than disappearing through a
    -- catalog join.
    SELECT pg_catalog.format('role=%s object=%s stage=database-audit',
             pg_catalog.split_part(expected.contract, '|', 2),
             pg_catalog.split_part(expected.contract, '|', 1))
      INTO offending_entry
        FROM pg_catalog.unnest(
               allowed_function_final_extra_acl_contracts)
             expected(contract)
        LEFT JOIN pg_catalog.pg_roles grantee_role
          ON grantee_role.rolname =
             pg_catalog.split_part(expected.contract, '|', 2)
        WHERE grantee_role.oid IS NULL
        ORDER BY pg_catalog.split_part(expected.contract, '|', 2),
                 pg_catalog.split_part(expected.contract, '|', 1)
        LIMIT 1;
    IF offending_entry IS NOT NULL THEN
        RAISE EXCEPTION
            'H1A006 missing expected protected function ACL grantee role: %',
            offending_entry USING ERRCODE = '42501';
    END IF;

    SELECT pg_catalog.format('%I.%s', namespace.nspname,
             function_row.oid::pg_catalog.regprocedure::text)
      INTO offending_entry
        FROM pg_catalog.pg_proc function_row
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = function_row.pronamespace
        WHERE function_row.proowner = boundary_oid
          AND pg_catalog.format('%I.%s', namespace.nspname,
                function_row.oid::pg_catalog.regprocedure::text) =
              ANY (allowed_function_signatures)
          AND EXISTS (
            WITH actual_acl(grantee, privilege_type, is_grantable) AS (
              SELECT acl.grantee, acl.privilege_type, acl.is_grantable
              FROM pg_catalog.aclexplode(function_row.proacl) acl
            ),
            expected_acl(grantee, privilege_type, is_grantable) AS (
              SELECT function_row.proowner, 'EXECUTE'::text, false
              UNION
              SELECT grantee_role.oid, 'EXECUTE'::text, false
              FROM pg_catalog.unnest(
                     allowed_function_final_extra_acl_contracts)
                   expected(contract)
              LEFT JOIN pg_catalog.pg_roles grantee_role
                ON grantee_role.rolname =
                   pg_catalog.split_part(expected.contract, '|', 2)
              WHERE pg_catalog.split_part(expected.contract, '|', 1) =
                    pg_catalog.format('%I.%s', namespace.nspname,
                      function_row.oid::pg_catalog.regprocedure::text)
            )
            SELECT 1 FROM (
              (SELECT * FROM actual_acl EXCEPT SELECT * FROM expected_acl)
              UNION ALL
              (SELECT * FROM expected_acl EXCEPT SELECT * FROM actual_acl)
            ) difference
          )
        ORDER BY 1
        LIMIT 1;
    IF offending_entry IS NOT NULL THEN
        RAISE EXCEPTION
            'H1A006 exact protected function ACL mismatch object=% stage=database-audit',
            offending_entry USING ERRCODE = '42501';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_class relation,
        LATERAL pg_catalog.aclexplode(coalesce(
            relation.relacl,
            pg_catalog.acldefault(
                CASE relation.relkind
                  WHEN 'S' THEN 'S'::"char" ELSE 'r'::"char" END,
                relation.relowner))) acl
        WHERE relation.relowner = boundary_oid
          AND relation.relkind IN ('r', 'p', 'v', 'm', 'f', 'S')
          AND (acl.grantee = 0 OR acl.is_grantable)) OR EXISTS (
        SELECT 1
        FROM pg_catalog.pg_proc function_row,
        LATERAL pg_catalog.aclexplode(coalesce(
            function_row.proacl,
            pg_catalog.acldefault('f', function_row.proowner))) acl
        WHERE function_row.proowner <> boundary_oid
          AND acl.grantee = boundary_oid) OR EXISTS (
        SELECT 1
        FROM pg_catalog.pg_class relation,
        LATERAL pg_catalog.aclexplode(coalesce(
            relation.relacl,
            pg_catalog.acldefault(
                CASE relation.relkind
                  WHEN 'S' THEN 'S'::"char" ELSE 'r'::"char" END,
                relation.relowner))) acl
        WHERE relation.relowner <> boundary_oid
          AND acl.grantee = boundary_oid) THEN
        RAISE EXCEPTION 'H1A006 boundary relation or inbound ACL mismatch'
            USING ERRCODE = '42501';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_default_acl default_acl
        JOIN pg_catalog.pg_roles owner_role
          ON owner_role.oid = default_acl.defaclrole,
        LATERAL pg_catalog.aclexplode(default_acl.defaclacl) acl
        WHERE owner_role.rolname = ANY (ARRAY[
            'campaign_operations_h1_boundary_authority',
            'campaign_operations_owner',
            'campaign_operations_scheduler_protocol_evidence_owner'])
          AND acl.grantee <> default_acl.defaclrole) OR EXISTS (
        SELECT 1
        FROM unnest(ARRAY[
            'campaign_operations_h1_boundary_authority',
            'campaign_operations_owner',
            'campaign_operations_scheduler_protocol_evidence_owner'])
            expected_owner(role_name)
        JOIN pg_catalog.pg_roles owner_role
          ON owner_role.rolname = expected_owner.role_name
        WHERE NOT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_default_acl default_acl
            WHERE default_acl.defaclrole = owner_role.oid
              AND default_acl.defaclnamespace = 0
              AND default_acl.defaclobjtype = 'f')) THEN
        RAISE EXCEPTION 'H1A007 default ACL matrix mismatch'
            USING ERRCODE = '42501';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_proc function_row
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = function_row.pronamespace
        WHERE function_row.proname = ANY (allowed_functions)
          AND namespace.nspname NOT LIKE 'pg\_%' ESCAPE '\'
          AND namespace.nspname <> 'information_schema'
        GROUP BY function_row.proname
        HAVING count(*) <> 1 OR bool_or(namespace.nspname <> 'public')) THEN
        RAISE EXCEPTION 'H1A005 protected function alternate schema or overload'
            USING ERRCODE = '42501';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_proc wrapper
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = wrapper.pronamespace
        WHERE namespace.nspname NOT LIKE 'pg\_%' ESCAPE '\'
          AND namespace.nspname NOT IN ('information_schema', 'public')
          AND (wrapper.prosecdef OR EXISTS (
              SELECT 1
              FROM pg_catalog.aclexplode(coalesce(
                  wrapper.proacl,
                  pg_catalog.acldefault('f', wrapper.proowner))) acl
              WHERE acl.grantee = 0))
          AND wrapper.prosrc ~
              '(record_campaign_operations_production_|transition_campaign_operations_request_dispatch_production_v2|campaign_operations_production_transition_context)') THEN
        RAISE EXCEPTION 'H1A005 alternate-schema protected wrapper'
            USING ERRCODE = '42501';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_proc transition
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = transition.pronamespace
        WHERE namespace.nspname = 'public'
          AND transition.proname = ANY (ARRAY[
              'record_campaign_operations_production_enable_v1',
              'record_campaign_operations_production_disable_v1',
              'transition_campaign_operations_request_dispatch_production_v2'])
          AND (transition.proowner <> boundary_oid OR
               NOT transition.prosecdef OR transition.proleakproof OR
               transition.provolatile <> 'v' OR transition.proparallel <> 'u' OR
               transition.pronargdefaults <> 0 OR
               transition.provariadic <> 0 OR
               transition.proconfig IS DISTINCT FROM
                   ARRAY['search_path=pg_catalog, public']::text[])) THEN
        RAISE EXCEPTION 'H1A008 fixed transition catalog mismatch'
            USING ERRCODE = '55000';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_proc protected_function
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = protected_function.pronamespace
        JOIN pg_catalog.pg_language language
          ON language.oid = protected_function.prolang
        WHERE pg_catalog.format('%I.%s', namespace.nspname,
                protected_function.oid::pg_catalog.regprocedure::text) =
              ANY (allowed_function_signatures)
          AND (protected_function.proowner <> boundary_oid OR
               protected_function.prokind <> 'f' OR
               protected_function.proleakproof OR
               (pg_catalog.format('%I.%s', namespace.nspname,
                    protected_function.oid::pg_catalog.regprocedure::text) ||
                '|' || language.lanname || '|' ||
                protected_function.prosecdef::text || '|' ||
                protected_function.provolatile::text || '|' ||
                protected_function.proparallel::text || '|' ||
                protected_function.pronargdefaults::text || '|' ||
                protected_function.provariadic::text || '|' ||
                coalesce(pg_catalog.array_to_string(
                    protected_function.proconfig, ','), 'NULL')) <>
                   ALL (allowed_function_contracts))) THEN
        RAISE EXCEPTION 'H1A008 protected function catalog mismatch'
            USING ERRCODE = '55000';
    END IF;

    SELECT pg_catalog.format('%I.%s', namespace.nspname,
             protected_function.oid::pg_catalog.regprocedure::text)
      INTO offending_entry
        FROM pg_catalog.pg_proc protected_function
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = protected_function.pronamespace
        JOIN pg_catalog.pg_type return_type
          ON return_type.oid = protected_function.prorettype
        JOIN pg_catalog.pg_namespace return_namespace
          ON return_namespace.oid = return_type.typnamespace
        WHERE pg_catalog.format('%I.%s', namespace.nspname,
                protected_function.oid::pg_catalog.regprocedure::text) =
              ANY (allowed_function_signatures)
          AND NOT EXISTS (
              SELECT 1
              FROM pg_catalog.unnest(allowed_function_return_contracts)
                   expected(contract)
              WHERE pg_catalog.split_part(expected.contract, '|', 1) =
                    pg_catalog.format('%I.%s', namespace.nspname,
                      protected_function.oid::pg_catalog.regprocedure::text)
                AND pg_catalog.pg_get_function_identity_arguments(
                      protected_function.oid) = coalesce((
                    SELECT pg_catalog.split_part(identity.contract, '|', 2)
                    FROM pg_catalog.unnest(
                           allowed_function_identity_argument_contracts)
                         identity(contract)
                    WHERE pg_catalog.split_part(identity.contract, '|', 1) =
                          pg_catalog.format('%I.%s', namespace.nspname,
                            protected_function.oid::pg_catalog.regprocedure::text)), '')
                AND pg_catalog.split_part(expected.contract, '|', 2) =
                    pg_catalog.format('%I.%I', return_namespace.nspname,
                      return_type.typname)
                AND pg_catalog.split_part(expected.contract, '|', 3)::boolean =
                    protected_function.proretset
                AND pg_catalog.split_part(expected.contract, '|', 4) =
                    CASE WHEN protected_function.proallargtypes IS NULL THEN 'NULL'
                    ELSE (SELECT pg_catalog.string_agg(
                              pg_catalog.format('%I.%I', argument_namespace.nspname,
                                argument_type.typname), ',' ORDER BY argument.ordinality)
                          FROM pg_catalog.unnest(
                                 protected_function.proallargtypes)
                               WITH ORDINALITY argument(type_oid, ordinality)
                          JOIN pg_catalog.pg_type argument_type
                            ON argument_type.oid = argument.type_oid
                          JOIN pg_catalog.pg_namespace argument_namespace
                            ON argument_namespace.oid = argument_type.typnamespace)
                    END
                AND pg_catalog.split_part(expected.contract, '|', 5) =
                    coalesce(pg_catalog.array_to_string(
                      protected_function.proargmodes, ','), 'NULL')
                AND pg_catalog.split_part(expected.contract, '|', 6) =
                    CASE WHEN protected_function.proargmodes IS NULL THEN 'NULL'
                    ELSE (SELECT pg_catalog.string_agg(
                              protected_function.proargnames[argument.ordinality], ','
                              ORDER BY argument.ordinality)
                          FROM pg_catalog.unnest(
                                 protected_function.proargmodes)
                               WITH ORDINALITY argument(mode, ordinality)
                          WHERE argument.mode IN ('o', 'b', 't'))
                    END)
        ORDER BY 1
        LIMIT 1;
    IF offending_entry IS NOT NULL THEN
        RAISE EXCEPTION
            'H1A008 protected function return contract mismatch object=% stage=database-audit',
            offending_entry USING ERRCODE = '55000';
    END IF;

    IF require_migration_ledger THEN
        IF expected_migration_checksum IS NULL OR
           expected_migration_checksum !~ '^[0-9a-f]{64}$' OR
           NOT EXISTS (
              SELECT 1 FROM public.schema_migrations migration
              WHERE migration.version = '055'
                AND migration.filename =
                    '055_campaign_operations_production_admission_foundation.sql'
                AND migration.checksum = expected_migration_checksum) THEN
            RAISE EXCEPTION 'H1A009 migration checksum or ledger mismatch'
                USING ERRCODE = '55000';
        END IF;
    END IF;

    IF require_historical_bytes THEN
        IF pg_catalog.to_regclass(
              'public.phase_h1_pre055_historical_bytes') IS NULL OR
           NOT EXISTS (
              SELECT 1
              FROM public.phase_h1_pre055_historical_bytes snapshot
              JOIN public.campaign_operations_dispatch_attempt attempt
                ON attempt.dispatch_attempt_id = 7001
              JOIN public.campaign_operations_completion_event completion
                ON completion.completion_event_id = 7001
              WHERE snapshot.attempt_bytes = pg_catalog.convert_to(
                        attempt.attempt_identity_canonical, 'UTF8')
                AND snapshot.attempt_hash = attempt.attempt_identity_hash
                AND snapshot.request_bytes = pg_catalog.convert_to(
                        completion.request_evidence_canonical, 'UTF8')
                AND snapshot.request_hash = completion.request_evidence_hash
                AND snapshot.completion_bytes = pg_catalog.convert_to(
                        completion.completion_identity_canonical, 'UTF8')
                AND snapshot.completion_hash = completion.completion_identity_hash
                AND position(
                      pg_catalog.convert_to(
                        attempt.attempt_identity_canonical, 'UTF8') IN
                      snapshot.request_bytes) > 0) THEN
            RAISE EXCEPTION 'H1A010 historical canonical/hash bytes mismatch'
                USING ERRCODE = '23514';
        END IF;
    END IF;

    RETURN true;
END;
$$;
ALTER FUNCTION campaign_operations_h1_deployment_audit_v1(
    text, boolean, boolean)
    OWNER TO campaign_operations_h1_boundary_authority;
REVOKE ALL PRIVILEGES ON FUNCTION
    campaign_operations_h1_deployment_audit_v1(text, boolean, boolean)
FROM PUBLIC, pqxx,
    campaign_operations_production_enabler,
    campaign_operations_production_disabler,
    campaign_operations_production_dispatcher,
    campaign_operations_production_phase5_transactional,
    campaign_operations_production_reader,
    campaign_operations_scheduler_protocol_evidence_owner,
    campaign_operations_scheduler_protocol_evidence_reader;

-- A migration-time postcondition uses the same read-only catalog audit used
-- after upgrade and restore.  The ledger is recorded by the migration runner,
-- so it is deliberately not required inside the migration transaction.
SELECT campaign_operations_h1_deployment_audit_v1(NULL, false, false);

DO $$
DECLARE production_schema text := current_schema();
BEGIN
    EXECUTE format(
        'GRANT USAGE ON SCHEMA %I TO '
        'campaign_operations_production_reader, '
        'campaign_operations_scheduler_protocol_evidence_reader',
        production_schema);
END $$;

DO $$
DECLARE role_name text;
BEGIN
    FOREACH role_name IN ARRAY ARRAY[
        'campaign_operations_production_enabler',
        'campaign_operations_production_disabler',
        'campaign_operations_production_dispatcher',
        'campaign_operations_production_phase5_transactional',
        'campaign_operations_production_reader',
        'campaign_operations_scheduler_protocol_evidence_owner',
        'campaign_operations_scheduler_protocol_evidence_reader']
    LOOP
        IF pg_has_role('pqxx', role_name, 'MEMBER') OR
           (SELECT rolcanlogin OR rolsuper OR rolcreatedb OR rolcreaterole OR
                   rolreplication OR rolbypassrls
            FROM pg_roles WHERE rolname = role_name) THEN
        RAISE EXCEPTION 'H1A002 migration 055 role hardening mismatch for %',
                role_name USING ERRCODE = '42501';
        END IF;
    END LOOP;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
        WHERE rolname = 'campaign_operations_h1_boundary_authority'
          AND NOT rolcanlogin AND rolsuper AND NOT rolcreatedb
          AND NOT rolcreaterole AND NOT rolreplication) OR EXISTS (
        SELECT 1 FROM pg_catalog.pg_auth_members membership
        JOIN pg_catalog.pg_roles granted ON granted.oid = membership.roleid
        JOIN pg_catalog.pg_roles member_role ON member_role.oid = membership.member
        WHERE granted.rolname IN (
            'campaign_operations_h1_boundary_authority',
            'campaign_operations_production_enabler',
            'campaign_operations_production_disabler',
            'campaign_operations_production_dispatcher',
            'campaign_operations_production_phase5_transactional',
            'campaign_operations_production_reader',
            'campaign_operations_scheduler_protocol_evidence_owner',
            'campaign_operations_scheduler_protocol_evidence_reader')
           OR member_role.rolname IN (
            'campaign_operations_h1_boundary_authority',
            'campaign_operations_production_enabler',
            'campaign_operations_production_disabler',
            'campaign_operations_production_dispatcher',
            'campaign_operations_production_phase5_transactional',
            'campaign_operations_production_reader',
            'campaign_operations_scheduler_protocol_evidence_owner',
            'campaign_operations_scheduler_protocol_evidence_reader')) THEN
        RAISE EXCEPTION 'H1A003 migration 055 sealed role graph mismatch'
            USING ERRCODE = '42501';
    END IF;
    IF EXISTS (SELECT 1 FROM
            campaign_operations_production_transition_context)
       OR EXISTS (SELECT 1 FROM
            campaign_operations_request_production_admission)
       OR EXISTS (SELECT 1 FROM
            campaign_operations_production_enablement_event)
       OR EXISTS (SELECT 1 FROM campaign_operations_operational_request
                  WHERE production_dispatch_enabled)
       OR EXISTS (SELECT 1 FROM campaign_operations_dispatch_attempt
                  WHERE attempt_contract_version = 2) THEN
        RAISE EXCEPTION 'H1A010 migration 055 must not backfill production evidence'
            USING ERRCODE = '23514';
    END IF;
END $$;
