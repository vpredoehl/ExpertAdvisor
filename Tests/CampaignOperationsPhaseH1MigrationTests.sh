#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -eq 7 && "$2" == attest &&
      "$1" =~ ^(emit_runtime_result|emit_pre_enablement_runtime|emit_invariant_runtime)$ ]]; then
  exec python3 "$repo_root/Scripts/CampaignOperationsH1EvidencePayloadGenerator.py" \
    generate "$3" "$4" "$5" "$6" "$7"
fi
"$repo_root/Scripts/CampaignOperationsH1ManifestValidator.sh"
database="expertadvisor_campaign_operations_phase3_test_h1_${$}"
failure_database="expertadvisor_campaign_operations_phase3_test_h1_failure_${$}"
lock_database="expertadvisor_campaign_operations_phase3_test_h1_lock_${$}"
broad_database="expertadvisor_campaign_operations_phase5_regression_test_${$}"
cluster_root="$(mktemp -d /tmp/ea-h1-pg.XXXXXX)"
cluster_data="$cluster_root/data"
cluster_socket="$cluster_root/s"
mkdir -p "$cluster_socket"
runtime_artifacts="$cluster_root/runtime-artifacts"
mkdir -p "$runtime_artifacts"
runtime_run_id="h1-$(date -u +%Y%m%dT%H%M%SZ)-${$}"
runtime_results="$cluster_root/h1-runtime-results.tsv"
printf '%s\n' "$runtime_run_id" > "$cluster_root/run-id"
printf '%s\n' 'result_format_version	run_id	fixture_id	requirement_id	test_source	timestamp	actual_status	sqlstate	diagnostic	object_identity	stage	operation_id	artifact_path	artifact_digest	lock_outcome	cycle_detected	cleanup_result	generator_id	generator_version	generator_implementation	generator_entry_point	emitted_runtime_record_id	output_artifact_id	record_digest' \
  > "$runtime_results"

emit_runtime_result() {
  local requirement="$1" fixture="$2" test_source="$3" actual_status="$4"
  local sqlstate="$5" diagnostic="$6" object_identity="$7" stage="$8"
  local operation_id="$9" artifact_label="${10}" evidence_source="${11}"
  local lock_outcome="${12:-not-applicable}" cycle_detected="${13:-false}"
  local generator_id="${14:-GEN-TRACE}" generator_implementation generator_entry_point
  case "$generator_id" in
    GEN-TRACE) generator_implementation=Tests/CampaignOperationsPhaseH1MigrationTests.sh; generator_entry_point=emit_runtime_result ;;
    GEN-LOCK) generator_implementation=Scripts/CampaignOperationsH1LockEvidence.py; generator_entry_point=generate ;;
    GEN-ACL-MANIFEST) generator_implementation=Tests/CampaignOperationsPhaseH1MigrationTests.sh; generator_entry_point=emit_acl_catalog_runtime ;;
    *) echo "unsupported runtime generator ${generator_id}" >&2; exit 1 ;;
  esac
  [[ -s "$evidence_source" ]] || {
    echo "runtime evidence missing for ${requirement}/${fixture}: ${evidence_source}" >&2
    exit 1
  }
  local artifact_path="runtime-artifacts/records/$fixture/$artifact_label"
  local output_artifact_id="ART-RECORD-${fixture}"
  [[ "$generator_id" == GEN-TRACE ]] || output_artifact_id="ART-RUNTIME-OBS-${fixture}"
  mkdir -p "$(dirname "$cluster_root/$artifact_path")"
  if [[ ! -e "$cluster_root/$artifact_path" ]]; then
    cp "$evidence_source" "$cluster_root/$artifact_path"
  fi
  local artifact_digest
  artifact_digest="$(shasum -a 256 "$cluster_root/$artifact_path" | awk '{print $1}')"
  local record_line record_digest
  record_line="$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
    h1-runtime-result-v2 "$runtime_run_id" "$fixture" "$requirement" \
    "$test_source" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$actual_status" \
    "$sqlstate" "$diagnostic" "$object_identity" "$stage" "$operation_id" \
    "$artifact_path" "$artifact_digest" "$lock_outcome" "$cycle_detected" PASS \
    "$generator_id" h1-generator-registry-v2 "$generator_implementation" \
    "$generator_entry_point" "RT-${fixture}" "$output_artifact_id")"
  record_digest="$(printf '%s' "$record_line" | shasum -a 256 | awk '{print $1}')"
  printf '%s\t%s\n' "$record_line" "$record_digest" >> "$runtime_results"
}
cleanup() {
    if [[ -n "${H1_EVIDENCE_CAPTURE_DIR:-}" && -n "${evidence_root:-}" && -d "$evidence_root" ]]; then
        mkdir -p "$H1_EVIDENCE_CAPTURE_DIR"
        cp -R "$evidence_root"/. "$H1_EVIDENCE_CAPTURE_DIR/" 2>/dev/null || true
    fi
    if [[ -n "${active_distinct_data:-}" && -f "$active_distinct_data/postmaster.pid" ]]; then
        pg_ctl -D "$active_distinct_data" -m immediate stop >/dev/null 2>&1 || true
    fi
    if [[ -n "${restore_data:-}" && -f "$restore_data/postmaster.pid" ]]; then
        pg_ctl -D "$restore_data" -m immediate stop >/dev/null 2>&1 || true
    fi
    if [[ -n "${H1_PRESERVE_CLUSTER_ROOT:-}" ]]; then
        printf '%s\n' "$cluster_root" > "$H1_PRESERVE_CLUSTER_ROOT"
        return
    fi
    if [[ -f "$cluster_data/postmaster.pid" ]]; then
        pg_ctl -D "$cluster_data" -m immediate stop >/dev/null 2>&1 || true
    fi
    rm -rf -- "$cluster_root"
}
trap cleanup EXIT

initdb -D "$cluster_data" -U campaign_manager_login \
  --auth=trust --no-instructions >/dev/null
pg_ctl -D "$cluster_data" -o "-F -h '' -k $cluster_socket" \
    -w start >/dev/null
target=(-h "$cluster_socket" -p 5432 -U campaign_manager_login)
createdb "${target[@]}" "$database"

# pg_dump excludes cluster-global roles.  Recreate only the inert prerequisite
# roles before restoring the schema's authoritative campaign_operations_owner
# assignments; migrations 053-055 create and harden additional capabilities.
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<'SQL'
CREATE ROLE pqxx NOLOGIN;
CREATE ROLE vjp NOLOGIN;
CREATE ROLE campaign_operations_owner NOLOGIN;
CREATE ROLE campaign_operations_campaign_creator NOLOGIN;
CREATE ROLE campaign_operations_authorizer NOLOGIN;
CREATE ROLE campaign_operations_auditor NOLOGIN;
CREATE ROLE campaign_operations_reader NOLOGIN;
CREATE ROLE campaign_operations_budget_administrator NOLOGIN;
CREATE ROLE campaign_operations_request_acceptor NOLOGIN;
CREATE ROLE campaign_operations_dispatcher NOLOGIN;
CREATE ROLE campaign_operations_phase5_transactional NOLOGIN;
SQL
# Seed exclusively from the repository-owned schema-049 backup.  The H1
# assurance suite must never inspect or clone the live production database.
pg_restore --schema-only --no-privileges \
    --file=- "$repo_root/Database/backups/LSTM_latest.dump" |
    psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database"

for prerequisite_migration in \
    050_experiment_current_operation_canonicalization.sql \
    051_scheduler_ownership_and_worker_attempts.sql \
    052_scheduler_protocol_and_exact_attempt_hardening.sql
do
    psql "${target[@]}" -q -1 -v ON_ERROR_STOP=1 "$database" \
      -f "$repo_root/Database/migrations/$prerequisite_migration"
done

psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<'SQL'
SET session_replication_role = replica;
INSERT INTO experiment_scheduler_protocol(
  singleton,required_generation,cutover_state,cutover_completed_at,
  cutover_completed_by,cutover_executable_path,cutover_process_evidence)
VALUES(true,52,'complete','2026-07-31T12:34:56.123456Z',
  'scheduler.owner','/Applications/LSTM_Release','cutover-process-evidence')
ON CONFLICT (singleton) DO UPDATE SET
  required_generation = EXCLUDED.required_generation,
  cutover_state = EXCLUDED.cutover_state,
  cutover_completed_at = EXCLUDED.cutover_completed_at,
  cutover_completed_by = EXCLUDED.cutover_completed_by,
  cutover_executable_path = EXCLUDED.cutover_executable_path,
  cutover_process_evidence = EXCLUDED.cutover_process_evidence;
INSERT INTO experiment_recommendation_campaign_materialization(
  recommendation_campaign_materialization_id,
  recommendation_campaign_approval_id,materialization_contract_version,
  approval_identity_canonical,approval_identity_hash,
  recommendation_ranking_snapshot_id,ranking_snapshot_identity_canonical,
  ranking_snapshot_identity_hash,planning_policy_canonical,
  planning_policy_hash,planning_scope_canonical,
  campaign_plan_identity_canonical,campaign_plan_identity_hash,
  campaign_review_identity_canonical,campaign_review_identity_hash,
  approval_decision,approval_reviewer_identity,approval_reason_text,
  materialized_by,materialization_reason_text,selected_member_count,
  initially_created_proposal_count,initially_reused_proposal_count,
  materialization_identity_canonical,materialization_identity_hash)
VALUES(7,7,1,'approval-canonical-v1','fnv1a64:a111111111111111',
  7,'ranking-snapshot-canonical-v1','fnv1a64:a222222222222222',
  'planning-policy-canonical-v1','fnv1a64:a333333333333333',
  'planning-scope-canonical-v1','campaign-plan-canonical-v1',
  'fnv1a64:a444444444444444','campaign-review-canonical-v1',
  'fnv1a64:a555555555555555','approved','fixture.reviewer',
  'fixture approval','fixture.materializer','fixture materialization',
  1,1,0,'materialization-canonical-v1','fnv1a64:1111111111111111');
INSERT INTO campaign_operations_campaign(
  operational_campaign_id,recommendation_campaign_materialization_id,
  materialization_contract_version,materialization_identity_canonical,
  materialization_identity_hash,materialization_member_count,origin_kind,
  action_kind,action_contract_version,scope_kind,scope_contract_version,
  campaign_contract_version,campaign_identity_canonical,
  campaign_identity_hash)
VALUES(7,7,1,'materialization-canonical-v1',
  'fnv1a64:1111111111111111',1,'phase4d_materialization_v1',
  'dispatch_full_materialization',1,'complete_materialization',1,1,
  'campaign-canonical-v1','fnv1a64:2222222222222222');
INSERT INTO campaign_operations_authorization_event(
  authorization_event_id,operational_campaign_id,campaign_identity_canonical,
  previous_event_id,previous_event_identity_canonical,
  previous_event_identity_hash,chain_version,event_kind,action_kind,
  action_contract_version,scope_kind,scope_contract_version,
  prerequisite_policy,governance_provenance_event_id,
  provenance_identity_canonical,provenance_identity_hash,authorization_role,
  actor_identity,reason,not_before,expires_at,authorization_contract_version,
  authorization_identity_canonical,authorization_identity_hash)
VALUES(7,7,'campaign-canonical-v1',NULL,NULL,NULL,1,'granted',
  'dispatch_full_materialization',1,'complete_materialization',1,
  'phase4d_materialization_only_v1',NULL,NULL,NULL,
  'campaign_operations_authorizer','fixture.actor','fixture authorization',
  '2026-01-01T00:00:00Z',NULL,1,'authorization-canonical-v1',
  'fnv1a64:4444444444444444');
INSERT INTO campaign_operations_budget_ledger_entry(
  budget_ledger_entry_id,operational_campaign_id,campaign_identity_canonical,
  previous_entry_id,previous_entry_identity_canonical,
  previous_entry_identity_hash,ledger_version,entry_kind,ledger_status,
  budget_unit,delta,prior_total,resulting_total,administrator_identity,reason,
  budget_contract_version,budget_identity_canonical,budget_identity_hash)
VALUES(7,7,'campaign-canonical-v1',NULL,NULL,NULL,1,'grant','active',
  'materialized_member_dispatch',1,0,1,'fixture.actor','fixture budget',1,
  'budget-canonical-v1','fnv1a64:8888888888888888');
INSERT INTO campaign_operations_reservation(
  reservation_id,operational_campaign_id,campaign_identity_canonical,
  logical_operation_contract_version,logical_operation_canonical,
  logical_operation_hash,authorization_event_id,
  authorization_identity_canonical,authorization_identity_hash,
  budget_ledger_entry_id,budget_ledger_version,budget_identity_canonical,
  budget_identity_hash,action_kind,action_contract_version,
  recommendation_campaign_materialization_id,materialization_contract_version,
  materialization_identity_canonical,materialization_identity_hash,scope_kind,
  scope_contract_version,materialization_member_count,amount,budget_unit,
  expires_at,reservation_contract_version,reservation_identity_canonical,
  reservation_identity_hash,reservation_state,state_version)
VALUES(7,7,'campaign-canonical-v1',1,'operation-canonical-v1',
  'fnv1a64:3333333333333333',7,'authorization-canonical-v1',
  'fnv1a64:4444444444444444',7,1,'budget-canonical-v1',
  'fnv1a64:8888888888888888','dispatch_full_materialization',1,7,1,
  'materialization-canonical-v1','fnv1a64:1111111111111111',
  'complete_materialization',1,1,1,'materialized_member_dispatch',NULL,1,
  'reservation-canonical-v1','fnv1a64:5555555555555555','held',1);
INSERT INTO campaign_operations_operational_request(
  operational_request_id,operational_campaign_id,
  campaign_identity_canonical,logical_operation_contract_version,
  logical_operation_canonical,logical_operation_hash,authorization_event_id,
  authorization_identity_canonical,authorization_identity_hash,reservation_id,
  reservation_identity_canonical,reservation_identity_hash,action_kind,
  action_contract_version,recommendation_campaign_materialization_id,
  materialization_contract_version,materialization_identity_canonical,
  materialization_identity_hash,ordered_scope_digest,
  materialization_member_count,accepting_actor_identity,reason,
  prerequisite_policy,request_contract_version,request_identity_canonical,
  request_identity_hash,request_state,state_version,
  production_dispatch_enabled)
VALUES(7,7,'campaign-canonical-v1',1,'operation-canonical-v1',
  'fnv1a64:3333333333333333',7,'authorization-canonical-v1',
  'fnv1a64:4444444444444444',7,'reservation-canonical-v1',
  'fnv1a64:5555555555555555','dispatch_full_materialization',1,7,1,
  'materialization-canonical-v1','fnv1a64:1111111111111111',
  'fnv1a64:6666666666666666',1,'fixture.actor','fixture reason',
  'phase4d_materialization_only_v1',1,'request-canonical-v1',
  'fnv1a64:7777777777777777','ready',3,false);
SET session_replication_role = origin;
SQL

psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" \
  -f "$repo_root/Database/migrations/053_campaign_operations_controls_cancellation_reconciliation.sql"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" \
  -f "$repo_root/Database/migrations/054_campaign_operations_completion_and_audit.sql"

# Authoritative historical evidence is created while the database is still at
# schema 054.  Exact UTF-8 bytes and hashes are captured before 055 exists.
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<'SQL'
SET session_replication_role = replica;
INSERT INTO campaign_operations_dispatch_attempt(
  dispatch_attempt_id,operational_request_id,request_identity_canonical,
  attempt_ordinal,expected_request_version,resulting_request_version,
  lease_token_digest,lease_expires_at,dispatcher_identity,
  attempt_contract_version,attempt_identity_canonical,attempt_identity_hash)
VALUES(7001,7,'request-canonical-v1',1,3,4,
  'fnv1a64:0123456789abcdef','2026-07-31T12:39:56.123456Z',
  'isolated.test',1,
  'campaign_operations_dispatch_attempt_v1;request_id=7;request_identity_canonical=20:request-canonical-v1;attempt_ordinal=1;expected_request_version=3;resulting_request_version=4;lease_token_digest=24:fnv1a64:0123456789abcdef;lease_expires_at=27:2026-07-31T12:39:56.123456Z;dispatcher_identity=13:isolated.test',
  campaign_operations_tagged_fnv1a64(
  'campaign_operations_dispatch_attempt_v1;request_id=7;request_identity_canonical=20:request-canonical-v1;attempt_ordinal=1;expected_request_version=3;resulting_request_version=4;lease_token_digest=24:fnv1a64:0123456789abcdef;lease_expires_at=27:2026-07-31T12:39:56.123456Z;dispatcher_identity=13:isolated.test'));
DO $$
DECLARE candidate campaign_operations_completion_event%ROWTYPE;
BEGIN
  candidate.completion_event_id := 7001;
  candidate.operational_campaign_id := 7;
  candidate.campaign_identity_canonical := 'campaign-canonical-v1';
  candidate.operation_key := 'historical-completion-001';
  candidate.administrative_terminal_state := 'terminal_failed';
  candidate.completion_classification := 'operational_request_failed';
  candidate.budget_ledger_entry_id := 7;
  candidate.budget_ledger_version := 1;
  candidate.budget_resulting_total := 1;
  candidate.budget_ever_reserved := 1;
  candidate.budget_committed := 0;
  candidate.budget_released_or_expired := 1;
  candidate.budget_held := 0;
  candidate.budget_unallocated := 1;
  candidate.scope_member_count := 1;
  candidate.completed_member_count := 0;
  candidate.failed_member_count := 1;
  candidate.cancelled_or_never_dispatched_member_count := 0;
  candidate.reservation_count := 1;
  candidate.request_count := 1;
  candidate.binding_count := 0;
  candidate.control_owner_count := 0;
  candidate.cancellation_request_count := 0;
  candidate.cancellation_settlement_count := 0;
  candidate.unresolved_blocking_observation_count := 0;
  candidate.authorization_evidence_canonical :=
    campaign_operations_completion_evidence_text(7,'authorization');
  candidate.budget_evidence_canonical :=
    campaign_operations_completion_evidence_text(7,'budget');
  candidate.reservation_evidence_canonical :=
    campaign_operations_completion_evidence_text(7,'reservation');
  candidate.request_evidence_canonical :=
    campaign_operations_completion_evidence_text(7,'request');
  candidate.binding_evidence_canonical :=
    campaign_operations_completion_evidence_text(7,'binding');
  candidate.lifecycle_evidence_canonical :=
    campaign_operations_completion_evidence_text(7,'lifecycle');
  candidate.cancellation_evidence_canonical :=
    campaign_operations_completion_evidence_text(7,'cancellation');
  candidate.reconciliation_evidence_canonical :=
    campaign_operations_completion_evidence_text(7,'reconciliation');
  candidate.authorization_evidence_hash :=
    campaign_operations_tagged_fnv1a64(candidate.authorization_evidence_canonical);
  candidate.budget_evidence_hash :=
    campaign_operations_tagged_fnv1a64(candidate.budget_evidence_canonical);
  candidate.reservation_evidence_hash :=
    campaign_operations_tagged_fnv1a64(candidate.reservation_evidence_canonical);
  candidate.request_evidence_hash :=
    campaign_operations_tagged_fnv1a64(candidate.request_evidence_canonical);
  candidate.binding_evidence_hash :=
    campaign_operations_tagged_fnv1a64(candidate.binding_evidence_canonical);
  candidate.lifecycle_evidence_hash :=
    campaign_operations_tagged_fnv1a64(candidate.lifecycle_evidence_canonical);
  candidate.cancellation_evidence_hash :=
    campaign_operations_tagged_fnv1a64(candidate.cancellation_evidence_canonical);
  candidate.reconciliation_evidence_hash :=
    campaign_operations_tagged_fnv1a64(candidate.reconciliation_evidence_canonical);
  candidate.actor_identity := 'fixture.actor';
  candidate.capability := 'campaign_operations_completion_writer';
  candidate.reason := 'Historical schema-054 fixture.';
  candidate.completion_contract_version := 1;
  candidate.completion_identity_canonical :=
    'campaign_operations_completion_v1' ||
    ';campaign=' || octet_length(candidate.campaign_identity_canonical) || ':' || candidate.campaign_identity_canonical ||
    ';operation_key=' || octet_length(candidate.operation_key) || ':' || candidate.operation_key ||
    ';terminal_state=' || candidate.administrative_terminal_state ||
    ';classification=' || candidate.completion_classification ||
    ';budget_ledger_entry_id=' || candidate.budget_ledger_entry_id ||
    ';budget_ledger_version=' || candidate.budget_ledger_version ||
    ';budget_resulting_total=' || candidate.budget_resulting_total ||
    ';budget_ever_reserved=' || candidate.budget_ever_reserved ||
    ';budget_committed=' || candidate.budget_committed ||
    ';budget_released_or_expired=' || candidate.budget_released_or_expired ||
    ';budget_held=' || candidate.budget_held ||
    ';budget_unallocated=' || candidate.budget_unallocated ||
    ';scope_member_count=' || candidate.scope_member_count ||
    ';completed_member_count=' || candidate.completed_member_count ||
    ';failed_member_count=' || candidate.failed_member_count ||
    ';cancelled_or_never_dispatched_member_count=' || candidate.cancelled_or_never_dispatched_member_count ||
    ';reservation_count=' || candidate.reservation_count ||
    ';request_count=' || candidate.request_count ||
    ';binding_count=' || candidate.binding_count ||
    ';control_owner_count=' || candidate.control_owner_count ||
    ';cancellation_request_count=' || candidate.cancellation_request_count ||
    ';cancellation_settlement_count=' || candidate.cancellation_settlement_count ||
    ';unresolved_blocking_observation_count=' || candidate.unresolved_blocking_observation_count ||
    ';authorization_evidence=' || octet_length(candidate.authorization_evidence_canonical) || ':' || candidate.authorization_evidence_canonical ||
    ';budget_evidence=' || octet_length(candidate.budget_evidence_canonical) || ':' || candidate.budget_evidence_canonical ||
    ';reservation_evidence=' || octet_length(candidate.reservation_evidence_canonical) || ':' || candidate.reservation_evidence_canonical ||
    ';request_evidence=' || octet_length(candidate.request_evidence_canonical) || ':' || candidate.request_evidence_canonical ||
    ';binding_evidence=' || octet_length(candidate.binding_evidence_canonical) || ':' || candidate.binding_evidence_canonical ||
    ';lifecycle_evidence=' || octet_length(candidate.lifecycle_evidence_canonical) || ':' || candidate.lifecycle_evidence_canonical ||
    ';cancellation_evidence=' || octet_length(candidate.cancellation_evidence_canonical) || ':' || candidate.cancellation_evidence_canonical ||
    ';reconciliation_evidence=' || octet_length(candidate.reconciliation_evidence_canonical) || ':' || candidate.reconciliation_evidence_canonical ||
    ';actor=' || octet_length(candidate.actor_identity) || ':' || candidate.actor_identity ||
    ';capability=campaign_operations_completion_writer' ||
    ';reason=' || octet_length(candidate.reason) || ':' || candidate.reason;
  candidate.completion_identity_hash :=
    campaign_operations_tagged_fnv1a64(candidate.completion_identity_canonical);
  candidate.recorded_at := transaction_timestamp();
  IF NOT campaign_operations_completion_identity_valid(candidate) OR
     position((SELECT attempt_identity_canonical FROM
       campaign_operations_dispatch_attempt WHERE dispatch_attempt_id=7001)
       IN candidate.request_evidence_canonical) = 0 THEN
    RAISE EXCEPTION 'schema-054 historical fixture is not authoritative';
  END IF;
  INSERT INTO campaign_operations_completion_event SELECT candidate.*;
  INSERT INTO campaign_operations_completion_audit_reference_event(
    completion_audit_reference_event_id,operational_campaign_id,
    completion_event_id,actor_identity,capability,reason,outcome,
    replay_disposition,diagnostic_code)
  VALUES(7001,7,7001,'fixture.actor',
    'campaign_operations_completion_writer','Historical schema-054 fixture.',
    'recorded','recorded','all_completion_prerequisites_proven');
END $$;
CREATE TABLE phase_h1_pre055_historical_bytes AS
SELECT convert_to(attempt.attempt_identity_canonical,'UTF8') AS attempt_bytes,
       attempt.attempt_identity_hash AS attempt_hash,
       convert_to(completion.request_evidence_canonical,'UTF8') AS request_bytes,
       completion.request_evidence_hash AS request_hash,
       convert_to(completion.completion_identity_canonical,'UTF8') AS completion_bytes,
       completion.completion_identity_hash AS completion_hash
FROM campaign_operations_dispatch_attempt attempt
CROSS JOIN campaign_operations_completion_event completion
WHERE attempt.dispatch_attempt_id=7001 AND completion.completion_event_id=7001;
SET session_replication_role = origin;
SQL

# Prove a failing 054 -> 055 upgrade is fully transactional.  A deliberately
# incompatible private-context relation is introduced only in the disposable
# failure database so migration 055 fails after beginning its DDL.
createdb "${target[@]}" -T "$database" "$failure_database"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$failure_database" <<'SQL'
CREATE TABLE campaign_operations_production_transition_context(
  incompatible_column integer);
SQL
if psql "${target[@]}" -q -1 -v ON_ERROR_STOP=1 "$failure_database" \
    -f "$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql" \
    >/dev/null 2>&1; then
  echo "migration 055 failure fixture unexpectedly succeeded" >&2
  exit 1
fi
if [[ "$(psql "${target[@]}" -At "$failure_database" -c \
  "SELECT to_regclass('campaign_operations_production_transition_context') IS NOT NULL
          AND to_regclass('campaign_operations_production_enablement_event') IS NULL
          AND NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname=
            'campaign_operations_h1_boundary_authority');")" != "t" ]]; then
  echo "migration 055 failure did not roll back completely" >&2
  exit 1
fi
dropdb "${target[@]}" "$failure_database"

# CREATE OR REPLACE must never normalize an unexpected exact protected entry
# point owned by an ordinary role.
protected_function_database="expertadvisor_h1_protected_function_${$}"
createdb "${target[@]}" -T "$database" "$protected_function_database"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$protected_function_database" <<'SQL'
CREATE FUNCTION campaign_operations_h1_deployment_audit_v1(
  text,boolean,boolean) RETURNS boolean LANGUAGE sql AS $$SELECT true$$;
SQL
if psql "${target[@]}" -q -1 -v ON_ERROR_STOP=1 -v VERBOSITY=verbose \
    "$protected_function_database" -f \
    "$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql" \
    >"$cluster_root/055-protected-function-owner.log" 2>&1; then
  echo "migration 055 replaced an unexpected protected function" >&2
  exit 1
fi
if ! rg -q '42501: H1A004 incompatible pre-existing protected function owner' \
    "$cluster_root/055-protected-function-owner.log"; then
  tail -n 30 "$cluster_root/055-protected-function-owner.log" >&2
  echo "protected function owner fixture reached the wrong failure" >&2
  exit 1
fi
dropdb "${target[@]}" "$protected_function_database"

# A default on any protected signature is rejected before migration DDL.
protected_default_database="expertadvisor_h1_protected_default_${$}"
createdb "${target[@]}" -T "$database" "$protected_default_database"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$protected_default_database" \
  -c "ALTER FUNCTION campaign_operations_tagged_fnv1a64(text)
      RENAME TO campaign_operations_tagged_fnv1a64_without_default"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$protected_default_database" <<'SQL'
CREATE FUNCTION campaign_operations_tagged_fnv1a64(candidate text DEFAULT '')
RETURNS text LANGUAGE sql IMMUTABLE AS $$SELECT 'fnv1a64:0000000000000000'$$;
SQL
if psql "${target[@]}" -q -1 -v ON_ERROR_STOP=1 -v VERBOSITY=verbose \
    "$protected_default_database" -f \
    "$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql" \
    >"$cluster_root/055-protected-function-default.log" 2>&1; then
  echo "migration 055 normalized a protected default argument" >&2
  exit 1
fi
if ! rg -q '42501: H1A005 protected function default or variadic mismatch' \
    "$cluster_root/055-protected-function-default.log"; then
  tail -n 30 "$cluster_root/055-protected-function-default.log" >&2
  echo "protected function default fixture reached the wrong failure" >&2
  exit 1
fi
dropdb "${target[@]}" "$protected_default_database"

# An unsafe pre-existing direct membership must fail rather than be silently
# retained or rewritten. Roles are cluster-global, so remove this disposable
# probe before the supported install.
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<'SQL'
CREATE ROLE campaign_operations_production_reader NOLOGIN;
CREATE ROLE campaign_operations_h1_unsafe_login LOGIN;
GRANT campaign_operations_production_reader
  TO campaign_operations_h1_unsafe_login;
SQL
if psql "${target[@]}" -q -1 -v ON_ERROR_STOP=1 "$database" \
    -f "$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql" \
    >"$cluster_root/055-unsafe-role.log" 2>&1; then
  echo "migration 055 retained unsafe role membership" >&2
  exit 1
fi
if ! rg -q 'H1A003 prohibited H1 role-graph edge' \
    "$cluster_root/055-unsafe-role.log"; then
  tail -n 50 "$cluster_root/055-unsafe-role.log" >&2
  echo "migration 055 unsafe role test reached the wrong failure" >&2
  exit 1
fi
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<'SQL'
REVOKE campaign_operations_production_reader
  FROM campaign_operations_h1_unsafe_login;
DROP ROLE campaign_operations_h1_unsafe_login;
DROP ROLE campaign_operations_production_reader;
SQL

# Every boundary-role attribute is a separate fail-closed upgrade fixture.  The
# exact H1A002 branch and SQLSTATE must be visible; a syntax/fixture failure is
# not accepted as a negative-test pass.
run_boundary_attribute_fixture() {
  local label="$1"
  local mutation_sql="$2"
  local fixture_log="$cluster_root/055-role-attribute-${label}.log"
  psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
    "CREATE ROLE campaign_operations_h1_boundary_authority
       NOLOGIN SUPERUSER INHERIT NOCREATEDB NOCREATEROLE NOREPLICATION
       NOBYPASSRLS CONNECTION LIMIT -1 PASSWORD NULL;
     ${mutation_sql}"
  if psql "${target[@]}" -q -1 -v ON_ERROR_STOP=1 -v VERBOSITY=verbose \
      "$database" -f \
      "$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql" \
      >"$fixture_log" 2>&1; then
    echo "boundary attribute fixture ${label} unexpectedly succeeded" >&2
    exit 1
  fi
  if ! rg -q '42501: H1A002 role identity mismatch: campaign_operations_h1_boundary_authority' \
      "$fixture_log"; then
    tail -n 30 "$fixture_log" >&2
    echo "boundary attribute fixture ${label} reached the wrong failure" >&2
    exit 1
  fi
  local requirement fixture
  case "$label" in
    login) requirement=H1-ROLE-LOGIN; fixture=H1ROLE001 ;;
    superuser) requirement=H1-ROLE-SUPERUSER; fixture=H1ROLE002 ;;
    inherit) requirement=H1-ROLE-INHERIT; fixture=H1ROLE003 ;;
    createdb) requirement=H1-ROLE-CREATEDB; fixture=H1ROLE004 ;;
    createrole) requirement=H1-ROLE-CREATEROLE; fixture=H1ROLE005 ;;
    replication) requirement=H1-ROLE-REPLICATION; fixture=H1ROLE006 ;;
    bypassrls) requirement=H1-ROLE-BYPASSRLS; fixture=H1ROLE007 ;;
    connlimit) requirement=H1-ROLE-CONNLIMIT; fixture=H1ROLE008 ;;
    password) requirement=H1-ROLE-PASSWORD; fixture=H1ROLE009 ;;
    validity) requirement=H1-ROLE-VALIDITY; fixture=H1ROLE010 ;;
    config) requirement=H1-ROLE-CONFIG; fixture=H1ROLE011 ;;
    *) echo "unknown boundary attribute runtime fixture: $label" >&2; exit 1 ;;
  esac
  emit_runtime_result "$requirement" "$fixture" migration055-role-preflight \
    EXPECTED_FAILURE 42501 H1A002 campaign_operations_h1_boundary_authority \
    preflight "$fixture" "055-role-attribute-${label}.log" "$fixture_log"
  psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
    "DROP ROLE campaign_operations_h1_boundary_authority;"
}

run_boundary_attribute_fixture login \
  'ALTER ROLE campaign_operations_h1_boundary_authority LOGIN'
run_boundary_attribute_fixture superuser \
  'ALTER ROLE campaign_operations_h1_boundary_authority NOSUPERUSER'
run_boundary_attribute_fixture inherit \
  'ALTER ROLE campaign_operations_h1_boundary_authority NOINHERIT'
run_boundary_attribute_fixture createdb \
  'ALTER ROLE campaign_operations_h1_boundary_authority CREATEDB'
run_boundary_attribute_fixture createrole \
  'ALTER ROLE campaign_operations_h1_boundary_authority CREATEROLE'
run_boundary_attribute_fixture replication \
  'ALTER ROLE campaign_operations_h1_boundary_authority REPLICATION'
run_boundary_attribute_fixture bypassrls \
  'ALTER ROLE campaign_operations_h1_boundary_authority BYPASSRLS'
run_boundary_attribute_fixture connlimit \
  'ALTER ROLE campaign_operations_h1_boundary_authority CONNECTION LIMIT 7'
run_boundary_attribute_fixture password \
  "ALTER ROLE campaign_operations_h1_boundary_authority PASSWORD 'fixture-only'"
run_boundary_attribute_fixture validity \
  "ALTER ROLE campaign_operations_h1_boundary_authority VALID UNTIL '2035-01-01'"
run_boundary_attribute_fixture config \
  "ALTER ROLE campaign_operations_h1_boundary_authority SET search_path TO public"

run_boundary_graph_fixture() {
  local label="$1"
  local graph_sql="$2"
  local cleanup_sql="$3"
  local fixture_log="$cluster_root/055-role-graph-${label}.log"
  psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
    "CREATE ROLE campaign_operations_h1_boundary_authority
       NOLOGIN SUPERUSER INHERIT NOCREATEDB NOCREATEROLE NOREPLICATION
       NOBYPASSRLS CONNECTION LIMIT -1 PASSWORD NULL;
     ${graph_sql}"
  if psql "${target[@]}" -q -1 -v ON_ERROR_STOP=1 -v VERBOSITY=verbose \
      "$database" -f \
      "$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql" \
      >"$fixture_log" 2>&1; then
    echo "role-graph fixture ${label} unexpectedly succeeded" >&2
    exit 1
  fi
  if ! rg -q '42501: H1A003 prohibited H1 role-graph edge' "$fixture_log"; then
    tail -n 30 "$fixture_log" >&2
    echo "role-graph fixture ${label} reached the wrong failure" >&2
    exit 1
  fi
  local requirement="" fixture=""
  case "$label" in
    direct) requirement=H1-GRAPH-DIRECT; fixture=H1GRAPH001 ;;
    two_level) requirement=H1-GRAPH-TWO; fixture=H1GRAPH002 ;;
    three_level) requirement=H1-GRAPH-THREE; fixture=H1GRAPH003 ;;
    boundary_inherits) requirement=H1-GRAPH-OUTWARD; fixture=H1GRAPH005 ;;
  esac
  if [[ -n "$requirement" ]]; then
    emit_runtime_result "$requirement" "$fixture" migration055-graph-preflight \
      EXPECTED_FAILURE 42501 H1A003 \
      campaign_operations_h1_boundary_authority preflight "$fixture" \
      "055-role-graph-${label}.log" "$fixture_log"
  fi
  psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
    "${cleanup_sql}; DROP ROLE campaign_operations_h1_boundary_authority;"
}

run_boundary_graph_fixture direct \
  'CREATE ROLE h1_graph_login LOGIN;
   GRANT campaign_operations_h1_boundary_authority TO h1_graph_login' \
  'REVOKE campaign_operations_h1_boundary_authority FROM h1_graph_login;
   DROP ROLE h1_graph_login'
run_boundary_graph_fixture two_level \
  'CREATE ROLE h1_graph_mid NOLOGIN; CREATE ROLE h1_graph_login LOGIN;
   GRANT campaign_operations_h1_boundary_authority TO h1_graph_mid;
   GRANT h1_graph_mid TO h1_graph_login' \
  'REVOKE h1_graph_mid FROM h1_graph_login;
   REVOKE campaign_operations_h1_boundary_authority FROM h1_graph_mid;
   DROP ROLE h1_graph_login; DROP ROLE h1_graph_mid'
run_boundary_graph_fixture three_level \
  'CREATE ROLE h1_graph_mid1 NOLOGIN; CREATE ROLE h1_graph_mid2 NOLOGIN;
   CREATE ROLE h1_graph_login LOGIN;
   GRANT campaign_operations_h1_boundary_authority TO h1_graph_mid1;
   GRANT h1_graph_mid1 TO h1_graph_mid2;
   GRANT h1_graph_mid2 TO h1_graph_login' \
  'REVOKE h1_graph_mid2 FROM h1_graph_login;
   REVOKE h1_graph_mid1 FROM h1_graph_mid2;
   REVOKE campaign_operations_h1_boundary_authority FROM h1_graph_mid1;
   DROP ROLE h1_graph_login; DROP ROLE h1_graph_mid2; DROP ROLE h1_graph_mid1'
run_boundary_graph_fixture admin_option \
  'CREATE ROLE h1_graph_login LOGIN;
   GRANT campaign_operations_h1_boundary_authority TO h1_graph_login
     WITH ADMIN OPTION' \
  'REVOKE campaign_operations_h1_boundary_authority FROM h1_graph_login;
   DROP ROLE h1_graph_login'
run_boundary_graph_fixture boundary_inherits \
  'CREATE ROLE h1_graph_other NOLOGIN;
   GRANT h1_graph_other TO campaign_operations_h1_boundary_authority' \
  'REVOKE h1_graph_other FROM campaign_operations_h1_boundary_authority;
   DROP ROLE h1_graph_other'

# Unexpected ownership and alternate-schema entry points fail before DDL.  The
# migration must neither transfer nor repair either hostile object.
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<'SQL'
CREATE ROLE campaign_operations_h1_boundary_authority
  NOLOGIN SUPERUSER INHERIT NOCREATEDB NOCREATEROLE NOREPLICATION
  NOBYPASSRLS CONNECTION LIMIT -1 PASSWORD NULL;
CREATE TABLE campaign_operations_h1_unexpected_owner_probe(value integer);
ALTER TABLE campaign_operations_h1_unexpected_owner_probe
  OWNER TO campaign_operations_h1_boundary_authority;
SQL
if psql "${target[@]}" -q -1 -v ON_ERROR_STOP=1 -v VERBOSITY=verbose \
    "$database" -f \
    "$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql" \
    >"$cluster_root/055-unexpected-owner.log" 2>&1; then
  echo "unexpected boundary-owned object fixture succeeded" >&2
  exit 1
fi
if ! rg -q '42501: H1A004 unexpected boundary-owned relation' \
    "$cluster_root/055-unexpected-owner.log"; then
  tail -n 30 "$cluster_root/055-unexpected-owner.log" >&2
  echo "unexpected boundary-owned object reached the wrong failure" >&2
  exit 1
fi
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<'SQL'
ALTER TABLE campaign_operations_h1_unexpected_owner_probe
  OWNER TO campaign_manager_login;
DROP TABLE campaign_operations_h1_unexpected_owner_probe;
DROP ROLE campaign_operations_h1_boundary_authority;
CREATE SCHEMA h1_hostile_alternate;
CREATE FUNCTION h1_hostile_alternate.record_campaign_operations_production_enable_v1()
RETURNS text LANGUAGE sql AS 'SELECT current_user::text';
SQL
if psql "${target[@]}" -q -1 -v ON_ERROR_STOP=1 -v VERBOSITY=verbose \
    "$database" -f \
    "$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql" \
    >"$cluster_root/055-alternate-entry.log" 2>&1; then
  echo "alternate-schema protected-name fixture succeeded" >&2
  exit 1
fi
if ! rg -q '42501: H1A005 protected function alternate schema or overload' \
    "$cluster_root/055-alternate-entry.log"; then
  tail -n 30 "$cluster_root/055-alternate-entry.log" >&2
  echo "alternate-schema fixture reached the wrong failure" >&2
  exit 1
fi
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
  'DROP SCHEMA h1_hostile_alternate CASCADE;'

# An unsafe pre-existing default ACL is also an upgrade blocker.  Exercise the
# catalog interpretation before allowing migration 055 to install its frozen
# defaults, then remove only the disposable grant.
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<'SQL'
ALTER DEFAULT PRIVILEGES FOR ROLE campaign_operations_owner
  GRANT EXECUTE ON FUNCTIONS TO pqxx;
SQL
if psql "${target[@]}" -q -1 -v ON_ERROR_STOP=1 "$database" \
    -f "$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql" \
    >"$cluster_root/055-unsafe-default-acl.log" 2>&1; then
  echo "migration 055 retained an unsafe pre-existing default ACL" >&2
  exit 1
fi
if ! rg -q 'H1A007 unsafe pre-existing default ACL' \
    "$cluster_root/055-unsafe-default-acl.log"; then
  echo "migration 055 unsafe default-ACL test reached the wrong failure" >&2
  exit 1
fi
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<'SQL'
ALTER DEFAULT PRIVILEGES FOR ROLE campaign_operations_owner
  REVOKE EXECUTE ON FUNCTIONS FROM pqxx;
SQL

if ! psql "${target[@]}" -q -1 -v ON_ERROR_STOP=1 "$database" \
    -f "$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql" \
    >"$cluster_root/055-install.log" 2>&1; then
  tail -n 100 "$cluster_root/055-install.log" >&2
  exit 1
fi
if rg -q 'will be truncated' "$cluster_root/055-install.log"; then
  echo "migration 055 emitted an identifier truncation notice" >&2
  exit 1
fi
# Supported replay before any H1 evidence must be idempotent and transactional.
if ! psql "${target[@]}" -q -1 -v ON_ERROR_STOP=1 "$database" \
    -f "$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql" \
    >"$cluster_root/055-replay.log" 2>&1; then
  tail -n 100 "$cluster_root/055-replay.log" >&2
  exit 1
fi
if rg -q 'will be truncated' "$cluster_root/055-replay.log"; then
  echo "migration 055 replay emitted an identifier truncation notice" >&2
  exit 1
fi

# Capture the immutable post-replay ACL/default state before later negative
# tests deliberately mutate catalog objects.  The trusted generator consults
# PostgreSQL only; expected manifests are introduced by its comparator after
# the observation snapshots and execution receipt exist.
python3 "$repo_root/Scripts/CampaignOperationsH1CatalogPipeline.py" \
  "$cluster_socket" 5432 campaign_manager_login "$database" "$runtime_run_id" "$cluster_root"
tail -n +2 "$cluster_root/h1-acl-generic-runtime.tsv" >> "$runtime_results"

# Even the sealed owner cannot accidentally commit a context row: the deferred
# constraint rejects it and rollback leaves the table empty.
if psql "${target[@]}" -q -1 -v ON_ERROR_STOP=1 "$database" -c \
  "INSERT INTO campaign_operations_production_transition_context(
   backend_pid,transaction_id,transition_kind,operational_request_id,
   operation_key) VALUES(pg_backend_pid(),txid_current(),'enable',NULL,
   'context-commit-probe');" >"$cluster_root/055-context.log" 2>&1; then
  echo "migration 055 context commit probe unexpectedly succeeded" >&2
  exit 1
fi
if ! rg -q 'production transition context must be empty at transaction end' \
    "$cluster_root/055-context.log" ||
   [[ "$(psql "${target[@]}" -At "$database" -c \
       'SELECT count(*) FROM campaign_operations_production_transition_context;')" != "0" ]]; then
  echo "migration 055 context commit probe reached the wrong failure" >&2
  exit 1
fi

checksum="$(shasum -a 256 \
  "$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql" |
  awk '{print $1}')"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
  "INSERT INTO schema_migrations(version,filename,checksum) VALUES(
   '055','055_campaign_operations_production_admission_foundation.sql',
   '${checksum}') ON CONFLICT(version) DO UPDATE SET checksum=excluded.checksum;"

psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<'SQL'
DO $$
DECLARE candidate campaign_operations_completion_event%ROWTYPE;
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM phase_h1_pre055_historical_bytes snapshot
    JOIN campaign_operations_dispatch_attempt attempt
      ON attempt.dispatch_attempt_id=7001
    JOIN campaign_operations_completion_event completion
      ON completion.completion_event_id=7001
    WHERE snapshot.attempt_bytes =
            convert_to(attempt.attempt_identity_canonical,'UTF8')
      AND snapshot.attempt_hash = attempt.attempt_identity_hash
      AND snapshot.request_bytes =
            convert_to(completion.request_evidence_canonical,'UTF8')
      AND snapshot.request_hash = completion.request_evidence_hash
      AND snapshot.completion_bytes =
            convert_to(completion.completion_identity_canonical,'UTF8')
      AND snapshot.completion_hash = completion.completion_identity_hash
      AND campaign_operations_completion_identity_valid(completion)) THEN
    RAISE EXCEPTION '054 -> 055 historical canonical/hash bytes changed';
  END IF;
  SELECT completion.* INTO STRICT candidate
  FROM campaign_operations_completion_event completion
  WHERE completion.completion_event_id=7001;
  candidate.request_evidence_canonical :=
    candidate.request_evidence_canonical || 'changed';
  IF campaign_operations_completion_identity_valid(candidate) THEN
    RAISE EXCEPTION
      'request evidence same-hash/different-canonical was accepted';
  END IF;
  SELECT completion.* INTO STRICT candidate
  FROM campaign_operations_completion_event completion
  WHERE completion.completion_event_id=7001;
  candidate.completion_identity_canonical :=
    candidate.completion_identity_canonical || 'changed';
  IF campaign_operations_completion_identity_valid(candidate) THEN
    RAISE EXCEPTION
      'Completion V1 same-hash/different-canonical was accepted';
  END IF;
END $$;
SQL

# Genuine schema-054 Attempt V1 and Completion V1 bytes must survive the
# supported role-recreation plus owner-preserving database restore workflow.
restore_root="$cluster_root/restore"
restore_data="$restore_root/data"
restore_socket="$restore_root/s"
mkdir -p "$restore_socket"
initdb -D "$restore_data" -U h1_restore_admin \
  --auth=trust --no-instructions >/dev/null
pg_ctl -D "$restore_data" -o "-F -h '' -k $restore_socket -p 5433" \
  -w start >/dev/null
restore_target=(-h "$restore_socket" -p 5433 -U h1_restore_admin)
pg_dumpall "${target[@]}" --roles-only -f "$restore_root/roles.sql"
pg_dump "${target[@]}" -Fc -f "$restore_root/database.dump" "$database"
psql "${restore_target[@]}" -q -v ON_ERROR_STOP=1 postgres \
  -f "$restore_root/roles.sql"
createdb "${restore_target[@]}" -O campaign_manager_login h1_restored
# The authoritative schema fixture intentionally contains synthetic rows loaded
# under session_replication_role=replica and omits unrelated recommendation
# history.  Restore schema constraints first, then restore the exact dumped data
# with triggers disabled so the restored fixture has the same deliberate shape.
pg_restore "${restore_target[@]}" -d h1_restored --section=pre-data \
  "$restore_root/database.dump"
pg_restore "${restore_target[@]}" -d h1_restored --section=post-data \
  "$restore_root/database.dump"
pg_restore "${restore_target[@]}" -d h1_restored --section=data \
  --disable-triggers "$restore_root/database.dump"
"$repo_root/Scripts/CampaignOperationsH1RestoreAclOrigin.sh" \
  --host "$restore_socket" --port 5433 --user h1_restore_admin \
  --database h1_restored

"$repo_root/Scripts/CampaignOperationsH1DeploymentAudit.sh" \
  --stage post-role-recreation --host "$restore_socket" --port 5433 \
  --user h1_restore_admin --database postgres
"$repo_root/Scripts/CampaignOperationsH1DeploymentAudit.sh" \
  --stage post-database-restore --host "$restore_socket" --port 5433 \
  --user h1_restore_admin --database h1_restored
if [[ "$(psql "${restore_target[@]}" -At -v ON_ERROR_STOP=1 h1_restored \
  -c "BEGIN READ ONLY; SELECT
      campaign_operations_h1_deployment_audit_v1('${checksum}',true,true);
      COMMIT;")" != *t* ]]; then
  echo "post-restore historical-byte audit did not return true" >&2
  exit 1
fi

# Restore scenarios D-F: incompatible role identity and graph state are blocked
# before deployment even though the database checksum and ledger are intact.
psql "${restore_target[@]}" -q -v ON_ERROR_STOP=1 postgres -c \
  'ALTER ROLE campaign_operations_h1_boundary_authority NOINHERIT;'
if "$repo_root/Scripts/CampaignOperationsH1DeploymentAudit.sh" \
    --stage pre-restore --host "$restore_socket" --port 5433 \
    --user h1_restore_admin --database postgres \
    >"$restore_root/scenario-f.log" 2>&1; then
  echo "restore scenario F accepted altered role attributes" >&2
  exit 1
fi
rg -q 'SQLSTATE=42501 diagnostic=H1A002 .*stage=pre-restore .*audit_version=h1-deployment-audit-v1' \
  "$restore_root/scenario-f.log"
psql "${restore_target[@]}" -q -v ON_ERROR_STOP=1 postgres <<'SQL'
ALTER ROLE campaign_operations_h1_boundary_authority INHERIT;
CREATE ROLE h1_restore_hostile_mid1 NOLOGIN;
CREATE ROLE h1_restore_hostile_mid2 NOLOGIN;
CREATE ROLE h1_restore_hostile_login LOGIN;
GRANT campaign_operations_h1_boundary_authority
  TO h1_restore_hostile_mid1;
GRANT h1_restore_hostile_mid1
  TO h1_restore_hostile_mid2 WITH ADMIN OPTION;
GRANT h1_restore_hostile_mid2 TO h1_restore_hostile_login;
SQL
if "$repo_root/Scripts/CampaignOperationsH1DeploymentAudit.sh" \
    --stage pre-restore --host "$restore_socket" --port 5433 \
    --user h1_restore_admin --database postgres \
    >"$restore_root/scenario-e.log" 2>&1; then
  echo "restore scenario E accepted ADMIN OPTION" >&2
  exit 1
fi
rg -q 'SQLSTATE=42501 diagnostic=H1A003 .*stage=pre-restore .*audit_version=h1-deployment-audit-v1' \
  "$restore_root/scenario-e.log"
rg -q 'direction=member_to_h1,root=campaign_operations_h1_boundary_authority,endpoint=h1_restore_hostile_login,depth=3,roles=h1_restore_hostile_login->h1_restore_hostile_mid2->h1_restore_hostile_mid1->campaign_operations_h1_boundary_authority' \
  "$restore_root/scenario-e.log"
rg -q 'edge_depth=1,edge=h1_restore_hostile_login->h1_restore_hostile_mid2,admin_option=false' \
  "$restore_root/scenario-e.log"
rg -q 'edge_depth=2,edge=h1_restore_hostile_mid2->h1_restore_hostile_mid1,admin_option=true' \
  "$restore_root/scenario-e.log"
rg -q 'edge_depth=3,edge=h1_restore_hostile_mid1->campaign_operations_h1_boundary_authority,admin_option=false' \
  "$restore_root/scenario-e.log"
if ! psql -X -qAt -v ON_ERROR_STOP=1 -h "$restore_socket" -p 5433 \
    -U h1_restore_hostile_login postgres \
    -c 'SET ROLE campaign_operations_h1_boundary_authority;
        SELECT current_user;' > "$restore_root/scenario-e-set-role.log" ||
   ! rg -q '^campaign_operations_h1_boundary_authority$' \
        "$restore_root/scenario-e-set-role.log"; then
  echo "restore ADMIN OPTION fixture did not prove actual SET ROLE reachability" >&2
  exit 1
fi
if ! psql -X -qAt -v ON_ERROR_STOP=1 -h "$restore_socket" -p 5433 \
    -U h1_restore_hostile_login postgres \
    -c 'BEGIN; SET LOCAL ROLE campaign_operations_h1_boundary_authority;
        SELECT current_user; ROLLBACK;' > "$restore_root/scenario-e-set-local-role.log" ||
   ! rg -q '^campaign_operations_h1_boundary_authority$' \
        "$restore_root/scenario-e-set-local-role.log"; then
  echo "restore ADMIN OPTION fixture did not prove SET LOCAL ROLE reachability" >&2
  exit 1
fi
emit_runtime_result H1-GRAPH-ADMIN H1GRAPH004 deployment-audit \
  EXPECTED_FAILURE 42501 H1A003 campaign_operations_h1_boundary_authority \
  pre-restore H1GRAPH004 scenario-e.log "$restore_root/scenario-e.log"
emit_runtime_result H1-SET-ROLE H1GRAPH006 role-assumption-probe SUCCESS \
  00000 role-assumption-observed campaign_operations_h1_boundary_authority \
  role-assumption H1GRAPH006 scenario-e-set-role.log \
  "$restore_root/scenario-e-set-role.log"
emit_runtime_result H1-SET-LOCAL-ROLE H1GRAPH007 role-assumption-probe SUCCESS \
  00000 role-assumption-observed campaign_operations_h1_boundary_authority \
  role-assumption H1GRAPH007 scenario-e-set-local-role.log \
  "$restore_root/scenario-e-set-local-role.log"
psql "${restore_target[@]}" -q -v ON_ERROR_STOP=1 postgres <<'SQL'
REVOKE h1_restore_hostile_mid2 FROM h1_restore_hostile_login;
REVOKE h1_restore_hostile_mid1 FROM h1_restore_hostile_mid2;
REVOKE campaign_operations_h1_boundary_authority
  FROM h1_restore_hostile_mid1;
DROP ROLE h1_restore_hostile_login;
DROP ROLE h1_restore_hostile_mid2;
DROP ROLE h1_restore_hostile_mid1;
SQL

# Restore scenarios G-J use independent cloned databases so each audit reaches
# the intended catalog branch without an earlier unrelated defect.
run_restored_catalog_fixture() {
  local label="$1"
  local mutation_sql="$2"
  local expected_code="$3"
  local expected_fragment="${4:-}"
  local fixture_database="h1_restore_${label}"
  createdb "${restore_target[@]}" -T h1_restored "$fixture_database"
  psql "${restore_target[@]}" -q -v ON_ERROR_STOP=1 \
    "$fixture_database" -c "$mutation_sql"
  if "$repo_root/Scripts/CampaignOperationsH1DeploymentAudit.sh" \
      --stage post-database-restore --host "$restore_socket" --port 5433 \
      --user h1_restore_admin --database "$fixture_database" \
      >"$restore_root/scenario-${label}.log" 2>&1; then
    echo "restore scenario ${label} unexpectedly passed" >&2
    exit 1
  fi
  if ! rg -q "$expected_code" "$restore_root/scenario-${label}.log"; then
    tail -n 30 "$restore_root/scenario-${label}.log" >&2
    echo "restore scenario ${label} reached the wrong audit branch" >&2
    exit 1
  fi
  if ! rg -q '42501:|SQLSTATE=42501' "$restore_root/scenario-${label}.log"; then
    tail -n 30 "$restore_root/scenario-${label}.log" >&2
    echo "restore scenario ${label} did not expose SQLSTATE 42501" >&2
    exit 1
  fi
  if [[ -n "$expected_fragment" ]] &&
     ! rg -q "$expected_fragment" "$restore_root/scenario-${label}.log"; then
    tail -n 30 "$restore_root/scenario-${label}.log" >&2
    echo "restore scenario ${label} lacked exact mismatch direction/identity" >&2
    exit 1
  fi
  local requirement="" fixture="" object_identity=""
  case "$label" in
    g) requirement=H1-OBJ-TABLE; fixture=H1OBJ001; object_identity=h1_unexpected ;;
    j) requirement=H1-OBJ-ALT-NAME; fixture=H1OBJ016; object_identity=h1_alternate.record_campaign_operations_production_disable_v1 ;;
    explicit_acl_extra) requirement=H1-ACL-EXTRA; fixture=H1ACL002; object_identity=production_enablement_event ;;
    explicit_acl_missing) requirement=H1-ACL-MISSING; fixture=H1ACL003; object_identity=production_enablement_event ;;
    default_acl_missing) requirement=H1-DEFAULT-MISSING; fixture=H1ACL005; object_identity=pg_default_acl ;;
    default_acl_grant_option) requirement=H1-DEFAULT-GRANTOPTION; fixture=H1ACL006; object_identity=pg_default_acl ;;
    procedure) requirement=H1-OBJ-PROCEDURE; fixture=H1OBJ008; object_identity=h1_unexpected_procedure ;;
    direct_wrapper) requirement=H1-OBJ-DIRECT-WRAPPER; fixture=H1OBJ014; object_identity=h1_direct_wrapper.call_boundary ;;
    other_owner_wrapper) requirement=H1-OBJ-OTHER-WRAPPER; fixture=H1OBJ015; object_identity=h1_other_wrapper.call_boundary ;;
    overload) requirement=H1-OBJ-OVERLOAD; fixture=H1OBJ017; object_identity='record_campaign_operations_production_disable_v1(integer)' ;;
    default_variant) requirement=H1-OBJ-DEFAULT-VARIANT; fixture=H1OBJ018; object_identity=h1_default_variant.record_campaign_operations_production_disable_v1 ;;
    function_dependency) requirement=H1-OBJ-DEPENDENCY; fixture=H1OBJ019; object_identity=h1_protected_dependency ;;
    rule) requirement=H1-OBJ-RULE; fixture=H1OBJ009; object_identity=h1_protected_rule ;;
    operator) requirement=H1-OBJ-OPERATOR; fixture=H1OBJ010; object_identity=public.## ;;
    cast) requirement=H1-OBJ-CAST; fixture=H1OBJ025; object_identity=public.h1_cast_source-AS-public.h1_cast_target ;;
    ordinary_trigger) requirement=H1-OBJ-TRIGGER; fixture=H1OBJ011; object_identity=h1_protected_trigger ;;
    constraint_trigger) requirement=H1-OBJ-CONSTRAINT-TRIGGER; fixture=H1OBJ012; object_identity=h1_protected_constraint_trigger ;;
    event_trigger) requirement=H1-OBJ-EVENT-TRIGGER; fixture=H1OBJ013; object_identity=h1_event_trigger ;;
    sequence) requirement=H1-OBJ-SEQUENCE; fixture=H1OBJ002; object_identity=h1_unexpected_sequence ;;
    view) requirement=H1-OBJ-VIEW; fixture=H1OBJ003; object_identity=h1_unexpected_view ;;
    materialized_view) requirement=H1-OBJ-MATVIEW; fixture=H1OBJ004; object_identity=h1_unexpected_materialized_view ;;
    type) requirement=H1-OBJ-TYPE; fixture=H1OBJ005; object_identity=h1_unexpected_type ;;
    domain) requirement=H1-OBJ-DOMAIN; fixture=H1OBJ006; object_identity=h1_unexpected_domain ;;
    schema) requirement=H1-OBJ-SCHEMA; fixture=H1OBJ007; object_identity=h1_unexpected_schema ;;
    aggregate) requirement=H1-OBJ-AGGREGATE; fixture=H1OBJ020; object_identity=h1_unexpected_aggregate ;;
    foreign_table) requirement=H1-OBJ-FOREIGN-TABLE; fixture=H1OBJ021; object_identity=h1_unexpected_foreign_table ;;
    publication) requirement=H1-OBJ-PUBLICATION; fixture=H1OBJ022; object_identity=h1_unexpected_publication ;;
    subscription) requirement=H1-OBJ-SUBSCRIPTION; fixture=H1OBJ023; object_identity=h1_unexpected_subscription ;;
    large_object) requirement=H1-OBJ-LARGE-OBJECT; fixture=H1OBJ024; object_identity=919055 ;;
    null_origin_schema) requirement=H1-ACL-ORIGIN-SCHEMA; fixture=H1ACLORIGIN001; object_identity=public ;;
    null_origin_table) requirement=H1-ACL-ORIGIN-TABLE; fixture=H1ACLORIGIN002; object_identity=public.campaign_operations_production_transition_context ;;
    null_origin_view) requirement=H1-ACL-ORIGIN-VIEW; fixture=H1ACLORIGIN003; object_identity=public.campaign_operations_production_status_v1 ;;
    null_origin_sequence) requirement=H1-ACL-ORIGIN-SEQUENCE; fixture=H1ACLORIGIN004; object_identity=public.campaign_operations_dispatch_attempt_dispatch_attempt_id_seq ;;
    null_origin_function) requirement=H1-ACL-ORIGIN-FUNCTION; fixture=H1ACLORIGIN005; object_identity='public.campaign_operations_h1_deployment_audit_v1(text,boolean,boolean)' ;;
    explicit_origin_type) requirement=H1-ACL-ORIGIN-TYPE; fixture=H1ACLORIGIN006; object_identity=public.campaign_operations_production_transition_context ;;
    null_origin_column) requirement=H1-ACL-ORIGIN-COLUMN-EXPLICIT; fixture=H1ACLORIGIN007; object_identity=public.campaign_operations_operational_request.request_state ;;
    explicit_origin_column) requirement=H1-ACL-ORIGIN-COLUMN-NULL; fixture=H1ACLORIGIN008; object_identity=public.campaign_operations_production_transition_context.backend_pid ;;
  esac
  if [[ -n "$requirement" ]]; then
    local runtime_stage=post-database-restore
    case "$label" in
      null_origin_schema|null_origin_table|null_origin_view|\
      null_origin_sequence|null_origin_function|explicit_origin_type)
        runtime_stage=database-audit ;;
    esac
    emit_runtime_result "$requirement" "$fixture" deployment-audit \
      EXPECTED_FAILURE 42501 "$expected_code" "$object_identity" \
      "$runtime_stage" "$fixture" "scenario-${label}.log" \
      "$restore_root/scenario-${label}.log"
  fi
  if [[ "$label" == "subscription" ]]; then
    psql "${restore_target[@]}" -q -v ON_ERROR_STOP=1 \
      "$fixture_database" -c \
      'ALTER SUBSCRIPTION h1_unexpected_subscription SET (slot_name=NONE);
       DROP SUBSCRIPTION h1_unexpected_subscription'
  fi
  dropdb "${restore_target[@]}" "$fixture_database"
}

run_restored_catalog_fixture g \
  'CREATE TABLE h1_unexpected(value integer);
   ALTER TABLE h1_unexpected OWNER TO campaign_operations_h1_boundary_authority' \
  'H1A004'
run_restored_catalog_fixture h \
  'ALTER DEFAULT PRIVILEGES FOR ROLE
     campaign_operations_h1_boundary_authority
     GRANT EXECUTE ON FUNCTIONS TO PUBLIC' \
  'H1A007'
run_restored_catalog_fixture i \
  'GRANT EXECUTE ON FUNCTION
     record_campaign_operations_production_disable_v1(
       text,bigint,text,integer,text,text) TO PUBLIC' \
  'H1A006'
run_restored_catalog_fixture j \
  'CREATE SCHEMA h1_alternate;
   CREATE FUNCTION
     h1_alternate.record_campaign_operations_production_disable_v1()
     RETURNS integer LANGUAGE sql AS $$SELECT 1$$' \
  'H1A005'
run_restored_catalog_fixture explicit_acl_extra \
  'GRANT UPDATE ON public.campaign_operations_production_enablement_event TO campaign_operations_production_reader' \
  'H1A006' \
  'actual_minus_expected.*campaign_operations_production_enablement_event.*campaign_operations_production_reader.*UPDATE'
run_restored_catalog_fixture explicit_acl_missing \
  'REVOKE SELECT ON public.campaign_operations_production_enablement_event FROM campaign_operations_production_reader' \
  'H1A006' \
  'expected_minus_actual.*campaign_operations_production_enablement_event.*campaign_operations_production_reader.*SELECT'
run_restored_catalog_fixture default_acl_missing \
  'ALTER DEFAULT PRIVILEGES FOR ROLE campaign_operations_owner REVOKE USAGE ON TYPES FROM campaign_operations_owner' \
  'H1A007' \
  'expected_minus_actual.*H1-DEFAULT-GLOBAL-TYPE.*campaign_operations_owner'
run_restored_catalog_fixture default_acl_grant_option \
  'ALTER DEFAULT PRIVILEGES FOR ROLE campaign_operations_owner GRANT USAGE ON TYPES TO campaign_operations_owner WITH GRANT OPTION' \
  'H1A007' \
  'actual_minus_expected.*H1-DEFAULT-GLOBAL-TYPE.*campaign_operations_owner'

# Exact ACL-origin fixtures.  These direct catalog mutations are confined to
# cloned databases in this disposable cluster.  Owner-effective privileges
# remain available, so an origin-only mismatch cannot be hidden by an earlier
# missing-owner privilege failure.
run_restored_catalog_fixture null_origin_schema \
  "UPDATE pg_namespace SET nspacl=NULL WHERE nspname='public'" \
  'H1A006' 'exact ACL origin mismatch object=public stage=database-audit'
run_restored_catalog_fixture null_origin_table \
  "UPDATE pg_class SET relacl=NULL WHERE oid='public.campaign_operations_production_transition_context'::regclass" \
  'H1A006' 'campaign_operations_production_transition_context'
run_restored_catalog_fixture null_origin_view \
  "UPDATE pg_class SET relacl=NULL WHERE oid='public.campaign_operations_production_status_v1'::regclass" \
  'H1A006' 'campaign_operations_production_status_v1'
run_restored_catalog_fixture null_origin_sequence \
  "UPDATE pg_class SET relacl=NULL WHERE oid='public.campaign_operations_dispatch_attempt_dispatch_attempt_id_seq'::regclass" \
  'H1A006' 'campaign_operations_dispatch_attempt_dispatch_attempt_id_seq'
run_restored_catalog_fixture null_origin_function \
  "UPDATE pg_proc SET proacl=NULL WHERE oid='public.campaign_operations_h1_deployment_audit_v1(text,boolean,boolean)'::regprocedure" \
  'H1A006' 'campaign_operations_h1_deployment_audit_v1'
run_restored_catalog_fixture explicit_origin_type \
  "UPDATE pg_type SET typacl=acldefault('T',typowner) WHERE oid='public.campaign_operations_production_transition_context'::regtype" \
  'H1A006' 'exact ACL origin mismatch object='
run_restored_catalog_fixture null_origin_column \
  "UPDATE pg_attribute SET attacl=NULL WHERE attrelid='public.campaign_operations_operational_request'::regclass AND attname='request_state'" \
  'H1A006' 'campaign_operations_operational_request.request_state'
run_restored_catalog_fixture explicit_origin_column \
  "UPDATE pg_attribute attribute SET attacl=ARRAY[(quote_ident(pg_get_userbyid(class_row.relowner))||'=r/'||quote_ident(pg_get_userbyid(class_row.relowner)))::aclitem] FROM pg_class class_row WHERE attribute.attrelid=class_row.oid AND class_row.oid='public.campaign_operations_production_transition_context'::regclass AND attribute.attname='backend_pid'" \
  'H1A006' 'campaign_operations_production_transition_context.backend_pid'

# ADR-0019B all-schema/object-class authenticity fixtures.  Each clone starts
# from the passing restored catalog, so no prerequisite defect can mask the
# intended audit branch.
run_restored_catalog_fixture procedure \
  'CREATE PROCEDURE public.h1_unexpected_procedure() LANGUAGE sql AS $$SELECT 1$$;
   ALTER PROCEDURE public.h1_unexpected_procedure() OWNER TO campaign_operations_h1_boundary_authority' \
  'H1A004'
run_restored_catalog_fixture direct_wrapper \
  'CREATE SCHEMA h1_direct_wrapper;
   CREATE FUNCTION h1_direct_wrapper.call_boundary() RETURNS text LANGUAGE sql SECURITY DEFINER AS $$SELECT current_user::text$$;
   ALTER FUNCTION h1_direct_wrapper.call_boundary() OWNER TO campaign_operations_h1_boundary_authority' \
  'H1A004'
run_restored_catalog_fixture other_owner_wrapper \
  'CREATE SCHEMA h1_other_wrapper;
   CREATE FUNCTION h1_other_wrapper.call_boundary() RETURNS integer LANGUAGE sql SECURITY DEFINER AS $$SELECT 1 /* conservative reference: record_campaign_operations_production_disable_v1 */$$' \
  'H1A005'
run_restored_catalog_fixture overload \
  'CREATE FUNCTION public.record_campaign_operations_production_disable_v1(integer) RETURNS integer LANGUAGE sql AS $$SELECT $1$$' \
  'H1A005'
run_restored_catalog_fixture default_variant \
  'CREATE SCHEMA h1_default_variant;
   CREATE FUNCTION h1_default_variant.record_campaign_operations_production_disable_v1(value integer DEFAULT 1) RETURNS integer LANGUAGE sql AS $$SELECT value$$' \
  'H1A005'
run_restored_catalog_fixture function_dependency \
  'CREATE FUNCTION public.h1_protected_dependency(value text) RETURNS text LANGUAGE sql BEGIN ATOMIC SELECT public.campaign_operations_tagged_fnv1a64(value); END' \
  'H1A005'
run_restored_catalog_fixture rule \
  'CREATE TABLE public.h1_rule_source(value text);
   CREATE TABLE public.h1_rule_sink(value text);
   CREATE RULE h1_protected_rule AS ON INSERT TO public.h1_rule_source DO ALSO INSERT INTO public.h1_rule_sink VALUES(public.campaign_operations_tagged_fnv1a64(NEW.value))' \
  'H1A005'
run_restored_catalog_fixture operator \
  'CREATE OPERATOR public.## (RIGHTARG=text, FUNCTION=public.campaign_operations_tagged_fnv1a64)' \
  'H1A005'
run_restored_catalog_fixture cast \
  'CREATE TYPE public.h1_cast_source AS ENUM ($q$source$q$);
   CREATE TYPE public.h1_cast_target AS ENUM ($q$target$q$);
   CREATE FUNCTION public.h1_boundary_cast(public.h1_cast_source)
     RETURNS public.h1_cast_target LANGUAGE sql IMMUTABLE
     AS $$SELECT $q$target$q$::public.h1_cast_target$$;
   ALTER FUNCTION public.h1_boundary_cast(public.h1_cast_source)
     OWNER TO campaign_operations_h1_boundary_authority;
   CREATE CAST (public.h1_cast_source AS public.h1_cast_target)
     WITH FUNCTION public.h1_boundary_cast(public.h1_cast_source)' \
  'H1A005' \
  'DETAIL:  cast [0-9]+'
run_restored_catalog_fixture ordinary_trigger \
  'CREATE TABLE public.h1_trigger_target(value integer);
   CREATE TRIGGER h1_protected_trigger BEFORE INSERT ON public.h1_trigger_target FOR EACH ROW EXECUTE FUNCTION public.reject_campaign_operations_production_mutation()' \
  'H1A005'
run_restored_catalog_fixture constraint_trigger \
  'CREATE TABLE public.h1_constraint_trigger_target(value integer);
   CREATE CONSTRAINT TRIGGER h1_protected_constraint_trigger AFTER INSERT ON public.h1_constraint_trigger_target DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION public.reject_campaign_operations_production_mutation()' \
  'H1A005'
run_restored_catalog_fixture event_trigger \
  'CREATE FUNCTION public.h1_event_trigger_function() RETURNS event_trigger LANGUAGE plpgsql AS $$BEGIN END$$;
   CREATE EVENT TRIGGER h1_event_trigger ON ddl_command_end WHEN TAG IN ($q$CREATE TABLE$q$) EXECUTE FUNCTION public.h1_event_trigger_function();
   ALTER EVENT TRIGGER h1_event_trigger OWNER TO campaign_operations_h1_boundary_authority' \
  'H1A004'
run_restored_catalog_fixture sequence \
  'CREATE SEQUENCE public.h1_unexpected_sequence;
   ALTER SEQUENCE public.h1_unexpected_sequence OWNER TO campaign_operations_h1_boundary_authority' \
  'H1A004'
run_restored_catalog_fixture view \
  'CREATE VIEW public.h1_unexpected_view AS SELECT 1 AS value;
   ALTER VIEW public.h1_unexpected_view OWNER TO campaign_operations_h1_boundary_authority' \
  'H1A004'
run_restored_catalog_fixture materialized_view \
  'CREATE MATERIALIZED VIEW public.h1_unexpected_materialized_view AS SELECT 1 AS value;
   ALTER MATERIALIZED VIEW public.h1_unexpected_materialized_view OWNER TO campaign_operations_h1_boundary_authority' \
  'H1A004'
run_restored_catalog_fixture type \
  'CREATE TYPE public.h1_unexpected_type AS ENUM ($q$one$q$);
   ALTER TYPE public.h1_unexpected_type OWNER TO campaign_operations_h1_boundary_authority' \
  'H1A004'
run_restored_catalog_fixture domain \
  'CREATE DOMAIN public.h1_unexpected_domain AS text;
   ALTER DOMAIN public.h1_unexpected_domain OWNER TO campaign_operations_h1_boundary_authority' \
  'H1A004'
run_restored_catalog_fixture schema \
  'CREATE SCHEMA h1_unexpected_schema AUTHORIZATION campaign_operations_h1_boundary_authority' \
  'H1A004'
run_restored_catalog_fixture aggregate \
  'CREATE FUNCTION public.h1_sum_state(integer,integer) RETURNS integer LANGUAGE sql IMMUTABLE AS $$SELECT coalesce($1,0)+coalesce($2,0)$$;
   CREATE AGGREGATE public.h1_unexpected_aggregate(integer) (SFUNC=public.h1_sum_state, STYPE=integer, INITCOND=0);
   ALTER AGGREGATE public.h1_unexpected_aggregate(integer) OWNER TO campaign_operations_h1_boundary_authority' \
  'H1A004'
run_restored_catalog_fixture foreign_table \
  'CREATE FOREIGN DATA WRAPPER h1_fixture_fdw NO HANDLER;
   CREATE SERVER h1_fixture_server FOREIGN DATA WRAPPER h1_fixture_fdw;
   CREATE FOREIGN TABLE public.h1_unexpected_foreign_table(value integer) SERVER h1_fixture_server;
   ALTER FOREIGN TABLE public.h1_unexpected_foreign_table OWNER TO campaign_operations_h1_boundary_authority' \
  'H1A004'
run_restored_catalog_fixture publication \
  'CREATE TABLE public.h1_publication_table(value integer);
   CREATE PUBLICATION h1_unexpected_publication FOR TABLE public.h1_publication_table;
   ALTER PUBLICATION h1_unexpected_publication OWNER TO campaign_operations_h1_boundary_authority' \
  'H1A004'
run_restored_catalog_fixture subscription \
  'CREATE SUBSCRIPTION h1_unexpected_subscription CONNECTION $q$host=/tmp dbname=postgres$q$ PUBLICATION h1_missing_publication WITH (connect=false, create_slot=false, enabled=false);
   ALTER SUBSCRIPTION h1_unexpected_subscription OWNER TO campaign_operations_h1_boundary_authority' \
  'H1A004'
run_restored_catalog_fixture large_object \
  'SELECT pg_catalog.lo_create(919055);
   ALTER LARGE OBJECT 919055 OWNER TO campaign_operations_h1_boundary_authority' \
  'H1A004'

pg_ctl -D "$restore_data" -m immediate stop >/dev/null

# Scenarios A and C are separate target-cluster workflows.  The role recreation
# SQL is generated from exact source catalog attributes rather than reusing the
# roles-only dump exercised by Scenario B.
role_recreation_sql="$restore_root/exact-role-recreation.sql"
psql "${target[@]}" -X -qAt postgres > "$role_recreation_sql" <<'SQL'
SELECT format(
  'CREATE ROLE %I WITH %s %s %s %s %s %s %s CONNECTION LIMIT %s PASSWORD NULL;',
  rolname,
  CASE WHEN rolcanlogin THEN 'LOGIN' ELSE 'NOLOGIN' END,
  CASE WHEN rolsuper THEN 'SUPERUSER' ELSE 'NOSUPERUSER' END,
  CASE WHEN rolinherit THEN 'INHERIT' ELSE 'NOINHERIT' END,
  CASE WHEN rolcreatedb THEN 'CREATEDB' ELSE 'NOCREATEDB' END,
  CASE WHEN rolcreaterole THEN 'CREATEROLE' ELSE 'NOCREATEROLE' END,
  CASE WHEN rolreplication THEN 'REPLICATION' ELSE 'NOREPLICATION' END,
  CASE WHEN rolbypassrls THEN 'BYPASSRLS' ELSE 'NOBYPASSRLS' END,
  rolconnlimit)
FROM pg_authid
WHERE rolname !~ '^pg_'
ORDER BY rolname;
SQL

run_distinct_restore_success() {
  local scenario="$1" port="$2" first_audit_stage="$3"
  local scenario_root="$restore_root/scenario-${scenario}"
  local scenario_data="$scenario_root/data" scenario_socket="$scenario_root/s"
  mkdir -p "$scenario_socket"
  active_distinct_data="$scenario_data"
  initdb -D "$scenario_data" -U "h1_${scenario}_admin" \
    --auth=trust --no-instructions >/dev/null
  pg_ctl -D "$scenario_data" -o "-F -h '' -k $scenario_socket -p $port" \
    -w start >/dev/null
  local scenario_target=(-h "$scenario_socket" -p "$port" -U "h1_${scenario}_admin")
  psql "${scenario_target[@]}" -q -v ON_ERROR_STOP=1 postgres \
    -f "$role_recreation_sql"
  "$repo_root/Scripts/CampaignOperationsH1DeploymentAudit.sh" \
    --stage "$first_audit_stage" --host "$scenario_socket" --port "$port" \
    --user "h1_${scenario}_admin" --database postgres \
    > "$scenario_root/${first_audit_stage}.log"
  createdb "${scenario_target[@]}" -O campaign_manager_login "h1_restore_${scenario}"
  pg_restore "${scenario_target[@]}" -d "h1_restore_${scenario}" \
    --section=pre-data "$restore_root/database.dump"
  pg_restore "${scenario_target[@]}" -d "h1_restore_${scenario}" \
    --section=post-data "$restore_root/database.dump"
  pg_restore "${scenario_target[@]}" -d "h1_restore_${scenario}" \
    --section=data --disable-triggers "$restore_root/database.dump"
  "$repo_root/Scripts/CampaignOperationsH1RestoreAclOrigin.sh" \
    --host "$scenario_socket" --port "$port" --user "h1_${scenario}_admin" \
    --database "h1_restore_${scenario}"
  "$repo_root/Scripts/CampaignOperationsH1DeploymentAudit.sh" \
    --stage post-role-recreation --host "$scenario_socket" --port "$port" \
    --user "h1_${scenario}_admin" --database postgres \
    > "$scenario_root/post-role-recreation.log"
  "$repo_root/Scripts/CampaignOperationsH1DeploymentAudit.sh" \
    --stage post-database-restore --host "$scenario_socket" --port "$port" \
    --user "h1_${scenario}_admin" --database "h1_restore_${scenario}" \
    > "$scenario_root/post-database-restore.log"
  [[ "$(psql "${scenario_target[@]}" -At -v ON_ERROR_STOP=1 \
    "h1_restore_${scenario}" -c "SELECT campaign_operations_h1_deployment_audit_v1('${checksum}',true,true);")" == "t" ]]
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$scenario" source-exact "empty-target" catalog-role-recreation database-only \
    success true success success success 00000 restore-supported exact \
    supported "pending-digest" > "$restore_root/restore-scenario-${scenario}.tsv"
  pg_ctl -D "$scenario_data" -m immediate stop >/dev/null
  active_distinct_data=""
}

run_distinct_restore_success A 5434 post-role-recreation
run_distinct_restore_success C 5435 pre-restore

# Scenario D proves rejection before any database restore or target-object
# creation.  Every other H1 role is exact so H1A002 cannot be masked by H1A001.
scenario_d_root="$restore_root/scenario-D"
scenario_d_data="$scenario_d_root/data"
scenario_d_socket="$scenario_d_root/s"
mkdir -p "$scenario_d_socket"
active_distinct_data="$scenario_d_data"
initdb -D "$scenario_d_data" -U h1_D_admin --auth=trust --no-instructions >/dev/null
pg_ctl -D "$scenario_d_data" -o "-F -h '' -k $scenario_d_socket -p 5436" \
  -w start >/dev/null
scenario_d_target=(-h "$scenario_d_socket" -p 5436 -U h1_D_admin)
psql "${scenario_d_target[@]}" -q -v ON_ERROR_STOP=1 postgres \
  -f "$role_recreation_sql"
psql "${scenario_d_target[@]}" -q -v ON_ERROR_STOP=1 postgres -c \
  'ALTER ROLE campaign_operations_h1_boundary_authority NOINHERIT'
if "$repo_root/Scripts/CampaignOperationsH1DeploymentAudit.sh" \
    --stage pre-restore --host "$scenario_d_socket" --port 5436 \
    --user h1_D_admin --database postgres > "$scenario_d_root/pre-restore.log" 2>&1; then
  echo "restore scenario D accepted incompatible pre-existing role" >&2
  exit 1
fi
rg -q 'SQLSTATE=42501 diagnostic=H1A002 .*object=attributes:campaign_operations_h1_boundary_authority .*stage=pre-restore .*audit_version=h1-deployment-audit-v1' \
  "$scenario_d_root/pre-restore.log"
[[ "$(psql "${scenario_d_target[@]}" -At postgres -c \
  "SELECT count(*) FROM pg_database WHERE datname='h1_restore_D';")" == 0 ]]
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  D source-exact incompatible-boundary catalog-role-recreation none failed \
  false not-attempted not-applicable not-applicable 42501 H1A002 not-applicable \
  rejected-before-restore pending-digest > "$restore_root/restore-scenario-D.tsv"
pg_ctl -D "$scenario_d_data" -m immediate stop >/dev/null
active_distinct_data=""

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  B source-exact empty-target roles-only-dump custom-database success true \
  success success success 00000 restore-supported exact supported pending-digest \
  > "$restore_root/restore-scenario-B.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  E source-exact admin-option roles-only-dump none failed false not-attempted \
  not-applicable not-applicable 42501 H1A003 not-applicable rejected pending-digest \
  > "$restore_root/restore-scenario-E.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  F source-exact altered-attribute roles-only-dump none failed false not-attempted \
  not-applicable not-applicable 42501 H1A002 not-applicable rejected pending-digest \
  > "$restore_root/restore-scenario-F.tsv"
for restore_case in G H I J; do
  lower_case="$(tr '[:upper:]' '[:lower:]' <<<"$restore_case")"
  case "$restore_case" in
    G) restore_code=H1A004 ;;
    H) restore_code=H1A007 ;;
    I) restore_code=H1A006 ;;
    J) restore_code=H1A005 ;;
  esac
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$restore_case" source-exact exact-roles roles-only-dump custom-database \
    success true success success failed 42501 "$restore_code" exact blocked \
    pending-digest > "$restore_root/restore-scenario-${restore_case}.tsv"
  [[ -s "$restore_root/scenario-${lower_case}.log" ]]
done

emit_runtime_result H1-RESTORE-A H1RESTOREA restore-harness SUCCESS 00000 \
  restore-supported restored-catalog post-database-restore H1RESTOREA \
  restore-scenario-A.tsv "$restore_root/restore-scenario-A.tsv"
emit_runtime_result H1-RESTORE-B H1RESTOREB restore-harness SUCCESS 00000 \
  restore-supported restored-catalog post-database-restore H1RESTOREB \
  restore-scenario-B.tsv "$restore_root/restore-scenario-B.tsv"
emit_runtime_result H1-RESTORE-C H1RESTOREC restore-harness SUCCESS 00000 \
  restore-supported restored-catalog post-database-restore H1RESTOREC \
  restore-scenario-C.tsv "$restore_root/restore-scenario-C.tsv"
emit_runtime_result H1-RESTORE-D H1RESTORED restore-harness EXPECTED_FAILURE \
  42501 H1A002 campaign_operations_h1_boundary_authority pre-restore H1RESTORED \
  restore-scenario-D.tsv "$restore_root/restore-scenario-D.tsv"
emit_runtime_result H1-RESTORE-E H1RESTOREE restore-harness EXPECTED_FAILURE \
  42501 H1A003 campaign_operations_h1_boundary_authority pre-restore H1RESTOREE \
  restore-scenario-E.tsv "$restore_root/restore-scenario-E.tsv"
emit_runtime_result H1-RESTORE-F H1RESTOREF restore-harness EXPECTED_FAILURE \
  42501 H1A002 campaign_operations_h1_boundary_authority pre-restore H1RESTOREF \
  restore-scenario-F.tsv "$restore_root/restore-scenario-F.tsv"
emit_runtime_result H1-RESTORE-G H1RESTOREG restore-harness EXPECTED_FAILURE \
  42501 H1A004 h1_unexpected post-database-restore H1RESTOREG \
  restore-scenario-G.tsv "$restore_root/restore-scenario-G.tsv"
emit_runtime_result H1-RESTORE-H H1RESTOREH restore-harness EXPECTED_FAILURE \
  42501 H1A007 pg_default_acl post-database-restore H1RESTOREH \
  restore-scenario-H.tsv "$restore_root/restore-scenario-H.tsv"
emit_runtime_result H1-RESTORE-I H1RESTOREI restore-harness EXPECTED_FAILURE \
  42501 H1A006 record_campaign_operations_production_disable_v1 \
  post-database-restore H1RESTOREI restore-scenario-I.tsv \
  "$restore_root/restore-scenario-I.tsv"
emit_runtime_result H1-RESTORE-J H1RESTOREJ restore-harness EXPECTED_FAILURE \
  42501 H1A005 h1_alternate.record_campaign_operations_production_disable_v1 \
  post-database-restore H1RESTOREJ restore-scenario-J.tsv \
  "$restore_root/restore-scenario-J.tsv"

restore_runtime="$cluster_root/h1-restore-runtime.tsv"
printf '%s\n' 'format_version	run_id	scenario_id	source_cluster_state	target_cluster_pre_state	role_creation_source	dump_type	pre_restore_audit	restore_attempted	restore_result	post_role_audit	post_database_audit	expected_sqlstate	expected_diagnostic	actual_sqlstate	actual_diagnostic	historical_bytes	final_disposition	artifact_id	artifact_path	artifact_digest	record_digest' \
  > "$restore_runtime"
append_restore_runtime() {
  local scenario="$1" source_state="$2" target_state="$3" role_source="$4"
  local dump_type="$5" pre_audit="$6" attempted="$7" restore_result="$8"
  local post_role="$9" post_database="${10}" expected_state="${11}"
  local expected_diagnostic="${12}" actual_state="${13}"
  local actual_diagnostic="${14}" historical="${15}" disposition="${16}"
  local artifact_source="${17}" artifact_digest artifact_path record_line record_digest
  artifact_digest="$(shasum -a 256 "$artifact_source" | awk '{print $1}')"
  artifact_path="runtime-artifacts/records/H1RESTORE${scenario}/restore-scenario-${scenario}.tsv"
  [[ -s "$cluster_root/$artifact_path" ]] || {
    echo "missing canonical restore artifact $artifact_path" >&2; exit 1;
  }
  artifact_digest="$(shasum -a 256 "$cluster_root/$artifact_path" | awk '{print $1}')"
  record_line="$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
    h1-restore-runtime-v2 "$runtime_run_id" "$scenario" "$source_state" "$target_state" \
    "$role_source" "$dump_type" "$pre_audit" "$attempted" "$restore_result" \
    "$post_role" "$post_database" "$expected_state" "$expected_diagnostic" \
    "$actual_state" "$actual_diagnostic" "$historical" "$disposition" \
    "ART-RECORD-H1RESTORE${scenario}" "$artifact_path" "$artifact_digest")"
  record_digest="$(printf '%s' "$record_line" | shasum -a 256 | awk '{print $1}')"
  printf '%s\t%s\n' "$record_line" "$record_digest" >> "$restore_runtime"
}
append_restore_runtime A exact-h1-source empty-target catalog-role-recreation \
  database-only success true success success success 00000 restore-supported \
  00000 restore-supported exact supported "$restore_root/restore-scenario-A.tsv"
append_restore_runtime B exact-h1-source empty-target roles-only-dump \
  custom-database success true success success success 00000 restore-supported \
  00000 restore-supported exact supported "$restore_root/restore-scenario-B.tsv"
append_restore_runtime C exact-h1-source safe-pre-existing-roles \
  catalog-role-recreation database-only success true success success success \
  00000 restore-supported 00000 restore-supported exact supported \
  "$restore_root/restore-scenario-C.tsv"
append_restore_runtime D exact-h1-source incompatible-boundary \
  catalog-role-recreation none failed false not-attempted not-applicable \
  not-applicable 42501 H1A002 42501 H1A002 not-applicable \
  rejected-before-restore "$restore_root/restore-scenario-D.tsv"
append_restore_runtime E exact-h1-source admin-option roles-only-dump none failed \
  false not-attempted not-applicable not-applicable 42501 H1A003 42501 H1A003 \
  not-applicable rejected-before-restore "$restore_root/restore-scenario-E.tsv"
append_restore_runtime F exact-h1-source altered-attribute roles-only-dump none \
  failed false not-attempted not-applicable not-applicable 42501 H1A002 42501 \
  H1A002 not-applicable rejected-before-restore \
  "$restore_root/restore-scenario-F.tsv"
append_restore_runtime G exact-h1-source exact-roles roles-only-dump \
  custom-database success true success success failed 42501 H1A004 42501 H1A004 \
  exact blocked "$restore_root/restore-scenario-G.tsv"
append_restore_runtime H exact-h1-source exact-roles roles-only-dump \
  custom-database success true success success failed 42501 H1A007 42501 H1A007 \
  exact blocked "$restore_root/restore-scenario-H.tsv"
append_restore_runtime I exact-h1-source exact-roles roles-only-dump \
  custom-database success true success success failed 42501 H1A006 42501 H1A006 \
  exact blocked "$restore_root/restore-scenario-I.tsv"
append_restore_runtime J exact-h1-source exact-roles roles-only-dump \
  custom-database success true success success failed 42501 H1A005 42501 H1A005 \
  exact blocked "$restore_root/restore-scenario-J.tsv"
"$repo_root/Tests/CampaignOperationsPhaseH1RestoreArtifactTests.sh" \
  "$restore_runtime"

psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<'SQL'
SET session_replication_role = replica;
DELETE FROM campaign_operations_completion_audit_reference_event
 WHERE completion_event_id=7001;
DELETE FROM campaign_operations_completion_event WHERE completion_event_id=7001;
DELETE FROM campaign_operations_dispatch_attempt WHERE dispatch_attempt_id=7001;
SET session_replication_role = origin;
DROP TABLE phase_h1_pre055_historical_bytes;
SQL

psql "${target[@]}" -v ON_ERROR_STOP=1 "$database" \
  -f "$repo_root/Tests/CampaignOperationsPhaseH1MigrationTests.sql" | \
  tee "$cluster_root/migration-sql-results"
"$repo_root/Scripts/CampaignOperationsH1DeploymentAudit.sh" \
  --stage post-upgrade --host "$cluster_socket" --port 5432 \
  --user campaign_manager_login --database "$database" | \
  tee "$cluster_root/055-post-upgrade-audit.log"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" \
  -f "$repo_root/Tests/SchedulerOwnershipMigrationTests.sql"
echo "Scheduler ownership migration/policy SQL tests passed"
cp "$cluster_root/055-post-upgrade-audit.log" \
  "$cluster_root/explicit-acl-set-difference"
cp "$cluster_root/055-post-upgrade-audit.log" \
  "$cluster_root/default-acl-set-difference"
cp "$cluster_root/migration-sql-results" "$cluster_root/future-object-probe-results"
{
  cat "$cluster_root/055-install.log"
  cat "$cluster_root/055-replay.log"
} > "$cluster_root/055-install-and-replay.log"
cp "$restore_root/scenario-A/post-database-restore.log" \
  "$cluster_root/historical-byte-results"

emit_runtime_result H1-ACL-EXACT H1ACL001 h1-acl-manifest-v1 SUCCESS 00000 \
  exact-acl-state all-H1-boundary-objects post-upgrade H1ACL001 \
  explicit-acl-set-difference "$cluster_root/explicit-acl-set-difference"
emit_runtime_result H1-DEFAULT-EXACT H1ACL004 h1-acl-manifest-v1 SUCCESS 00000 \
  exact-default-acl-state pg_default_acl post-upgrade H1ACL004 \
  default-acl-set-difference "$cluster_root/default-acl-set-difference"
emit_runtime_result H1-FUTURE-PROBES H1ACL007 future-object-probes SUCCESS 00000 \
  future-object-defaults-exact future-object-ACLs migration-test H1ACL007 \
  future-object-probe-results "$cluster_root/future-object-probe-results"
emit_runtime_result H1-FIXED-TRANSITIONS H1SQL001 fixed-transition-catalog-audit \
  SUCCESS 00000 fixed-transitions-exact three-fixed-transitions post-upgrade \
  H1SQL001 migration-sql-results "$cluster_root/migration-sql-results"
emit_runtime_result H1-SUPPORTING-DEFINERS H1SQL002 function-manifest SUCCESS \
  00000 supporting-definers-exact retained-supporting-functions post-upgrade \
  H1SQL002 migration-sql-results "$cluster_root/migration-sql-results"
emit_runtime_result H1-RECURSIVE-TRIGGER H1SQL004 guarded-triggers \
  EXPECTED_FAILURE 42501 fixed-transition-context-required nested-trigger \
  trigger H1SQL004 migration-sql-results "$cluster_root/migration-sql-results"
emit_runtime_result H1-REPLAY-ENABLE-EXACT H1SQL005A enable-replay-helper \
  SUCCESS 00000 production-enable-exact-replay enablement-event replay \
  H1SQL005A migration-sql-results "$cluster_root/migration-sql-results"
emit_runtime_result H1-REPLAY-ENABLE-CONFLICT H1SQL005B enable-replay-helper \
  EXPECTED_FAILURE 23505 production-enable-conflicting-replay enablement-event \
  replay H1SQL005B migration-sql-results "$cluster_root/migration-sql-results"
emit_runtime_result H1-REPLAY-DISABLE-EXACT H1SQL006A disable-replay-helper \
  SUCCESS 00000 production-disable-exact-replay enablement-event replay \
  H1SQL006A migration-sql-results "$cluster_root/migration-sql-results"
emit_runtime_result H1-REPLAY-DISABLE-CONFLICT H1SQL006B disable-replay-helper \
  EXPECTED_FAILURE 23505 production-disable-conflicting-replay enablement-event \
  replay H1SQL006B migration-sql-results "$cluster_root/migration-sql-results"
emit_runtime_result H1-REPLAY-ACQUIRE-EXACT H1SQL007A acquire-replay-helper \
  SUCCESS 00000 production-acquisition-exact-replay admission-attempt-audit \
  replay H1SQL007A migration-sql-results "$cluster_root/migration-sql-results"
emit_runtime_result H1-REPLAY-ACQUIRE-CONFLICT H1SQL007B acquire-replay-helper \
  EXPECTED_FAILURE 23505 production-acquisition-conflicting-replay \
  admission-attempt-audit replay H1SQL007B migration-sql-results \
  "$cluster_root/migration-sql-results"
emit_runtime_result H1-REPLAY-ACQUIRE-CORRUPT H1SQL007C acquire-replay-helper \
  EXPECTED_FAILURE 23514 production-acquisition-replay-evidence-corrupt \
  admission-attempt-audit replay H1SQL007C migration-sql-results \
  "$cluster_root/migration-sql-results"
emit_runtime_result H1-CANONICAL-FIVE-LEVELS H1SQL008 identity-validators \
  EXPECTED_FAILURE 23514 corrupt-evidence all-five-canonical-levels validation \
  H1SQL008 migration-sql-results "$cluster_root/migration-sql-results"
emit_runtime_result H1-POST-COMPLETION H1SQL009 post-completion-guards \
  EXPECTED_FAILURE 23514 post-completion-rejection four-production-objects \
  trigger H1SQL009 migration-sql-results "$cluster_root/migration-sql-results"
emit_runtime_result H1-MIGRATION-LEDGER H1RESTORE001 deployment-audit SUCCESS \
  00000 migration-ledger-exact schema_migrations post-upgrade H1RESTORE001 \
  055-install-and-replay.log "$cluster_root/055-install-and-replay.log"
emit_runtime_result H1-HISTORICAL-BYTES H1RESTORE003 historical-byte-audit \
  SUCCESS 00000 historical-bytes-exact Attempt-V1-Completion-V1 \
  post-database-restore H1RESTORE003 historical-byte-results \
  "$cluster_root/historical-byte-results"
emit_runtime_result H1-INERTNESS H1SQL010 migration055-postcondition SUCCESS \
  00000 h1-inert-default-off H1-evidence-tables postcondition H1SQL010 \
  migration-sql-results "$cluster_root/migration-sql-results"

psql "${target[@]}" -qAt -v ON_ERROR_STOP=1 "$database" \
  >"$cluster_root/055-savepoint-session.log" <<'SQL'
BEGIN;
SAVEPOINT h1_context_probe;
SELECT production_enablement_event_id
FROM record_campaign_operations_production_disable_v1(
  'savepoint-disable-001',
  (SELECT production_enablement_event_id
   FROM campaign_operations_production_enablement_event
   ORDER BY resulting_version DESC LIMIT 1),
  (SELECT enablement_identity_canonical
   FROM campaign_operations_production_enablement_event
   ORDER BY resulting_version DESC LIMIT 1),
  (SELECT resulting_version
   FROM campaign_operations_production_enablement_event
   ORDER BY resulting_version DESC LIMIT 1),
  'operator@example.test','Savepoint cleanup proof.');
ROLLBACK TO SAVEPOINT h1_context_probe;
SELECT count(*) FROM campaign_operations_production_transition_context;
COMMIT;
SELECT count(*) FROM campaign_operations_production_transition_context;
SQL
if [[ "$(tail -n 2 "$cluster_root/055-savepoint-session.log" | tr '\n' ',')" \
      != "0,0," ]]; then
  echo "savepoint/session-reuse context cleanup mismatch" >&2
  exit 1
fi
emit_runtime_result H1-CONTEXT-LIFECYCLE H1SQL003 production-transition-context \
  EXPECTED_FAILURE 42501 context-required production_transition_context \
  transition H1SQL003 055-savepoint-session.log \
  "$cluster_root/055-savepoint-session.log"

# Deterministic lock-order evidence.  The clone is disposable and contains the
# completed H1 fixture; production evidence is removed only in that clone.
createdb "${target[@]}" -T "$database" "$lock_database"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$lock_database" <<'SQL'
SET session_replication_role = replica;
DELETE FROM campaign_operations_dispatch_audit_reference_event
 WHERE request_production_admission_id IS NOT NULL;
DELETE FROM campaign_operations_dispatch_attempt WHERE attempt_contract_version=2;
DELETE FROM campaign_operations_request_production_admission;
UPDATE campaign_operations_operational_request
   SET request_state='ready',state_version=3,lease_token_hash=NULL,
       lease_expires_at=NULL,dispatcher_identity=NULL,
       production_dispatch_enabled=false
 WHERE operational_request_id=7;
SET session_replication_role = origin;
SQL

workflow_lock_test="$cluster_root/CampaignOperationsPhaseH1WorkflowLockTests"
workflow_lock_adapters="$cluster_root/CampaignOperationsPhaseH1WorkflowAdapters.o"
clang++ -std=c++20 -Wall -Wextra -Werror -Dmain=H1WorkflowAdapterUnusedMain \
  -I "$repo_root/Sources" -I "$repo_root/Headers" \
  -I /opt/homebrew/opt/libpqxx@7.10.1/include \
  -I /opt/homebrew/opt/libpq/include -c \
  "$repo_root/Tests/CampaignOperationsRepositoryTests.cpp" \
  -o "$workflow_lock_adapters"
clang++ -std=c++20 -Wall -Wextra -Werror \
  -I "$repo_root/Sources" -I "$repo_root/Headers" \
  -I /opt/homebrew/opt/libpqxx@7.10.1/include \
  -I /opt/homebrew/opt/libpq/include \
  "$repo_root/Tests/CampaignOperationsPhaseH1WorkflowLockTests.cpp" \
  "$workflow_lock_adapters" \
  "$repo_root/Sources/CampaignOperations.cpp" \
  "$repo_root/Sources/CampaignOperationsDispatch.cpp" \
  "$repo_root/Sources/CampaignOperationsControl.cpp" \
  "$repo_root/Sources/CampaignOperationsCompletion.cpp" \
  "$repo_root/Sources/CampaignOperationsRepository.cpp" \
  "$repo_root/Sources/CampaignOperationsService.cpp" \
  "$repo_root/Sources/CampaignOperationsDispatchRepository.cpp" \
  "$repo_root/Sources/CampaignOperationsManager.cpp" \
  "$repo_root/Sources/CampaignOperationsDispatchService.cpp" \
  "$repo_root/Sources/CampaignOperationsBindingRepository.cpp" \
  "$repo_root/Sources/CampaignOperationsControlRepository.cpp" \
  "$repo_root/Sources/CampaignOperationsControlService.cpp" \
  "$repo_root/Sources/CampaignOperationsCompletionRepository.cpp" \
  "$repo_root/Sources/CampaignOperationsCompletionService.cpp" \
  "$repo_root/Sources/CampaignOperationsProductionAdmission.cpp" \
  "$repo_root/Sources/CampaignOperationsProductionAdmissionRepository.cpp" \
  "$repo_root/Sources/CampaignOperationsProductionAdmissionService.cpp" \
  "$repo_root/Sources/ExperimentRecommendation.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCampaignActivation.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCampaignActivationRepository.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCampaignHandoff.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCampaignHandoffRepository.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionExecutionRepository.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionActivation.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionActivationRepository.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCampaignExecution.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCampaignExecutionRepository.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCampaignLaunch.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCampaignLaunchRepository.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionProposalReview.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionProposalReviewRepository.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCandidateGenerator.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionRepository.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionWorkflow.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionWorkflowRepository.cpp" \
  -L /opt/homebrew/opt/libpqxx@7.10.1/lib \
  -L /opt/homebrew/opt/libpq/lib -lpqxx -lpq -o "$workflow_lock_test"

"$workflow_lock_test" \
  "host=$cluster_socket port=5432 dbname=$lock_database user=campaign_manager_login" \
  fixture-setup >"$cluster_root/h1-lock-fixture-setup.log"

run_lock_order_probe() {
  local stage="$1"
  local blocker_sql="$2"
  local must_hold="$3"
  local must_not_hold="$4"
  local expected_gate="$5"
  local blocker_app="h1_lock_blocker_${stage}_${$}"
  local acquire_app="h1_lock_acquire_${stage}_${$}"
  PGAPPNAME="$blocker_app" psql "${target[@]}" -q -v ON_ERROR_STOP=1 \
    "$lock_database" -c "BEGIN; ${blocker_sql}; SELECT pg_sleep(30);" \
    >"$cluster_root/${stage}-blocker.log" 2>&1 &
  local blocker_job=$!
  local blocker_pid=""
  for _ in $(seq 1 200); do
    blocker_pid="$(psql "${target[@]}" -At "$lock_database" -c \
      "SELECT pid FROM pg_stat_activity WHERE application_name='${blocker_app}'
       AND wait_event='PgSleep' LIMIT 1;")"
    [[ -n "$blocker_pid" ]] && break
    sleep 0.02
  done
  if [[ -z "$blocker_pid" ]]; then
    echo "lock-order blocker did not reach ${stage}" >&2
    exit 1
  fi
  PGAPPNAME="$acquire_app" psql "${target[@]}" -q -v ON_ERROR_STOP=1 \
    "$lock_database" -c "SELECT * FROM
      transition_campaign_operations_request_dispatch_production_v2(
        71,3,'fnv1a64:0123456789abcdef',
        transaction_timestamp()+interval '5 minutes','lock-${stage}-001',
        'manager@example.test',campaign_operations_manager_build_canonical_v1(
          'campaign-operations-production-dispatch-and-manager-run-once-v1',
          repeat('a',40),'Apple clang 21.0.0',
          'sha256:'||repeat('b',64)));" \
    >"$cluster_root/${stage}-acquire.log" 2>&1 &
  local acquire_job=$!
  local acquire_pid=""
  local blocked="f"
  for _ in $(seq 1 200); do
    acquire_pid="$(psql "${target[@]}" -At "$lock_database" -c \
      "SELECT pid FROM pg_stat_activity WHERE application_name='${acquire_app}'
       LIMIT 1;")"
    if [[ -n "$acquire_pid" ]]; then
      blocked="$(psql "${target[@]}" -At "$lock_database" -c \
        "SELECT ${blocker_pid}=ANY(pg_blocking_pids(${acquire_pid}));")"
      [[ "$blocked" == "t" ]] && break
    fi
    sleep 0.02
  done
  if [[ "$blocked" != "t" ]]; then
    echo "lock-order acquisition was not catalog-blocked at ${stage}" >&2
    exit 1
  fi
  if [[ "$(psql "${target[@]}" -At "$lock_database" -c \
    "SELECT ${acquire_pid}=ANY(pg_blocking_pids(${blocker_pid}));")" != "f" ]]; then
    echo "lock-order prohibited reverse wait detected at ${stage}" >&2
    exit 1
  fi
  if [[ "$(psql "${target[@]}" -At "$lock_database" -c \
    "WITH RECURSIVE edges(waiter,blocker) AS (
       SELECT pid,unnest(pg_blocking_pids(pid)) FROM pg_stat_activity),
     walk(origin,node,path,cycle) AS (
       SELECT waiter,blocker,ARRAY[waiter,blocker],waiter=blocker FROM edges
       UNION ALL SELECT walk.origin,edges.blocker,walk.path||edges.blocker,
         edges.blocker=ANY(walk.path) FROM walk JOIN edges ON edges.waiter=walk.node
         WHERE NOT walk.cycle)
     SELECT EXISTS(SELECT 1 FROM walk WHERE cycle);" )" != "f" ]]; then
    echo "lock-order wait-for cycle detected at ${stage}" >&2
    exit 1
  fi
  psql "${target[@]}" -X -qAt -F $'\t' "$lock_database" -c \
    "SELECT '${stage}',activity.application_name,locks.pid,locks.locktype,
       coalesce(locks.relation::regclass::text,''),locks.mode,locks.granted,
       coalesce(locks.classid::text,''),coalesce(locks.objid::text,''),
       array_to_string(pg_blocking_pids(locks.pid),','),activity.state,
       coalesce(activity.wait_event_type,''),coalesce(activity.wait_event,''),
       coalesce(activity.backend_xid::text,''),
       coalesce(locks.transactionid::text,'')
     FROM pg_locks locks JOIN pg_stat_activity activity USING(pid)
     WHERE locks.pid IN (${blocker_pid},${acquire_pid})
       AND locks.locktype IN ('relation','tuple','transactionid','advisory')
     ORDER BY activity.application_name,locks.granted DESC,locks.locktype,
              locks.relation,locks.classid,locks.objid" \
    >>"$cluster_root/h1-lock-catalog-evidence.tsv"
  if [[ -n "$must_hold" ]] && [[ "$(psql "${target[@]}" -At "$lock_database" -c \
    "WITH held AS (
       SELECT relation::regclass::text AS relation_name
       FROM pg_locks WHERE pid=${acquire_pid} AND granted
         AND mode='RowShareLock')
     SELECT bool_and(name=ANY(ARRAY(SELECT relation_name FROM held)))
     FROM unnest(string_to_array('${must_hold}',',')) name;")" != "t" ]]; then
    echo "lock-order prior boundary missing at ${stage}" >&2
    exit 1
  fi
  if [[ -n "$must_not_hold" ]] && [[ "$(psql "${target[@]}" -At \
    "$lock_database" -c \
    "WITH held AS (
       SELECT relation::regclass::text AS relation_name
       FROM pg_locks WHERE pid=${acquire_pid} AND granted
         AND mode='RowShareLock')
     SELECT bool_or(name=ANY(ARRAY(SELECT relation_name FROM held)))
     FROM unnest(string_to_array('${must_not_hold}',',')) name;")" != "f" ]]; then
    echo "lock-order later boundary acquired early at ${stage}" >&2
    exit 1
  fi
  if [[ "$(psql "${target[@]}" -At "$lock_database" -c \
    "SELECT EXISTS(SELECT 1 FROM pg_locks WHERE pid=${acquire_pid}
       AND locktype='advisory' AND classid=19055 AND objid=1
       AND mode='ShareLock' AND granted); ")" != "$expected_gate" ]]; then
    echo "lock-order production gate state mismatch at ${stage}" >&2
    exit 1
  fi
  psql "${target[@]}" -q "$lock_database" -c \
    "SELECT pg_terminate_backend(${acquire_pid});
     SELECT pg_terminate_backend(${blocker_pid});" >/dev/null
  wait "$acquire_job" 2>/dev/null || true
  wait "$blocker_job" 2>/dev/null || true
  if [[ "$(psql "${target[@]}" -At "$lock_database" -c \
    "SELECT EXISTS(SELECT 1 FROM pg_stat_activity
       WHERE application_name IN ('${blocker_app}','${acquire_app}')) OR
       EXISTS(SELECT 1 FROM pg_locks WHERE pid IN (${blocker_pid},${acquire_pid}));")" != "f" ]]; then
    echo "lock-order rollback did not release backend locks at ${stage}" >&2
    exit 1
  fi
}

run_lock_order_probe scheduler_evidence \
  "SELECT * FROM experiment_scheduler_protocol WHERE singleton FOR UPDATE" \
  "experiment_scheduler_protocol" \
  "campaign_operations_authorization_event,campaign_operations_budget_ledger_entry,campaign_operations_campaign,campaign_operations_reservation,campaign_operations_operational_request" \
  "f"
run_lock_order_probe production_enable_domain \
  "SELECT pg_advisory_xact_lock(19055,1)" \
  "experiment_scheduler_protocol" \
  "campaign_operations_authorization_event,campaign_operations_budget_ledger_entry,campaign_operations_campaign,campaign_operations_reservation,campaign_operations_operational_request" \
  "f"
run_lock_order_probe authorization \
  "SELECT * FROM campaign_operations_authorization_event WHERE operational_campaign_id=71 AND action_kind='dispatch_full_materialization' AND action_contract_version=1 AND scope_kind='complete_materialization' AND scope_contract_version=1 ORDER BY chain_version DESC LIMIT 1 FOR UPDATE" \
  "experiment_scheduler_protocol,campaign_operations_authorization_event" \
  "campaign_operations_budget_ledger_entry,campaign_operations_campaign,campaign_operations_reservation,campaign_operations_operational_request" \
  "t"
run_lock_order_probe budget \
  "SELECT * FROM campaign_operations_budget_ledger_entry WHERE operational_campaign_id=71 ORDER BY ledger_version DESC LIMIT 1 FOR UPDATE" \
  "experiment_scheduler_protocol,campaign_operations_authorization_event,campaign_operations_budget_ledger_entry" \
  "campaign_operations_campaign,campaign_operations_reservation,campaign_operations_operational_request" \
  "t"
run_lock_order_probe campaign \
  "SELECT * FROM campaign_operations_campaign WHERE operational_campaign_id=71 FOR UPDATE" \
  "experiment_scheduler_protocol,campaign_operations_authorization_event,campaign_operations_budget_ledger_entry,campaign_operations_campaign" \
  "campaign_operations_reservation,campaign_operations_operational_request" \
  "t"
run_lock_order_probe reservation \
  "SELECT * FROM campaign_operations_reservation WHERE reservation_id=71 FOR UPDATE" \
  "experiment_scheduler_protocol,campaign_operations_authorization_event,campaign_operations_budget_ledger_entry,campaign_operations_campaign,campaign_operations_reservation" \
  "campaign_operations_operational_request" \
  "t"
run_lock_order_probe request \
  "SELECT * FROM campaign_operations_operational_request WHERE operational_request_id=71 FOR UPDATE" \
  "experiment_scheduler_protocol,campaign_operations_authorization_event,campaign_operations_budget_ledger_entry,campaign_operations_campaign,campaign_operations_reservation,campaign_operations_operational_request" \
  "" \
  "t"

# Complete workflow-versus-workflow probes.  A test hook pauses only after the
# production workflow has acquired its normal locks; it neither substitutes a
# lock nor bypasses any production statement.
run_complete_workflow_probe() {
  local lock_id="$1" mode="$2" classification="$3"
  local probe_db="h1lock_${$}_${lock_id#H1LOCK}"
  local counterpart_app="h1_${lock_id}_counterpart_${$}"
  local acquire_app="h1_${lock_id}_acquisition_${$}"
  local counterpart_log="$cluster_root/${lock_id}-counterpart.log"
  local acquire_log="$cluster_root/${lock_id}-acquisition.log"
  createdb "${target[@]}" -T "$lock_database" "$probe_db"

  case "$mode" in
    enable)
      psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$probe_db" -c \
        "SELECT record_campaign_operations_production_disable_v1(
          'h1-${lock_id}-prepare-disable',
          production_enablement_event_id,enablement_identity_canonical,
          resulting_version,'assurance@example.test','Prepare enable probe.')
         FROM campaign_operations_production_enablement_event
         ORDER BY resulting_version DESC LIMIT 1" >/dev/null
      PGAPPNAME="$counterpart_app" psql "${target[@]}" -q -v ON_ERROR_STOP=1 \
        "$probe_db" -c "BEGIN; SELECT record_campaign_operations_production_enable_v1(
          'h1-${lock_id}-enable',
          (SELECT resulting_version FROM campaign_operations_production_enablement_event ORDER BY resulting_version DESC LIMIT 1),
          (SELECT evidence_canonical FROM campaign_operations_scheduler_protocol_evidence_snapshot_v1()),
          'H1-LOCK-independent-verification','assurance@example.test',
          'campaign-operations-production-dispatch-and-manager-run-once-v1',
          campaign_operations_manager_build_canonical_v1(
            'campaign-operations-production-dispatch-and-manager-run-once-v1',repeat('a',40),'Apple clang 21.0.0','sha256:'||repeat('b',64)),
          repeat('a',40),'Apple clang 21.0.0','sha256:'||repeat('b',64),
          'Complete enable workflow probe.'); SELECT pg_sleep(4); COMMIT" \
        >"$counterpart_log" 2>&1 & ;;
    disable)
      PGAPPNAME="$counterpart_app" psql "${target[@]}" -q -v ON_ERROR_STOP=1 \
        "$probe_db" -c "BEGIN; SELECT record_campaign_operations_production_disable_v1(
          'h1-${lock_id}-disable',production_enablement_event_id,
          enablement_identity_canonical,resulting_version,
          'assurance@example.test','Complete disable workflow probe.')
          FROM campaign_operations_production_enablement_event
          ORDER BY resulting_version DESC LIMIT 1; SELECT pg_sleep(4); COMMIT" \
        >"$counterpart_log" 2>&1 & ;;
    recovery)
      psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$probe_db" -c \
        "SELECT transition_campaign_operations_request_dispatch_production_v2(
          71,3,'fnv1a64:0123456789abcdef',transaction_timestamp()+interval '5 minutes',
          'h1-${lock_id}-prepare','manager@example.test',
          campaign_operations_manager_build_canonical_v1(
            'campaign-operations-production-dispatch-and-manager-run-once-v1',repeat('a',40),'Apple clang 21.0.0','sha256:'||repeat('b',64)))" >/dev/null
      psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$probe_db" -c \
        "SET session_replication_role=replica; UPDATE campaign_operations_operational_request
         SET lease_expires_at=transaction_timestamp()-interval '1 minute'
         WHERE operational_request_id=71; SET session_replication_role=origin" >/dev/null
      PGAPPNAME="$counterpart_app" "$workflow_lock_test" \
        "host=$cluster_socket port=5432 dbname=$probe_db user=campaign_manager_login" \
        recovery >"$counterpart_log" 2>&1 & ;;
    reconciliation|reservation-expiration)
      psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$probe_db" -c \
        "SET session_replication_role=replica; UPDATE campaign_operations_reservation
         SET expires_at=transaction_timestamp()-interval '1 minute'
         WHERE reservation_id=71; SET session_replication_role=origin" >/dev/null
      PGAPPNAME="$counterpart_app" "$workflow_lock_test" \
        "host=$cluster_socket port=5432 dbname=$probe_db user=campaign_manager_login" \
        reconciliation >"$counterpart_log" 2>&1 & ;;
    *)
      PGAPPNAME="$counterpart_app" "$workflow_lock_test" \
        "host=$cluster_socket port=5432 dbname=$probe_db user=campaign_manager_login" \
        "$mode" >"$counterpart_log" 2>&1 & ;;
  esac
  local counterpart_job=$!
  local counterpart_pid=""
  for _ in $(seq 1 300); do
    counterpart_pid="$(psql "${target[@]}" -At "$probe_db" -c \
      "SELECT pid FROM pg_stat_activity WHERE application_name='${counterpart_app}'
       AND (wait_event='PgSleep' OR state='idle in transaction') LIMIT 1")"
    [[ -n "$counterpart_pid" ]] && break
    sleep 0.02
  done
  [[ -n "$counterpart_pid" ]] || {
    cat "$counterpart_log" >&2; echo "$lock_id counterpart did not pause" >&2; exit 1;
  }

  acquire_sql="SELECT transition_campaign_operations_request_dispatch_production_v2(
      71,3,'fnv1a64:0123456789abcdef',transaction_timestamp()+interval '5 minutes',
      'h1-${lock_id}-acquire','manager@example.test',
      campaign_operations_manager_build_canonical_v1(
        'campaign-operations-production-dispatch-and-manager-run-once-v1',repeat('a',40),'Apple clang 21.0.0','sha256:'||repeat('b',64)))"
  [[ "$mode" != budget-mutation ]] || \
    acquire_sql="BEGIN; ${acquire_sql}; SELECT pg_sleep(4); COMMIT"
  PGAPPNAME="$acquire_app" psql "${target[@]}" -q -v ON_ERROR_STOP=1 \
    -v VERBOSITY=verbose "$probe_db" -c "$acquire_sql" \
    >"$acquire_log" 2>&1 &
  local acquire_job=$!
  local acquire_pid="" blocked="f" actual_counterpart_pid=""
  for _ in $(seq 1 300); do
    acquire_pid="$(psql "${target[@]}" -At "$probe_db" -c \
      "SELECT pid FROM pg_stat_activity WHERE application_name='${acquire_app}' LIMIT 1")"
    if [[ -n "$acquire_pid" ]]; then
      if [[ "$mode" == budget-mutation ]]; then
        if [[ "$(psql "${target[@]}" -At "$probe_db" -c \
          "SELECT wait_event='PgSleep' AND cardinality(pg_blocking_pids(pid))=0
             FROM pg_stat_activity WHERE pid=${acquire_pid}")" == t ]]; then
          blocked=t; break
        fi
      else
        actual_counterpart_pid="$(psql "${target[@]}" -At "$probe_db" -c \
          "SELECT blocker.pid FROM unnest(pg_blocking_pids(${acquire_pid})) pid
           JOIN pg_stat_activity blocker USING(pid)
           WHERE blocker.application_name='${counterpart_app}' LIMIT 1")"
        if [[ -n "$actual_counterpart_pid" ]]; then
          counterpart_pid="$actual_counterpart_pid"
          blocked=t
          break
        fi
      fi
    fi
    sleep 0.02
  done
  [[ "$blocked" == t ]] || {
    cat "$counterpart_log" "$acquire_log" >&2
    echo "$lock_id acquisition did not reach its expected catalog state against $mode" >&2
    exit 1
  }
  [[ "$(psql "${target[@]}" -At "$probe_db" -c \
      "SELECT ${acquire_pid}=ANY(pg_blocking_pids(${counterpart_pid}))")" == f ]] || {
    echo "$lock_id prohibited reverse wait observed" >&2; exit 1;
  }
  psql "${target[@]}" -X -qAt -F $'\t' "$probe_db" -c \
    "SELECT '${lock_id}',activity.application_name,locks.pid,locks.locktype,
       coalesce(locks.relation::regclass::text,''),locks.mode,locks.granted,
       coalesce(locks.classid::text,''),coalesce(locks.objid::text,''),
       array_to_string(pg_blocking_pids(locks.pid),','),activity.state,
       coalesce(activity.wait_event_type,''),coalesce(activity.wait_event,''),
       coalesce(activity.backend_xid::text,''),
       coalesce(locks.transactionid::text,'')
     FROM pg_locks locks JOIN pg_stat_activity activity USING(pid)
     WHERE locks.pid IN (${counterpart_pid},${acquire_pid})
       AND locks.locktype IN ('relation','tuple','transactionid','advisory')
     ORDER BY activity.application_name,locks.granted DESC,locks.locktype,
              locks.relation,locks.classid,locks.objid" \
    >>"$cluster_root/h1-lock-catalog-evidence.tsv"
  local counterpart_status=0 acquire_status=0
  wait "$counterpart_job" || counterpart_status=$?
  wait "$acquire_job" || acquire_status=$?
  [[ "$counterpart_status" == 0 ]] || {
    cat "$counterpart_log" >&2; echo "$lock_id counterpart failed" >&2; exit 1;
  }
  local first_outcome=committed
  [[ "$acquire_status" == 0 ]] || first_outcome=expected-rejection
  release_verified="$(psql "${target[@]}" -At "$probe_db" -c \
    "SELECT NOT EXISTS(SELECT 1 FROM pg_stat_activity WHERE application_name
      IN ('${counterpart_app}','${acquire_app}'))")"
  [[ "$release_verified" == t ]] || {
      echo "$lock_id workflow locks were not released" >&2; exit 1;
    }
  partial_evidence_count="$(psql "${target[@]}" -At "$probe_db" -c \
    "SELECT count(*) FROM campaign_operations_request_production_admission a
       LEFT JOIN campaign_operations_dispatch_attempt d
         ON d.request_production_admission_id=a.request_production_admission_id
      WHERE d.dispatch_attempt_id IS NULL")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$lock_id" "$acquire_pid" "$counterpart_pid" "$acquire_app" \
    "$counterpart_app" "$first_outcome" committed true \
    "$partial_evidence_count" PASS partial-admission-query-v1 \
    backend-release-query-v1 >> "$cluster_root/h1-lock-special-pids.tsv"
  dropdb "${target[@]}" "$probe_db"
}

run_complete_workflow_probe H1LOCK001 enable full_workflow_vs_full_workflow
run_complete_workflow_probe H1LOCK002 disable full_workflow_vs_full_workflow
run_complete_workflow_probe H1LOCK003 completion full_workflow_vs_full_workflow
run_complete_workflow_probe H1LOCK004 cancellation full_workflow_vs_full_workflow
run_complete_workflow_probe H1LOCK005 recovery full_workflow_vs_full_workflow
run_complete_workflow_probe H1LOCK006 reconciliation full_workflow_vs_full_workflow
run_complete_workflow_probe H1LOCK008 budget-mutation full_workflow_vs_full_workflow
run_complete_workflow_probe H1LOCK010 reservation-release full_workflow_vs_full_workflow
run_complete_workflow_probe H1LOCK013 disable full_workflow_vs_full_workflow
run_complete_workflow_probe H1LOCK014 completion full_workflow_vs_full_workflow
run_complete_workflow_probe H1LOCK015 cancellation full_workflow_vs_full_workflow

run_uncommitted_replay_probe() {
  local mode="$1"
  local operation_key="uncommitted-${mode}-${$}"
  local first_app="h1_replay_first_${mode}_${$}"
  local second_app="h1_replay_second_${mode}_${$}"
  local lease_time
  lease_time="$(psql "${target[@]}" -At "$lock_database" -c \
    "SELECT to_char(transaction_timestamp()+interval '1 hour',
      'YYYY-MM-DD\"T\"HH24:MI:SS.USOF');")"
  psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$lock_database" <<'SQL'
SET session_replication_role = replica;
DELETE FROM campaign_operations_dispatch_audit_reference_event
 WHERE request_production_admission_id IS NOT NULL;
DELETE FROM campaign_operations_dispatch_attempt WHERE attempt_contract_version=2;
DELETE FROM campaign_operations_request_production_admission;
UPDATE campaign_operations_operational_request
   SET request_state='ready',state_version=3,lease_token_hash=NULL,
       lease_expires_at=NULL,dispatcher_identity=NULL,
       production_dispatch_enabled=false
 WHERE operational_request_id=71;
SET session_replication_role = origin;
SQL
  local build_sql="campaign_operations_manager_build_canonical_v1(
    'campaign-operations-production-dispatch-and-manager-run-once-v1',
    repeat('a',40),'Apple clang 21.0.0','sha256:'||repeat('b',64))"
  PGAPPNAME="$first_app" psql "${target[@]}" -qAt -v ON_ERROR_STOP=1 \
    "$lock_database" -c "BEGIN; SELECT * FROM
      transition_campaign_operations_request_dispatch_production_v2(
        71,3,'fnv1a64:0123456789abcdef','${lease_time}'::timestamptz,
        '${operation_key}','manager@example.test',${build_sql});
      SELECT pg_sleep(3); COMMIT;" \
    >"$cluster_root/${mode}-first.log" 2>&1 &
  local first_job=$!
  local first_pid=""
  for _ in $(seq 1 200); do
    first_pid="$(psql "${target[@]}" -At "$lock_database" -c \
      "SELECT pid FROM pg_stat_activity WHERE application_name='${first_app}'
       AND wait_event='PgSleep' LIMIT 1;")"
    [[ -n "$first_pid" ]] && break
    sleep 0.02
  done
  [[ -n "$first_pid" ]] || { echo "uncommitted ${mode} first operation did not pause" >&2; exit 1; }
  local second_token='fnv1a64:0123456789abcdef'
  [[ "$mode" == "conflicting_replay" ]] && second_token='fnv1a64:fedcba9876543210'
  PGAPPNAME="$second_app" psql "${target[@]}" -qAt -v ON_ERROR_STOP=1 \
    -v VERBOSITY=verbose "$lock_database" -c "SELECT * FROM
      transition_campaign_operations_request_dispatch_production_v2(
        71,3,'${second_token}','${lease_time}'::timestamptz,
        '${operation_key}','manager@example.test',${build_sql});" \
    >"$cluster_root/${mode}-second.log" 2>&1 &
  local second_job=$!
  local second_pid=""
  local blocked="f"
  for _ in $(seq 1 200); do
    second_pid="$(psql "${target[@]}" -At "$lock_database" -c \
      "SELECT pid FROM pg_stat_activity WHERE application_name='${second_app}' LIMIT 1;")"
    if [[ -n "$second_pid" ]]; then
      blocked="$(psql "${target[@]}" -At "$lock_database" -c \
        "SELECT ${first_pid}=ANY(pg_blocking_pids(${second_pid}));")"
      [[ "$blocked" == "t" ]] && break
    fi
    sleep 0.02
  done
  [[ "$blocked" == "t" ]] || { echo "uncommitted ${mode} did not wait" >&2; exit 1; }
  [[ "$(psql "${target[@]}" -At "$lock_database" -c \
    "SELECT ${second_pid}=ANY(pg_blocking_pids(${first_pid}));")" == "f" ]] ||
    { echo "uncommitted ${mode} reverse wait detected" >&2; exit 1; }
  local lock_id=H1LOCK017
  [[ "$mode" == "conflicting_replay" ]] && lock_id=H1LOCK018
  psql "${target[@]}" -X -qAt -F $'\t' "$lock_database" -c \
    "SELECT '${lock_id}',activity.application_name,locks.pid,locks.locktype,
       coalesce(locks.relation::regclass::text,''),locks.mode,locks.granted,
       coalesce(locks.classid::text,''),coalesce(locks.objid::text,''),
       array_to_string(pg_blocking_pids(locks.pid),','),activity.state,
       coalesce(activity.wait_event_type,''),coalesce(activity.wait_event,''),
       coalesce(activity.backend_xid::text,''),
       coalesce(locks.transactionid::text,'')
     FROM pg_locks locks JOIN pg_stat_activity activity USING(pid)
     WHERE locks.pid IN (${first_pid},${second_pid})
       AND locks.locktype IN ('relation','tuple','transactionid','advisory')
     ORDER BY activity.application_name,locks.granted DESC,locks.locktype,
              locks.relation,locks.classid,locks.objid" \
    >>"$cluster_root/h1-lock-catalog-evidence.tsv"
  wait "$first_job"
  local second_status=0
  wait "$second_job" || second_status=$?
  if [[ "$mode" == "exact_replay" ]]; then
    [[ "$second_status" == 0 ]] || { tail -n 30 "$cluster_root/${mode}-second.log" >&2; exit 1; }
    first_returned_identity="$(head -n 1 "$cluster_root/${mode}-first.log")"
    second_returned_identity="$(head -n 1 "$cluster_root/${mode}-second.log")"
    [[ -n "$first_returned_identity" &&
        "$first_returned_identity" == "$second_returned_identity" ]] || {
      echo "uncommitted exact replay returned a different evidence identity" >&2; exit 1;
    }
    [[ "$(psql "${target[@]}" -At "$lock_database" -c \
      "SELECT count(*)=1 FROM campaign_operations_request_production_admission
       WHERE dispatch_operation_key='${operation_key}';")" == "t" ]] ||
      { echo "uncommitted exact replay duplicated evidence" >&2; exit 1; }
  else
    [[ "$second_status" != 0 ]] &&
      rg -q '23505: production acquisition conflicting replay' \
        "$cluster_root/${mode}-second.log" ||
      { echo "uncommitted conflicting replay reached wrong outcome" >&2; exit 1; }
  fi
  local second_outcome=committed
  [[ "$mode" == conflicting_replay ]] && second_outcome=expected-rejection
  release_verified="$(psql "${target[@]}" -At "$lock_database" -c \
    "SELECT NOT EXISTS(SELECT 1 FROM pg_stat_activity WHERE application_name
      IN ('${first_app}','${second_app}'));")"
  [[ "$release_verified" == "t" ]] ||
    { echo "uncommitted ${mode} locks not released" >&2; exit 1; }
  partial_evidence_count="$(psql "${target[@]}" -At "$lock_database" -c \
    "SELECT count(*) FROM campaign_operations_request_production_admission a
       LEFT JOIN campaign_operations_dispatch_attempt d
         ON d.request_production_admission_id=a.request_production_admission_id
      WHERE d.dispatch_attempt_id IS NULL")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$lock_id" \
    "$first_pid" "$second_pid" "$first_app" "$second_app" committed \
    "$second_outcome" true "$partial_evidence_count" PASS \
    partial-admission-query-v1 backend-release-query-v1 \
    >> "$cluster_root/h1-lock-special-pids.tsv"
}

run_uncommitted_replay_probe exact_replay
run_uncommitted_replay_probe conflicting_replay

run_complete_acquisition_pair() {
  local lock_id="$1" first_request="$2" second_request="$3"
  local expect_blocking="$4" expected_evidence="$5"
  local probe_db="h1pair_${$}_${lock_id#H1LOCK}"
  local first_app="h1_${lock_id}_first_acquisition_${$}"
  local second_app="h1_${lock_id}_second_acquisition_${$}"
  createdb "${target[@]}" -T "$lock_database" "$probe_db"
  psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$probe_db" <<'SQL'
SET session_replication_role = replica;
DELETE FROM campaign_operations_dispatch_audit_reference_event
 WHERE operational_request_id IN (71,72);
DELETE FROM campaign_operations_dispatch_attempt
 WHERE operational_request_id IN (71,72);
DELETE FROM campaign_operations_request_production_admission
 WHERE operational_request_id IN (71,72);
UPDATE campaign_operations_operational_request
   SET request_state='ready',state_version=3,lease_token_hash=NULL,
       lease_expires_at=NULL,dispatcher_identity=NULL,
       production_dispatch_enabled=false
 WHERE operational_request_id IN (71,72);
SET session_replication_role = origin;
SQL
  local build_sql="campaign_operations_manager_build_canonical_v1(
    'campaign-operations-production-dispatch-and-manager-run-once-v1',
    repeat('a',40),'Apple clang 21.0.0','sha256:'||repeat('b',64))"
  PGAPPNAME="$first_app" psql "${target[@]}" -qAt -v ON_ERROR_STOP=1 \
    -v VERBOSITY=verbose "$probe_db" -c "BEGIN; SELECT
      transition_campaign_operations_request_dispatch_production_v2(
        ${first_request},3,'fnv1a64:0123456789abc001',
        transaction_timestamp()+interval '10 minutes',
        'h1-${lock_id}-first','manager@example.test',${build_sql});
      SELECT pg_sleep(4); COMMIT" \
    >"$cluster_root/${lock_id}-first.log" 2>&1 & local first_job=$!
  local first_pid=""
  for _ in $(seq 1 300); do
    first_pid="$(psql "${target[@]}" -At "$probe_db" -c \
      "SELECT pid FROM pg_stat_activity WHERE application_name='${first_app}'
       AND wait_event='PgSleep' LIMIT 1")"
    [[ -n "$first_pid" ]] && break
    sleep 0.02
  done
  [[ -n "$first_pid" ]] || { cat "$cluster_root/${lock_id}-first.log" >&2; exit 1; }
  PGAPPNAME="$second_app" psql "${target[@]}" -qAt -v ON_ERROR_STOP=1 \
    -v VERBOSITY=verbose "$probe_db" -c "BEGIN; SELECT
      transition_campaign_operations_request_dispatch_production_v2(
        ${second_request},3,'fnv1a64:0123456789abc002',
        transaction_timestamp()+interval '10 minutes',
        'h1-${lock_id}-second','manager@example.test',${build_sql});
      SELECT pg_sleep(4); COMMIT" \
    >"$cluster_root/${lock_id}-second.log" 2>&1 & local second_job=$!
  local second_pid="" blocked="f"
  for _ in $(seq 1 300); do
    second_pid="$(psql "${target[@]}" -At "$probe_db" -c \
      "SELECT pid FROM pg_stat_activity WHERE application_name='${second_app}' LIMIT 1")"
    if [[ -n "$second_pid" ]]; then
      blocked="$(psql "${target[@]}" -At "$probe_db" -c \
        "SELECT ${first_pid}=ANY(pg_blocking_pids(${second_pid}))")"
      if [[ "$expect_blocking" == true && "$blocked" == t ]]; then break; fi
      if [[ "$expect_blocking" == false ]] &&
         [[ "$(psql "${target[@]}" -At "$probe_db" -c \
           "SELECT wait_event='PgSleep' FROM pg_stat_activity WHERE pid=${second_pid}")" == t ]]; then break; fi
    fi
    sleep 0.02
  done
  if [[ "$expect_blocking" == true ]]; then
    [[ "$blocked" == t ]] || { echo "$lock_id did not serialize" >&2; exit 1; }
  else
    [[ "$blocked" == f ]] || { echo "$lock_id unexpectedly contended" >&2; exit 1; }
    [[ "$(psql "${target[@]}" -At "$probe_db" -c \
      "SELECT cardinality(pg_blocking_pids(${first_pid}))=0 AND
              cardinality(pg_blocking_pids(${second_pid}))=0")" == t ]] || {
        echo "$lock_id unrelated acquisitions had a wait edge" >&2; exit 1;
      }
  fi
  psql "${target[@]}" -X -qAt -F $'\t' "$probe_db" -c \
    "SELECT '${lock_id}',activity.application_name,locks.pid,locks.locktype,
       coalesce(locks.relation::regclass::text,''),locks.mode,locks.granted,
       coalesce(locks.classid::text,''),coalesce(locks.objid::text,''),
       array_to_string(pg_blocking_pids(locks.pid),','),activity.state,
       coalesce(activity.wait_event_type,''),coalesce(activity.wait_event,''),
       coalesce(activity.backend_xid::text,''),
       coalesce(locks.transactionid::text,'')
     FROM pg_locks locks JOIN pg_stat_activity activity USING(pid)
     WHERE locks.pid IN (${first_pid},${second_pid})
       AND locks.locktype IN ('relation','tuple','transactionid','advisory')
     ORDER BY activity.application_name,locks.granted DESC,locks.locktype,
              locks.relation,locks.classid,locks.objid" \
    >>"$cluster_root/h1-lock-catalog-evidence.tsv"
  local first_status=0 second_status=0
  wait "$first_job" || first_status=$?
  wait "$second_job" || second_status=$?
  [[ "$first_status" == 0 ]] || { cat "$cluster_root/${lock_id}-first.log" >&2; exit 1; }
  if [[ "$first_request" == "$second_request" ]]; then
    [[ "$second_status" != 0 ]] && rg -q '40001:' "$cluster_root/${lock_id}-second.log" || {
      cat "$cluster_root/${lock_id}-second.log" >&2; exit 1;
    }
  else
    [[ "$second_status" == 0 ]] || { cat "$cluster_root/${lock_id}-second.log" >&2; exit 1; }
    first_returned_identity="$(head -n 1 "$cluster_root/${lock_id}-first.log")"
    second_returned_identity="$(head -n 1 "$cluster_root/${lock_id}-second.log")"
    [[ -n "$first_returned_identity" && -n "$second_returned_identity" &&
        "$first_returned_identity" != "$second_returned_identity" ]] || {
      echo "$lock_id unrelated acquisitions did not return distinct evidence" >&2; exit 1;
    }
  fi
  [[ "$(psql "${target[@]}" -At "$probe_db" -c \
    "SELECT count(*) FROM campaign_operations_request_production_admission")" == "$expected_evidence" ]] || {
      echo "$lock_id evidence cardinality mismatch" >&2; exit 1;
    }
  local second_outcome=committed
  [[ "$second_status" != 0 ]] && second_outcome=expected-rejection
  psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$probe_db" <<'SQL'
SET session_replication_role = replica;
DELETE FROM campaign_operations_dispatch_audit_reference_event
 WHERE operational_request_id IN (71,72);
DELETE FROM campaign_operations_dispatch_attempt
 WHERE operational_request_id IN (71,72);
DELETE FROM campaign_operations_request_production_admission
 WHERE operational_request_id IN (71,72);
UPDATE campaign_operations_operational_request
   SET request_state='ready',state_version=3,lease_token_hash=NULL,
       lease_expires_at=NULL,dispatcher_identity=NULL,
       production_dispatch_enabled=false
 WHERE operational_request_id IN (71,72);
SET session_replication_role = origin;
SQL
  local rollback_app="h1_${lock_id}_rollback_acquisition_${$}"
  PGAPPNAME="$rollback_app" psql "${target[@]}" -qAt -v ON_ERROR_STOP=1 \
    "$probe_db" -c "BEGIN; SELECT
      transition_campaign_operations_request_dispatch_production_v2(
        ${first_request},3,'fnv1a64:0123456789abc003',
        transaction_timestamp()+interval '10 minutes',
        'h1-${lock_id}-rollback','manager@example.test',${build_sql});
      ROLLBACK" >"$cluster_root/${lock_id}-rollback.log"
  release_verified="$(psql "${target[@]}" -At "$probe_db" -c \
    "SELECT NOT EXISTS(SELECT 1 FROM campaign_operations_request_production_admission
       WHERE dispatch_operation_key='h1-${lock_id}-rollback') AND
            NOT EXISTS(SELECT 1 FROM campaign_operations_dispatch_attempt
       WHERE operation_key='h1-${lock_id}-rollback') AND
            EXISTS(SELECT 1 FROM campaign_operations_operational_request
       WHERE operational_request_id=${first_request} AND request_state='ready'
         AND state_version=3 AND lease_token_hash IS NULL) AND
            NOT EXISTS(SELECT 1 FROM pg_stat_activity
       WHERE application_name='${rollback_app}');")"
  [[ "$release_verified" == t ]] || {
    echo "$lock_id rollback cleanup failed" >&2; exit 1;
  }
  partial_evidence_count="$(psql "${target[@]}" -At "$probe_db" -c \
    "SELECT count(*) FROM campaign_operations_request_production_admission a
       LEFT JOIN campaign_operations_dispatch_attempt d
         ON d.request_production_admission_id=a.request_production_admission_id
      WHERE d.dispatch_attempt_id IS NULL")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$lock_id" "$first_pid" "$second_pid" "$first_app" "$second_app" \
    committed "$second_outcome" true "$partial_evidence_count" PASS \
    partial-admission-query-v1 backend-release-query-v1 \
    >> "$cluster_root/h1-lock-special-pids.tsv"
  dropdb "${target[@]}" "$probe_db"
}

run_complete_acquisition_pair H1LOCK011 71 71 true 1
run_complete_acquisition_pair H1LOCK016 71 72 false 2

lock_runtime="$cluster_root/h1-lock-runtime.tsv"
python3 "$repo_root/Scripts/CampaignOperationsH1LockEvidence.py" generate \
  "$repo_root/Tests/fixtures/CampaignOperationsH1LockPathMatrix.tsv" \
  "$cluster_root/h1-lock-catalog-evidence.tsv" \
  "$cluster_root/h1-lock-special-pids.tsv" "$lock_runtime" \
  "$cluster_root/raw-lock" "$runtime_run_id"

while IFS=$'\t' read -r lock_id pair _ _ _ _ permitted _ _; do
  [[ "$lock_id" == test_id ]] && continue
  emit_runtime_result "H1-LOCK-${lock_id#H1LOCK}" "$lock_id" \
    workflow-lock-probes SUCCESS 00000 no-deadlock-cycle "$pair" concurrency \
    "$lock_id" h1-lock-runtime.tsv "$lock_runtime" "$permitted" false GEN-LOCK
done < "$repo_root/Tests/fixtures/CampaignOperationsH1LockPathMatrix.tsv"
"$repo_root/Tests/CampaignOperationsPhaseH1LockArtifactTests.sh" \
  "$lock_runtime" "$cluster_root" "$runtime_run_id"
echo "Campaign Operations Phase H1 deterministic lock-order tests passed"

repository_test="$(mktemp -t CampaignOperationsPhaseH1RepositoryTests)"
trap 'rm -f "$repository_test"; cleanup' EXIT
clang++ -std=c++20 -I "$repo_root/Sources" -I "$repo_root/Headers" \
  -I /opt/homebrew/opt/libpqxx@7.10.1/include \
  -I /opt/homebrew/opt/libpq/include \
  "$repo_root/Tests/CampaignOperationsPhaseH1RepositoryTests.cpp" \
  "$repo_root/Sources/CampaignOperations.cpp" \
  "$repo_root/Sources/CampaignOperationsDispatch.cpp" \
  "$repo_root/Sources/CampaignOperationsProductionAdmission.cpp" \
  "$repo_root/Sources/CampaignOperationsProductionAdmissionRepository.cpp" \
  "$repo_root/Sources/CampaignOperationsProductionAdmissionService.cpp" \
  "$repo_root/Sources/ExperimentRecommendation.cpp" \
  -L /opt/homebrew/opt/libpqxx@7.10.1/lib \
  -L /opt/homebrew/opt/libpq/lib -lpqxx -lpq -o "$repository_test"
"$repository_test" \
  "host=$cluster_socket port=5432 dbname=$database user=campaign_manager_login" | \
  tee "$cluster_root/phase-h1-repository-results"
echo "Campaign Operations Phase H1 repository/service tests passed"
if [[ ! -s "$cluster_root/phase-h1-repository-results" ]]; then
  printf 'Campaign Operations Phase H1 repository/service tests passed\n' \
    > "$cluster_root/phase-h1-repository-results"
fi
emit_runtime_result H1-HYDRATION H1CPP001 repository-hydration EXPECTED_FAILURE \
  23514 hydration-corrupt Attempt-V2-and-enablement repository H1CPP001 \
  phase-h1-repository-results "$cluster_root/phase-h1-repository-results"

broad_repository_test="$(mktemp -t CampaignOperationsRepositoryTestsH1)"
createdb "${target[@]}" -T template0 -O campaign_manager_login "$broad_database"
clang++ -std=c++20 -Wall -Wextra -Werror \
  -I "$repo_root/Sources" -I "$repo_root/Headers" \
  -I /opt/homebrew/opt/libpqxx@7.10.1/include \
  -I /opt/homebrew/opt/libpq/include \
  "$repo_root/Tests/CampaignOperationsRepositoryTests.cpp" \
  "$repo_root/Sources/CampaignOperations.cpp" \
  "$repo_root/Sources/CampaignOperationsDispatch.cpp" \
  "$repo_root/Sources/CampaignOperationsControl.cpp" \
  "$repo_root/Sources/CampaignOperationsCompletion.cpp" \
  "$repo_root/Sources/CampaignOperationsProductionAdmission.cpp" \
  "$repo_root/Sources/CampaignOperationsRepository.cpp" \
  "$repo_root/Sources/CampaignOperationsService.cpp" \
  "$repo_root/Sources/CampaignOperationsDispatchRepository.cpp" \
  "$repo_root/Sources/CampaignOperationsManager.cpp" \
  "$repo_root/Sources/CampaignOperationsDispatchService.cpp" \
  "$repo_root/Sources/CampaignOperationsBindingRepository.cpp" \
  "$repo_root/Sources/CampaignOperationsControlRepository.cpp" \
  "$repo_root/Sources/CampaignOperationsControlService.cpp" \
  "$repo_root/Sources/CampaignOperationsCompletionRepository.cpp" \
  "$repo_root/Sources/CampaignOperationsCompletionService.cpp" \
  "$repo_root/Sources/CampaignOperationsProductionAdmissionRepository.cpp" \
  "$repo_root/Sources/CampaignOperationsProductionAdmissionService.cpp" \
  "$repo_root/Sources/ExperimentRecommendation.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCampaignActivation.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCampaignActivationRepository.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCampaignExecution.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCampaignExecutionRepository.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCampaignHandoff.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCampaignHandoffRepository.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCampaignLaunch.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCampaignLaunchRepository.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionExecutionRepository.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionActivation.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionActivationRepository.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionProposalReview.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionProposalReviewRepository.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCandidateGenerator.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionRepository.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionWorkflow.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionWorkflowRepository.cpp" \
  -L /opt/homebrew/opt/libpqxx@7.10.1/lib \
  -L /opt/homebrew/opt/libpq/lib -lpqxx -lpq \
  -o "$broad_repository_test"
LSTM_TEST_DB_NAME="$broad_database" \
LSTM_TEST_DB_HOST="$cluster_socket" \
LSTM_TEST_DB_PORT=5432 \
LSTM_TEST_DB_ADMIN_USER=campaign_manager_login \
  "$broad_repository_test"
echo "Campaign Operations Phase 1-5 repository/service/completion regression passed"

if rg -q \
  -- '--campaign-operations-manager-run-once|--campaign-operations-production-continuous' \
  "$repo_root/Sources"; then
  echo "H3/H4 production command leaked into H2 C++" >&2
  exit 1
fi
{
  printf 'command=rg scanned_paths=Sources patterns=H3,H4 result_rows=0 exit_status=1 scan_result=PASS\n'
  printf 'query=--campaign-operations-manager-run-once|--campaign-operations-production-continuous\n'
  rg --files "$repo_root/Sources" | sed "s#^$repo_root/##" | sort
} > "$cluster_root/phase-h1-cli-results"
  emit_runtime_result H1-H2-H4-EXCLUSION H1CPP002 CLI-and-source-policy SUCCESS \
  00000 no-H2-H3-H4-surface CLI-and-scheduler compile H1CPP002 \
  phase-h1-cli-results "$cluster_root/phase-h1-cli-results"

pre_runtime="$cluster_root/h1-pre-enablement-runtime.tsv"
printf 'version\trun_id\tevidence_id\trequirement_id\tproduction_surface\texecutable_workflow\tarchitectural_reason\tdisposition\tcleanup_result\n' \
  > "$pre_runtime"
while IFS=$'\t' read -r evidence requirement _ surface executable reason _ _ disposition; do
  [[ "$evidence" == evidence_id ]] && continue
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tPASS\n' \
    h1-pre-enablement-runtime-v1 "$runtime_run_id" "$evidence" "$requirement" \
    "$surface" "$executable" "$reason" "$disposition" >> "$pre_runtime"
done < "$repo_root/Tests/fixtures/CampaignOperationsH1PreEnablementEvidence.tsv"
mkdir -p "$runtime_artifacts/pre-enablement"
for pre_fixture in H1PRE001 H1PRE002; do
  awk -F '\t' -v fixture="$pre_fixture" 'NR==1 || $3==fixture' "$pre_runtime" \
    > "$runtime_artifacts/pre-enablement/${pre_fixture}.tsv"
done

invariant_runtime="$cluster_root/h1-uniqueness-invariant-runtime.tsv"
invariant_definition="$(psql "${target[@]}" -At "$database" -c \
  "SELECT pg_get_constraintdef(oid) FROM pg_constraint
    WHERE conrelid='public.campaign_operations_operational_request'::regclass
      AND conname='campaign_operations_request_operation_uidx' AND contype='u'")"
[[ "$invariant_definition" == \
   'UNIQUE (operational_campaign_id, action_kind, action_contract_version)' ]] || {
  echo "H1 uniqueness invariant catalog definition mismatch" >&2; exit 1;
}
printf 'version\trun_id\tinvariant_id\trequirement_id\tcatalog_object\tuniqueness_expression\tmigration_source\tdisposition\tcleanup_result\n' \
  > "$invariant_runtime"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tPASS\n' \
  h1-uniqueness-invariant-runtime-v1 "$runtime_run_id" H1INV001 H1-LOCK-012 \
  pg_constraint:public.campaign_operations_operational_request:campaign_operations_request_operation_uidx \
  "$invariant_definition" Database/migrations/047_campaign_operations_budget_request_acceptance.sql \
  accepted_non_executable_invariant >> "$invariant_runtime"
mkdir -p "$runtime_artifacts/invariants"
cp "$invariant_runtime" "$runtime_artifacts/invariants/H1INV001.tsv"

  acl_origin_runtime="$cluster_root/h1-acl-origin-runtime.tsv"
H1_EVIDENCE_RUN_ID="$runtime_run_id" \
"$repo_root/Tests/CampaignOperationsPhaseH1AclOriginTests.sh" \
  "$cluster_socket" 5432 campaign_manager_login "$database" "$acl_origin_runtime"
"$repo_root/Tests/CampaignOperationsPhaseH1AclOriginArtifactTests.sh" \
  "$acl_origin_runtime" "$cluster_root" \
  "$(awk -F '\t' 'NR==2{print $2}' "$acl_origin_runtime")"

"$repo_root/Tests/CampaignOperationsPhaseH1PreEnablementArtifactTests.sh" \
  "$pre_runtime" "$runtime_run_id"
"$repo_root/Tests/CampaignOperationsPhaseH1UniquenessInvariantTests.sh" \
  "$invariant_runtime" "$runtime_run_id"

# Convert every available production observation into a trusted generator
# execution, receipt, class-specific v2 raw envelope, and output snapshot.  The
# ACL/default rows already present were produced by the direct catalog query
# generator and are retained byte-for-byte by this reconciliation.
python3 "$repo_root/Scripts/CampaignOperationsH1TrustedEvidencePipeline.py" \
  "$cluster_root" "$runtime_run_id"

# Reconciliation operates on a dedicated evidence root.  Database files,
# transient compiler outputs, and pre-copy source logs never enter this root,
# so the recursive scanner can reject every unregistered file without broad
# filename or directory conventions.
evidence_root="$cluster_root/evidence-root"
mkdir -p "$evidence_root/runtime-artifacts"
cp "$runtime_results" "$cluster_root/run-id" "$lock_runtime" \
  "$restore_runtime" "$acl_origin_runtime" "$pre_runtime" \
  "$invariant_runtime" \
  "$cluster_root/h1-lock-catalog-evidence.tsv" \
  "$cluster_root/h1-lock-special-pids.tsv" \
  "$cluster_root/h1-acl-requirement-evidence.tsv" \
  "$cluster_root/h1-acl-catalog-observed.tsv" \
  "$cluster_root/h1-acl-catalog-payload.json" \
  "$cluster_root/h1-generator-execution-receipts.tsv" \
  "$cluster_root/h1-raw-evidence-envelopes.tsv" \
  "$cluster_root/h1-generator-output-snapshots.tsv" \
  "$evidence_root/"
cp -R "$runtime_artifacts"/. "$evidence_root/runtime-artifacts/"
cp -R "$cluster_root/raw-lock" "$cluster_root/raw-acl-origin" "$evidence_root/"

"$repo_root/Tests/CampaignOperationsPhaseH1TraceabilityTests.sh" \
  --validate "$evidence_root" "$runtime_run_id" | \
  tee "$evidence_root/h1-traceability-validation.log"
H1_ALLOW_PARTIAL_FINAL=1 "$repo_root/Tests/CampaignOperationsPhaseH1ReferenceGraphTests.sh" \
  "$evidence_root" "$runtime_run_id" | \
  tee "$evidence_root/h1-reference-graph.log"
# The graph-validation receipt is itself a registered artifact.  Reconcile
# once more after tee has finalized those bytes, then perform the byte-exact
# freshness validation against that stable input.
H1_ALLOW_PARTIAL_FINAL=1 "$repo_root/Tests/CampaignOperationsPhaseH1ReferenceGraphTests.sh" \
  "$evidence_root" "$runtime_run_id" >/dev/null
H1_ALLOW_PARTIAL_FINAL=1 "$repo_root/Tests/CampaignOperationsPhaseH1ReferenceGraphTests.sh" \
  "$evidence_root" "$runtime_run_id" \
  "$evidence_root/CampaignOperationsH1Traceability.md" >/dev/null
if [[ -n "${H1_EVIDENCE_CAPTURE_DIR:-}" ]]; then
  cp -R "$evidence_root"/. "$H1_EVIDENCE_CAPTURE_DIR/"
fi
