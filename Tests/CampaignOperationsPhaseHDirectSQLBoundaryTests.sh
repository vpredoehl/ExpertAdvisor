#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="$(mktemp -d /tmp/ea-h8-boundary.XXXXXX)"
cluster_root=""
cleanup() {
  if [[ -n "$cluster_root" && -f "$cluster_root/data/postmaster.pid" ]]; then
    pg_ctl -D "$cluster_root/data" -m immediate stop >/dev/null 2>&1 || true
  fi
  [[ -z "$cluster_root" || ! -d "$cluster_root" ]] || rm -rf -- "$cluster_root"
  rm -rf -- "$tmp_root"
}
trap cleanup EXIT

# The existing H2 disposable workflow proves the approved explicit-dispatch
# and H3-compatible production path, including atomic replay/bind evidence.
preserved="$tmp_root/preserved.txt"
H2_PRESERVE_AFTER_WORKFLOW_ROOT="$preserved" \
  H2_PRESERVE_CLUSTER_ROOT="$tmp_root/cluster-root" \
  bash "$repo_root/Tests/CampaignOperationsPhaseH2WorkflowTests.sh" \
  >"$tmp_root/workflow.log" 2>&1
rg -q 'H2_AFTER_WORKFLOW_QUIESCENT' "$tmp_root/workflow.log"
IFS='|' read -r cluster_root database cluster_socket < "$preserved"
target=(-h "$cluster_socket" -p 5432)

manager_sql() {
  psql -X -qAt -v ON_ERROR_STOP=1 "${target[@]}" -U h2_manager_login "$database" -c "$1"
}
dispatch_service_sql() {
  psql -X -qAt -v ON_ERROR_STOP=1 "${target[@]}" -U h2_dispatch_service_login "$database" -c "$1"
}
enabler_sql() {
  psql -X -qAt -v ON_ERROR_STOP=1 "${target[@]}" -U h2_enabler_login "$database" -c "$1"
}
admin_sql() {
  psql -X -qAt -v ON_ERROR_STOP=1 "${target[@]}" -U campaign_manager_login "$database" -c "$1"
}
admin_sql_db() {
  psql -X -qAt -v ON_ERROR_STOP=1 "${target[@]}" -U campaign_manager_login "$1" \
    -c "SET session_replication_role=replica; $2"
}
disabler_sql_db() {
  psql -X -qAt -v ON_ERROR_STOP=1 "${target[@]}" -U h2_disabler_login "$1" -c "$2"
}
expect_denied() {
  local user="$1" sql="$2" marker="$3"
  if psql -X -qAt -v ON_ERROR_STOP=1 "${target[@]}" -U "$user" "$database" \
      -c "$sql" >"$tmp_root/$marker.log" 2>&1; then
    echo "H8 boundary unexpectedly allowed $marker" >&2
    exit 1
  fi
  rg -q 'permission denied|42501|privilege' "$tmp_root/$marker.log"
  echo "H8_BOUNDARY $marker=PASS"
}

raw="transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamptz,text,text,text)"
authorized_name="campaign_operations_production_dispatch_authorized_v3"
authorized_sig="${authorized_name}(bigint,integer,text,timestamptz,text,text,text)"
expect_denied h2_manager_login \
  "SELECT public.transition_campaign_operations_request_dispatch_production_v2(71,3,'x',now(),'h8-raw','h8.direct','x');" \
  raw_manager_direct

admin_sql "CREATE ROLE h8_inherited_dispatcher NOLOGIN; GRANT campaign_operations_production_dispatcher TO h8_inherited_dispatcher; REVOKE campaign_operations_production_dispatcher FROM h2_manager_login; GRANT h8_inherited_dispatcher TO h2_manager_login;"
expect_denied h2_manager_login \
  "SELECT public.transition_campaign_operations_request_dispatch_production_v2(71,3,'x',now(),'h8-raw-inherited','h8.direct','x');" \
  raw_inherited_dispatcher
admin_sql "REVOKE h8_inherited_dispatcher FROM h2_manager_login; GRANT campaign_operations_production_dispatcher TO h2_manager_login; REVOKE campaign_operations_production_dispatcher FROM h8_inherited_dispatcher; DROP ROLE h8_inherited_dispatcher;"

expect_denied h2_manager_login \
  "SELECT public.$authorized_name(71,3,'x',now(),'h8-manager-direct','h8.direct','x');" \
  manager_wrapper_direct
expect_denied h2_manager_login \
  "SET ROLE campaign_operations_production_dispatch_service; SELECT 1;" \
  manager_cannot_set_role_service
expect_denied h2_dispatch_service_login \
  "SELECT public.transition_campaign_operations_request_dispatch_production_v2(71,3,'x',now(),'h8-raw-service','h8.direct','x');" \
  service_raw_direct

catalog="$(admin_sql "SELECT (NOT has_function_privilege('public','$raw','EXECUTE')) AND (NOT has_function_privilege('pqxx','$raw','EXECUTE')) AND has_function_privilege('campaign_operations_h1_boundary_authority','$raw','EXECUTE') AND (NOT has_function_privilege('h2_manager_login','$authorized_sig','EXECUTE')) AND (NOT has_function_privilege('campaign_operations_production_dispatcher','$authorized_sig','EXECUTE')) AND has_function_privilege('campaign_operations_production_dispatch_service','$authorized_sig','EXECUTE') AND (NOT has_function_privilege('public','$authorized_sig','EXECUTE')) AND (NOT has_function_privilege('pqxx','$authorized_sig','EXECUTE'));" )"
[[ "$catalog" == t ]]
echo 'H8_BOUNDARY catalog_acl=PASS public_raw=PASS pqxx_raw=PASS owner_raw=PASS manager_wrapper=DENIED dispatcher_wrapper=DENIED service_wrapper=PASS'

# Migration 059 is still under review and must be safe to replay before its
# production ledger entry is accepted.  Apply the exact file twice to a
# disposable template clone; the live fixture remains untouched.
replay_database="expertadvisor_h8_m059_replay_$$"
createdb "${target[@]}" -U campaign_manager_login -T "$database" "$replay_database"
psql -X -q -v ON_ERROR_STOP=1 "${target[@]}" -U campaign_manager_login \
  "$replay_database" -f "$repo_root/Database/migrations/059_campaign_operations_direct_sql_readiness_boundary.sql" >/dev/null
psql -X -q -v ON_ERROR_STOP=1 "${target[@]}" -U campaign_manager_login \
  "$replay_database" -f "$repo_root/Database/migrations/059_campaign_operations_direct_sql_readiness_boundary.sql" >/dev/null
dropdb "${target[@]}" -U campaign_manager_login --if-exists "$replay_database"
echo 'H8_BOUNDARY migration_059_replay_safe=PASS isolated_clone=PASS'

operation_key="$(admin_sql "SELECT operation_key FROM campaign_operations_dispatch_attempt WHERE operational_request_id=71 AND attempt_contract_version=2 ORDER BY attempt_ordinal LIMIT 1;")"
lease_digest="$(admin_sql "SELECT lease_token_digest FROM campaign_operations_dispatch_attempt WHERE operational_request_id=71 AND attempt_contract_version=2 ORDER BY attempt_ordinal LIMIT 1;")"
lease_expires_at="$(admin_sql "SELECT to_char(lease_expires_at AT TIME ZONE 'UTC','YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"') FROM campaign_operations_dispatch_attempt WHERE operational_request_id=71 AND attempt_contract_version=2 ORDER BY attempt_ordinal LIMIT 1;")"
prior_version="$(admin_sql "SELECT max(resulting_version) FROM campaign_operations_production_enablement_event;")"
enabler_sql "SELECT record_campaign_operations_production_enable_v1(
  'h8-readiness-enable', $prior_version, scheduler_protocol_evidence_canonical,
  independent_verification_reference, actor_identity,
  manager_service_contract, approved_build_contract_canonical,
  approved_build_source_commit, approved_build_compiler_contract,
  approved_build_executable_sha256, 'Restore readiness for direct-boundary regression.')
FROM campaign_operations_production_enablement_event
WHERE event_kind='enable' ORDER BY resulting_version LIMIT 1;" >/dev/null
build_contract="$(admin_sql "SELECT approved_build_contract_canonical FROM campaign_operations_production_enablement_event WHERE event_kind='enable' ORDER BY resulting_version DESC LIMIT 1;")"
dispatch_service_sql "SELECT public.$authorized_name(71,3,'$lease_digest','$lease_expires_at','$operation_key','h2.manager@example.test','$build_contract');" >/dev/null
echo 'H8_BOUNDARY approved_explicit_replay=PASS readiness_true=PASS'

expect_gate_denied() {
  local name="$1" mutation="$2"
  local clone="expertadvisor_h8_${name}_$$"
  createdb "${target[@]}" -U campaign_manager_login -T "$database" "$clone"
  admin_sql_db "$clone" "$mutation" >/dev/null
  if psql -X -qAt -v ON_ERROR_STOP=1 "${target[@]}" -U h2_dispatch_service_login "$clone" \
      -c "SELECT public.$authorized_name(71,3,'h8-blocked',now(),'$operation_key','h2.manager@example.test','$build_contract');" \
      >"$tmp_root/$name.log" 2>&1; then
    echo "H8 readiness blocker unexpectedly passed: $name" >&2
    exit 1
  fi
  echo "H8_BOUNDARY readiness_${name}=PASS"
  dropdb "${target[@]}" -U campaign_manager_login --if-exists "$clone"
}
expect_gate_denied_disable() {
  local name="$1"
  local clone="expertadvisor_h8_${name}_$$"
  createdb "${target[@]}" -U campaign_manager_login -T "$database" "$clone"
  disabler_sql_db "$clone" "SELECT record_campaign_operations_production_disable_v1(
    'h8-blocker-disable', production_enablement_event_id,
    enablement_identity_canonical, resulting_version,
    'h2.disabler@example.test', 'Disable the isolated readiness fixture.')
  FROM campaign_operations_production_enablement_event
  ORDER BY resulting_version DESC LIMIT 1;" >/dev/null
  if psql -X -qAt -v ON_ERROR_STOP=1 "${target[@]}" -U h2_dispatch_service_login "$clone" \
      -c "SELECT public.$authorized_name(71,3,'h8-blocked',now(),'$operation_key','h2.manager@example.test','$build_contract');" \
      >"$tmp_root/$name.log" 2>&1; then
    echo "H8 readiness blocker unexpectedly passed: $name" >&2
    exit 1
  fi
  echo "H8_BOUNDARY readiness_${name}=PASS"
  dropdb "${target[@]}" -U campaign_manager_login --if-exists "$clone"
}

# Each mutation is confined to a disposable clone and is performed by the
# administrative test principal; no live LSTM catalog or production evidence
# is changed.
if psql -X -qAt -v ON_ERROR_STOP=1 "${target[@]}" -U h2_dispatch_service_login "$database" \
    -c "SELECT public.$authorized_name(71,3,'h8-blocked',now(),'$operation_key','h8.manager@example.test','h8-build-mismatch');" \
    >"$tmp_root/build-mismatch.log" 2>&1; then
  echo 'H8 readiness build mismatch unexpectedly passed' >&2; exit 1
fi
echo 'H8_BOUNDARY readiness_build_mismatch=PASS'

expect_gate_denied scheduler_incomplete "UPDATE experiment_scheduler_protocol SET cutover_state='pending', cutover_completed_at=NULL, cutover_completed_by=NULL, cutover_executable_path=NULL, cutover_process_evidence=NULL WHERE singleton;"
expect_gate_denied_disable enablement_ineffective
expect_gate_denied canonical_corruption "UPDATE campaign_operations_production_enablement_event SET approved_build_contract_canonical='corrupt' WHERE event_kind='enable';"
expect_gate_denied completion_proof_invalid "SET session_replication_role=replica; UPDATE campaign_operations_dispatch_attempt SET request_production_admission_id=999999 WHERE dispatch_attempt_id=(SELECT max(dispatch_attempt_id) FROM campaign_operations_dispatch_attempt); SET session_replication_role=origin;"
expect_gate_denied reconciliation_blocker "INSERT INTO campaign_operations_reconciliation_cursor_event(run_key,prior_target_id,last_target_id,requested_limit,selected_count) VALUES('h8-blocker',0,71,100,1); INSERT INTO campaign_operations_reconciliation_observation(reconciliation_cursor_event_id,run_key,operational_campaign_id,operational_request_id,request_identity_canonical,expected_request_state,expected_request_version,reason_code,evidence_identity_canonical,evidence_identity_hash,recommended_service,recommended_action,diagnostic_code,observation_contract_version,observation_identity_canonical,observation_identity_hash) SELECT (SELECT max(reconciliation_cursor_event_id) FROM campaign_operations_reconciliation_cursor_event), 'h8-blocker',operational_campaign_id,operational_request_id,request_identity_canonical,'bound',state_version,'dispatch_outcome_unknown','h8-evidence','fnv1a64:0000000000000000','recovery','inspect','h8_blocker',1,'h8-observation','fnv1a64:0000000000000000' FROM campaign_operations_operational_request WHERE operational_request_id=71;"

echo 'H8_DIRECT_SQL_BOUNDARY_REGRESSION_OK raw_denied=PASS inherited_denied=PASS manager_wrapper_denied=PASS dispatcher_wrapper_denied=PASS service_path=PASS public_denied=PASS readiness_gate=PASS replay=PASS isolated_clones=PASS'
