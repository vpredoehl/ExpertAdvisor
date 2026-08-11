#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="$(mktemp -d /tmp/ea-h2-workflow.XXXXXX)"
preserve_marker="$tmp_root/h1-cluster-root"
cleanup() {
  if [[ -z "${H2_PRESERVE_CLUSTER_ROOT:-}" &&
        -n "${cluster_root:-}" && -f "$cluster_root/data/postmaster.pid" ]]; then
    pg_ctl -D "$cluster_root/data" -m immediate stop >/dev/null 2>&1 || true
  fi
  if [[ -n "${cluster_root:-}" && -d "$cluster_root" ]]; then
    if [[ -n "${H2_PRESERVE_CLUSTER_ROOT:-}" ]]; then
      printf '%s\n' "$cluster_root" > "$H2_PRESERVE_CLUSTER_ROOT"
    else
      rm -rf -- "$cluster_root"
    fi
  fi
  rm -rf -- "$tmp_root"
}
trap cleanup EXIT

# H1 owns the repository-native schema-054 fixture and 055 upgrade setup.
# Reuse its successfully reconciled disposable cluster before H2 workflow
# state is exercised.
set +e
H1_PRESERVE_CLUSTER_ROOT="$preserve_marker" \
  bash "$repo_root/Tests/CampaignOperationsPhaseH1MigrationTests.sh" \
  >"$tmp_root/h1-output.log" 2>&1
h1_status=$?
set -e
[[ "$h1_status" -eq 0 ]]
rg -q 'H1_ACL_CATALOG_V3_OK rows=41' "$tmp_root/h1-output.log"
[[ -s "$preserve_marker" ]]
cluster_root="$(sed -n '1p' "$preserve_marker")"
cluster_socket="$cluster_root/s"
target=(-h "$cluster_socket" -p 5432 -U campaign_manager_login)
database="$(psql "${target[@]}" -At postgres -c \
  "SELECT datname FROM pg_database WHERE datname ~ '^expertadvisor_campaign_operations_phase3_test_h1_[0-9]+$' ORDER BY datname DESC LIMIT 1")"
[[ -n "$database" ]]

echo "H2_FIXTURE schema=055 database=$database socket=$cluster_socket"
before_055="$(shasum -a 256 "$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql" | awk '{print $1}')"
[[ "$before_055" == 1b13d3a64336d7cbd55c935396ec42c4c06320105677829f0cf405733e5715fe ]]

# Make the predecessor ledger coherent for the real runner. The schema was
# restored through 049 and 050-055 were applied by the native H1 fixture.
for migration in "$repo_root"/Database/migrations/*.sql; do
  filename="$(basename "$migration")"
  version="${filename%%_*}"
  [[ "$version" != "$filename" ]] || version="${filename%.sql}"
  [[ "$version" != "056" && "$version" != "057" ]] || continue
  checksum="$(shasum -a 256 "$migration" | awk '{print $1}')"
  psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
    "INSERT INTO schema_migrations(version,filename,checksum)
       VALUES ('$version','$filename','$checksum')
       ON CONFLICT (version) DO UPDATE SET filename=EXCLUDED.filename,
       checksum=EXCLUDED.checksum;"
done

# Migration 056 is installed by the repository's actual runner.
LSTM_DB_HOST="$cluster_socket" \
LSTM_DB_NAME="$database" \
LSTM_DB_ADMIN_USER=campaign_manager_login \
  bash "$repo_root/migrate_lstm_db.sh" | tee "$tmp_root/migration-runner.log"
rg -q 'MIGRATION_APPLY,version=056,' "$tmp_root/migration-runner.log"
LSTM_DB_HOST="$cluster_socket" \
LSTM_DB_NAME="$database" \
LSTM_DB_ADMIN_USER=campaign_manager_login \
  bash "$repo_root/migrate_lstm_db.sh" | tee "$tmp_root/migration-replay.log"
rg -q 'MIGRATION_SKIP,version=056,' "$tmp_root/migration-replay.log"

after_055="$(shasum -a 256 "$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql" | awk '{print $1}')"
[[ "$before_055" == "$after_055" ]]
h1_sha="$(git -C "$repo_root" show HEAD:Database/migrations/055_campaign_operations_production_admission_foundation.sql | shasum -a 256 | awk '{print $1}')"
[[ "$after_055" == "$h1_sha" ]]
ledger_056="$(psql "${target[@]}" -At -v ON_ERROR_STOP=1 "$database" -c \
  "SELECT version||'|'||filename||'|'||checksum FROM schema_migrations WHERE version='056'")"
expected_056="056|$(basename "$repo_root"/Database/migrations/056_*.sql)|$(shasum -a 256 "$repo_root"/Database/migrations/056_*.sql | awk '{print $1}')"
[[ "$ledger_056" == "$expected_056" ]]
ledger_057="$(psql "${target[@]}" -At -v ON_ERROR_STOP=1 "$database" -c \
  "SELECT version||'|'||filename||'|'||checksum FROM schema_migrations WHERE version='057'")"
expected_057="057|$(basename "$repo_root"/Database/migrations/057_*.sql)|$(shasum -a 256 "$repo_root"/Database/migrations/057_*.sql | awk '{print $1}')"
[[ "$ledger_057" == "$expected_057" ]]
echo "H2_MIGRATION_RUNNER_OK ledger_055_to_057=PASS replay_noop=PASS 055_unchanged=PASS"

# Deployment-time membership is outside migrations. These synthetic LOGINs
# exactly match the accepted ADR-0019C identities.
# The executable conflict signature includes lifecycle evidence. Temporarily
# grant the existing owner role only inside this disposable workflow database;
# the grant is revoked immediately after the test executable.
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<'SQL'
SET session_replication_role = replica;
DELETE FROM campaign_operations_completion_audit_reference_event
 WHERE completion_event_id=7001;
DELETE FROM campaign_operations_completion_event WHERE completion_event_id=7001;
DELETE FROM campaign_operations_dispatch_attempt_outcome
 WHERE dispatch_attempt_id=7001;
DELETE FROM campaign_operations_dispatch_audit_reference_event
 WHERE dispatch_attempt_id=7001;
DELETE FROM campaign_operations_dispatch_attempt WHERE dispatch_attempt_id=7001;
UPDATE campaign_operations_operational_request
   SET request_state='ready', state_version=3, lease_token_hash=NULL,
       lease_expires_at=NULL, dispatcher_identity=NULL,
       production_dispatch_enabled=false
 WHERE operational_request_id=7;
SET session_replication_role = origin;
CREATE ROLE h2_enabler_login LOGIN;
CREATE ROLE h2_disabler_login LOGIN;
CREATE ROLE h2_manager_login LOGIN;
GRANT campaign_operations_production_enabler,
      campaign_operations_production_reader,
      campaign_operations_scheduler_protocol_evidence_reader
  TO h2_enabler_login;
GRANT campaign_operations_production_disabler,
      campaign_operations_production_reader
  TO h2_disabler_login;
GRANT campaign_operations_production_dispatcher,
      campaign_operations_production_phase5_transactional,
      campaign_operations_production_reader,
      campaign_operations_scheduler_protocol_evidence_reader
  TO h2_manager_login;
SQL

# Preserve the successful H1 enablement evidence as the first ledger event,
# then extend that exact canonical predecessor through the native disable
# transition.  H2's first enable therefore starts from the required disabled
# version-2 head instead of attempting a second genesis enable.
h1_enable_head="$(psql "${target[@]}" -At -v ON_ERROR_STOP=1 "$database" -c \
  "SELECT production_enablement_event_id||'|'||event_kind||'|'||resulting_version
     FROM campaign_operations_production_enablement_event
    WHERE production_enablement_event_id=1")"
[[ "$h1_enable_head" == "1|enable|1" ]]
echo "H2_H1_ENABLE_HEAD event_id=1 kind=enable resulting_version=1 preserved=PASS"

psql -h "$cluster_socket" -p 5432 -U h2_disabler_login -q -v ON_ERROR_STOP=1 \
  "$database" -c "SELECT record_campaign_operations_production_disable_v1(
    'h2-fixture-bridge-disable-v2',
    predecessor.production_enablement_event_id,
    predecessor.enablement_identity_canonical,
    predecessor.resulting_version,
    'h2.disabler@example.test',
    'Bridge the preserved H1 enablement head for H2 workflow.')
    FROM campaign_operations_production_enablement_event predecessor
   WHERE predecessor.production_enablement_event_id=1
     AND predecessor.event_kind='enable'
     AND predecessor.resulting_version=1;"
bridge_disable_head="$(psql "${target[@]}" -At -v ON_ERROR_STOP=1 "$database" -c \
  "SELECT production_enablement_event_id||'|'||event_kind||'|'||resulting_version||'|'||predecessor_event_id
     FROM campaign_operations_production_enablement_event
    ORDER BY resulting_version DESC LIMIT 1")"
[[ "$bridge_disable_head" == "2|disable|2|1" ]]
echo "H2_BRIDGE_DISABLE native=record_campaign_operations_production_disable_v1 predecessor_event=1 resulting_version=2 head=disable preserved_h1=PASS"

"$repo_root/Scripts/CampaignOperationsH2DeploymentAudit.sh" \
  --stage post-upgrade --host "$cluster_socket" --port 5432 \
  --user campaign_manager_login --database "$database" | tee "$tmp_root/h2-audit.log"
"$repo_root/Tests/CampaignOperationsPhaseH2PrivilegeDeploymentTests.sh" \
  --stage post-upgrade --host "$cluster_socket" --port 5432 \
  --user campaign_manager_login --database "$database" | tee "$tmp_root/h2-privilege.log"

# Recursive deployment-role hostile fixtures.  The clean audit above is the
# allowed H2 graph control.  These mutations are cluster-audit fixtures only;
# each is removed before the restored clean audit.
expect_recursive_audit_failure() {
  local marker="$1"
  local output="$tmp_root/${marker}.log"
  if "$repo_root/Scripts/CampaignOperationsH2DeploymentAudit.sh" \
      --stage post-upgrade --host "$cluster_socket" --port 5432 \
      --user campaign_manager_login --database "$database" \
      >"$output" 2>&1; then
    cat "$output" >&2
    echo "recursive role fixture unexpectedly passed: $marker" >&2
    exit 1
  fi
  rg -q 'H2A006|forbidden-effective-role|sealed-graph|capability-graph' "$output" || {
    cat "$output" >&2
    exit 1
  }
  echo "H2_ROLE_GRAPH_HOSTILE $marker=PASS"
}

hostile_a="h2_hostile_${$}_a"
hostile_b="h2_hostile_${$}_b"
hostile_c="h2_hostile_${$}_c"
hostile_d="h2_hostile_${$}_d"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<SQL
CREATE ROLE $hostile_a NOLOGIN;
GRANT pqxx TO $hostile_a;
GRANT $hostile_a TO h2_manager_login;
SQL
expect_recursive_audit_failure manager_intermediary_pqxx
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
  "REVOKE $hostile_a FROM h2_manager_login; REVOKE pqxx FROM $hostile_a; DROP ROLE $hostile_a;"

psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<SQL
CREATE ROLE $hostile_b NOLOGIN;
CREATE ROLE $hostile_c NOLOGIN;
GRANT pqxx TO $hostile_c;
GRANT $hostile_c TO $hostile_b;
GRANT $hostile_b TO h2_manager_login;
SQL
expect_recursive_audit_failure manager_two_intermediaries_pqxx
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
  "REVOKE $hostile_b FROM h2_manager_login; REVOKE $hostile_c FROM $hostile_b; REVOKE pqxx FROM $hostile_c; DROP ROLE $hostile_b, $hostile_c;"

psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<SQL
CREATE ROLE $hostile_d NOLOGIN;
GRANT campaign_operations_h1_boundary_authority TO $hostile_d;
GRANT $hostile_d TO h2_manager_login;
SQL
expect_recursive_audit_failure manager_intermediary_sealed_owner
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
  "REVOKE $hostile_d FROM h2_manager_login; REVOKE campaign_operations_h1_boundary_authority FROM $hostile_d; DROP ROLE $hostile_d;"

psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
  "GRANT pqxx TO h2_manager_login;"
expect_recursive_audit_failure manager_direct_pqxx
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
  "REVOKE pqxx FROM h2_manager_login;"

unrelated="h2_hostile_${$}_unrelated"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<SQL
CREATE ROLE $unrelated NOLOGIN;
GRANT pqxx TO $unrelated;
GRANT $unrelated TO h2_manager_login;
SQL
expect_recursive_audit_failure unrelated_intermediary_pqxx
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
  "REVOKE $unrelated FROM h2_manager_login; REVOKE pqxx FROM $unrelated; DROP ROLE $unrelated;"

"$repo_root/Scripts/CampaignOperationsH2DeploymentAudit.sh" \
  --stage post-upgrade --host "$cluster_socket" --port 5432 \
  --user campaign_manager_login --database "$database" | tee "$tmp_root/h2-audit-restored.log"
echo "H2_ROLE_GRAPH_RESTORED_CLEAN=PASS"

# Reuse the repository-native Phase E fixture builder for a valid, disposable
# request/materialization graph.  The builder inserts only into this cluster.
workflow_lock_test="$tmp_root/CampaignOperationsPhaseH1WorkflowLockTests"
workflow_lock_adapters="$tmp_root/CampaignOperationsPhaseH1WorkflowAdapters.o"
clang++ -std=c++20 -Wall -Wextra -Werror -Dmain=H1WorkflowAdapterUnusedMain \
  -I "$repo_root/Sources" -I "$repo_root/Headers" \
  -I /opt/homebrew/opt/libpqxx@7.10.1/include -I /opt/homebrew/opt/libpq/include -c \
  "$repo_root/Tests/CampaignOperationsRepositoryTests.cpp" \
  -o "$workflow_lock_adapters"
clang++ -std=c++20 -Wall -Wextra -Werror \
  -I "$repo_root/Sources" -I "$repo_root/Headers" \
  -I /opt/homebrew/opt/libpqxx@7.10.1/include -I /opt/homebrew/opt/libpq/include \
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
  "$repo_root/Sources/ExperimentRecommendationConversion.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionRepository.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionProposalReview.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionProposalReviewRepository.cpp" \
  "$repo_root/Sources/ExperimentRecommendationReview.cpp" \
  "$repo_root/Sources/ExperimentRecommendationRanking.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCampaignExecution.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCampaignExecutionRepository.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCampaignLaunch.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCampaignLaunchRepository.cpp" \
  "$repo_root/Sources/ExperimentRecommendationCandidateGenerator.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionWorkflow.cpp" \
  "$repo_root/Sources/ExperimentRecommendationConversionWorkflowRepository.cpp" \
  -L /opt/homebrew/opt/libpqxx@7.10.1/lib -L /opt/homebrew/opt/libpq/lib \
  -lpqxx -lpq -o "$workflow_lock_test"
"$workflow_lock_test" \
  "host=$cluster_socket port=5432 dbname=$database user=campaign_manager_login" \
  phase-e-fixture-setup | tee "$tmp_root/h2-phase-e-fixture.log"
rg -q 'workflow=phase-e-fixture-setup,outcome=completed' "$tmp_root/h2-phase-e-fixture.log"
rg -q 'H2_PHASE_E_FIXTURE_ASSERTIONS campaign_id=71 .*proposal=1 approved_review=1 execution=0 activation=0 materialization_member=1' \
  "$tmp_root/h2-phase-e-fixture.log"

if [[ -n "${H2_PRESERVE_BASELINE_ROOT:-}" ]]; then
  active_sessions="$(psql "${target[@]}" -At postgres -c \
    "SELECT pid||'|'||application_name||'|'||state
       FROM pg_stat_activity
      WHERE datname='${database}' AND pid <> pg_backend_pid()
      ORDER BY pid;" || true)"
  [[ -z "$active_sessions" ]] || {
    echo "H2_BASELINE_NOT_QUIESCENT sessions=$active_sessions" >&2
    exit 1
  }
  printf '%s|%s|%s\n' "$cluster_root" "$database" "$cluster_socket" \
    > "$H2_PRESERVE_BASELINE_ROOT"
  echo "H2_BASELINE_QUIESCENT database=$database socket=$cluster_socket sessions=0"
  exit 0
fi

connection_for() {
  local user="$1" label="$2"
  printf 'host=%s port=5432 dbname=%s user=%s application_name=h2-%s' \
    "$cluster_socket" "$database" "$user" "$label"
}
enabler_connection="$(connection_for h2_enabler_login enabler)"
disabler_connection="$(connection_for h2_disabler_login disabler)"
manager_connection="$(connection_for h2_manager_login manager)"
workflow_test="$tmp_root/CampaignOperationsPhaseH2WorkflowTests"
includes=(-I "$repo_root/Sources" -I "$repo_root/Headers"
  -I /opt/homebrew/opt/libpqxx@7.10.1/include -I /opt/homebrew/opt/libpq/include)
sources=(
  "$repo_root/Sources/CampaignOperations.cpp"
  "$repo_root/Sources/CampaignOperationsDispatch.cpp"
  "$repo_root/Sources/CampaignOperationsProductionAdmission.cpp"
  "$repo_root/Sources/CampaignOperationsRepository.cpp"
  "$repo_root/Sources/CampaignOperationsDispatchRepository.cpp"
  "$repo_root/Sources/CampaignOperationsManager.cpp"
  "$repo_root/Sources/CampaignOperationsDispatchService.cpp"
  "$repo_root/Sources/CampaignOperationsBindingRepository.cpp"
  "$repo_root/Sources/CampaignOperationsProductionAdmissionRepository.cpp"
  "$repo_root/Sources/CampaignOperationsProductionAdmissionService.cpp"
  "$repo_root/Sources/ExperimentRecommendation.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignActivation.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignActivationRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignMaterializationRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignHandoff.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignHandoffRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignFollowUpProposalRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationConversionExecutionRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationConversionActivation.cpp"
  "$repo_root/Sources/ExperimentRecommendationConversionActivationRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignExecution.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignExecutionRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignLaunch.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignLaunchRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationConversionProposalReview.cpp"
  "$repo_root/Sources/ExperimentRecommendationConversionProposalReviewRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationCandidateGenerator.cpp"
  "$repo_root/Sources/ExperimentRecommendationConversionRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationConversionWorkflow.cpp"
  "$repo_root/Sources/ExperimentRecommendationConversionWorkflowRepository.cpp")
# Add the owner capability only after the clean deployment audit and remove it
# immediately after the executable's read-only conflict-signature assertion.
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
  "GRANT campaign_operations_owner TO h2_manager_login;"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
  "GRANT SELECT ON campaign_operations_operational_request, campaign_operations_dispatch_attempt, campaign_operations_dispatch_attempt_outcome, campaign_operations_request_binding, campaign_operations_downstream_control_owner, campaign_operations_reservation_commitment, experiment_recommendation_conversion_execution, experiment_recommendation_conversion_activation, experiment, campaign_operations_production_enablement_event, experiment_lifecycle_cancellation_event TO campaign_operations_owner;"
clang++ -std=c++20 -Wall -Wextra -Werror -ffunction-sections -fdata-sections \
  -DCAMPAIGN_OPERATIONS_H2_TESTING \
  "${includes[@]}" "$repo_root/Tests/CampaignOperationsPhaseH2WorkflowTests.cpp" \
  "${sources[@]}" -L /opt/homebrew/opt/libpqxx@7.10.1/lib \
  -L /opt/homebrew/opt/libpq/lib -lpqxx -lpq -Wl,-dead_strip -o "$workflow_test"
"$workflow_test" "$enabler_connection" "$disabler_connection" \
  "$manager_connection" | tee "$tmp_root/h2-workflow.log"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
  "REVOKE campaign_operations_owner FROM h2_manager_login;"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
  "REVOKE SELECT ON campaign_operations_operational_request, campaign_operations_dispatch_attempt, campaign_operations_dispatch_attempt_outcome, campaign_operations_request_binding, campaign_operations_downstream_control_owner, campaign_operations_reservation_commitment, experiment_recommendation_conversion_execution, experiment_recommendation_conversion_activation, experiment, campaign_operations_production_enablement_event, experiment_lifecycle_cancellation_event FROM campaign_operations_owner;"
rg -q 'H2_WORKFLOW_CPP_OK enable=PASS disable=PASS acquisition=PASS handoff=PASS replay=PASS conflict=PASS' \
  "$tmp_root/h2-workflow.log"

count_query="SELECT
 (SELECT count(*) FROM campaign_operations_request_production_admission WHERE operational_request_id=71),
 (SELECT count(*) FROM campaign_operations_dispatch_attempt WHERE operational_request_id=71 AND attempt_contract_version=2),
 (SELECT count(*) FROM campaign_operations_dispatch_audit_reference_event WHERE operational_request_id=71 AND dispatch_attempt_id IS NOT NULL),
 (SELECT count(*) FROM campaign_operations_request_binding WHERE operational_request_id=71),
 (SELECT count(*) FROM experiment_recommendation_conversion_activation WHERE experiment_id IS NOT NULL AND recommendation_conversion_execution_id IS NOT NULL);"
counts="$(psql "${target[@]}" -At -v ON_ERROR_STOP=1 "$database" -c "$count_query")"
echo "H2_NO_DUPLICATE_COUNTS admission,attempt,audit,binding,activation=$counts"
state_contract="$(psql "${target[@]}" -At -v ON_ERROR_STOP=1 "$database" -c \
  "SELECT request_state||'|'||state_version::text||'|'||
          production_dispatch_enabled::text||'|'||
          (lease_token_hash IS NULL AND lease_expires_at IS NULL AND
           dispatcher_identity IS NULL)::text||'|'||
          (SELECT count(*) FROM campaign_operations_request_production_admission
             WHERE operational_request_id=71)||'|'||
          (SELECT count(*) FROM campaign_operations_dispatch_attempt
             WHERE operational_request_id=71 AND attempt_contract_version=2)
     FROM campaign_operations_operational_request
    WHERE operational_request_id=71;")"
[[ "$state_contract" == "bound|5|true|true|1|1" ]]
echo "H2_PRODUCTION_BIND_STATE request_state=bound state_version=5 production_dispatch_enabled=true lease=cleared admission=1 attempt_v2=1"

if [[ -n "${H2_PRESERVE_AFTER_WORKFLOW_ROOT:-}" ]]; then
  active_sessions="$(psql "${target[@]}" -At postgres -c \
    "SELECT pid||'|'||application_name||'|'||state
       FROM pg_stat_activity
      WHERE datname='${database}' AND pid <> pg_backend_pid()
      ORDER BY pid;" || true)"
  [[ -z "$active_sessions" ]] || {
    echo "H2_AFTER_WORKFLOW_NOT_QUIESCENT sessions=$active_sessions" >&2
    exit 1
  }
  printf '%s|%s|%s\n' "$cluster_root" "$database" "$cluster_socket" \
    > "$H2_PRESERVE_AFTER_WORKFLOW_ROOT"
  echo "H2_AFTER_WORKFLOW_QUIESCENT database=$database socket=$cluster_socket sessions=0"
  exit 0
fi

expect_role_failure() {
  local user="$1" sql="$2" marker="$3"
  local output="$tmp_root/${marker}.log"
  if psql -X -qAt -v ON_ERROR_STOP=1 -h "$cluster_socket" -p 5432 -U "$user" "$database" \
      -c "$sql" >"$output" 2>&1; then
    echo "negative role case unexpectedly succeeded: $marker" >&2
    exit 1
  fi
  rg -q '42501|permission denied|privilege' "$output" || { cat "$output" >&2; exit 1; }
  echo "H2_ROLE_NEGATIVE $marker=PASS"
}
expect_role_failure h2_manager_login \
  "SELECT record_campaign_operations_production_enable_v1('bad-enable',0,'x','x','x','x','x','x','x','x','x');" \
  dispatcher_cannot_enable
expect_role_failure h2_manager_login \
  "SELECT record_campaign_operations_production_disable_v1('bad-disable',1,'x',1,'x','x');" \
  dispatcher_cannot_disable
expect_role_failure h2_enabler_login \
  "SELECT transition_campaign_operations_request_dispatch_production_v2(7,3,'x',now(),'x','x','x');" \
  enabler_cannot_acquire
expect_role_failure h2_disabler_login \
  "SELECT transition_campaign_operations_request_dispatch_production_v2(7,3,'x',now(),'x','x','x');" \
  disabler_cannot_acquire
expect_role_failure h2_manager_login \
  "UPDATE experiment_scheduler_protocol SET failure_diagnostic='forbidden';" \
  manager_cannot_mutate_raw_scheduler
expect_role_failure h2_manager_login \
  "INSERT INTO campaign_operations_production_enablement_event SELECT * FROM campaign_operations_production_enablement_event LIMIT 1;" \
  manager_cannot_mutate_h1_table

sealed_owner="$(psql "${target[@]}" -At "$database" -c \
  "SELECT NOT pg_has_role('h2_manager_login','campaign_operations_h1_boundary_authority','USAGE')")"
[[ "$sealed_owner" == t ]]
echo "H2_SEALED_OWNER_UNREACHABLE=PASS"

# Deterministic blocker evidence for the frozen H2 global 0a/0b protocol lock.
block_a="$tmp_root/block-a.log"
block_b="$tmp_root/block-b.log"
PGAPPNAME=h2-lock-owner psql -X -qAt -h "$cluster_socket" -p 5432 -U campaign_manager_login "$database" \
  -c \
  "BEGIN; SELECT pg_backend_pid(); SELECT pg_advisory_xact_lock(19055,1); SELECT pg_sleep(5); COMMIT;" \
  >"$block_a" 2>&1 &
block_a_pid=$!
for _ in $(seq 1 100); do
  lock_owner="$(psql -X -qAt -h "$cluster_socket" -p 5432 -U campaign_manager_login "$database" \
    -c "SELECT pid FROM pg_stat_activity WHERE application_name='h2-lock-owner' AND state<>'idle' LIMIT 1" 2>/dev/null || true)"
  [[ -n "${lock_owner:-}" ]] && break
done
[[ -n "${lock_owner:-}" ]]
PGAPPNAME=h2-lock-waiter psql -X -qAt -h "$cluster_socket" -p 5432 -U campaign_manager_login "$database" \
  -c \
  "BEGIN; SELECT pg_backend_pid(); SELECT pg_advisory_xact_lock_shared(19055,1); COMMIT;" \
  >"$block_b" 2>&1 &
block_b_pid=$!
blocking_line=""
for _ in $(seq 1 100); do
  blocking_line="$(psql -X -qAt -h "$cluster_socket" -p 5432 -U campaign_manager_login "$database" \
    -c "SELECT a.pid||':'||array_to_string(pg_blocking_pids(a.pid),',') FROM pg_stat_activity a WHERE a.application_name='h2-lock-waiter' AND cardinality(pg_blocking_pids(a.pid))>0" 2>/dev/null || true)"
  [[ -n "$blocking_line" ]] && break
done
[[ -n "$blocking_line" ]]
echo "H2_BLOCKING_PIDS lock_owner=$lock_owner waiter=$blocking_line order=0a/0b_then_phase5 reverse_wait=absent"
wait "$block_a_pid"
wait "$block_b_pid"

# Wrong ledger checksum is tested in a cloned disposable database.
wrong_database="expertadvisor_campaign_operations_h2_wrong_checksum_${$}"
createdb "${target[@]}" -T "$database" "$wrong_database"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$wrong_database" -c \
  "UPDATE schema_migrations SET checksum='0000000000000000000000000000000000000000000000000000000000000000' WHERE version='056';"
if LSTM_DB_HOST="$cluster_socket" LSTM_DB_NAME="$wrong_database" \
  LSTM_DB_ADMIN_USER=campaign_manager_login bash "$repo_root/migrate_lstm_db.sh" \
  >"$tmp_root/wrong-checksum.log" 2>&1; then
  echo "wrong migration checksum unexpectedly accepted" >&2
  exit 1
fi
rg -q 'version=056.*checksum_mismatch' "$tmp_root/wrong-checksum.log"
dropdb "${target[@]}" "$wrong_database"

echo "H2_WORKFLOW_INTEGRATION_OK roles=PASS replay=PASS in_doubt=PASS locking=PASS no_duplicate=PASS migration_runner=PASS"
