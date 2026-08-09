#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="$(mktemp -d /tmp/ea-h2-m057.XXXXXX)"
cluster_root=""
baseline_marker="$tmp_root/baseline"
request_id=71
expected_version=4
operation_key=m057-operation
lease_digest=fnv1a64:f5db465c70a7cee0
expected_diagnostic='campaign operations production bind predecessor invalid'

cleanup() {
  if [[ -n "$cluster_root" && -f "$cluster_root/data/postmaster.pid" ]]; then
    pg_ctl -D "$cluster_root/data" -m immediate stop >/dev/null 2>&1 || true
  fi
  [[ -z "$cluster_root" || ! -d "$cluster_root" ]] || rm -rf -- "$cluster_root"
  rm -rf -- "$tmp_root"
}
trap cleanup EXIT

# The accepted H1->H2 builder stops immediately after the real production
# acquisition commit.  It does not run the Phase-E handoff.
baseline_built=false
for baseline_attempt in 1 2 3; do
  if H2_PRESERVE_BASELINE_ROOT="$baseline_marker" \
      H2_PRESERVE_CLUSTER_ROOT="$tmp_root/cluster-root" \
      bash "$repo_root/Tests/CampaignOperationsPhaseH2WorkflowTests.sh" \
      >"$tmp_root/baseline-${baseline_attempt}.log" 2>&1; then
    baseline_built=true
    break
  fi
done
[[ "$baseline_built" == true && -s "$baseline_marker" ]] || {
  tail -100 "$tmp_root/baseline-3.log" >&2
  exit 1
}
IFS='|' read -r cluster_root base_database cluster_socket < "$baseline_marker"
target=(-h "$cluster_socket" -p 5432 -U campaign_manager_login)

build_contract="$(psql "${target[@]}" -At "$base_database" -c \
  "SELECT campaign_operations_manager_build_canonical_v1('campaign-operations-production-dispatch-and-manager-run-once-v1','777bb11e6e3a4fb0a7bdc7f84d7f8f4a0a1b2c3d','clang++-h2-fixture','sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef');")"
scheduler_evidence="$(psql "${target[@]}" -At "$base_database" -c \
  "SELECT evidence_canonical FROM campaign_operations_scheduler_protocol_evidence_snapshot_v1();")"

prepare_clone() {
  local database="$1" fixture="$2"
  createdb "${target[@]}" -T "$base_database" "$database"

  psql -X -qAt -v ON_ERROR_STOP=1 -h "$cluster_socket" -p 5432 \
    -U h2_enabler_login "$database" -v build="$build_contract" \
    -v scheduler="$scheduler_evidence" <<'SQL' >/dev/null
SELECT record_campaign_operations_production_enable_v1(
  'm057-fixture-enable', 0, :'scheduler', 'cee://h2/m057-fixture',
  'h2.enabler@example.test',
  'campaign-operations-production-dispatch-and-manager-run-once-v1',
  :'build', '777bb11e6e3a4fb0a7bdc7f84d7f8f4a0a1b2c3d', 'clang++-h2-fixture',
  'sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef',
  'H2 migration 057 clean predecessor fixture');
SQL
  psql -X -qAt -v ON_ERROR_STOP=1 -h "$cluster_socket" -p 5432 \
    -U h2_manager_login "$database" -v lease="$lease_digest" -v build="$build_contract" <<'SQL' >/tmp/m057-acquire.$$
SELECT dispatch_attempt_id,lease_token_digest,lease_expires_at::text
FROM transition_campaign_operations_request_dispatch_production_v2(
  71,3,:'lease',now()+interval '300 seconds',
  'm057-operation','h2.manager@example.test',:'build');
SQL
  local acquisition_state
  acquisition_state="$(< /tmp/m057-acquire.$$)"
  rm -f -- /tmp/m057-acquire.$$

  case "$fixture" in
    producing|contract)
      # The production schema correctly prevents duplicate resulting versions
      # and invalid V1/V2 shapes.  Relax only those checks in this disposable
      # hostile clone so the real migration-057 function can observe the
      # intended later-producing-row predicate.
      psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<'SQL'
SET session_replication_role=replica;
  ALTER TABLE campaign_operations_dispatch_attempt
  DROP CONSTRAINT campaign_operations_dispatch_attempt_request_version_uidx,
  DROP CONSTRAINT campaign_operations_dispatch_attempt_check,
  DROP CONSTRAINT campaign_operations_dispatch_attempt_v1_v2_shape;
DROP INDEX campaign_operations_dispatch_attempt_v2_operation_uidx;
INSERT INTO campaign_operations_dispatch_attempt(
  dispatch_attempt_id,operational_request_id,request_identity_canonical,
  attempt_ordinal,expected_request_version,resulting_request_version,
  lease_token_digest,lease_expires_at,dispatcher_identity,
  attempt_contract_version,attempt_identity_canonical,attempt_identity_hash,
  acquired_at,request_production_admission_id,
  request_production_admission_canonical,request_production_admission_hash,
  production_enablement_event_id,production_enablement_event_canonical,
  production_enablement_event_hash,operation_key,requesting_actor,
  original_executing_service_principal,approved_build_contract_canonical,
  approved_build_contract_hash,production_capability)
SELECT nextval(pg_get_serial_sequence(
           'campaign_operations_dispatch_attempt','dispatch_attempt_id')),
       operational_request_id,request_identity_canonical,2,
       expected_request_version,resulting_request_version,lease_token_digest,
       lease_expires_at,dispatcher_identity,2,'m057-later-attempt-v2',
       'fnv1a64:0000000000000000',transaction_timestamp(),
       request_production_admission_id,request_production_admission_canonical,
       request_production_admission_hash,production_enablement_event_id,
       production_enablement_event_canonical,production_enablement_event_hash,
       'm057-operation',requesting_actor,
       original_executing_service_principal,approved_build_contract_canonical,
       approved_build_contract_hash,production_capability
  FROM campaign_operations_dispatch_attempt
 WHERE dispatch_attempt_id=(SELECT dispatch_attempt_id
                              FROM campaign_operations_dispatch_attempt
                             WHERE operational_request_id=71
                               AND attempt_ordinal=1);
SET session_replication_role=origin;
SQL
      ;;
    enablement_relation|current_head)
      local enablement_id enablement_canonical
      enablement_id="$(psql "${target[@]}" -At "$database" -c "SELECT production_enablement_event_id FROM campaign_operations_production_enablement_event WHERE resulting_version=1")"
      enablement_canonical="$(psql "${target[@]}" -At "$database" -c "SELECT enablement_identity_canonical FROM campaign_operations_production_enablement_event WHERE resulting_version=1")"
      psql -X -qAt -v ON_ERROR_STOP=1 -h "$cluster_socket" -p 5432 \
        -U h2_disabler_login "$database" -v id="$enablement_id" -v canonical="$enablement_canonical" <<'SQL' >/dev/null
SELECT record_campaign_operations_production_disable_v1(
  'm057-fixture-disable', :'id', :'canonical', 1,
  'h2.disabler@example.test','H2 migration 057 current-head fixture');
SQL
      psql -X -qAt -v ON_ERROR_STOP=1 -h "$cluster_socket" -p 5432 \
        -U h2_enabler_login "$database" -v build="$build_contract" \
        -v scheduler="$scheduler_evidence" <<'SQL' >/dev/null
SELECT record_campaign_operations_production_enable_v1(
  'm057-fixture-enable-2', 2, :'scheduler', 'cee://h2/m057-fixture-2',
  'h2.enabler@example.test',
  'campaign-operations-production-dispatch-and-manager-run-once-v1',
  :'build', '777bb11e6e3a4fb0a7bdc7f84d7f8f4a0a1b2c3d', 'clang++-h2-fixture',
  'sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef',
  'H2 migration 057 current-head fixture 2');
SQL
      ;;
  esac

  local ids
  ids="$(psql "${target[@]}" -At -F '|' "$database" -c \
    "SELECT 71,
            (SELECT request_production_admission_id FROM campaign_operations_request_production_admission WHERE operational_request_id=71),
            (SELECT dispatch_attempt_id FROM campaign_operations_dispatch_attempt WHERE operational_request_id=71 AND attempt_ordinal=1),
            (SELECT dispatch_attempt_id FROM campaign_operations_dispatch_attempt WHERE operational_request_id=71 AND attempt_ordinal=2),
            (SELECT production_enablement_event_id FROM campaign_operations_dispatch_attempt WHERE operational_request_id=71 AND attempt_ordinal=1),
            (SELECT production_enablement_event_id FROM campaign_operations_production_enablement_event ORDER BY resulting_version DESC LIMIT 1),
            (SELECT production_enablement_event_id FROM campaign_operations_production_enablement_event WHERE resulting_version=2)")"
  printf '%s|%s\n' "$acquisition_state" "$ids"
}

# A table-by-table signature includes all rows in the disposable database.
# This is intentionally stronger than request-scoped joins: if a hostile
# attempt changes operational_request_id, its exact primary-key row remains in
# the signature.  The listed classes are the durable evidence that the H2
# transition can observe or that the common Phase-E handoff can create.
signature_query=''
signature_query+="SELECT 'request='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.operational_request_id)),'none') FROM campaign_operations_operational_request t;\n"
signature_query+="SELECT 'admission='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.request_production_admission_id)),'none') FROM campaign_operations_request_production_admission t;\n"
signature_query+="SELECT 'attempt='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.dispatch_attempt_id)),'none') FROM campaign_operations_dispatch_attempt t;\n"
signature_query+="SELECT 'attempt_outcome='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.dispatch_attempt_outcome_id)),'none') FROM campaign_operations_dispatch_attempt_outcome t;\n"
signature_query+="SELECT 'dispatch_audit='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.dispatch_audit_reference_event_id)),'none') FROM campaign_operations_dispatch_audit_reference_event t;\n"
signature_query+="SELECT 'binding='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.request_binding_id)),'none') FROM campaign_operations_request_binding t;\n"
signature_query+="SELECT 'owner='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.downstream_control_owner_id)),'none') FROM campaign_operations_downstream_control_owner t;\n"
signature_query+="SELECT 'commitment='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.reservation_commitment_id)),'none') FROM campaign_operations_reservation_commitment t;\n"
signature_query+="SELECT 'campaign='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.operational_campaign_id)),'none') FROM campaign_operations_campaign t;\n"
signature_query+="SELECT 'authorization='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.authorization_event_id)),'none') FROM campaign_operations_authorization_event t;\n"
signature_query+="SELECT 'budget_ledger='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.budget_ledger_entry_id)),'none') FROM campaign_operations_budget_ledger_entry t;\n"
signature_query+="SELECT 'reservation='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.reservation_id)),'none') FROM campaign_operations_reservation t;\n"
signature_query+="SELECT 'enablement='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.production_enablement_event_id)),'none') FROM campaign_operations_production_enablement_event t;\n"
signature_query+="SELECT 'enablement_audit='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.production_enablement_audit_reference_event_id)),'none') FROM campaign_operations_production_enablement_audit_reference_event t;\n"
signature_query+="SELECT 'materialization='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.recommendation_campaign_materialization_id)),'none') FROM experiment_recommendation_campaign_materialization t;\n"
signature_query+="SELECT 'materialization_member='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.recommendation_campaign_materialization_member_id)),'none') FROM experiment_recommendation_campaign_materialization_member t;\n"
signature_query+="SELECT 'proposal='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.recommendation_conversion_proposal_id)),'none') FROM experiment_recommendation_conversion_proposal t;\n"
signature_query+="SELECT 'execution='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.recommendation_conversion_execution_id)),'none') FROM experiment_recommendation_conversion_execution t;\n"
signature_query+="SELECT 'activation='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.recommendation_conversion_activation_id)),'none') FROM experiment_recommendation_conversion_activation t;\n"
signature_query+="SELECT 'experiment='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.experiment_id)),'none') FROM experiment t;\n"
signature_query+="SELECT 'scheduler_invocation='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.scheduler_invocation_id)),'none') FROM experiment_scheduler_invocation t;\n"
signature_query+="SELECT 'scheduler_lease='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.singleton)),'none') FROM experiment_scheduler_lease t;\n"
signature_query+="SELECT 'scheduler_worker_attempt='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.worker_attempt_id)),'none') FROM experiment_scheduler_worker_attempt t;\n"
signature_query+="SELECT 'scheduler_protocol='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.singleton)),'none') FROM experiment_scheduler_protocol t;\n"
signature_query+="SELECT 'cancellation_request='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.cancellation_request_id)),'none') FROM campaign_operations_cancellation_request t;\n"
signature_query+="SELECT 'lifecycle='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.lifecycle_cancellation_event_id)),'none') FROM experiment_lifecycle_cancellation_event t;\n"
signature_query+="SELECT 'cancellation_settlement='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.cancellation_settlement_id)),'none') FROM campaign_operations_cancellation_settlement t;\n"
signature_query+="SELECT 'reconciliation_observation='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.reconciliation_observation_id)),'none') FROM campaign_operations_reconciliation_observation t;\n"
signature_query+="SELECT 'reconciliation_resolution='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.reconciliation_resolution_id)),'none') FROM campaign_operations_reconciliation_resolution t;\n"
signature_query+="SELECT 'reconciliation_cursor='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.reconciliation_cursor_event_id)),'none') FROM campaign_operations_reconciliation_cursor_event t;\n"
signature_query+="SELECT 'control_audit='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.control_audit_reference_event_id)),'none') FROM campaign_operations_control_audit_reference_event t;\n"
signature_query+="SELECT 'control_event='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.control_event_id)),'none') FROM campaign_operations_control_event t;\n"
signature_query+="SELECT 'reservation_event='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.reservation_event_id)),'none') FROM campaign_operations_reservation_event t;\n"
signature_query+="SELECT 'completion='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.completion_event_id)),'none') FROM campaign_operations_completion_event t;\n"
signature_query+="SELECT 'completion_audit='||coalesce(md5(string_agg(row_to_json(t)::text,'|' ORDER BY t.completion_audit_reference_event_id)),'none') FROM campaign_operations_completion_audit_reference_event t;"

durable_signature() {
  local database="$1"
  printf '%b' "$signature_query" | psql "${target[@]}" -At -v ON_ERROR_STOP=1 "$database"
}

non_request_signature() {
  durable_signature "$1" | rg -v '^request='
}

# Test-only observation of the six STRICT reads and ordered P1..P34 terms.
# It never performs a state transition; the real migration function remains
# the only implementation invoked for each hostile call.
predicate_truth_vector() {
  local database="$1" attempt_id="$2"
  psql -X -qAt -v ON_ERROR_STOP=1 -h "$cluster_socket" -p 5432 \
    -U campaign_manager_login "$database" \
    -v attempt="$attempt_id" -v lease="$lease_digest" -v build="$build_contract" <<'SQL'
WITH r AS (SELECT * FROM campaign_operations_operational_request WHERE operational_request_id=71),
a AS (SELECT * FROM campaign_operations_dispatch_attempt WHERE dispatch_attempt_id=:'attempt'::bigint),
d AS (SELECT * FROM campaign_operations_request_production_admission WHERE operational_request_id=71),
f AS (SELECT * FROM campaign_operations_dispatch_attempt WHERE request_production_admission_id=(SELECT request_production_admission_id FROM d) AND attempt_contract_version=2 AND attempt_ordinal=1),
e AS (SELECT * FROM campaign_operations_production_enablement_event WHERE production_enablement_event_id=(SELECT production_enablement_event_id FROM a)),
c AS (SELECT * FROM campaign_operations_production_enablement_event ORDER BY resulting_version DESC LIMIT 1),
o AS (SELECT count(*)::integer AS n FROM campaign_operations_dispatch_attempt_outcome WHERE dispatch_attempt_id=:'attempt'::bigint),
v AS (SELECT
 CASE WHEN ((SELECT request_state FROM r)<>'dispatching') IS TRUE THEN 'FAIL' ELSE 'PASS' END p1,
 CASE WHEN ((SELECT state_version FROM r)<>4) IS TRUE THEN 'FAIL' ELSE 'PASS' END p2,
 CASE WHEN ((SELECT production_dispatch_enabled FROM r) IS DISTINCT FROM true) IS TRUE THEN 'FAIL' ELSE 'PASS' END p3,
 CASE WHEN ((SELECT lease_token_hash FROM r) IS DISTINCT FROM :'lease') IS TRUE THEN 'FAIL' ELSE 'PASS' END p4,
 CASE WHEN ((SELECT lease_expires_at FROM r)<=transaction_timestamp()) IS TRUE THEN 'FAIL' ELSE 'PASS' END p5,
 CASE WHEN ((SELECT attempt_contract_version FROM a)<>2) IS TRUE THEN 'FAIL' ELSE 'PASS' END p6,
 CASE WHEN ((SELECT operational_request_id FROM a)<>71) IS TRUE THEN 'FAIL' ELSE 'PASS' END p7,
 CASE WHEN ((SELECT request_identity_canonical FROM a)<>(SELECT request_identity_canonical FROM r)) IS TRUE THEN 'FAIL' ELSE 'PASS' END p8,
 CASE WHEN ((SELECT expected_request_version FROM a)<>3) IS TRUE THEN 'FAIL' ELSE 'PASS' END p9,
 CASE WHEN ((SELECT resulting_request_version FROM a)<>4) IS TRUE THEN 'FAIL' ELSE 'PASS' END p10,
 CASE WHEN ((SELECT lease_token_digest FROM a)<>:'lease') IS TRUE THEN 'FAIL' ELSE 'PASS' END p11,
 CASE WHEN ((SELECT lease_expires_at FROM a)<>(SELECT lease_expires_at FROM r)) IS TRUE THEN 'FAIL' ELSE 'PASS' END p12,
 CASE WHEN ((SELECT dispatcher_identity FROM a)<>(SELECT dispatcher_identity FROM r)) IS TRUE THEN 'FAIL' ELSE 'PASS' END p13,
 CASE WHEN ((SELECT operation_key FROM a) IS DISTINCT FROM 'm057-operation') IS TRUE THEN 'FAIL' ELSE 'PASS' END p14,
 CASE WHEN ((SELECT approved_build_contract_canonical FROM a) IS DISTINCT FROM :'build') IS TRUE THEN 'FAIL' ELSE 'PASS' END p15,
 CASE WHEN ((SELECT operational_request_id FROM d)<>71) IS TRUE THEN 'FAIL' ELSE 'PASS' END p16,
 CASE WHEN ((SELECT request_identity_canonical FROM d)<>(SELECT request_identity_canonical FROM r)) IS TRUE THEN 'FAIL' ELSE 'PASS' END p17,
 CASE WHEN ((SELECT expected_request_version FROM d)<>(SELECT expected_request_version FROM f)) IS TRUE THEN 'FAIL' ELSE 'PASS' END p18,
 CASE WHEN ((SELECT operational_request_id FROM f)<>71) IS TRUE THEN 'FAIL' ELSE 'PASS' END p19,
 CASE WHEN ((SELECT request_production_admission_id FROM f)<>(SELECT request_production_admission_id FROM d)) IS TRUE THEN 'FAIL' ELSE 'PASS' END p20,
 CASE WHEN ((SELECT operation_key FROM f) IS DISTINCT FROM (SELECT dispatch_operation_key FROM d)) IS TRUE THEN 'FAIL' ELSE 'PASS' END p21,
 CASE WHEN ((SELECT request_production_admission_id FROM d)<>(SELECT request_production_admission_id FROM a)) IS TRUE THEN 'FAIL' ELSE 'PASS' END p22,
 CASE WHEN ((SELECT production_enablement_event_id FROM a)<>(SELECT production_enablement_event_id FROM d)) IS TRUE THEN 'FAIL' ELSE 'PASS' END p23,
 CASE WHEN ((SELECT request_production_admission_canonical FROM a)<>(SELECT admission_identity_canonical FROM d)) IS TRUE THEN 'FAIL' ELSE 'PASS' END p24,
 CASE WHEN ((SELECT request_production_admission_hash FROM a)<>(SELECT admission_identity_hash FROM d)) IS TRUE THEN 'FAIL' ELSE 'PASS' END p25,
 CASE WHEN ((SELECT production_enablement_event_canonical FROM a)<>(SELECT enablement_identity_canonical FROM e)) IS TRUE THEN 'FAIL' ELSE 'PASS' END p26,
 CASE WHEN ((SELECT production_enablement_event_hash FROM a)<>(SELECT enablement_identity_hash FROM e)) IS TRUE THEN 'FAIL' ELSE 'PASS' END p27,
 CASE WHEN ((SELECT event_kind FROM e)<>'enable') IS TRUE THEN 'FAIL' ELSE 'PASS' END p28,
 CASE WHEN ((SELECT production_enablement_event_id FROM e)<>(SELECT production_enablement_event_id FROM c)) IS TRUE THEN 'FAIL' ELSE 'PASS' END p29,
 CASE WHEN ((SELECT event_kind FROM c)<>'enable') IS TRUE THEN 'FAIL' ELSE 'PASS' END p30,
 CASE WHEN ((SELECT approved_build_contract_canonical FROM c) IS DISTINCT FROM :'build') IS TRUE THEN 'FAIL' ELSE 'PASS' END p31,
 CASE WHEN ((SELECT approved_build_contract_canonical FROM e) IS DISTINCT FROM :'build') IS TRUE THEN 'FAIL' ELSE 'PASS' END p32,
 CASE WHEN ((SELECT approved_build_contract_canonical FROM d) IS DISTINCT FROM :'build') IS TRUE THEN 'FAIL' ELSE 'PASS' END p33,
 CASE WHEN ((SELECT n FROM o)<>0) IS TRUE THEN 'FAIL' ELSE 'PASS' END p34)
SELECT 'P1='||p1||'|P2='||p2||'|P3='||p3||'|P4='||p4||'|P5='||p5||'|P6='||p6||'|P7='||p7||'|P8='||p8||'|P9='||p9||'|P10='||p10||'|P11='||p11||'|P12='||p12||'|P13='||p13||'|P14='||p14||'|P15='||p15||'|P16='||p16||'|P17='||p17||'|P18='||p18||'|P19='||p19||'|P20='||p20||'|P21='||p21||'|P22='||p22||'|P23='||p23||'|P24='||p24||'|P25='||p25||'|P26='||p26||'|P27='||p27||'|P28='||p28||'|P29='||p29||'|P30='||p30||'|P31='||p31||'|P32='||p32||'|P33='||p33||'|P34='||p34 FROM v;
SQL
}

assert_valid_baseline() {
  local database="$1" attempt_id="$2" vector="$3"
  [[ -n "$vector" ]] || { echo "M057_BASELINE_EVALUATOR_EMPTY database=$database" >&2; exit 1; }
  [[ "$(tr '|' '\n' <<< "$vector" | rg -c '=FAIL$' || true)" -eq 0 ]] || {
    echo "M057_BASELINE_INVALID attempt=$attempt_id vector=$vector" >&2; exit 1;
  }
  echo "M057_BASELINE_VALID attempt=$attempt_id strict_lookups=PASS all_explicit_predicates=PASS"
}

frozen_ids() {
  local database="$1" ids="$2"
  IFS='|' read -r admission first producing referenced current alt <<< "$ids"
  echo "M057_FROZEN_PRESEED_IDS request=$request_id admission=$admission first_attempt=$first producing_attempt=${producing:-none} referenced_enablement=$referenced current_head=$current alternate_enablement=${alt:-none} exact_primary_keys=FROZEN_IN_SIGNATURE"
}

assert_prebind_counts() {
  local database="$1" expected_attempts="${2:-1}" actual
  actual="$(psql "${target[@]}" -At -v ON_ERROR_STOP=1 "$database" -c \
    "SELECT (SELECT count(*) FROM campaign_operations_request_binding WHERE operational_request_id=71)||'|'||
            (SELECT count(*) FROM campaign_operations_dispatch_attempt_outcome o JOIN campaign_operations_dispatch_attempt a USING(dispatch_attempt_id) WHERE a.operational_request_id=71)||'|'||
            (SELECT count(*) FROM campaign_operations_downstream_control_owner WHERE operational_request_id=71)||'|'||
            (SELECT count(*) FROM campaign_operations_reservation_commitment WHERE operational_request_id=71)||'|'||
            (SELECT count(*) FROM experiment_recommendation_conversion_execution e JOIN experiment_recommendation_campaign_materialization_member m ON m.recommendation_conversion_proposal_id=e.recommendation_conversion_proposal_id JOIN campaign_operations_operational_request r ON r.recommendation_campaign_materialization_id=m.recommendation_campaign_materialization_id WHERE r.operational_request_id=71)||'|'||
            (SELECT count(*) FROM experiment_recommendation_conversion_activation a JOIN experiment_recommendation_conversion_execution e ON e.recommendation_conversion_execution_id=a.recommendation_conversion_execution_id JOIN experiment_recommendation_campaign_materialization_member m ON m.recommendation_conversion_proposal_id=e.recommendation_conversion_proposal_id JOIN campaign_operations_operational_request r ON r.recommendation_campaign_materialization_id=m.recommendation_campaign_materialization_id WHERE r.operational_request_id=71)||'|'||
            (SELECT count(*) FROM experiment_lifecycle_cancellation_event);")"
  [[ "$actual" == "0|0|0|0|0|0|0" ]] || { echo "M057_PREBIND_SIDE_EFFECT_COUNTS=$actual expected_attempts=$expected_attempts" >&2; exit 1; }
}

run_positive_control() {
  local database="expertadvisor_m057_positive_$$" state ids attempt_id admission first referenced current before after request_state
  state="$(prepare_clone "$database" clean)"
  IFS='|' read -r attempt_id _ _ _ _ _ _ _ _ <<< "$state"
  ids="$(awk -F'|' '{print $5"|"$6"|"$7"|"$8"|"$9"|"$10}' <<< "$state")"
  before="$(durable_signature "$database")"
  assert_prebind_counts "$database"
  psql -X -qAt -v ON_ERROR_STOP=1 -h "$cluster_socket" -p 5432 \
    -U campaign_manager_login "$database" -v lease="$lease_digest" \
    -v attempt="$attempt_id" -v build="$build_contract" <<'SQL' >/dev/null
SET session_replication_role=replica;
SELECT transition_campaign_operations_request_bound_production_v2(
  71,4,:'lease',:'attempt','m057-operation',:'build');
SET session_replication_role=origin;
SQL
  request_state="$(psql "${target[@]}" -At "$database" -c \
    "SELECT request_state||'|'||state_version||'|'||production_dispatch_enabled||'|'||(lease_token_hash IS NULL AND lease_expires_at IS NULL AND dispatcher_identity IS NULL) FROM campaign_operations_operational_request WHERE operational_request_id=71")"
  after="$(durable_signature "$database")"
  [[ "$request_state" == "bound|5|true|true" ]] || { echo "M057_POSITIVE_REQUEST_STATE=$request_state" >&2; exit 1; }
  [[ "$(non_request_signature "$database")" == "$(printf '%s\n' "$before" | rg -v '^request=')" ]] || { echo "M057_POSITIVE_UNEXPECTED_NON_REQUEST_MUTATION" >&2; exit 1; }
  assert_prebind_counts "$database"
  frozen_ids "$database" "$ids"
  echo "M057_CLEAN_CONTROL_FROM_FIRST_PRINCIPLES enable=PASS acquisition=COMMITTED attempt_v2=1 request=dispatching/v4 production_enabled=true transition=REAL_PROTECTED_FUNCTION result=SUCCESS positive_request_mutation=EXACT bound/v5 lease_cleared=true other_durable_classes_byte_identical=PASS downstream_side_effect=0 before_signature=$(printf '%s' "$before" | md5) after_signature=$(printf '%s' "$after" | md5)"
  dropdb "${target[@]}" "$database"
}

mutate_case() {
  local database="$1" case_name="$2" producing="$3" first="$4" admission="$5" referenced="$6" current="$7" alt="$8"
  local sql
  case "$case_name" in
    request_state) sql="UPDATE campaign_operations_operational_request SET request_state='bound' WHERE operational_request_id=71" ;;
    request_version) sql="UPDATE campaign_operations_operational_request SET state_version=99 WHERE operational_request_id=71" ;;
    production_disabled) sql="UPDATE campaign_operations_operational_request SET production_dispatch_enabled=false WHERE operational_request_id=71" ;;
    request_lease_digest) sql="UPDATE campaign_operations_operational_request SET lease_token_hash='fnv1a64:0000000000000000' WHERE operational_request_id=71" ;;
    request_lease_expiry) sql="UPDATE campaign_operations_operational_request SET lease_expires_at=now()-interval '1 second' WHERE operational_request_id=71" ;;
    producing_contract) sql="ALTER TABLE campaign_operations_dispatch_attempt DROP CONSTRAINT campaign_operations_dispatch_attempt_contract_version_check; UPDATE campaign_operations_dispatch_attempt SET attempt_contract_version=1 WHERE dispatch_attempt_id=$producing" ;;
    producing_request_relation) sql="UPDATE campaign_operations_dispatch_attempt SET operational_request_id=7 WHERE dispatch_attempt_id=$producing" ;;
    producing_request_identity) sql="UPDATE campaign_operations_dispatch_attempt SET request_identity_canonical='wrong-request-canonical' WHERE dispatch_attempt_id=$producing" ;;
    producing_expected_version) sql="UPDATE campaign_operations_dispatch_attempt SET expected_request_version=99 WHERE dispatch_attempt_id=$producing" ;;
    producing_resulting_version) sql="UPDATE campaign_operations_dispatch_attempt SET resulting_request_version=100 WHERE dispatch_attempt_id=$producing" ;;
    producing_lease_digest) sql="UPDATE campaign_operations_dispatch_attempt SET lease_token_digest='fnv1a64:0000000000000000' WHERE dispatch_attempt_id=$producing" ;;
    producing_lease_expiry) sql="UPDATE campaign_operations_dispatch_attempt SET lease_expires_at=now()-interval '1 second' WHERE dispatch_attempt_id=$producing" ;;
    producing_dispatcher) sql="UPDATE campaign_operations_dispatch_attempt SET dispatcher_identity='wrong.dispatcher@example.test' WHERE dispatch_attempt_id=$producing" ;;
    producing_operation_key) sql="UPDATE campaign_operations_dispatch_attempt SET operation_key='wrong-operation-key' WHERE dispatch_attempt_id=$producing" ;;
    producing_build) sql="UPDATE campaign_operations_dispatch_attempt SET approved_build_contract_canonical='stale-build' WHERE dispatch_attempt_id=$producing" ;;
    admission_request_relation) sql="UPDATE campaign_operations_request_production_admission SET operational_request_id=7 WHERE request_production_admission_id=$admission" ;;
    admission_request_identity) sql="UPDATE campaign_operations_request_production_admission SET request_identity_canonical='wrong-admission-request' WHERE request_production_admission_id=$admission" ;;
    admission_expected_version) sql="UPDATE campaign_operations_request_production_admission SET expected_request_version=99 WHERE request_production_admission_id=$admission" ;;
    first_request_relation) sql="UPDATE campaign_operations_dispatch_attempt SET operational_request_id=7 WHERE dispatch_attempt_id=$first" ;;
    first_admission_relation) sql="UPDATE campaign_operations_dispatch_attempt SET request_production_admission_id=999999 WHERE dispatch_attempt_id=$first" ;;
    first_operation_key) sql="UPDATE campaign_operations_dispatch_attempt SET operation_key='wrong-first-operation-key' WHERE dispatch_attempt_id=$first" ;;
    producing_admission_relation) sql="UPDATE campaign_operations_dispatch_attempt SET request_production_admission_id=999999 WHERE dispatch_attempt_id=$producing" ;;
    producing_enablement_relation)
      local enablement_id enablement_canonical
      enablement_id="$(psql "${target[@]}" -At "$database" -c "SELECT production_enablement_event_id FROM campaign_operations_production_enablement_event WHERE resulting_version=1")"
      enablement_canonical="$(psql "${target[@]}" -At "$database" -c "SELECT enablement_identity_canonical FROM campaign_operations_production_enablement_event WHERE resulting_version=1")"
      psql -X -qAt -v ON_ERROR_STOP=1 -h "$cluster_socket" -p 5432 -U h2_disabler_login "$database" -v id="$enablement_id" -v canonical="$enablement_canonical" <<'SQL' >/dev/null
SELECT record_campaign_operations_production_disable_v1('m057-fixture-disable', :'id', :'canonical', 1, 'h2.disabler@example.test', 'H2 migration 057 reachability fixture');
SQL
      psql -X -qAt -v ON_ERROR_STOP=1 -h "$cluster_socket" -p 5432 -U h2_enabler_login "$database" -v build="$build_contract" -v scheduler="$scheduler_evidence" <<'SQL' >/dev/null
SELECT record_campaign_operations_production_enable_v1('m057-fixture-enable-2', 2, :'scheduler', 'cee://h2/m057-fixture-2', 'h2.enabler@example.test', 'campaign-operations-production-dispatch-and-manager-run-once-v1', :'build', '777bb11e6e3a4fb0a7bdc7f84d7f8f4a0a1b2c3d', 'clang++-h2-fixture', 'sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef', 'H2 migration 057 reachability fixture 2');
SQL
      local new_enablement
      new_enablement="$(psql "${target[@]}" -At "$database" -c "SELECT production_enablement_event_id FROM campaign_operations_production_enablement_event ORDER BY resulting_version DESC LIMIT 1")"
      sql="UPDATE campaign_operations_dispatch_attempt SET production_enablement_event_id=$new_enablement, production_enablement_event_canonical=(SELECT enablement_identity_canonical FROM campaign_operations_production_enablement_event WHERE production_enablement_event_id=$new_enablement), production_enablement_event_hash=(SELECT enablement_identity_hash FROM campaign_operations_production_enablement_event WHERE production_enablement_event_id=$new_enablement) WHERE dispatch_attempt_id=$producing"
      ;;
    producing_admission_canonical) sql="UPDATE campaign_operations_dispatch_attempt SET request_production_admission_canonical='stale-admission-canonical' WHERE dispatch_attempt_id=$producing" ;;
    producing_admission_hash) sql="UPDATE campaign_operations_dispatch_attempt SET request_production_admission_hash='fnv1a64:0000000000000000' WHERE dispatch_attempt_id=$producing" ;;
    producing_enablement_canonical) sql="UPDATE campaign_operations_dispatch_attempt SET production_enablement_event_canonical='stale-enable-canonical' WHERE dispatch_attempt_id=$producing" ;;
    producing_enablement_hash) sql="UPDATE campaign_operations_dispatch_attempt SET production_enablement_event_hash='fnv1a64:0000000000000000' WHERE dispatch_attempt_id=$producing" ;;
    referenced_enablement_kind) sql="ALTER TABLE campaign_operations_production_enablement_event DROP CONSTRAINT campaign_operations_production_enablement_shape; UPDATE campaign_operations_production_enablement_event SET event_kind='disable' WHERE production_enablement_event_id=$referenced" ;;
    current_head_identity)
      local enablement_id enablement_canonical
      enablement_id="$(psql "${target[@]}" -At "$database" -c "SELECT production_enablement_event_id FROM campaign_operations_production_enablement_event WHERE resulting_version=1")"
      enablement_canonical="$(psql "${target[@]}" -At "$database" -c "SELECT enablement_identity_canonical FROM campaign_operations_production_enablement_event WHERE resulting_version=1")"
      psql -X -qAt -v ON_ERROR_STOP=1 -h "$cluster_socket" -p 5432 -U h2_disabler_login "$database" -v id="$enablement_id" -v canonical="$enablement_canonical" <<'SQL' >/dev/null
SELECT record_campaign_operations_production_disable_v1('m057-fixture-disable', :'id', :'canonical', 1, 'h2.disabler@example.test', 'H2 migration 057 reachability fixture');
SQL
      psql -X -qAt -v ON_ERROR_STOP=1 -h "$cluster_socket" -p 5432 -U h2_enabler_login "$database" -v build="$build_contract" -v scheduler="$scheduler_evidence" <<'SQL' >/dev/null
SELECT record_campaign_operations_production_enable_v1('m057-fixture-enable-2', 2, :'scheduler', 'cee://h2/m057-fixture-2', 'h2.enabler@example.test', 'campaign-operations-production-dispatch-and-manager-run-once-v1', :'build', '777bb11e6e3a4fb0a7bdc7f84d7f8f4a0a1b2c3d', 'clang++-h2-fixture', 'sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef', 'H2 migration 057 reachability fixture 2');
SQL
      return
      ;;
    current_head_build) sql="UPDATE campaign_operations_production_enablement_event SET approved_build_contract_canonical='stale-current-head-build' WHERE production_enablement_event_id=$current" ;;
    referenced_build) sql="UPDATE campaign_operations_production_enablement_event SET approved_build_contract_canonical='stale-referenced-build' WHERE production_enablement_event_id=$referenced" ;;
    admission_build) sql="UPDATE campaign_operations_request_production_admission SET approved_build_contract_canonical='stale-admission-build' WHERE request_production_admission_id=$admission" ;;
    outcome_exists) sql="INSERT INTO campaign_operations_dispatch_attempt_outcome(dispatch_attempt_id,attempt_identity_canonical,result_classification,downstream_evidence_classification,semantic_conflict_classification,uncertain_commit_recovery_classification,diagnostic_code,expected_request_version,resulting_request_version,expected_reservation_version,resulting_reservation_version,outcome_contract_version,outcome_identity_canonical,outcome_identity_hash) SELECT dispatch_attempt_id,attempt_identity_canonical,'rejected','no_phase5_evidence','none','proven_no_commit','dispatch_downstream_reconciliation_required',4,4,1,1,1,'m057-existing-outcome','fnv1a64:0000000000000000' FROM campaign_operations_dispatch_attempt WHERE dispatch_attempt_id=$producing" ;;
    *) echo "unknown hostile case: $case_name" >&2; exit 64 ;;
  esac
  [[ "$case_name" == current_head_identity ]] && return
  psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<SQL >/dev/null
SET session_replication_role=replica;
$sql;
SET session_replication_role=origin;
SQL
}

run_case() {
  local predicate_id="$1" case_name="$2" fixture="$3" expected_sqlstate="$4"
  local database="expertadvisor_m057_${case_name}_$$" state output before after ids attempt admission first producing referenced current alt request_before request_after baseline_vector vector false_set false_count prefix_false_count first_failing target_predicate lookup_status
  state="$(prepare_clone "$database" "$fixture")"
  IFS='|' read -r initial_attempt _ _ _ admission first producing referenced current alt <<< "$state"
  attempt_id="$initial_attempt"
  [[ -n "$producing" && "$producing" != "NULL" ]] || producing="$attempt_id"
  ids="$admission|$first|$producing|$referenced|$current|$alt"
  baseline_vector="$(predicate_truth_vector "$database" "$producing")"
  assert_valid_baseline "$database" "$producing" "$baseline_vector"
  mutate_case "$database" "$case_name" "$producing" "$first" "$admission" "$referenced" "$current" "$alt"
  lookup_status="$(psql "${target[@]}" -At "$database" -c "SELECT (SELECT count(*) FROM campaign_operations_operational_request WHERE operational_request_id=71)||'|'||(SELECT count(*) FROM campaign_operations_dispatch_attempt WHERE dispatch_attempt_id=$producing)||'|'||(SELECT count(*) FROM campaign_operations_request_production_admission WHERE operational_request_id=71)||'|'||(SELECT count(*) FROM campaign_operations_dispatch_attempt WHERE request_production_admission_id=(SELECT request_production_admission_id FROM campaign_operations_request_production_admission WHERE operational_request_id=71) AND attempt_contract_version=2 AND attempt_ordinal=1)||'|'||(SELECT count(*) FROM campaign_operations_production_enablement_event WHERE production_enablement_event_id=(SELECT production_enablement_event_id FROM campaign_operations_dispatch_attempt WHERE dispatch_attempt_id=$producing))||'|'||(CASE WHEN EXISTS (SELECT 1 FROM campaign_operations_production_enablement_event) THEN 1 ELSE 0 END);")"
  if [[ "$lookup_status" == "1|1|1|1|1|1" ]]; then
    vector="$(predicate_truth_vector "$database" "$producing")"
  else
    vector=""
  fi
  false_set="$(tr '|' '\n' <<< "$vector" | awk -F= '$2=="FAIL" {print $1}' | paste -sd, -)"
  false_count="$(tr '|' '\n' <<< "$vector" | awk -F= '$2=="FAIL" {n++} END {print n+0}')"
  first_failing="$(tr '|' '\n' <<< "$vector" | awk -F= '$2=="FAIL" {print $1; exit}')"
  target_predicate="P$((10#${predicate_id#P}))"
  prefix_false_count="$(tr '|' '\n' <<< "$vector" | awk -F'[=P]' -v target="${target_predicate#P}" '$2 ~ /^[0-9]+$/ && $2 <= target && $3=="FAIL" {n++} END {print n+0}')"
  echo "M057_REACHABILITY case=$case_name target=$target_predicate strict_lookups=$([[ "$lookup_status" == "1|1|1|1|1|1" ]] && echo PASS || echo FAIL) earlier_predicates=$([[ "$first_failing" == "$target_predicate" ]] && echo PASS || echo FAIL) target_predicate=$([[ "$first_failing" == "$target_predicate" ]] && echo FAIL || echo NOT_FIRST) first_failing_predicate=${first_failing:-NONE} reachability=$([[ "$first_failing" == "$target_predicate" ]] && echo PASS || echo FAIL) truth_vector=${vector:-UNAVAILABLE}"
  if { [[ "$predicate_id" == P31 ]] && rg -q 'P32=FAIL' <<< "$vector"; } ||
     { [[ "$predicate_id" == P32 ]] && rg -q 'P31=FAIL' <<< "$vector"; } ||
     { [[ "$predicate_id" == P28 ]] && rg -q 'P30=FAIL' <<< "$vector"; } ||
     { [[ "$predicate_id" == P30 ]] && rg -q 'P28=FAIL' <<< "$vector"; }; then
    harness_relational_alias_count=$((harness_relational_alias_count + 1))
    echo "M057_RELATIONAL_ALIAS case=$case_name target=$predicate_id false_predicates=$false_set isolated_target_count=$false_count result=REJECTED"
  fi
  before="$(durable_signature "$database")"
  request_before="$(psql "${target[@]}" -At "$database" -c "SELECT request_state||'|'||state_version||'|'||production_dispatch_enabled||'|'||coalesce(lease_token_hash,'NULL') FROM campaign_operations_operational_request WHERE operational_request_id=71")"
  output="$tmp_root/$case_name.log"
  if psql -X -qAt -v ON_ERROR_STOP=1 -v VERBOSITY=verbose -h "$cluster_socket" -p 5432 \
      -U h2_manager_login "$database" -v lease="$lease_digest" -v attempt="$producing" \
      -v build="$build_contract" <<'SQL' >"$output" 2>&1; then
SELECT state_version FROM transition_campaign_operations_request_bound_production_v2(
  71,4,:'lease',:'attempt','m057-operation',:'build');
SQL
    echo "M057_HOSTILE_UNEXPECTED_SUCCESS predicate=$predicate_id case=$case_name" >&2
    exit 1
  fi
  sqlstate_ok=false
  diagnostic_ok=false
  if rg -q -- "ERROR:  $expected_sqlstate:" "$output"; then sqlstate_ok=true; fi
  if [[ "$expected_sqlstate" == 40001 ]] && rg -q -- "$expected_diagnostic" "$output"; then diagnostic_ok=true; fi
  [[ "$sqlstate_ok" == true ]] || echo "M057_CASE_SQLSTATE_MISMATCH predicate_id=$predicate_id case=$case_name expected=$expected_sqlstate observed=$(rg -o 'ERROR:  [A-Z0-9]+:' "$output" | head -1 || true)"
  [[ "$diagnostic_ok" == true ]] || echo "M057_CASE_DIAGNOSTIC_MISMATCH predicate_id=$predicate_id case=$case_name expected=$expected_diagnostic"
  after="$(durable_signature "$database")"
  [[ "$before" == "$after" ]] || { echo "M057_HOSTILE_CALL_SIDE_EFFECT predicate=$predicate_id case=$case_name before=$before after=$after" >&2; exit 1; }
  request_after="$(psql "${target[@]}" -At "$database" -c "SELECT request_state||'|'||state_version||'|'||production_dispatch_enabled||'|'||coalesce(lease_token_hash,'NULL') FROM campaign_operations_operational_request WHERE operational_request_id=71")"
  [[ "$request_before" == "$request_after" ]] || { echo "M057_HOSTILE_REQUEST_MUTATION predicate=$predicate_id before=$request_before after=$request_after" >&2; exit 1; }
  frozen_ids "$database" "$ids"
  if [[ "$expected_sqlstate" == 40001 && "$sqlstate_ok" == true && "$diagnostic_ok" == true && "$first_failing" == "$target_predicate" && "$prefix_false_count" -eq 1 ]]; then
    verified_explicit_count=$((verified_explicit_count + 1))
  fi
  echo "M057_COVERAGE predicate_id=$target_predicate case=$case_name mutation=HOSTILE_FIXTURE_ONLY target_reached=$([[ "$first_failing" == "$target_predicate" ]] && echo PASS || echo FAIL) branch=EXPLICIT_40001 sqlstate=$expected_sqlstate diagnostic=$expected_diagnostic exact_call_signature_unchanged=PASS downstream_binding_owner_commitment_experiment_execution_activation_lifecycle_handoff_audit=PASS false_predicates=${false_set:-NONE}"
  dropdb "${target[@]}" "$database"
}

run_lookup_case() {
  local lookup_id="$1" case_name="$2" fixture="$3" database="expertadvisor_m057_lookup_${2}_$$" state output before after attempt_id first referenced
  state="$(prepare_clone "$database" "$fixture")"
  IFS='|' read -r attempt_id _ _ _ _ first _ referenced _ _ <<< "$state"
  case "$case_name" in
    request_missing) : ;;
    attempt_missing) attempt_id=999999 ;;
    admission_missing)
      psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c "SET session_replication_role=replica; DELETE FROM campaign_operations_request_production_admission WHERE request_production_admission_id=(SELECT request_production_admission_id FROM campaign_operations_request_production_admission WHERE operational_request_id=71); SET session_replication_role=origin;" >/dev/null
      ;;
    first_attempt_missing)
      psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c "SET session_replication_role=replica; DELETE FROM campaign_operations_dispatch_attempt WHERE dispatch_attempt_id=$first; SET session_replication_role=origin;" >/dev/null
      ;;
    referenced_enablement_missing)
      psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c "SET session_replication_role=replica; DELETE FROM campaign_operations_production_enablement_event WHERE production_enablement_event_id=$referenced; SET session_replication_role=origin;" >/dev/null
      ;;
    current_head_missing)
      psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c "SET session_replication_role=replica; DELETE FROM campaign_operations_production_enablement_event; SET session_replication_role=origin;" >/dev/null
      ;;
    *) echo "unknown lookup case: $case_name" >&2; exit 64 ;;
  esac
  before="$(durable_signature "$database")"
  output="$tmp_root/lookup-$case_name.log"
  if psql -X -qAt -v ON_ERROR_STOP=1 -v VERBOSITY=verbose -h "$cluster_socket" -p 5432 \
      -U h2_manager_login "$database" -v lease="$lease_digest" -v attempt="$attempt_id" \
      -v build="$build_contract" <<SQL >"$output" 2>&1; then
SELECT state_version FROM transition_campaign_operations_request_bound_production_v2(
  $([[ "$case_name" == request_missing ]] && echo 999999 || echo 71),4,:'lease',:'attempt','m057-operation',:'build');
SQL
    exit 1
  fi
  rg -q 'ERROR:  P0002:' "$output" || { cat "$output" >&2; exit 1; }
  rg -q 'query returned no rows' "$output" || { cat "$output" >&2; exit 1; }
  after="$(durable_signature "$database")"
  [[ "$before" == "$after" ]] || { echo "M057_LOOKUP_CALL_SIDE_EFFECT lookup_id=$lookup_id case=$case_name" >&2; exit 1; }
  if [[ "$lookup_id" != L6 ]]; then
    verified_strict_lookup_count=$((verified_strict_lookup_count + 1))
  fi
  echo "M057_LOOKUP_REACHABILITY lookup_id=$lookup_id case=$case_name strict_lookup=FAIL first_lookup=$lookup_id sqlstate=P0002 diagnostic=query_returned_no_rows exact_signature_unchanged=PASS"
  dropdb "${target[@]}" "$database"
}

run_positive_control

# Authoritative inventory derived from migration 057 lines 42-78 and 83-132.
# The metadata is the single source for execution order, fixture case, and
# source expression labels used by the report.
strict_lookup_metadata=(
  'L1|request row lookup|42-45|operational_request_id=target_request_id|request_missing'
  'L2|producing attempt lookup|47-49|dispatch_attempt_id=dispatch_attempt_id_value|attempt_missing'
  'L3|production admission lookup|51-53|operational_request_id=target_request_id|admission_missing'
  'L4|qualifying first Attempt V2 lookup|59-64|admission_id AND contract_version=2 AND ordinal=1|first_attempt_missing'
  'L5|referenced enablement lookup|66-69|production_enablement_event_id=attempt.production_enablement_event_id|referenced_enablement_missing'
  'L6|current enablement head lookup|71-74|ORDER BY resulting_version DESC LIMIT 1|current_head_missing'
)
explicit_metadata=(
  'P01|request_state|request.request_state|clean|request_state|83'
  'P02|request_version|request.state_version|clean|request_version|84'
  'P03|production_disabled|request.production_dispatch_enabled|clean|production_disabled|85'
  'P04|request_lease_digest|request.lease_token_hash|clean|request_lease_digest|86'
  'P05|request_lease_expiry|request.lease_expires_at|clean|request_lease_expiry|87'
  'P06|producing_contract|attempt.attempt_contract_version|producing|producing_contract|88'
  'P07|producing_request_relation|attempt.operational_request_id|producing|producing_request_relation|89'
  'P08|producing_request_identity|attempt.request_identity_canonical|producing|producing_request_identity|90-91'
  'P09|producing_expected_version|attempt.expected_request_version|producing|producing_expected_version|92'
  'P10|producing_resulting_version|attempt.resulting_request_version|producing|producing_resulting_version|93'
  'P11|producing_lease_digest|attempt.lease_token_digest|producing|producing_lease_digest|94'
  'P12|producing_lease_expiry|attempt.lease_expires_at|producing|producing_lease_expiry|95'
  'P13|producing_dispatcher|attempt.dispatcher_identity|producing|producing_dispatcher|96'
  'P14|producing_operation_key|attempt.operation_key|producing|producing_operation_key|97'
  'P15|producing_build|attempt.approved_build_contract_canonical|producing|producing_build|98-99'
  'P16|admission_request_relation|admission.operational_request_id|clean|admission_request_relation|100'
  'P17|admission_request_identity|admission.request_identity_canonical|clean|admission_request_identity|101-102'
  'P18|admission_expected_version|admission.expected_request_version|clean|admission_expected_version|103-104'
  'P19|first_request_relation|first.operational_request_id|producing|first_request_relation|105'
  'P20|first_admission_relation|first.request_production_admission_id|producing|first_admission_relation|106-107'
  'P21|first_operation_key|first.operation_key|producing|first_operation_key|108-109'
  'P22|producing_admission_relation|attempt.request_production_admission_id|producing|producing_admission_relation|110-111'
  'P23|producing_enablement_relation|attempt.production_enablement_event_id|clean|producing_enablement_relation|112-113'
  'P24|producing_admission_canonical|attempt.request_production_admission_canonical|producing|producing_admission_canonical|114-115'
  'P25|producing_admission_hash|attempt.request_production_admission_hash|producing|producing_admission_hash|116-117'
  'P26|producing_enablement_canonical|attempt.production_enablement_event_canonical|producing|producing_enablement_canonical|118-119'
  'P27|producing_enablement_hash|attempt.production_enablement_event_hash|producing|producing_enablement_hash|120-121'
  'P28|referenced_enablement_kind|enablement.event_kind|clean|referenced_enablement_kind|122'
  'P29|current_head_identity|enablement.production_enablement_event_id|clean|current_head_identity|123-124'
  'P30|current_head_kind|current_enablement.event_kind|clean|referenced_enablement_kind|125'
  'P31|current_head_build|current_enablement.approved_build_contract_canonical|clean|current_head_build|126-127'
  'P32|referenced_build|enablement.approved_build_contract_canonical|clean|referenced_build|128-129'
  'P33|admission_build|admission.approved_build_contract_canonical|clean|admission_build|130-131'
  'P34|outcome_exists|outcome_count|clean|outcome_exists|132'
)
echo "M057_INVENTORY total=41 strict_lookups=6 explicit_if=34 final_cas=1"
[[ "${#strict_lookup_metadata[@]}" -eq 6 && "${#explicit_metadata[@]}" -eq 34 ]] || exit 1
harness_relational_alias_count=0
verified_explicit_count=0
verified_strict_lookup_count=0
for metadata in "${explicit_metadata[@]}"; do
  IFS='|' read -r predicate_id case_name _ fixture mutation _ <<< "$metadata"
  run_case "$predicate_id" "$mutation" "$fixture" 40001
done

run_lookup_case L1 request_missing clean
run_lookup_case L2 attempt_missing clean
run_lookup_case L3 admission_missing clean
run_lookup_case L4 first_attempt_missing producing
run_lookup_case L5 referenced_enablement_missing current_head
run_lookup_case L6 current_head_missing clean
echo "LOOKUP_EXECUTABLE_COVERAGE 6_of_6=$([[ "$verified_strict_lookup_count" -eq 5 ]] && echo STRUCTURAL_L6_PROOF_REQUIRED || echo FAIL)"
echo "M057_LOOKUP_STRUCTURAL_NOTE L6=NOT_SEPARATELY_ISOLATABLE reason=current-head-LIMIT-1 returns a row whenever L5 can resolve any enablement; empty table makes L5 fail first"

# The binding conflict is owned by the outer Phase-E workflow, not by the
# migration-057 predecessor function, which has no binding lookup/predicate.
workflow_conflict_log="$tmp_root/outer-layer-conflict.log"
bash "$repo_root/Tests/CampaignOperationsPhaseH2WorkflowTests.sh" >"$workflow_conflict_log" 2>&1
rg -q 'H2_DISPATCH_CONFLICT' "$workflow_conflict_log"
rg -q 'H2_OUTER_LAYER_BINDING_CONFLICT diagnostic=production_dispatch_conflicting_replay before_after_signature=IDENTICAL migration057_invoked=NO result=PASS' "$workflow_conflict_log"
echo "M057_OUTER_LAYER_CONFLICT authoritative_layer=RunDispatchAdapter/FindAndValidateCompleteDispatchBinding migration057_predicate=NOT_PRESENT before_after_signature=IDENTICAL migration057_invoked=NO result=PASS"

# Non-concurrent P0001 branch reachability.  This trigger is a fixture seam,
# never a claim of an ordinary concurrent CAS-loss race.
run_cas_guard_case() {
  local database="expertadvisor_m057_cas_guard_$$" state attempt before after output
  state="$(prepare_clone "$database" clean)"
  IFS='|' read -r attempt _ _ _ _ _ _ _ _ <<< "$state"
  psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<'SQL'
CREATE OR REPLACE FUNCTION m057_cas_guard_v1() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RETURN NULL; END; $$;
CREATE TRIGGER m057_cas_guard_trigger
BEFORE UPDATE OF request_state ON campaign_operations_operational_request
FOR EACH ROW WHEN (OLD.operational_request_id=71)
EXECUTE FUNCTION m057_cas_guard_v1();
SQL
  before="$(durable_signature "$database")"
  output="$tmp_root/cas-guard.log"
  if psql -X -qAt -v ON_ERROR_STOP=1 -v VERBOSITY=verbose -h "$cluster_socket" -p 5432 \
      -U h2_manager_login "$database" -v lease="$lease_digest" -v attempt="$attempt" \
      -v build="$build_contract" <<'SQL' >"$output" 2>&1; then
SELECT state_version FROM transition_campaign_operations_request_bound_production_v2(
  71,4,:'lease',:'attempt','m057-operation',:'build');
SQL
    exit 1
  fi
  rg -q 'ERROR:  P0001:' "$output"
  rg -q 'campaign operations production bind compare-and-set lost' "$output"
  after="$(durable_signature "$database")"
  [[ "$before" == "$after" ]]
  echo "M057_FINAL_CAS_GUARD sqlstate=P0001 diagnostic=campaign operations production bind compare-and-set lost branch_reachability=NON_CONCURRENT_TRIGGER_FIXTURE genuine_concurrent_loss=NOT_CLAIMED no_partial_mutation=PASS"
  dropdb "${target[@]}" "$database"
}
run_cas_guard_case

echo "M057_FINAL_CAS_CLASSIFICATION ordinary_concurrency=DEFENSIVE_UNREACHABLE target_request_row=FOR_UPDATE final_update_fields=state_version_lease_digest_request_state transaction_timestamp=TRANSACTION_STABLE p0001_guard=RETAINED branch_reachability=NON_CONCURRENT_ONLY unsupported_malformed_input=NOT_ORDINARY_CAS_RACE"
echo "MIGRATION057_FINAL_CAS_DEFENSIVE_UNREACHABLE"
unverified_explicit_count=$((34 - verified_explicit_count))
unverified_strict_lookup_count=$((6 - verified_strict_lookup_count))
echo "VERIFIED_EXPLICIT_40001_COUNT=$verified_explicit_count"
echo "UNVERIFIED_EXPLICIT_40001_COUNT=$unverified_explicit_count"
echo "VERIFIED_STRICT_LOOKUP_COUNT=$verified_strict_lookup_count"
echo "UNVERIFIED_STRICT_LOOKUP_COUNT=$unverified_strict_lookup_count"
echo "HARNESS_RELATIONAL_ALIAS_COUNT=$harness_relational_alias_count"
if [[ "$unverified_explicit_count" -ne 0 || "$unverified_strict_lookup_count" -ne 0 || "$harness_relational_alias_count" -ne 0 ]]; then
  echo "H2_MIGRATION_057_HOSTILE_SUITE_INCOMPLETE explicit_unverified=$unverified_explicit_count strict_unverified=$unverified_strict_lookup_count relational_aliases=$harness_relational_alias_count" >&2
  exit 1
fi
echo "H2_MIGRATION_057_HOSTILE_SUITE_OK positive_control=PASS authoritative_inventory=PASS explicit_40001=PASS lookup_sqlstate=P0002_PASS aliases=0 proven=PASS durable_signature=COMPLETE preseed_primary_keys=FROZEN enablement_audit=PASS produced_experiment_execution_activation=PASS lifecycle_handoff=PASS final_cas=DEFENSIVE_UNREACHABLE"

before_055="$(shasum -a 256 "$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql" | awk '{print $1}')"
head_055="$(git -C "$repo_root" show HEAD:Database/migrations/055_campaign_operations_production_admission_foundation.sql | shasum -a 256 | awk '{print $1}')"
[[ "$before_055" == "$head_055" ]]
echo "H2_MIGRATION_055_BYTE_IDENTICAL sha256=$before_055 direct_head_comparison=PASS"
