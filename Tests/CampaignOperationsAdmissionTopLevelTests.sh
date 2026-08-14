#!/usr/bin/env bash
set -euo pipefail

# This is an environment-backed regression for the real user-facing command.
# It deliberately requires an already prepared disposable/development database
# containing the approved materialization selected below.  It never creates or
# edits experiment fixtures and refuses to run if the campaign already exists.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${1:-$repo_root/DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release}"
database="${LSTM_DB_NAME:-LSTM}"
host="${LSTM_DB_HOST:-127.0.0.1}"
port="${LSTM_DB_PORT:-5432}"
materialization_id="${CAMPAIGN_OPERATIONS_ADMISSION_TEST_MATERIALIZATION_ID:-2}"
actor="${CAMPAIGN_OPERATIONS_ADMISSION_TEST_ACTOR:-vjp}"
reason="${CAMPAIGN_OPERATIONS_ADMISSION_TEST_REASON:-Admit approved CADCHF H4 Donchian-20 enabled versus zero-ablation materialization 2 into Campaign Operations}"
login="${CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER:?CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER is required}"

[[ -x "$binary" ]] || { echo "missing executable: $binary" >&2; exit 2; }

psql_admin=(psql -X -v ON_ERROR_STOP=1 -h "$host" -p "$port" -d "$database")
psql_login=(psql -X -v ON_ERROR_STOP=1 -h "$host" -p "$port" -U "$login" -d "$database")

query() { "${psql_admin[@]}" -Atqc "$1"; }
login_query() { "${psql_login[@]}" -Atqc "$1"; }

materialization_state="$(query "
SELECT count(DISTINCT m.recommendation_campaign_materialization_id) || '|' ||
       max(m.selected_member_count)::text || '|' || count(mm.recommendation_campaign_materialization_member_id)::text
FROM experiment_recommendation_campaign_materialization m
LEFT JOIN experiment_recommendation_campaign_materialization_member mm
  ON mm.recommendation_campaign_materialization_id =
     m.recommendation_campaign_materialization_id
WHERE m.recommendation_campaign_materialization_id = ${materialization_id};")"
IFS='|' read -r materialization_count expected_members actual_members <<<"$materialization_state"
[[ "$materialization_count" == 1 && "$expected_members" == "$actual_members" &&
   "$expected_members" != "" ]] || {
    echo "materialization ${materialization_id} is not a complete fixture: $materialization_state" >&2
    exit 2
}

campaign_before="$(query "SELECT count(*) FROM campaign_operations_campaign WHERE recommendation_campaign_materialization_id=${materialization_id};")"
[[ "$campaign_before" == 0 ]] || {
    echo "refusing to reuse materialization ${materialization_id}: campaign already exists" >&2
    exit 2
}

experiment_before="$(query "SELECT count(*) || '|' || coalesce(md5(string_agg(row_to_json(e)::text, '|' ORDER BY e.experiment_id)), 'none') FROM experiment e;")"
worker_before="$(query "SELECT CASE WHEN to_regclass('experiment_scheduler_worker_attempt') IS NULL THEN 'absent' ELSE (SELECT count(*) || '|' || coalesce(md5(string_agg(row_to_json(e)::text, '|' ORDER BY e.worker_attempt_id)), 'none') FROM experiment_scheduler_worker_attempt e) END;")"

output="$(CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER="$login" \
    LSTM_DB_NAME="$database" LSTM_DB_HOST="$host" LSTM_DB_PORT="$port" \
    "$binary" --campaign-operations-admit "$materialization_id" \
    --campaign-operations-actor "$actor" \
    --campaign-operations-reason "$reason" --yes 2>&1)" || {
    echo "$output" >&2
    echo "real Campaign Operations admission command failed" >&2
    exit 1
}
grep -Fq "disposition=recorded" <<<"$output"

state_after="$(query "
SELECT (SELECT count(*) FROM campaign_operations_campaign
        WHERE recommendation_campaign_materialization_id=${materialization_id}) || '|' ||
       (SELECT count(*) FROM campaign_operations_audit_reference_event audit
        JOIN campaign_operations_campaign campaign USING (operational_campaign_id)
        WHERE campaign.recommendation_campaign_materialization_id=${materialization_id}
          AND audit.cause_kind='campaign_created') || '|' ||
       (SELECT materialization_member_count::text FROM campaign_operations_campaign
        WHERE recommendation_campaign_materialization_id=${materialization_id}) || '|' ||
       (SELECT count(*)::text FROM campaign_operations_completion_event
        WHERE operational_campaign_id IN (
          SELECT operational_campaign_id FROM campaign_operations_campaign
          WHERE recommendation_campaign_materialization_id=${materialization_id}));")"
[[ "$state_after" == "1|1|$expected_members|0" ]] || {
    echo "unexpected admission state: $state_after" >&2
    exit 1
}

output_replay="$(CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER="$login" \
    LSTM_DB_NAME="$database" LSTM_DB_HOST="$host" LSTM_DB_PORT="$port" \
    "$binary" --campaign-operations-admit "$materialization_id" \
    --campaign-operations-actor "$actor" \
    --campaign-operations-reason "$reason" --yes 2>&1)" || {
    echo "$output_replay" >&2
    echo "idempotent admission replay failed" >&2
    exit 1
}
grep -Fq "disposition=existing_identical" <<<"$output_replay"

state_replay="$(query "SELECT (SELECT count(*) FROM campaign_operations_campaign WHERE recommendation_campaign_materialization_id=${materialization_id}) || '|' || (SELECT count(*) FROM campaign_operations_audit_reference_event audit JOIN campaign_operations_campaign campaign USING (operational_campaign_id) WHERE campaign.recommendation_campaign_materialization_id=${materialization_id} AND audit.cause_kind='campaign_created');")"
[[ "$state_replay" == "1|1" ]] || { echo "replay duplicated campaign/audit: $state_replay" >&2; exit 1; }

# Continue through the real pre-Phase-H budget service as the reviewed LOGIN.
# Its direct campaign-lock helper execution was sealed by H1 and restored by
# migration 064. Request acceptance is covered by the prerequisite-complete
# repository fixture because admission alone intentionally creates no
# operational authorization event.
campaign_id="$(query "SELECT operational_campaign_id FROM campaign_operations_campaign WHERE recommendation_campaign_materialization_id=${materialization_id};")"
budget_output="$(CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER="$login" \
    LSTM_DB_NAME="$database" LSTM_DB_HOST="$host" LSTM_DB_PORT="$port" \
    "$binary" --campaign-operations-budget-grant "$campaign_id" \
    --campaign-operations-expected-budget-version 0 \
    --campaign-operations-budget-value "$expected_members" \
    --campaign-operations-actor "$actor" \
    --campaign-operations-reason "${reason}; pre-Phase-H budget compatibility" \
    --yes 2>&1)" || {
    echo "$budget_output" >&2
    echo "real pre-Phase-H budget command failed" >&2
    exit 1
}
grep -Fq "replay_disposition=recorded" <<<"$budget_output"
[[ "$(query "SELECT count(*) FROM campaign_operations_budget_ledger_entry WHERE operational_campaign_id=${campaign_id} AND ledger_version=1;")" == 1 ]] || {
    echo "pre-Phase-H budget command did not record one initial ledger entry" >&2
    exit 1
}

[[ "$(query "SELECT count(*) || '|' || coalesce(md5(string_agg(row_to_json(e)::text, '|' ORDER BY e.experiment_id)), 'none') FROM experiment e;")" == "$experiment_before" ]] || {
    echo "admission mutated experiment rows" >&2; exit 1;
}
[[ "$(query "SELECT CASE WHEN to_regclass('experiment_scheduler_worker_attempt') IS NULL THEN 'absent' ELSE (SELECT count(*) || '|' || coalesce(md5(string_agg(row_to_json(e)::text, '|' ORDER BY e.worker_attempt_id)), 'none') FROM experiment_scheduler_worker_attempt e) END;")" == "$worker_before" ]] || {
    echo "admission mutated scheduler worker attempts" >&2; exit 1;
}

negative="$(login_query "SELECT rolsuper OR EXISTS (SELECT 1 FROM pg_roles r WHERE r.rolname = ANY(ARRAY['campaign_operations_production_enabler','campaign_operations_production_disabler','campaign_operations_production_dispatcher','campaign_operations_production_dispatch_service','campaign_operations_production_phase5_transactional','campaign_operations_production_reader','campaign_operations_scheduler_protocol_evidence_reader']) AND pg_has_role(current_user,r.rolname,'MEMBER')) FROM pg_roles WHERE rolname=current_user; SELECT has_table_privilege(current_user,'campaign_operations_completion_event','SELECT') OR has_table_privilege(current_user,'campaign_operations_completion_audit_reference_event','SELECT'); SELECT has_function_privilege(current_user,'public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)','EXECUTE') OR has_function_privilege(current_user,'public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)','EXECUTE') OR has_function_privilege(current_user,'public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)','EXECUTE') OR has_function_privilege(current_user,'public.campaign_operations_production_acquire_replay_v2(bigint,integer,text,timestamp with time zone,text,text,text)','EXECUTE');")"
[[ "$negative" == $'f\nf\nf' ]] || { echo "pre-Phase-H login gained forbidden authority: $negative" >&2; exit 1; }

echo "Campaign Operations top-level admission regression passed: materialization=${materialization_id} members=${expected_members} replay=existing_identical experiments_unchanged=true workers_unchanged=true"
