#!/usr/bin/env bash
set -euo pipefail

# Environment-backed regression for request acceptance through the reviewed
# pre-Phase-H LOGIN. The target must be a disposable fixture with an admitted
# campaign, effective authorization, active budget, and no request yet.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${1:-$repo_root/DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release}"
database="${LSTM_DB_NAME:-LSTM}"
host="${LSTM_DB_HOST:-127.0.0.1}"
port="${LSTM_DB_PORT:-5432}"
campaign_id="${CAMPAIGN_OPERATIONS_PRE_PHASE_H_REQUEST_TEST_CAMPAIGN_ID:?CAMPAIGN_OPERATIONS_PRE_PHASE_H_REQUEST_TEST_CAMPAIGN_ID is required}"
actor="${CAMPAIGN_OPERATIONS_PRE_PHASE_H_REQUEST_TEST_ACTOR:-pre-phase-h-request-test}"
reason="${CAMPAIGN_OPERATIONS_PRE_PHASE_H_REQUEST_TEST_REASON:-Exercise the pre-Phase-H request acceptor helper path}"
login="${CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER:?CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER is required}"

[[ -x "$binary" ]] || { echo "missing executable: $binary" >&2; exit 2; }

psql_admin=(psql -X -v ON_ERROR_STOP=1 -h "$host" -p "$port" -d "$database")
psql_login=(psql -X -v ON_ERROR_STOP=1 -h "$host" -p "$port" -U "$login" -d "$database")
query() { "${psql_admin[@]}" -Atqc "$1"; }

fixture_state="$(query "
SELECT count(*) || '|' ||
       count(*) FILTER (WHERE authorization.authorization_event_id IS NOT NULL) || '|' ||
       count(*) FILTER (WHERE budget.budget_ledger_entry_id IS NOT NULL) || '|' ||
       count(*) FILTER (WHERE request.operational_request_id IS NULL)
FROM campaign_operations_campaign campaign
LEFT JOIN LATERAL (
  SELECT authorization_event_id
  FROM campaign_operations_authorization_event authorization
  WHERE authorization.operational_campaign_id = campaign.operational_campaign_id
    AND authorization.event_kind = 'granted'
  ORDER BY authorization.chain_version DESC LIMIT 1
) authorization ON true
LEFT JOIN LATERAL (
  SELECT budget_ledger_entry_id
  FROM campaign_operations_budget_ledger_entry budget
  WHERE budget.operational_campaign_id = campaign.operational_campaign_id
    AND budget.ledger_status = 'active'
  ORDER BY budget.ledger_version DESC LIMIT 1
) budget ON true
LEFT JOIN campaign_operations_operational_request request
  ON request.operational_campaign_id = campaign.operational_campaign_id
WHERE campaign.operational_campaign_id = ${campaign_id};")"
[[ "$fixture_state" == "1|1|1|1" ]] || {
    echo "campaign ${campaign_id} is not a request-acceptance fixture: ${fixture_state}" >&2
    exit 2
}

output="$(CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER="$login" \
    LSTM_DB_NAME="$database" LSTM_DB_HOST="$host" LSTM_DB_PORT="$port" \
    "$binary" --campaign-operations-accept-request "$campaign_id" \
    --campaign-operations-actor "$actor" \
    --campaign-operations-reason "$reason" --yes 2>&1)" || {
    echo "$output" >&2
    echo "real pre-Phase-H request acceptance command failed" >&2
    exit 1
}
grep -Fq "CAMPAIGN_OPERATIONS_REQUEST_ACCEPTED" <<<"$output"
grep -Fq "replay_disposition=recorded" <<<"$output"
[[ "$(query "SELECT count(*) FROM campaign_operations_operational_request WHERE operational_campaign_id=${campaign_id};")" == 1 ]] || {
    echo "pre-Phase-H request command did not record one request" >&2
    exit 1
}

forbidden="$("${psql_login[@]}" -Atqc "SELECT rolsuper OR EXISTS (SELECT 1 FROM pg_roles r WHERE r.rolname LIKE 'campaign_operations_production_%' AND pg_has_role(current_user,r.rolname,'MEMBER')) FROM pg_roles WHERE rolname=current_user; SELECT has_function_privilege(current_user,'public.campaign_operations_production_acquire_replay_v2(bigint,integer,text,timestamp with time zone,text,text,text)','EXECUTE');")"
[[ "$forbidden" == $'f\nf' ]] || {
    echo "pre-Phase-H request LOGIN gained forbidden authority: ${forbidden}" >&2
    exit 1
}

echo "Campaign Operations pre-Phase-H request acceptance regression passed: campaign=${campaign_id} helper=campaign-lock login=${login} production_authority=false"
