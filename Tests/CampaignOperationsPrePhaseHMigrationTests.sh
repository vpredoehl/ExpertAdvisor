#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 HOST PORT ADMIN_USER DATABASE" >&2
  exit 64
}
[[ $# -eq 4 ]] || usage

host="$1"
port="$2"
admin_user="$3"
database="$4"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target=(-X -q -v ON_ERROR_STOP=1 -h "$host" -p "$port" -U "$admin_user" "$database")
migration="$repo_root/Database/migrations/064_campaign_operations_pre_phase_h_helper_acl_reconciliation.sql"
overlay="$repo_root/Database/manifests/064_campaign_operations_pre_phase_h_acl_manifest.sql"

# Deployment membership is intentionally outside migration 064. Create the
# reviewed synthetic LOGINs only when the disposable fixture does not already
# provide them, using the accepted H2 deployment-role contract so the
# post-upgrade wrapper sees the deployed post-H1 graph rather than an empty
# capability graph.
psql "${target[@]}" <<'SQL' >/dev/null
DO $$
DECLARE role_name text;
BEGIN
  FOREACH role_name IN ARRAY ARRAY[
      'campaign_operations_authorizer',
      'campaign_operations_auditor',
      'campaign_operations_reader',
      'campaign_operations_controller',
      'campaign_operations_cancellation_coordinator',
      'campaign_operations_reconciler',
      'campaign_operations_recovery',
      'campaign_operations_completion_writer',
      'campaign_operations_production_enabler',
      'campaign_operations_production_disabler',
      'campaign_operations_production_dispatcher',
      'campaign_operations_production_dispatch_service',
      'campaign_operations_production_phase5_transactional',
      'campaign_operations_production_reader',
      'campaign_operations_scheduler_protocol_evidence_reader']
  LOOP
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = role_name) THEN
      EXECUTE format('CREATE ROLE %I NOLOGIN INHERIT', role_name);
    END IF;
  END LOOP;
  FOREACH role_name IN ARRAY ARRAY[
      'campaign_operations_enabler_login',
      'campaign_operations_disabler_login',
      'campaign_operations_manager_login',
      'campaign_operations_dispatch_service_login',
      'campaign_operations_pre_phase_h_login']
  LOOP
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = role_name) THEN
      EXECUTE format('CREATE ROLE %I LOGIN INHERIT', role_name);
    END IF;
  END LOOP;
END
$$;
GRANT campaign_operations_production_enabler,
      campaign_operations_production_reader,
      campaign_operations_scheduler_protocol_evidence_reader
  TO campaign_operations_enabler_login;
GRANT campaign_operations_production_disabler,
      campaign_operations_production_reader
  TO campaign_operations_disabler_login;
GRANT campaign_operations_production_dispatcher,
      campaign_operations_production_phase5_transactional,
      campaign_operations_production_reader,
      campaign_operations_scheduler_protocol_evidence_reader
  TO campaign_operations_manager_login;
GRANT campaign_operations_production_dispatch_service,
      campaign_operations_production_phase5_transactional
  TO campaign_operations_dispatch_service_login;
GRANT campaign_operations_campaign_creator
  TO campaign_operations_pre_phase_h_login;
GRANT campaign_operations_budget_administrator
  TO campaign_operations_pre_phase_h_login;
GRANT campaign_operations_request_acceptor
  TO campaign_operations_pre_phase_h_login;
SQL

# The caller supplies a disposable predecessor database containing migration
# 055 and its H1 roles. Capture the frozen audit definition and confirm the
# normal sealed-055 predecessor has neither compatibility ACL yet.
frozen_h1_audit_definition="$(psql "${target[@]}" -Atqc \
  "SELECT md5(pg_catalog.pg_get_functiondef(\
      'public.campaign_operations_h1_deployment_audit_v1(text,boolean,boolean)'::regprocedure))")"
[[ -n "$frozen_h1_audit_definition" ]] || {
  echo "frozen H1 audit function is missing from the 055 predecessor" >&2
  exit 1
}
sealed_predecessor_grants="$(psql "${target[@]}" -Atqc "
SELECT count(*)
FROM pg_catalog.pg_proc function_row
CROSS JOIN LATERAL pg_catalog.aclexplode(function_row.proacl) acl
JOIN pg_catalog.pg_roles grantee ON grantee.oid = acl.grantee
WHERE function_row.oid =
    'public.lock_campaign_operations_campaign(bigint)'::regprocedure
  AND grantee.rolname IN (
    'campaign_operations_budget_administrator',
    'campaign_operations_request_acceptor')
  AND acl.privilege_type = 'EXECUTE';")"
[[ "$sealed_predecessor_grants" == 0 ]] || {
  echo "055 predecessor unexpectedly already contains 064 compatibility ACL: $sealed_predecessor_grants" >&2
  exit 1
}
[[ "$(psql "${target[@]}" -Atqc "SELECT count(*) FROM schema_migrations WHERE version='064';")" == 0 ]] || {
  echo "064 migration ledger row unexpectedly exists before focused test" >&2
  exit 1
}

# First apply from the normal sealed-055 predecessor.  The second apply is
# the partially applied live predecessor: both grants exist, but schema 064
# is still absent.  The third apply proves ordinary replay safety.
psql "${target[@]}" -f "$migration" >/dev/null
psql "${target[@]}" -f "$migration" >/dev/null
psql "${target[@]}" -f "$migration" >/dev/null

frozen_h1_audit_after="$(psql "${target[@]}" -Atqc \
  "SELECT md5(pg_catalog.pg_get_functiondef(\
      'public.campaign_operations_h1_deployment_audit_v1(text,boolean,boolean)'::regprocedure))")"
[[ "$frozen_h1_audit_after" == "$frozen_h1_audit_definition" ]] || {
  echo "migration 064 changed the frozen H1 audit definition" >&2
  exit 1
}
[[ "$(psql "${target[@]}" -Atqc "SELECT count(*) FROM schema_migrations WHERE version='064';")" == 0 ]] || {
  echo "migration 064 unexpectedly wrote its ledger row" >&2
  exit 1
}

overlay_result="$(psql "${target[@]}" -At -f "$overlay")"
[[ -z "$overlay_result" ]] || {
  printf '%s\n' "$overlay_result" >&2
  echo "pre-Phase-H ACL overlay rejected the corrected state" >&2
  exit 1
}

# Exercise the helper as the reviewed LOGIN identity, not only through SET
# ROLE capability fixtures. The SELECT is empty-safe for catalog-only clones;
# populated predecessor fixtures perform the actual row-locking call.
psql "${target[@]}" <<'SQL' >/dev/null
SELECT to_regclass('public.campaign_operations_campaign') IS NOT NULL
    AS has_campaign_schema \gset
\if :has_campaign_schema
SET SESSION AUTHORIZATION campaign_operations_pre_phase_h_login;
SELECT lock_campaign_operations_campaign(operational_campaign_id)
FROM campaign_operations_campaign
ORDER BY operational_campaign_id
LIMIT 1;
RESET SESSION AUTHORIZATION;
\endif
SQL

psql "${target[@]}" -f "$repo_root/Tests/CampaignOperationsPrePhaseHPrivilegeTests.sql" >/dev/null

# Record 064 only after the forward migration has completed, then exercise the
# deployment wrapper. With the compatibility overlay recorded but H2 absent,
# the ordinary deployed LOGIN graph must fail closed rather than be delegated
# without an installed migration-056 authority.
migration_checksum="$(shasum -a 256 "$migration" | awk '{print $1}')"
psql "${target[@]}" -v migration_checksum="$migration_checksum" <<'SQL' >/dev/null
INSERT INTO schema_migrations(version, filename, checksum)
VALUES ('064',
        '064_campaign_operations_pre_phase_h_helper_acl_reconciliation.sql',
        :'migration_checksum');
SQL
if audit_output="$(bash "$repo_root/Scripts/CampaignOperationsH1DeploymentAudit.sh" \
  --stage post-upgrade --host "$host" --port "$port" \
  --user "$admin_user" --database "$database" 2>&1)"; then
  echo "post-upgrade audit accepted ordinary graph without migration 056" >&2
  exit 1
fi
grep -Fq 'diagnostic=H2A004' <<<"$audit_output"
grep -Fq 'post-upgrade role-graph delegation requires H2 / migration-056 authority' \
  <<<"$audit_output"

# H1 boundary authority remains sealed even though later deployment LOGIN
# memberships are delegated to the later-phase authority at post-upgrade.
hostile_login="campaign_operations_pre_phase_h_h1_boundary_${$}"
hostile_audit_output="$(mktemp -t campaign_operations_pre_phase_h_h1_boundary.XXXXXX)"
trap 'rm -f "$hostile_audit_output"' EXIT
psql "${target[@]}" <<SQL >/dev/null
CREATE ROLE $hostile_login LOGIN INHERIT;
GRANT campaign_operations_h1_boundary_authority TO $hostile_login;
SQL
if bash "$repo_root/Scripts/CampaignOperationsH1DeploymentAudit.sh" \
    --stage post-upgrade --host "$host" --port "$port" \
    --user "$admin_user" --database "$database" \
    > "$hostile_audit_output" 2>&1; then
  cat "$hostile_audit_output" >&2
  echo "post-upgrade audit accepted prohibited H1 boundary membership" >&2
  exit 1
fi
grep -Fq 'diagnostic=H1A003' \
  "$hostile_audit_output"
psql "${target[@]}" -c \
  "REVOKE campaign_operations_h1_boundary_authority FROM $hostile_login; DROP ROLE $hostile_login;" \
  >/dev/null

echo "CAMPAIGN_OPERATIONS_PRE_PHASE_H_MIGRATION_TEST_OK migration064=PASS partial_live_predecessor=PASS replay=PASS frozen_h1=UNCHANGED login_helper=PASS acl_boundary=PASS deployed_login_graph=H2_REQUIRED boundary_isolation=PASS audit_wrapper=FAIL_CLOSED"
