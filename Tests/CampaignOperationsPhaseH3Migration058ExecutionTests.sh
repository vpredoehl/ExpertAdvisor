#!/usr/bin/env bash
set -euo pipefail

# Execute 058 against the repository-native disposable H2 predecessor.  This
# specifically reaches the compatibility-function REVOKE: the former typo
# named a function that the migration has not created, so psql -f exits nonzero.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="$(mktemp -d /tmp/ea-h3-migration058.XXXXXX)"
cluster_root=""

cleanup() {
  if [[ -n "$cluster_root" && -f "$cluster_root/data/postmaster.pid" ]]; then
    pg_ctl -D "$cluster_root/data" -m immediate stop >/dev/null 2>&1 || true
  fi
  rm -rf -- "$tmp_root"
}
trap cleanup EXIT

baseline="$tmp_root/baseline"
H2_PRESERVE_CLUSTER_ROOT="$tmp_root/cluster-root" \
H2_PRESERVE_BASELINE_ROOT="$baseline" \
  bash "$repo_root/Tests/CampaignOperationsPhaseH2WorkflowTests.sh" \
  >"$tmp_root/h2-predecessor.log" 2>&1

IFS='|' read -r cluster_root database socket < "$baseline"
target=(-h "$socket" -p 5432 -U campaign_manager_login)

# The H2 fixture has the accepted 057 schema.  Its runner pre-records later
# repository bytes only to prevent their installation, so remove that synthetic
# 058 ledger entry and prove no H3 object exists before the actual execution.
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<'SQL'
DELETE FROM schema_migrations WHERE version = '058';
SQL
[[ "$(psql "${target[@]}" -At -v ON_ERROR_STOP=1 "$database" -c \
  "SELECT EXISTS (SELECT 1 FROM schema_migrations WHERE version = '057') AND\
          NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = '058') AND\
          to_regclass('campaign_operations_h2_manager_key_compatibility') IS NULL AND\
          to_regprocedure('public.reject_campaign_operations_h2_manager_key_compat_mutation()') IS NULL;")" == t ]]

# Execute the former spelling against an isolated clone to prove this regression
# rejects the exact pre-correction bytes at the unconditional REVOKE.
former_typo_database="expertadvisor_h3_058_typo_$$"
former_migration="$tmp_root/058-former-typo.sql"
cp "$repo_root/Database/migrations/058_campaign_operations_h3_manager_run_once.sql" "$former_migration"
perl -0pi -e 's/(REVOKE ALL PRIVILEGES ON FUNCTION\s+)reject_campaign_operations_h2_manager_key_compat_mutation\(\)/$1reject_campaign_operations_h2_manager_key_compatibility_mutation()/g' \
  "$former_migration"
createdb "${target[@]}" -T "$database" "$former_typo_database"
if psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$former_typo_database" -f "$former_migration" \
    >"$tmp_root/former-typo.log" 2>&1; then
  echo "former 058 compatibility-function typo unexpectedly installed" >&2
  exit 1
fi
rg -q 'reject_campaign_operations_h2_manager_key_compatibility_mutation' \
  "$tmp_root/former-typo.log"
rg -q 'does not exist' "$tmp_root/former-typo.log"
dropdb "${target[@]}" "$former_typo_database"

psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -f \
  "$repo_root/Database/migrations/058_campaign_operations_h3_manager_run_once.sql"

checksum="$(shasum -a 256 "$repo_root/Database/migrations/058_campaign_operations_h3_manager_run_once.sql" | awk '{print $1}')"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
  "INSERT INTO schema_migrations(version,filename,checksum) VALUES \
   ('058','058_campaign_operations_h3_manager_run_once.sql','$checksum');"
[[ "$(psql "${target[@]}" -At -v ON_ERROR_STOP=1 "$database" -c \
  "SELECT to_regclass('campaign_operations_h2_manager_key_compatibility') IS NOT NULL AND\
          to_regprocedure('public.reject_campaign_operations_h2_manager_key_compat_mutation()') IS NOT NULL AND\
          EXISTS (SELECT 1 FROM schema_migrations WHERE version = '058' AND\
              filename = '058_campaign_operations_h3_manager_run_once.sql' AND checksum = '$checksum');")" == t ]]

echo "H3_MIGRATION058_EXECUTION_REGRESSION_OK predecessor=057 former_typo_rejected=PASS disposable_install=PASS compatibility_revoke=PASS"
