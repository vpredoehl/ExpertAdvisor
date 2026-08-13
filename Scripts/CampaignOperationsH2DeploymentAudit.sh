#!/usr/bin/env bash
set -euo pipefail
audit_version="h2-deployment-audit-v1"
stage=""
host=""
port=""
user_name=""
database_name=""
while (($#)); do
  case "$1" in
    --stage) stage="$2"; shift 2 ;;
    --host) host="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --user) user_name="$2"; shift 2 ;;
    --database) database_name="$2"; shift 2 ;;
    *) exit 64 ;;
  esac
done
repo_root="$(cd "$(dirname "$BASH_SOURCE")/.." && pwd)"
h1="$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql"
h2="$repo_root/Database/migrations/056_campaign_operations_h2_privilege_deployment_contract.sql"
h2_bind="$repo_root/Database/migrations/057_campaign_operations_h2_production_bind_state_contract.sql"
h8="$repo_root/Database/migrations/059_campaign_operations_direct_sql_readiness_boundary.sql"
h8_acl="$repo_root/Database/manifests/059_campaign_operations_direct_sql_boundary_explicit_acl.tsv"
h8_manifest="$repo_root/Database/manifests/059_campaign_operations_direct_sql_boundary_manifest.sha256"
h1_sum="$(shasum -a 256 "$h1" | awk '{print $1}')"
h2_sum="$(shasum -a 256 "$h2" | awk '{print $1}')"
h2_bind_sum="$(shasum -a 256 "$h2_bind" | awk '{print $1}')"
h8_sum="$(shasum -a 256 "$h8" | awk '{print $1}')"
h8_acl_sum="$(shasum -a 256 "$h8_acl" | awk '{print $1}')"
fail() { printf 'SQLSTATE=%s diagnostic=%s object=%s stage=%s audit_version=%s detail=%s\n' "$1" "$2" "$3" "$stage" "$audit_version" "$4" >&2; exit 1; }
[[ "$h1_sum" == "1b13d3a64336d7cbd55c935396ec42c4c06320105677829f0cf405733e5715fe" ]] || fail 55000 H2A004 local-055 "migration-055-bytes-changed"
[[ "$h8_acl_sum" == "$(awk '{print $2}' "$h8_manifest")" ]] || fail 55000 H8A005 local-059 "manifest-checksum-mismatch"
"$repo_root/Scripts/CampaignOperationsH2ManifestValidator.sh" || fail 55000 H2A004 "manifest-set" "manifest-digest-or-row-count"
result="$(psql -X -qAt -v ON_ERROR_STOP=1 -h "$host" -p "$port" -U "$user_name" "$database_name" -v h1_sum="$h1_sum" -v h2_sum="$h2_sum" -v h2_bind_sum="$h2_bind_sum" -v h8_sum="$h8_sum" -f "$repo_root/Tests/CampaignOperationsPhaseH2DeploymentAudit.sql" 2>&1)" || {
  first="$(printf '%s\n' "$result" | sed -n '1p')"
  code="$(printf '%s' "$first" | cut -d: -f1)"
  case "$code" in H2A004|H2A005|H8A005) fail 55000 "$code" "$first" "$first";; H2A001|H2A002|H2A003|H2A006) fail 42501 "$code" "$first" "$first";; *) fail 55000 H2A004 "$first" "$result";; esac
}
if [[ -n "$result" ]]; then
  first="$(printf '%s\n' "$result" | sed -n '1p')"
  code="$(printf '%s' "$first" | cut -d: -f1)"
  case "$code" in H2A004|H2A005|H8A005) fail 55000 "$code" "$first" "$first";; H2A001|H2A002|H2A003|H2A006) fail 42501 "$code" "$first" "$first";; *) fail 55000 H2A004 "$first" "$result";; esac
fi
printf 'H2_DEPLOYMENT_AUDIT_OK stage=%s audit_version=%s h1_checksum=%s h2_checksum=%s h2_bind_checksum=%s h8_checksum=%s\n' "$stage" "$audit_version" "$h1_sum" "$h2_sum" "$h2_bind_sum" "$h8_sum"
