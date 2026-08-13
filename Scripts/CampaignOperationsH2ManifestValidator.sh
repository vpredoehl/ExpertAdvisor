#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
migration="$repo_root/Database/migrations/056_campaign_operations_h2_privilege_deployment_contract.sql"
manifest="$repo_root/Database/manifests/056_campaign_operations_h2_explicit_acl.tsv"
digest_file="$repo_root/Database/manifests/056_campaign_operations_h2_manifest.sha256"

migration_sum="$(shasum -a 256 "$migration" | awk '{print $1}')"
manifest_sum="$(shasum -a 256 "$manifest" | awk '{print $1}')"
expected_sum="$(printf '%s\n%s\n' "$migration_sum" "$manifest_sum" | shasum -a 256 | awk '{print $1}')"
actual_sum="$(awk 'NF == 2 && $1 == "h2-manifest-set-v1" { print $2 }' "$digest_file")"
[[ "$actual_sum" == "$expected_sum" ]] || {
  printf 'H2A004 manifest digest mismatch expected=%s actual=%s\n' "$expected_sum" "$actual_sum" >&2
  exit 1
}

[[ "$(awk 'NR > 1 { count++ } END { print count + 0 }' "$manifest")" == 12 &&
   "$(awk -F '\t' 'NR > 1 && NF != 8 { bad++ } END { print bad + 0 }' "$manifest")" == 0 ]] || {
  printf 'H2A004 manifest row count mismatch\n' >&2
  exit 1
}

printf 'H2_MANIFEST_OK migration_checksum=%s manifest_checksum=%s set_checksum=%s\n' \
  "$migration_sum" "$manifest_sum" "$expected_sum"
