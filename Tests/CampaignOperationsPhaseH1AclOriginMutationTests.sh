#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixtures="$repo_root/Tests/fixtures/CampaignOperationsH1AclOriginFixtures.tsv"
tool="$repo_root/Scripts/CampaignOperationsH1AclEvidence.py"
validator="$repo_root/Tests/CampaignOperationsPhaseH1AclOriginArtifactTests.sh"
scratch="$(mktemp -d /tmp/ea-h1-acl-origin-mutations.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT
run_id=h1-acl-mutation
base="$scratch/base"; mkdir -p "$base/raw-acl-origin"
while IFS=$'\t' read -r fixture requirement class schema identity column expected actual direction; do
  [[ "$fixture" == fixture_id ]] && continue
  if [[ "$actual" == null ]]; then raw_null=true; raw_acl=""
  else raw_null=false; raw_acl="{}"; fi
  raw="$base/raw-acl-origin/$fixture.tsv"
  printf 'format\th1-acl-origin-raw-v3\n' > "$raw"
  printf 'meta\t%s\t%s\tGEN-ACL-ORIGIN\th1-generator-registry-v2\tScripts/CampaignOperationsH1AclEvidence.py\tgenerate\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tfixture_owner\tr\t42501\tH1A006 exact ACL origin mismatch object=%s stage=database-audit\t%s\tdatabase-audit\tPASS\n' \
    "$run_id" "ART-ACL-RAW-${fixture}" \
    "$fixture" "$requirement" "$class" "$schema" "$identity" "$column" \
    "$expected" "$actual" "$direction" "$raw_null" "$raw_acl" "$identity" \
    "$identity" >> "$raw"
  printf 'acl\tfixture_owner\tSELECT\tfalse\n' >> "$raw"
done < "$fixtures"
python3 "$tool" generate "$fixtures" "$base/raw-acl-origin" \
  "$base/h1-acl-origin-runtime.tsv" "$run_id"
"$validator" "$base/h1-acl-origin-runtime.tsv" "$base" "$run_id" >/dev/null

run_runtime_case() {
  local name="$1" expression="$2" expected="$3" root="$scratch/$1"
  cp -R "$base" "$root"; perl -i -pe "$expression" "$root/h1-acl-origin-runtime.tsv"
  if "$validator" "$root/h1-acl-origin-runtime.tsv" "$root" "$run_id" >"$root.log" 2>&1; then exit 1; fi
  rg -q "H1O001 key=${expected} stage=acl-origin-runtime-reconciliation" "$root.log" || { cat "$root.log" >&2; exit 1; }
  printf 'H1_MUTATION_CASE\tacl-%s\th1-acl-origin-runtime.tsv\tH1O001:%s\tacl-origin-runtime-reconciliation\tH1O001:%s\tacl-origin-runtime-reconciliation\tPASS\n' \
    "$name" "$expected" "$expected"
}
run_runtime_case digest 'if (/\tH1AO001\t/) { s/[0-9a-f]{64}(?=\t[0-9a-f]{64}$)/ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff/ }' raw-digest:H1AO001
run_runtime_case owner 'if (/\tH1AO001\t/) { s/\tfixture_owner\t/\twrong_owner\t/ }' normalized:H1AO001:owner
run_runtime_case object 'if (/\tH1AO001\t/) { s/\th1_acl_origin_schema\t/\twrong_object\t/ }' normalized:H1AO001:object_identity
run_runtime_case expected_origin 'if (/\tH1AO001\t/) { s/\texplicit\tnull\tA\t/\tnull\tnull\tA\t/ }' normalized:H1AO001:expected_origin
run_runtime_case actual_origin 'if (/\tH1AO001\t/) { s/\texplicit\tnull\tA\t/\texplicit\texplicit\tA\t/ }' normalized:H1AO001:actual_origin
run_runtime_case direction 'if (/\tH1AO001\t/) { s/\texplicit\tnull\tA\t/\texplicit\tnull\tB\t/ }' normalized:H1AO001:direction
run_runtime_case sqlstate 'if (/\tH1AO001\t/) { s/\t42501\t42501\t/\t42501\t00000\t/ }' normalized:H1AO001:actual_sqlstate
run_runtime_case diagnostic 'if (/\tH1AO001\t/) { s/\tH1A006\tH1A006\t/\tH1A006\tH1A007\t/ }' normalized:H1AO001:actual_diagnostic
run_runtime_case stage 'if (/\tH1AO001\t/) { s/\tdatabase-audit\tdatabase-audit\t/\tdatabase-audit\twrong-stage\t/ }' normalized:H1AO001:actual_stage

root="$scratch/missing"; cp -R "$base" "$root"; rm "$root/raw-acl-origin/H1AO001.tsv"
if "$validator" "$root/h1-acl-origin-runtime.tsv" "$root" "$run_id" >"$root.log" 2>&1; then exit 1; fi
rg -q 'H1O001 key=missing-raw:H1AO001 ' "$root.log"
printf 'H1_MUTATION_CASE\tacl-missing-raw\traw-acl-origin/H1AO001.tsv\tH1O001:missing-raw:H1AO001\tacl-origin-runtime-reconciliation\tH1O001:missing-raw:H1AO001\tacl-origin-runtime-reconciliation\tPASS\n'

for mutation in raw_acl expanded_tuple raw_owner raw_object; do
  root="$scratch/$mutation"; cp -R "$base" "$root"; raw="$root/raw-acl-origin/H1AO001.tsv"
  case "$mutation" in
    raw_acl) perl -i -pe 'if (/^meta\th1-acl-mutation\tART-ACL-RAW-H1AO001\t/) { s/\ttrue\t\tfixture_owner/\ttrue\t{}\tfixture_owner/ }' "$raw" ;;
    expanded_tuple) perl -i -pe 'if (/^acl\t/) { s/\tSELECT\t/\tUPDATE\t/ }' "$raw" ;;
    raw_owner) perl -i -pe 'if (/^meta\th1-acl-mutation\tART-ACL-RAW-H1AO001\t/) { s/\tfixture_owner\tr\t/\twrong_owner\tr\t/ }' "$raw" ;;
    raw_object) perl -i -pe 'if (/^meta\th1-acl-mutation\tART-ACL-RAW-H1AO001\t/) { s/object=h1_acl_origin_schema/object=wrong_object/ }' "$raw" ;;
  esac
  digest="$(shasum -a 256 "$raw" | awk '{print $1}')"
  perl -i -pe "if (/\\tH1AO001\\t/) { s/[0-9a-f]{64}(?=\\t[0-9a-f]{64}\$)/${digest}/ }" "$root/h1-acl-origin-runtime.tsv"
  if "$validator" "$root/h1-acl-origin-runtime.tsv" "$root" "$run_id" >"$root.log" 2>&1; then
    echo "ACL raw mutation $mutation unexpectedly passed" >&2; exit 1
  fi
  printf 'H1_MUTATION_CASE\tacl-raw-%s\traw-acl-origin/H1AO001.tsv\tH1O001:raw-reconciliation\tacl-origin-runtime-reconciliation\tH1O001:raw-reconciliation\tacl-origin-runtime-reconciliation\tPASS\n' "$mutation"
done
echo "Campaign Operations H1 ACL-origin mutation tests passed cases=14"
