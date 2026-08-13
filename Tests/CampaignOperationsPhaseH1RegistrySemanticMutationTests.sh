#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tool="$repo_root/Scripts/CampaignOperationsH1EvidenceGraph.py"
scratch="$(mktemp -d /tmp/ea-h1-registry-semantic.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT
base="$scratch/base"
mkdir -p "$base"
cp "$repo_root"/Tests/fixtures/CampaignOperationsH1{Requirements,Fixtures,Generators,RuntimeRecords,Validators,ReportEntries,Artifacts,Edges,RegistryDigests}.tsv "$base/"

validate() { H1_REGISTRY_ROOT="$1" python3 "$tool" validate-registries; }
validate "$base" >/dev/null
refresh() {
  local root="$1" file="$2" value
  value="$(shasum -a 256 "$root/$file" | awk '{print $1}')"
  perl -i -pe "if (/\\t\\Q${file}\\E\\t/) { s/[0-9a-f]{64}\$/${value}/ }" \
    "$root/CampaignOperationsH1RegistryDigests.tsv"
}
run_case() {
  local name="$1" code="$2" detail="$3" mutation="$4" root="$scratch/$1"
  cp -R "$base" "$root"
  eval "$mutation"
  if validate "$root" >"$root.log" 2>&1; then
    echo "registry semantic mutation $name unexpectedly passed" >&2; exit 1
  fi
  rg -q "H1R${code} .*stage=registry-semantic-validation detail=${detail}" "$root.log" || {
    cat "$root.log" >&2; exit 1;
  }
  printf 'H1_MUTATION_CASE\tregistry-%s\tregistry:%s\tH1R%s:%s\tregistry-semantic-validation\tH1R%s:%s\tregistry-semantic-validation\tPASS\n' \
    "$name" "$name" "$code" "$detail" "$code" "$detail"
}
run_authority_case() {
  local name="$1" code="$2" key="$3" detail="$4" mutation="$5" root="$scratch/$1"
  cp -R "$base" "$root"
  eval "$mutation"
  if validate "$root" >"$root.log" 2>&1; then
    echo "authority mutation $name unexpectedly passed" >&2; exit 1
  fi
  rg -q "H1A${code} key=${key} stage=normative-requirement-reconciliation detail=${detail}" "$root.log" || {
    cat "$root.log" >&2; exit 1;
  }
  printf 'H1_MUTATION_CASE\tregistry-%s\tregistry:%s\tH1A%s:%s\tnormative-requirement-reconciliation\tH1A%s:%s\tnormative-requirement-reconciliation\tPASS\n' \
    "$name" "$name" "$code" "$detail" "$code" "$detail"
}

run_case forged_generator_path 003 invalid-generator-implementation \
  "perl -i -pe 'if (/GEN-TRACE/) { s@Tests/CampaignOperationsPhaseH1MigrationTests.sh@Tests/NonexistentGenerator.sh@ }' \"\$root/CampaignOperationsH1Generators.tsv\"; refresh \"\$root\" CampaignOperationsH1Generators.tsv"
run_case forged_validator_path 003 invalid-validator-implementation \
  "perl -i -pe 'if (/VAL-TRACE/) { s@Scripts/CampaignOperationsH1EvidenceValidator.py@Scripts/NonexistentValidator.py@ }' \"\$root/CampaignOperationsH1Validators.tsv\"; refresh \"\$root\" CampaignOperationsH1Validators.tsv"
run_authority_case requirement_placeholder 108 H1-ROLE-LOGIN 'authority-mismatch:description' \
  "perl -i -pe 'if (/H1-ROLE-LOGIN/) { my @f=split(/\\t/); \$f[3]=qq{placeholder}; \$_=join(qq{\\t},@f) }' \"\$root/CampaignOperationsH1Requirements.tsv\"; refresh \"\$root\" CampaignOperationsH1Requirements.tsv"
run_authority_case requirement_cardinality_99 108 H1-ROLE-LOGIN 'authority-mismatch:cardinality' \
  "perl -i -pe 'if (/H1-ROLE-LOGIN/) { my @f=split(/\\t/); \$f[9]=99; \$_=join(qq{\\t},@f) }' \"\$root/CampaignOperationsH1Requirements.tsv\"; refresh \"\$root\" CampaignOperationsH1Requirements.tsv"
run_case fixture_semantic 003 invalid-fixture-semantic \
  "perl -i -pe 'if (/H1ROLE001/) { s/\\texecutable\\t/\\tforged\\t/ }' \"\$root/CampaignOperationsH1Fixtures.tsv\"; refresh \"\$root\" CampaignOperationsH1Fixtures.tsv"
run_case generator_node_type 003 wrong-generator-node-type \
  "perl -i -pe 'if (/GEN-TRACE/) { s/\\tgeneric_runtime_record\\t/\\treport_entry\\t/ }' \"\$root/CampaignOperationsH1Generators.tsv\"; refresh \"\$root\" CampaignOperationsH1Generators.tsv"
run_case runtime_record_type 003 invalid-runtime-semantic \
  "perl -i -pe 'if (/RT-H1ROLE001/) { s/\\truntime\\texecutable\\t/\\tgeneric_forgery\\texecutable\\t/ }' \"\$root/CampaignOperationsH1RuntimeRecords.tsv\"; refresh \"\$root\" CampaignOperationsH1RuntimeRecords.tsv"
run_case validator_node_type 003 wrong-validator-semantic \
  "perl -i -pe 'if (/VAL-TRACE/) { s/\\tvalidator_result\\t/\\treport_entry\\t/ }' \"\$root/CampaignOperationsH1Validators.tsv\"; refresh \"\$root\" CampaignOperationsH1Validators.tsv"
run_case report_policy 003 invalid-report-semantic \
  "perl -i -pe 'if (/REP-H1ROLE001/) { s/reconciled_validator_status/forged_status/ }' \"\$root/CampaignOperationsH1ReportEntries.tsv\"; refresh \"\$root\" CampaignOperationsH1ReportEntries.tsv"
run_case artifact_namespace 003 contradictory-runtime-edge:artifact_id \
  "perl -i -pe 'if (/RT-H1ROLE001/) { s/ART-RECORD-H1ROLE001/ART-RECORD-H1ROLE002/ }' \"\$root/CampaignOperationsH1RuntimeRecords.tsv\"; refresh \"\$root\" CampaignOperationsH1RuntimeRecords.tsv"
run_authority_case undeclared_final_assurance 107 H1-ASSURANCE-WORKTREE 'missing-assurance-control' \
  "perl -i -ne 'print unless /H1-ASSURANCE-WORKTREE/' \"\$root/CampaignOperationsH1Requirements.tsv\"; refresh \"\$root\" CampaignOperationsH1Requirements.tsv"

# The reverse edge is an independent row.  Its removal must not be healed from
# the remaining forward row.
root="$scratch/missing_reverse_edge"; cp -R "$base" "$root"
perl -i -ne 'print unless /EDGE-000002\t/' "$root/CampaignOperationsH1Edges.tsv"
refresh "$root" CampaignOperationsH1Edges.tsv
if validate "$root" >"$root.log" 2>&1; then echo "missing reverse edge unexpectedly passed" >&2; exit 1; fi
rg -q 'H1R009 .*stage=explicit-edge-validation detail=missing-explicit-reverse-edge' "$root.log" || { cat "$root.log" >&2; exit 1; }
printf 'H1_MUTATION_CASE\tregistry-missing-reverse-edge\tCampaignOperationsH1Edges.tsv\tH1R009:missing-explicit-reverse-edge\texplicit-edge-validation\tH1R009:missing-explicit-reverse-edge\texplicit-edge-validation\tPASS\n'

echo "Campaign Operations H1 registry semantic mutation tests passed cases=12"
