#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
validator="$repo_root/Tests/CampaignOperationsPhaseH1TraceabilityTests.sh"
base_manifest="$repo_root/Tests/fixtures/CampaignOperationsH1Traceability.tsv"
base_locks="$repo_root/Tests/fixtures/CampaignOperationsH1LockPathMatrix.tsv"
scratch="$(mktemp -d /tmp/ea-h1-trace-mutations.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT
base="$scratch/base"
mkdir -p "$base"
printf 'h1-mutation-baseline\n' > "$base/run-id"
cp "$base_manifest" "$base/trace.tsv"
cp "$base_locks" "$base/locks.tsv"
printf '%s\n' 'result_format_version	run_id	fixture_id	requirement_id	test_source	timestamp	actual_status	sqlstate	diagnostic	object_identity	stage	operation_id	artifact_path	artifact_digest	lock_outcome	cycle_detected	cleanup_result' \
    > "$base/h1-runtime-results.tsv"
printf 'header\n' > "$base/h1-lock-runtime.tsv"
printf 'header\n' > "$base/h1-restore-runtime.tsv"
printf 'header\n' > "$base/h1-acl-origin-runtime.tsv"

while IFS=$'\t' read -r requirement _ implementation fixture _ _ _ sqlstate \
    diagnostic object stage artifact expected_status; do
    [[ "$requirement" == "requirement_id" ]] && continue
    if [[ ! -e "$base/$artifact" ]]; then
        printf 'runtime artifact for %s\n' "$artifact" > "$base/$artifact"
    fi
    digest="$(shasum -a 256 "$base/$artifact" | awk '{print $1}')"
    operation_id="operation-$fixture"
    lock_outcome="not-applicable"
    if [[ "$fixture" == H1LOCK* ]]; then
        operation_id="$fixture"
        lock_outcome="$(awk -F '\t' -v id="$fixture" '$1==id{print $7}' "$base/locks.tsv")"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        h1-runtime-result-v2 h1-mutation-baseline "$fixture" "$requirement" \
        "$implementation" 2026-08-01T12:00:00Z "$expected_status" "$sqlstate" \
        "$diagnostic" "$object" "$stage" "$operation_id" "$artifact" \
        "$digest" "$lock_outcome" false PASS >> "$base/h1-runtime-results.tsv"
done < "$base/trace.tsv"

# The synthetic parser fixture is emitted only as v2.  Expand the convenient
# 17-field construction above before invoking the acceptance parser.
awk -F '\t' 'BEGIN { OFS="\t" }
  NR==1 {
    print $0,"generator_id","generator_version","generator_implementation",
      "generator_entry_point","emitted_runtime_record_id","output_artifact_id",
      "record_digest";
    next
  }
  {
    print $0,"GEN-TRACE","h1-generator-registry-v2",
      "Tests/CampaignOperationsPhaseH1TraceabilityMutationTests.sh","synthetic",
      "RT-" $3,"ART-RECORD-" $3,
      "0000000000000000000000000000000000000000000000000000000000000000"
  }' "$base/h1-runtime-results.tsv" > "$base/h1-runtime-results-v2.tsv"
mv "$base/h1-runtime-results-v2.tsv" "$base/h1-runtime-results.tsv"

H1_TRACEABILITY_MANIFEST="$base/trace.tsv" H1_LOCK_MANIFEST="$base/locks.tsv" \
H1_RUNTIME_RESULTS="$base/h1-runtime-results.tsv" \
    "$validator" --validate "$base" h1-mutation-baseline >/dev/null

run_case() {
    local label="$1" expected_code="$2" mutation="$3"
    local case_root="$scratch/$label"
    cp -R "$base" "$case_root"
    eval "$mutation"
    local log="$case_root/validator.log"
    if H1_TRACEABILITY_MANIFEST="$case_root/trace.tsv" \
       H1_LOCK_MANIFEST="$case_root/locks.tsv" \
       H1_RUNTIME_RESULTS="$case_root/h1-runtime-results.tsv" \
       "$validator" --validate "$case_root" h1-mutation-baseline >"$log" 2>&1; then
        echo "traceability mutation $label unexpectedly passed" >&2
        exit 1
    fi
    rg -q "H1T${expected_code} .*stage=runtime-traceability-reconciliation" "$log" || {
        cat "$log" >&2; echo "traceability mutation $label reached wrong failure" >&2; exit 1;
    }
    printf 'H1_MUTATION_CASE\tparser-%s\ttraceability-parser\tH1T%s\truntime-traceability-reconciliation\tH1T%s\truntime-traceability-reconciliation\tPASS\n' \
      "$label" "$expected_code" "$expected_code"
}

run_case changed_sqlstate 009 \
  "perl -i -pe 'if (/^H1-ROLE-LOGIN\\t/) { s/42501/55000/ }' \"\$case_root/trace.tsv\""
run_case changed_diagnostic 009 \
  "perl -i -pe 'if (/^H1-ROLE-LOGIN\\t/) { s/H1A002/H1A003/ }' \"\$case_root/trace.tsv\""
run_case changed_object 009 \
  "perl -i -pe 'if (/^H1-ROLE-LOGIN\\t/) { s/campaign_operations_h1_boundary_authority/wrong_role/ }' \"\$case_root/trace.tsv\""
run_case changed_stage 009 \
  "perl -i -pe 'if (/^H1-ROLE-LOGIN\\t/) { s/\\tpreflight\\t/\\twrong-stage\\t/ }' \"\$case_root/trace.tsv\""
run_case missing_fixture 009 \
  "perl -i -pe 'if (/^H1-ROLE-LOGIN\\t/) { s/H1ROLE001/H1ROLE999/ }' \"\$case_root/trace.tsv\""
run_case missing_runtime 009 \
  "perl -i -ne 'print unless /\\tH1ROLE001\\tH1-ROLE-LOGIN\\t/' \"\$case_root/h1-runtime-results.tsv\""
run_case duplicate_runtime 006 \
  "rg '\\tH1ROLE001\\tH1-ROLE-LOGIN\\t' \"\$case_root/h1-runtime-results.tsv\" >> \"\$case_root/h1-runtime-results.tsv\""
run_case runtime_disagreement 009 \
  "perl -i -pe 'if (/\\tH1ROLE001\\tH1-ROLE-LOGIN\\t/) { s/\\tEXPECTED_FAILURE\\t/\\tSUCCESS\\t/ }' \"\$case_root/h1-runtime-results.tsv\""
run_case deleted_manifest_row 009 \
  "perl -i -ne 'print unless /^H1-ROLE-LOGIN\\t/' \"\$case_root/trace.tsv\""
run_case duplicate_requirement 002 \
  "perl -i -pe 'if (/^H1-ROLE-SUPERUSER\\t/) { s/^H1-ROLE-SUPERUSER/H1-ROLE-LOGIN/ }' \"\$case_root/trace.tsv\""
run_case unknown_status 006 \
  "perl -i -pe 'if (/\\tH1ROLE001\\tH1-ROLE-LOGIN\\t/) { s/\\tEXPECTED_FAILURE\\t/\\tUNKNOWN\\t/ }' \"\$case_root/h1-runtime-results.tsv\""
run_case stale_run_id 006 \
  "perl -i -pe 'if (/\\tH1ROLE001\\tH1-ROLE-LOGIN\\t/) { s/h1-mutation-baseline/h1-stale/ }' \"\$case_root/h1-runtime-results.tsv\""
run_case missing_artifact 007 \
  "artifact=\$(awk -F '\\t' '\$3==\"H1ROLE001\"{print \$13}' \"\$case_root/h1-runtime-results.tsv\"); rm -f -- \"\$case_root/\$artifact\""
run_case artifact_mismatch 008 \
  "artifact=\$(awk -F '\\t' '\$3==\"H1ROLE001\"{print \$13}' \"\$case_root/h1-runtime-results.tsv\"); printf changed >> \"\$case_root/\$artifact\""
run_case pass_without_evidence 009 \
  "perl -i -pe 'if (/\\tH1ROLE001\\tH1-ROLE-LOGIN\\t/) { s/\\tEXPECTED_FAILURE\\t/\\tSUCCESS\\t/; s/\\t42501\\t/\\t00000\\t/ }' \"\$case_root/h1-runtime-results.tsv\""
run_case reversed_lock_direction 011 \
  "perl -i -pe 'if (/^H1LOCK001\\t/) { my @f=split(/\\t/); (\$f[6],\$f[7])=(\$f[7],\$f[6]); \$_=join(qq{\\t},@f) }' \"\$case_root/locks.tsv\""
run_case wrong_lock_identity 009 \
  "perl -i -pe 'if (/\\tH1LOCK001\\tH1-LOCK-001\\t/) { s/acquisition\\/enable/acquisition\\/wrong/ }' \"\$case_root/h1-runtime-results.tsv\""
run_case incorrect_cycle 009 \
  "perl -i -pe 'if (/\\tH1LOCK001\\tH1-LOCK-001\\t/) { s/\\tfalse\\tPASS\\t/\\ttrue\\tPASS\\t/ }' \"\$case_root/h1-runtime-results.tsv\""

# This suite exercises the v2 traceability parser with synthetic rows and is
# excluded from ADR-0019B acceptance evidence.  Report
# freshness is tested against a complete authentic evidence graph by
# CampaignOperationsPhaseH1ReferenceMutationTests.sh.
echo "Campaign Operations H1 traceability parser-unit mutation tests passed cases=18 acceptance_evidence=false"
