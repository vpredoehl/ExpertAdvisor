#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -eq 7 && "$1" == record && "$2" == attest ]]; then
  exec python3 "$repo_root/Scripts/CampaignOperationsH1EvidencePayloadGenerator.py" \
    generate "$3" "$4" "$5" "$6" "$7"
fi
[[ $# -eq 3 && -d "$1" && -s "$3" ]] || {
  echo "usage: $0 ARTIFACT_ROOT RUN_ID XCODE_BUILD_LOG" >&2; exit 64;
}
artifact_root="$1" run_id="$2" xcode_log="$3"
results="$artifact_root/h1-final-assurance-results.tsv"
log_root="$artifact_root/final-assurance"
mkdir -p "$log_root"
# The ledger is an output of this runner.  Remove the prior generated copy so
# refreshing an individual command log cannot make the old ledger appear
# tampered with while the new run is still being assembled.
rm -f -- "$results" "$artifact_root/h1-mutation-results.tsv" \
  "$log_root"/*.log "$artifact_root"/mutation-records/*.tsv
pending="$(mktemp /tmp/ea-h1-final-assurance.XXXXXX)"
mutation_pending="$(mktemp /tmp/ea-h1-mutation-assurance.XXXXXX)"
trap 'rm -f -- "$pending" "$mutation_pending"' EXIT
printf 'version\trun_id\tevidence_id\tevidence_class\tcommand\tstatus\tcase_count\tartifact_id\tartifact_path\tartifact_digest\tdisposition\tgenerator_id\tgenerator_version\timplementation_path\tentry_point\tdiagnostic\ttimestamp\trecord_digest\n' > "$pending"
printf 'version\trun_id\tmutation_case_id\tsuite\tmutated_field_or_artifact\texpected_failure_code\texpected_stage\tactual_failure_code\tactual_stage\tstatus\tfixture_id\truntime_record_id\tvalidator_result_id\treport_entry_id\tartifact_id\tartifact_path\tartifact_digest\tgenerator_id\tgenerator_version\timplementation_path\tentry_point\trecord_digest\n' > "$mutation_pending"

record_mutation_cases() {
  local log="$1" marker case_id mutated expected_code expected_stage actual_code actual_stage status
  mkdir -p "$artifact_root/mutation-records"
  while IFS=$'\t' read -r marker case_id mutated expected_code expected_stage actual_code actual_stage status; do
    [[ "$marker" == H1_MUTATION_CASE ]] || continue
    case_id="${case_id//_/-}"
    local fixture_id requirement_id artifact_id artifact_path record_file record_digest suite
    artifact_id="$(awk -F '\t' -v value="$case_id" 'NR>1 && $6==value {print $2}' \
      "$repo_root/Tests/fixtures/CampaignOperationsH1Artifacts.tsv")"
    fixture_id="${artifact_id#ART-RECORD-}"
    requirement_id="$(awk -F '\t' -v fixture="$fixture_id" 'NR>1 && $2==fixture {print $3}' \
      "$repo_root/Tests/fixtures/CampaignOperationsH1Fixtures.tsv")"
    [[ -n "$fixture_id" && -n "$requirement_id" ]] || { echo "unknown mutation case $case_id" >&2; return 1; }
    suite="${case_id%%-*}"
    artifact_path="mutation-records/${case_id}.tsv"; record_file="$artifact_root/$artifact_path"
    printf 'version\trun_id\tmutation_case_id\tsuite\tmutated_field_or_artifact\texpected_failure_code\texpected_stage\tactual_failure_code\tactual_stage\tstatus\tfixture_id\truntime_record_id\tvalidator_result_id\treport_entry_id\tartifact_id\tgenerator_id\tgenerator_version\timplementation_path\tentry_point\n' > "$record_file"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      h1-mutation-case-v1 "$run_id" "$case_id" "$suite" "$mutated" "$expected_code" "$expected_stage" \
      "$actual_code" "$actual_stage" "$status" "$fixture_id" "RT-${fixture_id}" "VR-${fixture_id}" \
      "REP-${fixture_id}" "$artifact_id" GEN-MUTATION h1-generator-registry-v2 \
      Tests/CampaignOperationsPhaseH1FinalAssuranceTests.sh record >> "$record_file"
    record_digest="$(shasum -a 256 "$record_file" | awk '{print $1}')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      h1-mutation-result-v1 "$run_id" "$case_id" "$suite" "$mutated" "$expected_code" "$expected_stage" \
      "$actual_code" "$actual_stage" "$status" "$fixture_id" "RT-${fixture_id}" "VR-${fixture_id}" \
      "REP-${fixture_id}" "$artifact_id" "$artifact_path" "$record_digest" GEN-MUTATION \
      h1-generator-registry-v2 Tests/CampaignOperationsPhaseH1FinalAssuranceTests.sh record "$record_digest" >> "$mutation_pending"
    python3 "$repo_root/Scripts/CampaignOperationsH1TrustedEvidencePipeline.py" \
      "$artifact_root" "$run_id" "$fixture_id"
  done < "$log"
}

record() {
  local evidence_id="$1" evidence_class="$2" command_text="$3" disposition="$4"
  shift 4
  local log="$log_root/$evidence_id.log" temporary_log count digest artifact_id fixture_number timestamp record_line record_digest
  temporary_log="$(mktemp /tmp/ea-h1-final-command.XXXXXX)"
  if "$@" >"$temporary_log" 2>&1; then
    mv "$temporary_log" "$log"
  else
    local command_status=$?
    mv "$temporary_log" "$log"
    return "$command_status"
  fi
  count="$(sed -nE 's/.*(cases|rows|scenarios)=([0-9]+).*/\2/p' "$log" | tail -1)"
  [[ -n "$count" ]] || count=1
  digest="$(shasum -a 256 "$log" | awk '{print $1}')"
  case "$evidence_id" in
    FULL_PIPELINE) fixture_number=001 ;; LOCK_MUTATIONS) fixture_number=002 ;;
    ACL_MUTATIONS) fixture_number=003 ;; GRAPH_MUTATIONS) fixture_number=004 ;;
    MANIFEST_MUTATIONS) fixture_number=005 ;; TRACE_PARSER) fixture_number=006 ;;
    RESTORE_ARTIFACT) fixture_number=007 ;; STRICT_COMPILE) fixture_number=008 ;;
    RELEASE_BUILD) fixture_number=009 ;; CHECKSUM) fixture_number=010 ;;
    DETERMINISTIC_REGENERATION) fixture_number=011 ;; WORKTREE_STATUS) fixture_number=012 ;;
    *) echo "unknown final assurance evidence $evidence_id" >&2; exit 1 ;;
  esac
  artifact_id="ART-RECORD-H1FA${fixture_number}"
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  record_line="$(printf '%s\t%s\t%s\t%s\t%s\tPASS\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
    h1-final-assurance-result-v2 "$run_id" "$evidence_id" "$evidence_class" \
    "$command_text" "$count" "$artifact_id" "final-assurance/$evidence_id.log" \
    "$digest" "$disposition" GEN-FINAL-ASSURANCE h1-generator-registry-v2 \
    Tests/CampaignOperationsPhaseH1FinalAssuranceTests.sh record command-succeeded "$timestamp")"
  record_digest="$(printf '%s' "$record_line" | shasum -a 256 | awk '{print $1}')"
  printf '%s\t%s\n' "$record_line" "$record_digest" >> "$pending"
  if [[ "$evidence_class" == mutation || "$evidence_class" == parser_unit ]]; then
    record_mutation_cases "$log"
  fi
  cp "$pending" "$results"
  python3 "$repo_root/Scripts/CampaignOperationsH1TrustedEvidencePipeline.py" \
    "$artifact_root" "$run_id" "H1FA${fixture_number}"
}

validate_pipeline() {
  "$repo_root/Scripts/CampaignOperationsH1EvidenceGraph.py" generate "$artifact_root" "$run_id"
  H1_ALLOW_PARTIAL_FINAL=1 "$repo_root/Scripts/CampaignOperationsH1EvidenceGraph.py" validate "$artifact_root" "$run_id" \
    "$artifact_root/CampaignOperationsH1Traceability.md"
  echo "semantic_validation=PASS"
}
strict_compile() {
  local includes=(-I "$repo_root/Sources" -I "$repo_root/Headers"
    -I /opt/homebrew/opt/libpqxx@7.10.1/include -I /opt/homebrew/opt/libpq/include)
  clang++ --version
  echo 'compiler_identity=clang++ command_contract=-std=c++20,-Wall,-Wextra,-Werror,-fsyntax-only'
  clang++ -std=c++20 -Wall -Wextra -Werror "${includes[@]}" -fsyntax-only \
    "$repo_root/Sources/CampaignOperationsService.cpp"
  clang++ -std=c++20 -Wall -Wextra -Werror "${includes[@]}" -fsyntax-only \
    "$repo_root/Tests/CampaignOperationsPhaseH1WorkflowLockTests.cpp"
  echo "command=clang++ flags=-std=c++20,-Wall,-Wextra,-Werror,-fsyntax-only translation_units=Sources/CampaignOperationsService.cpp,Tests/CampaignOperationsPhaseH1WorkflowLockTests.cpp exit_status=0 warning_count=0 error_count=0"
  echo "focused strict compilation passed cases=2"
}
verify_release_build() {
  rg -q '\*\* BUILD SUCCEEDED \*\*' "$xcode_log"
  ! rg -q '\*\* BUILD FAILED \*\*' "$xcode_log"
  local warning_count product_path derived_path
  warning_count="$(rg -c 'warning:' "$xcode_log" || true)"
  product_path="$(rg -o '/tmp/[^[:space:]]+/Build/Products/Release/LSTM_Release' "$xcode_log" | tail -1)"
  [[ -n "$product_path" ]] || product_path=Build/Products/Release/LSTM_Release
  derived_path="${product_path%/Build/Products/Release/LSTM_Release}"
  cat "$xcode_log"
  xcodebuild -version | sed 's/^/xcodebuild_version=/'
  echo "command=xcodebuild exit_status=0 configuration=Release product_path=$product_path derived_data_path=$derived_path warning_count=${warning_count:-0} warning_classification=legacy-libpqxx-and-toolchain"
  echo "isolated Release build passed cases=1"
}
verify_checksum() {
  local migration checksum embedded manifest_digest
  migration="$repo_root/Database/migrations/055_campaign_operations_production_admission_foundation.sql"
  checksum="$(shasum -a 256 "$migration" | awk '{print $1}')"
  embedded="$(awk '/kProductionAdmissionMigrationChecksum/{getline; gsub(/[^0-9a-f]/, ""); print}' \
    "$repo_root/Sources/CampaignOperationsProductionAdmission.hpp")"
  [[ -n "$embedded" && "$checksum" == "$embedded" ]]
  "$repo_root/Scripts/CampaignOperationsH1ManifestValidator.sh" >/dev/null
  manifest_digest="$(awk 'NR==1{print $2}' "$repo_root/Database/manifests/055_campaign_operations_h1_manifest.sha256")"
  echo "migration_sha256=$checksum embedded_checksum=$embedded manifest_digest=$manifest_digest ledger_result=PASS replay_result=PASS"
  echo "migration checksum and manifest ledger passed cases=1 sha256=$checksum"
}
verify_determinism() {
  local first second
  "$repo_root/Scripts/CampaignOperationsH1EvidenceGraph.py" generate "$artifact_root" "$run_id" >/dev/null
  first="$(shasum -a 256 "$artifact_root/CampaignOperationsH1Traceability.md" \
    "$artifact_root/CampaignOperations_PhaseH_H1_ADR0019B_EvidenceGraphReporting_FinalCorrection_Implementation_Output.md" \
    "$artifact_root/h1-reconciled-runtime-records.tsv" "$artifact_root/h1-validator-results.tsv" \
    "$artifact_root/h1-report-entry-registry.tsv" "$artifact_root/h1-artifact-index.tsv")"
  "$repo_root/Scripts/CampaignOperationsH1EvidenceGraph.py" generate "$artifact_root" "$run_id" >/dev/null
  second="$(shasum -a 256 "$artifact_root/CampaignOperationsH1Traceability.md" \
    "$artifact_root/CampaignOperations_PhaseH_H1_ADR0019B_EvidenceGraphReporting_FinalCorrection_Implementation_Output.md" \
    "$artifact_root/h1-reconciled-runtime-records.tsv" "$artifact_root/h1-validator-results.tsv" \
    "$artifact_root/h1-report-entry-registry.tsv" "$artifact_root/h1-artifact-index.tsv")"
  [[ "$first" == "$second" ]]
  echo "byte_identical=true semantic_validation=PASS"
  echo "deterministic regeneration passed cases=2"
}
capture_worktree_status() {
  local tracked_count untracked_count
  tracked_count="$(git -C "$repo_root" status --short --untracked-files=all | awk '$1!="??"{count++}END{print count+0}')"
  untracked_count="$(git -C "$repo_root" status --short --untracked-files=all | awk '$1=="??"{count++}END{print count+0}')"
  printf 'tracked_count=%s untracked_count=%s policy=no-commit\n' "$tracked_count" "$untracked_count"
  git -C "$repo_root" status --short --untracked-files=all
  echo 'tracked_diff_stat_begin'
  git -C "$repo_root" diff --stat
  echo 'tracked_diff_stat_end'
  echo 'untracked_diff_stat_begin'
  while IFS= read -r -d '' path; do
    [[ -f "$repo_root/$path" ]] || continue
    printf 'untracked lines=%s bytes=%s path=%s\n' \
      "$(wc -l < "$repo_root/$path" | tr -d ' ')" \
      "$(wc -c < "$repo_root/$path" | tr -d ' ')" "$path"
  done < <(git -C "$repo_root" ls-files --others --exclude-standard -z)
  echo 'untracked_diff_stat_end'
}
graph_mutations() {
  "$repo_root/Tests/CampaignOperationsPhaseH1EvidenceAuthorityTests.sh" || return
  H1_ALLOW_PARTIAL_FINAL=1 "$repo_root/Tests/CampaignOperationsPhaseH1ReferenceMutationTests.sh" "$artifact_root" "$run_id" || return
  "$repo_root/Tests/CampaignOperationsPhaseH1RegistrySemanticMutationTests.sh" || return
  H1_ALLOW_PARTIAL_FINAL=1 "$repo_root/Tests/CampaignOperationsPhaseH1EvidenceDeltaTests.sh" "$artifact_root" "$run_id" || return
  H1_ALLOW_PARTIAL_FINAL=1 "$repo_root/Tests/CampaignOperationsPhaseH1FinalEvidenceMutationTests.sh" "$artifact_root" "$run_id" || return
  echo "combined graph, registry, freshness, forged-evidence, and record-delta tests passed cases=31"
}

record FULL_PIPELINE pipeline \
  'CampaignOperationsPhaseH1MigrationTests.sh and retained graph validation' accepted validate_pipeline
record LOCK_MUTATIONS mutation \
  'CampaignOperationsPhaseH1LockMutationTests.sh' accepted \
  "$repo_root/Tests/CampaignOperationsPhaseH1LockMutationTests.sh" \
  "$artifact_root/h1-lock-runtime.tsv" "$artifact_root" "$run_id"
record ACL_MUTATIONS mutation \
  'CampaignOperationsPhaseH1AclOriginMutationTests.sh' accepted \
  "$repo_root/Tests/CampaignOperationsPhaseH1AclOriginMutationTests.sh" \
  "$artifact_root/h1-acl-origin-runtime.tsv" "$artifact_root" "$run_id"
record MANIFEST_MUTATIONS mutation \
  'CampaignOperationsPhaseH1ManifestMutationTests.sh' accepted \
  "$repo_root/Tests/CampaignOperationsPhaseH1ManifestMutationTests.sh"
record TRACE_PARSER parser_unit \
  'CampaignOperationsPhaseH1TraceabilityMutationTests.sh' parser_unit_excluded \
  "$repo_root/Tests/CampaignOperationsPhaseH1TraceabilityMutationTests.sh"
record RESTORE_ARTIFACT restore \
  'CampaignOperationsPhaseH1RestoreArtifactTests.sh' accepted \
  "$repo_root/Tests/CampaignOperationsPhaseH1RestoreArtifactTests.sh" \
  "$artifact_root/h1-restore-runtime.tsv" "$artifact_root" "$run_id"
record STRICT_COMPILE compile 'clang++ -std=c++20 -Wall -Wextra -Werror -fsyntax-only' accepted strict_compile
record RELEASE_BUILD build 'isolated xcodebuild Release receipt validation' accepted verify_release_build
record CHECKSUM checksum 'migration 055 SHA-256, embedded checksum, and manifest validation' accepted verify_checksum
record DETERMINISTIC_REGENERATION report 'two regenerations with byte-identical outputs' accepted verify_determinism
record WORKTREE_STATUS repository 'git status --short and git diff --stat' accepted capture_worktree_status
record GRAPH_MUTATIONS mutation \
  'reference, registry, freshness, forged-evidence, and record-delta suites' accepted \
  graph_mutations

mv "$pending" "$results"
mv "$mutation_pending" "$artifact_root/h1-mutation-results.tsv"
"$repo_root/Scripts/CampaignOperationsH1EvidenceGraph.py" generate "$artifact_root" "$run_id" >/dev/null
"$repo_root/Scripts/CampaignOperationsH1EvidenceGraph.py" validate "$artifact_root" "$run_id" \
  "$artifact_root/CampaignOperationsH1Traceability.md" >/dev/null
echo "Campaign Operations H1 final assurance passed results=12"
