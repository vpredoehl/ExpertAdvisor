#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -ge 2 && $# -le 3 && -d "$1" ]] || {
  echo "usage: $0 ARTIFACT_ROOT RUN_ID [REPORT]" >&2; exit 64;
}
root="$1" run_id="$2" report="${3:-}"
H1_RUNTIME_RESULTS="$root/h1-runtime-results.tsv" \
  "$repo_root/Tests/CampaignOperationsPhaseH1TraceabilityTests.sh" \
  --validate "$root" "$run_id" >/dev/null
"$repo_root/Tests/CampaignOperationsPhaseH1LockArtifactTests.sh" \
  "$root/h1-lock-runtime.tsv" "$root" "$run_id" >/dev/null
"$repo_root/Tests/CampaignOperationsPhaseH1RestoreArtifactTests.sh" \
  "$root/h1-restore-runtime.tsv" "$root" "$run_id" >/dev/null
"$repo_root/Tests/CampaignOperationsPhaseH1AclOriginArtifactTests.sh" \
  "$root/h1-acl-origin-runtime.tsv" "$root" "$run_id" >/dev/null
"$repo_root/Tests/CampaignOperationsPhaseH1PreEnablementArtifactTests.sh" \
  "$root/h1-pre-enablement-runtime.tsv" "$run_id" >/dev/null
"$repo_root/Tests/CampaignOperationsPhaseH1UniquenessInvariantTests.sh" \
  "$root/h1-uniqueness-invariant-runtime.tsv" "$run_id" >/dev/null
mode=generate; [[ -n "$report" ]] && mode=validate
python3 "$repo_root/Scripts/CampaignOperationsH1EvidenceGraph.py" \
  "$mode" "$root" "$run_id" ${report:+"$report"}
