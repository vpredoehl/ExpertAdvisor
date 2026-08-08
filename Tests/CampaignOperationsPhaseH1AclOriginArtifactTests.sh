#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -ge 1 && $# -le 3 && -s "$1" ]] || {
  echo "usage: $0 RUNTIME [ARTIFACT_ROOT] [RUN_ID]" >&2; exit 64;
}
root="${2:-$(cd "$(dirname "$1")" && pwd)}"
run_id="${3:-$(awk -F '\t' 'NR==2{print $2}' "$1")}"
python3 "$repo_root/Scripts/CampaignOperationsH1AclEvidence.py" validate \
  "$repo_root/Tests/fixtures/CampaignOperationsH1AclOriginFixtures.tsv" \
  "$1" "$root" "$run_id"
