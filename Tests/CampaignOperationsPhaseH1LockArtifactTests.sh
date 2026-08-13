#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -ge 1 && $# -le 3 && -s "$1" ]] || {
  echo "usage: $0 RUNTIME_TSV [ARTIFACT_ROOT] [RUN_ID]" >&2; exit 64;
}
artifact_root="${2:-$(cd "$(dirname "$1")" && pwd)}"
python3 "$repo_root/Scripts/CampaignOperationsH1LockEvidence.py" validate \
  "$repo_root/Tests/fixtures/CampaignOperationsH1LockPathMatrix.tsv" \
  "$1" "$artifact_root" ${3:+"$3"}
