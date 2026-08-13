#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 1 && $# -le 3 && -s "$1" ]] || {
  echo "usage: $0 /path/to/h1-restore-runtime.tsv [ARTIFACT_ROOT] [RUN_ID]" >&2; exit 64;
}
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="${2:-$(cd "$(dirname "$1")" && pwd)}"
python3 "$repo_root/Scripts/CampaignOperationsH1RestoreEvidence.py" \
  --snapshot-bundle "$1" "$root" ${3:+"$3"}
