#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
/usr/bin/python3 "${repo_root}/Tests/SemanticWorkerHistoricalTrainingCandidatePublisherTests.py"
printf '%s\n' "SemanticWorkerHistoricalTrainingCandidatePublisherTests passed"
