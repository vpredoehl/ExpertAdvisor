#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_reference.XXXXXX)"
trap 'rm -rf "${test_dir}"' EXIT
python3 "${repo_root}/Scripts/GenerateExpertAdvisorReference.py" --check
python3 "${repo_root}/Scripts/GenerateExpertAdvisorReference.py" --output "${test_dir}/reference.md"
cmp "${repo_root}/docs/ai/ExpertAdvisorReference.md" "${test_dir}/reference.md"
fixture="${test_dir}/repo"
mkdir -p "${fixture}/Headers" "${fixture}/Builds/SemanticWorkers"
cp "${repo_root}/Headers/ModelInputExpansion.hpp" "${fixture}/Headers/"
cp "${repo_root}/Headers/ModelInputContract.hpp" "${fixture}/Headers/"
cp "${repo_root}/Builds/SemanticWorkers/registry.json" "${fixture}/Builds/SemanticWorkers/"
python3 - "${fixture}/Builds/SemanticWorkers/registry.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text()); d["current_layout"]+=1
p.write_text(json.dumps(d)+"\n")
PY
if python3 "${repo_root}/Scripts/GenerateExpertAdvisorReference.py" --repo-root "${fixture}" --output "${test_dir}/bad.md" >"${test_dir}/out" 2>"${test_dir}/err"; then
  echo "expected semantic-layout mismatch to fail" >&2; exit 1
fi
grep -q "semantic layout mismatch" "${test_dir}/err"
printf '%s\n' "ExpertAdvisorReferenceTests passed"
