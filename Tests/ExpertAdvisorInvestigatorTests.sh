#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d /tmp/ea_investigator.XXXXXX)"
trap 'rm -rf "${tmp}"' EXIT

investigator="${repo_root}/Scripts/ExpertAdvisorInvestigator.py"

python3 "${investigator}" authorities >"${tmp}/authorities.txt"
grep -q '^physical_schema: Database/LSTM_schema.sql$' "${tmp}/authorities.txt"
grep -q '^missing: none$' "${tmp}/authorities.txt"

python3 "${investigator}" semantics --format json >"${tmp}/semantics.json"
python3 - "${tmp}/semantics.json" <<'PY'
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["kind"] == "semantics"
assert d["current_layout"] > 0
assert d["model_input_width"] > d["return_suffix_width"]
assert d["tensor_feature_width"] == d["model_input_width"] - d["return_suffix_width"]
assert d["layout_chain"][-1]["layout"] == d["current_layout"]
PY

python3 "${investigator}" workers --format json >"${tmp}/workers.json"
python3 - "${tmp}/workers.json" <<'PY'
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["kind"] == "workers"
assert sorted(w["role"] for w in d["workers"]) == ["infer", "train"]
assert all(w["layout"] == d["current_layout"] for w in d["workers"])
assert len({w["model_input_width"] for w in d["workers"]}) == 1
PY

python3 "${investigator}" status --format json >"${tmp}/status.json"
python3 - "${tmp}/status.json" <<'PY'
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["kind"] == "status"
assert len(d["head"]) == 40
assert isinstance(d["clean"], bool)
assert d["reference_current"] is True
PY

# The investigator is read-only: these commands must not alter tracked content.
before="$(git -C "${repo_root}" diff --no-ext-diff --binary HEAD | shasum -a 256)"
python3 "${investigator}" authorities >/dev/null
python3 "${investigator}" semantics >/dev/null
python3 "${investigator}" workers >/dev/null
python3 "${investigator}" status >/dev/null
after="$(git -C "${repo_root}" diff --no-ext-diff --binary HEAD | shasum -a 256)"
test "${before}" = "${after}"

printf '%s\n' "ExpertAdvisorInvestigatorTests passed"
