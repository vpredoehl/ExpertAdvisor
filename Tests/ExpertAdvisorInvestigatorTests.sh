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

python3 "${investigator}" domains --format json >"${tmp}/domains.json"
python3 - "${tmp}/domains.json" <<'PY'
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["kind"] == "domains"
expected={"data-pipeline","labels","model","training","inference","experiment-lifecycle",
          "recommendations","profitability","research-automation","scheduler","database"}
assert set(d["domains"]) == expected
assert d["missing"] == []
PY

python3 "${investigator}" evidence scheduler --format json >"${tmp}/scheduler.json"
python3 - "${tmp}/scheduler.json" <<'PY'
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["kind"] == "evidence"
assert d["domain"] == "scheduler"
assert d["domain_authority"]["path"] == "docs/architecture/Volume_XI_Scheduler.md"
assert any(x["section"] == "10" for x in d["constitutional_authority"])
assert any("ADR-0004-" in x["path"] for x in d["accepted_decisions"])
for group in (d["constitutional_authority"], d["domain_authority"]["sections"]):
    assert all(x["line_start"] > 0 and x["line_end"] >= x["line_start"] for x in group)
PY

python3 "${investigator}" evidence database --format json >"${tmp}/database.json"
python3 - "${tmp}/database.json" <<'PY'
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["kind"] == "evidence"
assert d["domain_authority"]["path"] == "docs/architecture/Volume_XII_Database.md"
assert "Database/LSTM_schema.sql" in d["implementation_authority"]
assert any("ADR-0001-" in x["path"] for x in d["accepted_decisions"])
PY

# Every routed domain must resolve without missing authority.
python3 - "${tmp}/domains.json" "${investigator}" <<'PY'
import json, subprocess, sys
from pathlib import Path
domains=json.loads(Path(sys.argv[1]).read_text())["domains"]
for domain in domains:
    subprocess.run(
        ["python3", sys.argv[2], "evidence", domain, "--format", "json"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
PY

# Preserve Q2's read-only regression boundary: commands must not alter tracked content.
before="$(git -C "${repo_root}" diff --no-ext-diff --binary HEAD | shasum -a 256)"
python3 "${investigator}" authorities >/dev/null
python3 "${investigator}" semantics >/dev/null
python3 "${investigator}" workers >/dev/null
python3 "${investigator}" status >/dev/null
python3 "${investigator}" domains >/dev/null
for domain in data-pipeline labels model training inference experiment-lifecycle \
              recommendations profitability research-automation scheduler database; do
    python3 "${investigator}" evidence "${domain}" >/dev/null
done
after="$(git -C "${repo_root}" diff --no-ext-diff --binary HEAD | shasum -a 256)"
test "${before}" = "${after}"

printf '%s\n' "ExpertAdvisorInvestigatorTests passed"
