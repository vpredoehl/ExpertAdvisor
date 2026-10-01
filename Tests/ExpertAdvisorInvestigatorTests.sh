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


python3 "${investigator}" investigations --format json >"${tmp}/investigations.json"
python3 - "${tmp}/investigations.json" <<'PY'
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["kind"] == "investigations"
expected={"semantic-compatibility","scheduler-dispatch","experiment-identity",
          "training-configuration","inference-evaluation","profitability-evaluation",
          "recommendation-governance","research-automation"}
assert set(d["investigations"]) == expected
assert d["invalid_domains"] == []
PY

python3 "${investigator}" plan semantic-compatibility --format json >"${tmp}/plan.json"
python3 - "${tmp}/plan.json" "${investigator}" <<'PY'
import json, subprocess, sys
from pathlib import Path
plan=json.loads(Path(sys.argv[1]).read_text())
assert plan["kind"] == "plan"
assert plan["investigation"] == "semantic-compatibility"
assert plan["domains"] == ["model","training","inference","scheduler",
                           "experiment-lifecycle","database"]
assert len(plan["evidence"]) == len(plan["domains"])
for domain, bundle in zip(plan["domains"], plan["evidence"]):
    assert bundle["domain"] == domain
    direct=json.loads(subprocess.check_output(
        ["python3", sys.argv[2], "evidence", domain, "--format", "json"],
        text=True,
    ))
    assert bundle == direct
assert "not a diagnosis" in plan["boundary"]
assert "live-state" in plan["boundary"]
assert "authorization" in plan["boundary"]
PY

# Q4 profiles may select Q3 domain names only; they do not carry authority paths.
python3 - "${investigator}" <<'PY'
import importlib.util, sys
from pathlib import Path

path=Path(sys.argv[1])
sys.path.insert(0, str(path.parent))

spec=importlib.util.spec_from_file_location("investigator_q4", path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

for name, definition in m.INVESTIGATION_PROFILES.items():
    assert set(definition) == {"purpose", "domains"}
    assert definition["domains"]
    assert len(definition["domains"]) == len(set(definition["domains"]))
    for domain in definition["domains"]:
        assert domain in m.DOMAIN_ROUTES
        assert "/" not in domain
PY

# Every Q4 plan must resolve and exactly compose its declared Q3 domains.
python3 - "${tmp}/investigations.json" "${investigator}" <<'PY'
import json, subprocess, sys
from pathlib import Path
names=json.loads(Path(sys.argv[1]).read_text())["investigations"]
for name in names:
    raw=subprocess.check_output(
        ["python3", sys.argv[2], "plan", name, "--format", "json"],
        text=True,
    )
    plan=json.loads(raw)
    assert plan["kind"] == "plan"
    assert plan["investigation"] == name
    assert [x["domain"] for x in plan["evidence"]] == plan["domains"]
PY

if python3 "${investigator}" plan definitely-not-a-profile >/dev/null 2>&1; then
    echo "unknown Q4 investigation profile unexpectedly succeeded" >&2
    exit 1
fi

# Preserve Q2's read-only regression boundary: commands must not alter tracked content.
before="$(git -C "${repo_root}" diff --no-ext-diff --binary HEAD | shasum -a 256)"
python3 "${investigator}" authorities >/dev/null
python3 "${investigator}" semantics >/dev/null
python3 "${investigator}" workers >/dev/null
python3 "${investigator}" status >/dev/null
python3 "${investigator}" domains >/dev/null
python3 "${investigator}" investigations >/dev/null
for profile in semantic-compatibility scheduler-dispatch experiment-identity \
               training-configuration inference-evaluation profitability-evaluation \
               recommendation-governance research-automation; do
    python3 "${investigator}" plan "${profile}" >/dev/null
done
for domain in data-pipeline labels model training inference experiment-lifecycle \
              recommendations profitability research-automation scheduler database; do
    python3 "${investigator}" evidence "${domain}" >/dev/null
done
after="$(git -C "${repo_root}" diff --no-ext-diff --binary HEAD | shasum -a 256)"
test "${before}" = "${after}"

printf '%s\n' "ExpertAdvisorInvestigatorTests passed"
