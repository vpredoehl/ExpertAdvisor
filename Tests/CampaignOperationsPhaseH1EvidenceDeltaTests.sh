#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 && -d "$1" ]] || { echo "usage: $0 ARTIFACT_ROOT RUN_ID" >&2; exit 64; }
source_root="$1" run_id="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$(mktemp -d /tmp/ea-h1-evidence-delta.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT

assert_local_delta() {
  local name="$1" target="$2" mutation="$3" root
  root="$scratch/$name"
  cp -R "$source_root" "$root"
  python3 "$repo_root/Scripts/CampaignOperationsH1EvidenceGraph.py" generate "$root" "$run_id" >/dev/null
  cp "$root/h1-reconciled-runtime-records.tsv" "$scratch/$name.runtime.before"
  cp "$root/h1-validator-results.tsv" "$scratch/$name.validator.before"
  cp "$root/h1-report-entry-registry.tsv" "$scratch/$name.report.before"
  eval "$mutation"
  python3 "$repo_root/Scripts/CampaignOperationsH1EvidenceGraph.py" generate "$root" "$run_id" >/dev/null
  python3 "$repo_root/Scripts/CampaignOperationsH1EvidenceGraph.py" validate "$root" "$run_id" \
    "$root/CampaignOperationsH1Traceability.md" >/dev/null
  python3 - "$root" "$target" "$scratch" "$name" <<'PY'
import csv, pathlib, sys
root, target, scratch, name = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3]), sys.argv[4]
checks = [
    (scratch / f"{name}.runtime.before", root / "h1-reconciled-runtime-records.tsv", "runtime_record_id", "RT-" + target.removeprefix("REP-")),
    (scratch / f"{name}.validator.before", root / "h1-validator-results.tsv", "validator_result_id", "VR-" + target.removeprefix("REP-")),
    (scratch / f"{name}.report.before", root / "h1-report-entry-registry.tsv", "report_entry_id", target),
]
for before_path, after_path, key, expected in checks:
    def load(path):
        with path.open(newline="") as source:
            rows = {row[key]: row for row in csv.DictReader(source, delimiter="\t")}
            if key == "validator_result_id":
                for row in rows.values(): row.pop("validator_execution_id", None)
            return rows
    before, after = load(before_path), load(after_path)
    if before.keys() != after.keys():
        raise SystemExit(f"delta-node-set-changed:{after_path.name}")
    changed = [identifier for identifier in before if before[identifier] != after[identifier]]
    if changed != [expected]:
        raise SystemExit(f"delta-not-local:{after_path.name}:{changed}:{expected}")
PY
  rm -rf -- "$root"
}

mutate_tsv='python3 - "$root" "$name" <<'"'"'PY'"'"'
import csv, hashlib, pathlib, sys
root, case = pathlib.Path(sys.argv[1]), sys.argv[2]
def rewrite(path, key, value, changes):
    with path.open(newline="") as source:
        reader=csv.DictReader(source, delimiter="\t"); fields=reader.fieldnames; rows=list(reader)
    row=next(item for item in rows if item[key] == value)
    for field, replacement in changes.items(): row[field]=replacement
    row["record_digest"]=hashlib.sha256("\t".join(row[field] for field in fields[:-1]).encode()).hexdigest()
    with path.open("w", newline="") as target:
        writer=csv.DictWriter(target, fieldnames=fields, delimiter="\t", lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
if case == "generic":
    rewrite(root/"h1-runtime-results.tsv", "fixture_id", "H1ROLE001", {"timestamp":"2026-08-02T03:13:22Z"})
elif case == "lock":
    raw=root/"raw-lock/H1LOCK001.tsv"; rows=list(csv.reader(raw.open(newline=""), delimiter="\t")); rows[1][10]="rolled-back"
    with raw.open("w", newline="") as target: csv.writer(target, delimiter="\t", lineterminator="\n").writerows(rows)
    rewrite(root/"h1-lock-runtime.tsv", "h1lock_id", "H1LOCK001", {"first_transaction_outcome":"rolled-back", "raw_artifact_digest":hashlib.sha256(raw.read_bytes()).hexdigest()})
elif case == "acl":
    raw=root/"raw-acl-origin/H1AO001.tsv"; rows=list(csv.reader(raw.open(newline=""), delimiter="\t")); rows[1][21]="H1A006-DELTA exact ACL origin mismatch object=h1_acl_origin_schema stage=database-audit"
    with raw.open("w", newline="") as target: csv.writer(target, delimiter="\t", lineterminator="\n").writerows(rows)
    rewrite(root/"h1-acl-origin-runtime.tsv", "fixture_id", "H1AO001", {"actual_diagnostic":"H1A006-DELTA", "raw_artifact_digest":hashlib.sha256(raw.read_bytes()).hexdigest()})
elif case == "restore":
    artifact=root/"runtime-artifacts/records/H1RESTOREA/restore-scenario-A.tsv"; artifact.write_text(artifact.read_text().replace("pending-digest", "delta-digest"))
    digest=hashlib.sha256(artifact.read_bytes()).hexdigest()
    rewrite(root/"h1-restore-runtime.tsv", "scenario_id", "A", {"artifact_digest":digest})
    rewrite(root/"h1-runtime-results.tsv", "fixture_id", "H1RESTOREA", {"artifact_digest":digest})
elif case == "final":
    rewrite(root/"h1-final-assurance-results.tsv", "evidence_id", "CHECKSUM", {"timestamp":"2026-08-02T03:23:42Z"})
else: raise SystemExit(case)
PY'

assert_local_delta generic REP-H1ROLE001 "$mutate_tsv"
assert_local_delta lock REP-H1LOCK001 "$mutate_tsv"
assert_local_delta acl REP-H1AO001 "$mutate_tsv"
assert_local_delta restore REP-H1RESTOREA "$mutate_tsv"
assert_local_delta final REP-H1FA010 "$mutate_tsv"

echo "Campaign Operations H1 one-record delta tests passed cases=5"
