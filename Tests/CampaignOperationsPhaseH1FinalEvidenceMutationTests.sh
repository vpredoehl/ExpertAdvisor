#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 && -d "$1" ]] || { echo "usage: $0 ARTIFACT_ROOT RUN_ID" >&2; exit 64; }
source_root="$1" run_id="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tool="$repo_root/Scripts/CampaignOperationsH1EvidenceGraph.py"
scratch="$(mktemp -d /tmp/ea-h1-final-evidence-mutations.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT

refresh_ledger_row() {
  python3 - "$1" "$2" "$3" <<'PY'
import csv, hashlib, sys
ledger, evidence_id, artifact = sys.argv[1:]
with open(ledger, newline="") as source:
    reader=csv.DictReader(source, delimiter="\t"); fields=reader.fieldnames; rows=list(reader)
row=next(item for item in rows if item["evidence_id"] == evidence_id)
row["artifact_digest"]=hashlib.sha256(open(artifact,"rb").read()).hexdigest()
row["record_digest"]=hashlib.sha256("\t".join(row[field] for field in fields[:-1]).encode()).hexdigest()
with open(ledger,"w",newline="") as target:
    writer=csv.DictWriter(target,fieldnames=fields,delimiter="\t",lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
PY
}

root="$scratch/build"; cp -R "$source_root" "$root"
printf '** BUILD FAILED **\ncommand=xcodebuild exit_status=0 configuration=Release product_path=/tmp/forged derived_data_path=/tmp warning_count=0\n' > "$root/final-assurance/RELEASE_BUILD.log"
refresh_ledger_row "$root/h1-final-assurance-results.tsv" RELEASE_BUILD "$root/final-assurance/RELEASE_BUILD.log"
if python3 "$tool" validate "$root" "$run_id" "$root/CampaignOperationsH1Traceability.md" >"$scratch/build.log" 2>&1; then
  echo "forged failed-build log unexpectedly passed" >&2; exit 1
fi
rg -q 'H1V703 key=RELEASE_BUILD stage=authentic-runtime-validation detail=release-build-not-succeeded' "$scratch/build.log"
printf 'H1_MUTATION_CASE\tfinal-forged-build\tfinal-assurance/RELEASE_BUILD.log\tH1V703:release-build-not-succeeded\tauthentic-runtime-validation\tH1V703:release-build-not-succeeded\tauthentic-runtime-validation\tPASS\n'

root="$scratch/restore"; cp -R "$source_root" "$root"
python3 - "$root/h1-restore-runtime.tsv" <<'PY'
import csv, hashlib, sys
path=sys.argv[1]
with open(path,newline="") as source:
    reader=csv.DictReader(source,delimiter="\t"); fields=reader.fieldnames; rows=list(reader)
row=next(item for item in rows if item["scenario_id"] == "A")
row["expected_diagnostic"]=row["actual_diagnostic"]="forged-restore-supported"
row["record_digest"]=hashlib.sha256("\t".join(row[field] for field in fields[:-1]).encode()).hexdigest()
with open(path,"w",newline="") as target:
    writer=csv.DictWriter(target,fieldnames=fields,delimiter="\t",lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
PY
if python3 "$tool" validate "$root" "$run_id" "$root/CampaignOperationsH1Traceability.md" >"$scratch/restore.log" 2>&1; then
  echo "forged restore row unexpectedly passed" >&2; exit 1
fi
rg -q 'H1V801 key=h1-restore-runtime.tsv stage=authentic-runtime-validation detail=restore-semantic-validation-failed' "$scratch/restore.log"
printf 'H1_MUTATION_CASE\tfinal-forged-restore\th1-restore-runtime.tsv\tH1V801:restore-semantic-validation-failed\tauthentic-runtime-validation\tH1V801:restore-semantic-validation-failed\tauthentic-runtime-validation\tPASS\n'

echo "Campaign Operations H1 final evidence mutation tests passed cases=2"
