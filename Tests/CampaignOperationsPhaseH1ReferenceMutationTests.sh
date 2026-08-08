#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 2 && -d "$1" ]] || { echo "usage: $0 ARTIFACT_ROOT RUN_ID" >&2; exit 64; }
source_root="$1" run_id="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tool="$repo_root/Scripts/CampaignOperationsH1EvidenceGraph.py"
scratch="$(mktemp -d /tmp/ea-h1-reference-mutations.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT
base="$scratch/base"; cp -R "$source_root" "$base"

generate() { python3 "$tool" generate "$1" "$run_id" >/dev/null; }
validate() { python3 "$tool" validate "$1" "$run_id" "$1/CampaignOperationsH1Traceability.md"; }
generate "$base"; validate "$base" >/dev/null

run_case() {
  local name="$1" code="$2" stage="$3" detail="$4" mutation="$5" root="$scratch/$1"
  cp -R "$base" "$root"; eval "$mutation"
  if validate "$root" >"$root.log" 2>&1; then echo "$name unexpectedly passed" >&2; exit 1; fi
  rg -q "H1R${code} .*stage=${stage} detail=${detail}" "$root.log" || { cat "$root.log" >&2; exit 1; }
  printf 'H1_MUTATION_CASE\tgraph-%s\t%s\tH1R%s:%s\t%s\tH1R%s:%s\t%s\tPASS\n' \
    "$name" "$name" "$code" "$detail" "$stage" "$code" "$detail" "$stage"
}
run_authority_case() {
  local name="$1" code="$2" stage="$3" detail="$4" mutation="$5" root="$scratch/$1"
  cp -R "$base" "$root"; eval "$mutation"
  if validate "$root" >"$root.log" 2>&1; then echo "$name unexpectedly passed" >&2; exit 1; fi
  rg -q "H1A${code} .*stage=${stage} detail=${detail}" "$root.log" || { cat "$root.log" >&2; exit 1; }
  printf 'H1_MUTATION_CASE\tgraph-%s\t%s\tH1A%s:%s\t%s\tH1A%s:%s\t%s\tPASS\n' \
    "$name" "$name" "$code" "$detail" "$stage" "$code" "$detail" "$stage"
}

run_authority_case rogue_top_level 309 filesystem-representation-validation unknown-evidence-file \
  "printf 'rogue\n' > \"\$root/rogue.txt\""
run_authority_case rogue_nested 306 directory-policy-validation unregistered-directory \
  "mkdir -p \"\$root/unregistered/nested\"; printf 'rogue\n' > \"\$root/unregistered/nested/rogue.txt\""
run_authority_case filename_inferred_artifact 309 filesystem-representation-validation unknown-evidence-file \
  "printf 'forged\n' > \"\$root/raw-lock/H1ROLE001.tsv\""
run_authority_case unindexed_generated_registry 309 filesystem-representation-validation unknown-evidence-file \
  "printf 'generated\n' > \"\$root/h1-unindexed-generated.tsv\""
run_authority_case unindexed_validation_log 309 filesystem-representation-validation unknown-evidence-file \
  "printf 'PASS\n' > \"\$root/unindexed-validation.log\""
run_authority_case unindexed_final_log 309 filesystem-representation-validation unknown-evidence-file \
  "mkdir -p \"\$root/final-assurance\"; printf 'PASS\n' > \"\$root/final-assurance/UNDECLARED.log\""
run_case stale_generated_report 006 report-provenance-freshness stale-generated-report \
  "printf '\nstale\n' >> \"\$root/CampaignOperationsH1Traceability.md\""
run_case missing_report_entry 006 report-provenance-freshness stale-generated-registry \
  "perl -i -ne 'print unless /REP-H1ROLE001/' \"\$root/h1-report-entry-registry.tsv\""
run_case extra_report_entry 006 report-provenance-freshness stale-generated-registry \
  "rg 'REP-H1ROLE001' \"\$root/h1-report-entry-registry.tsv\" >> \"\$root/h1-report-entry-registry.tsv\""

root="$scratch/stale_validator_result"; cp -R "$base" "$root"
perl -i -pe 'if (/VR-H1ROLE001/) { s/\tequal\t/\tforged\t/ }' "$root/h1-validator-results.tsv"
if validate "$root" >"$root.log" 2>&1; then echo "stale validator unexpectedly passed" >&2; exit 1; fi
rg -q 'H1V007 key=h1-validator-results.tsv stage=authentic-runtime-validation detail=stale-validator-results' "$root.log" || { cat "$root.log" >&2; exit 1; }
printf 'H1_MUTATION_CASE\tgraph-stale-validator-result\th1-validator-results.tsv\tH1V007:stale-validator-results\tauthentic-runtime-validation\tH1V007:stale-validator-results\tauthentic-runtime-validation\tPASS\n'

# Replace a semantically parsed H2/H3/H4-exclusion artifact, then refresh both
# valid SHA-256 fields.  Digest agreement must not make the false claim pass.
root="$scratch/forged_exclusion"; cp -R "$base" "$root"
printf 'H2 H3 H4 mutation entry points are present\n' > "$root/runtime-artifacts/records/H1CPP002/phase-h1-cli-results"
python3 - "$root/h1-runtime-results.tsv" "$root/runtime-artifacts/records/H1CPP002/phase-h1-cli-results" <<'PY'
import csv, hashlib, sys
ledger, artifact = sys.argv[1:]
with open(ledger, newline="") as source:
    reader = csv.DictReader(source, delimiter="\t"); fields = reader.fieldnames; rows = list(reader)
value = hashlib.sha256(open(artifact, "rb").read()).hexdigest()
for row in rows:
    if row["fixture_id"] == "H1CPP002":
        row["artifact_digest"] = value
        row["record_digest"] = hashlib.sha256("\t".join(row[field] for field in fields[:-1]).encode()).hexdigest()
with open(ledger, "w", newline="") as target:
    writer = csv.DictWriter(target, fieldnames=fields, delimiter="\t", lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
PY
if validate "$root" >"$root.log" 2>&1; then echo "forged exclusion unexpectedly passed" >&2; exit 1; fi
rg -q 'H1V106 key=H1CPP002 stage=authentic-runtime-validation detail=prohibited-h2-h3-h4-entry-point' "$root.log" || { cat "$root.log" >&2; exit 1; }
printf 'H1_MUTATION_CASE\tgraph-forged-exclusion\truntime-artifacts/records/H1CPP002/phase-h1-cli-results\tH1V106:prohibited-h2-h3-h4-entry-point\tauthentic-runtime-validation\tH1V106:prohibited-h2-h3-h4-entry-point\tauthentic-runtime-validation\tPASS\n'

# A syntactically valid replacement of a generic runtime semantic is rejected
# even after its record digest is recomputed.
root="$scratch/generic_semantic"; cp -R "$base" "$root"
python3 - "$root/h1-runtime-results.tsv" <<'PY'
import csv, hashlib, sys
path = sys.argv[1]
with open(path, newline="") as source:
    reader = csv.DictReader(source, delimiter="\t"); fields = reader.fieldnames; rows = list(reader)
for row in rows:
    if row["fixture_id"] == "H1ROLE001":
        row["diagnostic"] = "H1A999"
        row["record_digest"] = hashlib.sha256("\t".join(row[field] for field in fields[:-1]).encode()).hexdigest()
with open(path, "w", newline="") as target:
    writer = csv.DictWriter(target, fieldnames=fields, delimiter="\t", lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
PY
if validate "$root" >"$root.log" 2>&1; then echo "generic semantic forgery unexpectedly passed" >&2; exit 1; fi
rg -q 'H1V105 key=H1ROLE001 stage=authentic-runtime-validation detail=semantic-mismatch:diagnostic' "$root.log" || { cat "$root.log" >&2; exit 1; }
printf 'H1_MUTATION_CASE\tgraph-generic-semantic\th1-runtime-results.tsv\tH1V105:semantic-mismatch:diagnostic\tauthentic-runtime-validation\tH1V105:semantic-mismatch:diagnostic\tauthentic-runtime-validation\tPASS\n'

echo "Campaign Operations H1 reference graph mutation tests passed cases=12"
