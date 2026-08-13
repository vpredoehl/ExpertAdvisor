#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tool="$repo_root/Scripts/CampaignOperationsH1LockEvidence.py"
matrix="$repo_root/Tests/fixtures/CampaignOperationsH1LockPathMatrix.tsv"
scratch="$(mktemp -d /tmp/ea-h1-lock-mutations.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT
catalog="$scratch/catalog.tsv" meta="$scratch/meta.tsv"
: > "$catalog"; : > "$meta"
while IFS=$'\t' read -r id _ _ _ _ _ permit _ _ _ _ _ _ _; do
  [[ "$id" == test_id ]] && continue
  first_app="first-$id"; second_app="second-$id"
  printf '%s\t%s\t101\trelation\tpublic.campaign_operations_operational_request\tRowShareLock\tt\t\t\t%s\tactive\tClient\tClientRead\t501\t\n' \
    "$id" "$first_app" "$([[ "$permit" == 'second->first' ]] && echo '' || [[ "$permit" == none ]] && echo '' || echo 202)" >> "$catalog"
  printf '%s\t%s\t202\trelation\tpublic.campaign_operations_campaign\tRowShareLock\tt\t\t\t%s\tactive\tClient\tClientRead\t502\t\n' \
    "$id" "$second_app" "$([[ "$permit" == 'second->first' ]] && echo 101 || echo '')" >> "$catalog"
  if [[ "$permit" != none ]]; then
    if [[ "$permit" == 'second->first' ]]; then app="$second_app"; pid=202; xid=501
    else app="$first_app"; pid=101; xid=502; fi
    printf '%s\t%s\t%s\ttransactionid\t\tShareLock\tf\t\t\t%s\tactive\tLock\ttransactionid\t\t%s\n' \
      "$id" "$app" "$pid" "$([[ "$pid" == 101 ]] && echo 202 || echo 101)" "$xid" >> "$catalog"
  fi
  printf '%s\t101\t202\t%s\t%s\tcommitted\tcommitted\ttrue\t0\tPASS\tpartial-admission-query-v1\tbackend-release-query-v1\n' \
    "$id" "$first_app" "$second_app" >> "$meta"
done < "$matrix"
mkdir -p "$scratch/base/raw-lock"
python3 "$tool" generate "$matrix" "$catalog" "$meta" \
  "$scratch/base/h1-lock-runtime.tsv" "$scratch/base/raw-lock" h1-lock-mutation-run
validator="$repo_root/Tests/CampaignOperationsPhaseH1LockArtifactTests.sh"
"$validator" "$scratch/base/h1-lock-runtime.tsv" "$scratch/base" h1-lock-mutation-run >/dev/null

python3 - "$scratch/base/h1-lock-runtime.tsv" <<'PY'
import csv, sys
with open(sys.argv[1], newline="") as source:
    rows = {row["h1lock_id"]: row for row in csv.DictReader(source, delimiter="\t")}
assert rows["H1LOCK001"]["observed_wait_direction"] == rows["H1LOCK001"]["permitted_wait_direction"]
assert rows["H1LOCK008"]["observed_wait_direction"] == "none"
assert rows["H1LOCK001"]["format_version"] == "h1-lock-runtime-v3"
assert rows["H1LOCK001"]["run_id"] == "h1-lock-mutation-run"
assert rows["H1LOCK001"]["raw_artifact_id"] == "ART-LOCK-RAW-H1LOCK001"
PY

run_runtime_case() {
  local name="$1" expression="$2" expected="$3"
  local root="$scratch/$name"; cp -R "$scratch/base" "$root"
  perl -i -pe "$expression" "$root/h1-lock-runtime.tsv"
  if "$validator" "$root/h1-lock-runtime.tsv" "$root" h1-lock-mutation-run >"$root.log" 2>&1; then
    echo "lock mutation $name unexpectedly passed" >&2; exit 1
  fi
  rg -q "H1L001 key=${expected} stage=lock-runtime-reconciliation" "$root.log" || {
    cat "$root.log" >&2; exit 1;
  }
  printf 'H1_MUTATION_CASE\tlock-%s\th1-lock-runtime.tsv\tH1L001:%s\tlock-runtime-reconciliation\tH1L001:%s\tlock-runtime-reconciliation\tPASS\n' \
    "$name" "$expected" "$expected"
}

run_runtime_case stale_digest \
  'if (/\tH1LOCK001\t/) { s/[0-9a-f]{64}(?=\t[0-9a-f]{64}$)/ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff/ }' \
  raw-digest:H1LOCK001
run_runtime_case waiting \
  'if (/\tH1LOCK001\t/) { s/\tfalse\t202\t/\ttrue\t202\t/ }' \
  normalized:H1LOCK001:second_waiting
run_runtime_case direction \
  'if (/\tH1LOCK001\t/) { s/\tacquisition->enable\tacquisition->enable\tenable->acquisition\t/\tacquisition->enable\tnone\tenable->acquisition\t/ }' \
  normalized:H1LOCK001:observed_wait_direction
run_runtime_case cycle \
  'if (/\tH1LOCK001\t/) { s/\tfalse\tactive:/\ttrue\tactive:/ }' \
  normalized:H1LOCK001:cycle_detected
run_runtime_case release \
  'if (/\tH1LOCK001\t/) { s/\ttrue\t0\tPASS\t/\tfalse\t0\tPASS\t/ }' \
  normalized:H1LOCK001:lock_release_verified
run_runtime_case evidence_count \
  'if (/\tH1LOCK001\t/) { s/\ttrue\t0\tPASS\t/\ttrue\t1\tPASS\t/ }' \
  normalized:H1LOCK001:partial_evidence_count
run_runtime_case wrong_pid \
  'if (/\tH1LOCK001\t/) { s/\t101\t202\t/\t999\t202\t/ }' \
  normalized:H1LOCK001:first_pid
run_runtime_case lock_identity \
  'if (/\tH1LOCK001\t/) { s/transactionid:502:ShareLock/transactionid:999:ShareLock/ }' \
  normalized:H1LOCK001:requested_lock
run_runtime_case stale_run_id \
  'if (/\tH1LOCK001\t/) { s/h1-lock-mutation-run/h1-stale-run/ }' \
  stale-run:H1LOCK001

run_raw_case() {
  local name="$1" expression="$2" expected="$3"
  local root="$scratch/$name" digest
  cp -R "$scratch/base" "$root"
  perl -i -pe "$expression" "$root/raw-lock/H1LOCK001.tsv"
  digest="$(shasum -a 256 "$root/raw-lock/H1LOCK001.tsv" | awk '{print $1}')"
  perl -i -pe "if (/\\tH1LOCK001\\t/) { s/[0-9a-f]{64}(?=\\t[0-9a-f]{64}\$)/${digest}/ }" \
    "$root/h1-lock-runtime.tsv"
  if "$validator" "$root/h1-lock-runtime.tsv" "$root" h1-lock-mutation-run >"$root.log" 2>&1; then
    echo "raw lock mutation $name unexpectedly passed" >&2; exit 1
  fi
  rg -q "H1L001 key=${expected} stage=lock-runtime-reconciliation" "$root.log" || {
    cat "$root.log" >&2; exit 1;
  }
  printf 'H1_MUTATION_CASE\tlock-raw-%s\traw-lock/H1LOCK001.tsv\tH1L001:%s\tlock-runtime-reconciliation\tH1L001:%s\tlock-runtime-reconciliation\tPASS\n' \
    "$name" "$expected" "$expected"
}

# Raw first-backend blocking is permitted for H1LOCK001.  Move the blocker to
# the second backend: it must normalize to enable->acquisition and be rejected.
run_raw_case prohibited_reverse \
  'if (/^catalog\tH1LOCK001\tfirst-H1LOCK001\t/) { s/\t202\tactive/\t\tactive/ } elsif (/^catalog\tH1LOCK001\tsecond-H1LOCK001\t202\trelation/) { s/\t\tactive/\t101\tactive/ }' \
  prohibited-direction:H1LOCK001
run_raw_case impossible_mutual \
  'if (/^catalog\tH1LOCK001\tsecond-H1LOCK001\t202\trelation/) { s/\t\tactive/\t101\tactive/ }' \
  mutual-blocking:H1LOCK001
run_raw_case stale_pid \
  'if (/^meta\th1-lock-mutation-run\tH1LOCK001\t/) { s/\t101\t202\t/\t999\t202\t/ }' \
  raw-identity:H1LOCK001
run_raw_case wrong_application \
  'if (/^meta\th1-lock-mutation-run\tH1LOCK001\t/) { s/\tfirst-H1LOCK001\tsecond-H1LOCK001\t/\twrong-application\tsecond-H1LOCK001\t/ }' \
  raw-identity:H1LOCK001

root="$scratch/missing-raw"; cp -R "$scratch/base" "$root"
rm "$root/raw-lock/H1LOCK001.tsv"
if "$validator" "$root/h1-lock-runtime.tsv" "$root" h1-lock-mutation-run >"$root.log" 2>&1; then exit 1; fi
rg -q 'H1L001 key=missing-raw:H1LOCK001 ' "$root.log"
printf 'H1_MUTATION_CASE\tlock-missing-raw\traw-lock/H1LOCK001.tsv\tH1L001:missing-raw:H1LOCK001\tlock-runtime-reconciliation\tH1L001:missing-raw:H1LOCK001\tlock-runtime-reconciliation\tPASS\n'

# Alter raw evidence and replace its digest with the new valid SHA-256.  The
# normalized row must still fail because its values no longer match the raw data.
root="$scratch/altered-raw"; cp -R "$scratch/base" "$root"
perl -i -pe 'if (/^catalog\tH1LOCK001\tfirst-H1LOCK001\t101\ttransactionid/) { s/\t502$/\t999/ }' \
  "$root/raw-lock/H1LOCK001.tsv"
digest="$(shasum -a 256 "$root/raw-lock/H1LOCK001.tsv" | awk '{print $1}')"
perl -i -pe "if (/\\tH1LOCK001\\t/) { s/[0-9a-f]{64}(?=\\t[0-9a-f]{64}\$)/${digest}/ }" "$root/h1-lock-runtime.tsv"
if "$validator" "$root/h1-lock-runtime.tsv" "$root" h1-lock-mutation-run >"$root.log" 2>&1; then exit 1; fi
rg -q 'H1L001 key=normalized:H1LOCK001:' "$root.log"
printf 'H1_MUTATION_CASE\tlock-altered-raw\traw-lock/H1LOCK001.tsv\tH1L001:normalized:H1LOCK001\tlock-runtime-reconciliation\tH1L001:normalized:H1LOCK001\tlock-runtime-reconciliation\tPASS\n'

echo "Campaign Operations H1 lock mutation tests passed cases=15 raw_direction_cases=7"
