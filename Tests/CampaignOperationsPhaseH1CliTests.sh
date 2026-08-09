#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -ne 1 || ! -x "$1" ]]; then
    echo "usage: $0 /path/to/LSTM_Release" >&2
    exit 2
fi

binary=$1
temporary_directory=$(mktemp -d)
trap 'rm -rf "$temporary_directory"' EXIT

expect_invalid() {
    local expected=$1
    shift
    local output_file="$temporary_directory/output"
    local status=0
    "$binary" "$@" >"$output_file" 2>&1 || status=$?
    if [[ $status -ne 1 ]] || ! rg -Fq -- "$expected" "$output_file"; then
        echo "unexpected H1 CLI result ($status): $*" >&2
        sed -n '1,30p' "$output_file" >&2
        exit 1
    fi
}

expect_invalid \
  "Campaign Operations production readiness/status accepts neither --dry-run nor --yes" \
  --campaign-operations-production-readiness --yes
expect_invalid \
  "Campaign Operations production readiness/status accepts neither --dry-run nor --yes" \
  --campaign-operations-production-status --dry-run
expect_invalid "duplicate --campaign-operations-production-readiness" \
  --campaign-operations-production-readiness \
  --campaign-operations-production-readiness
expect_invalid "duplicate --campaign-operations-production-status" \
  --campaign-operations-production-status \
  --campaign-operations-production-status
expect_invalid "production mutation requires --campaign-operations-operation-key" \
  --campaign-operations-production-disable \
  --campaign-operations-expected-production-version 1 \
  --campaign-operations-actor operator@example.test \
  --campaign-operations-reason rollback --yes
expect_invalid "invalid --campaign-operations-operation-key for production" \
  --campaign-operations-production-disable \
  --campaign-operations-operation-key "bad key" \
  --campaign-operations-expected-production-version 1 \
  --campaign-operations-actor operator@example.test \
  --campaign-operations-reason rollback --yes
expect_invalid "--dry-run is not valid for durable Campaign Operations mutations" \
  --campaign-operations-production-disable \
  --campaign-operations-operation-key disable-001 \
  --campaign-operations-expected-production-version 1 \
  --campaign-operations-actor operator@example.test \
  --campaign-operations-reason rollback --dry-run --yes
expect_invalid "Campaign Operations mutation requires --yes" \
  --campaign-operations-production-disable \
  --campaign-operations-operation-key disable-001 \
  --campaign-operations-expected-production-version 1 \
  --campaign-operations-actor operator@example.test \
  --campaign-operations-reason rollback
expect_invalid "production dispatch requires request ID and expected request version" \
  --campaign-operations-dispatch-request \
  --campaign-operations-operation-key dispatch-001 \
  --campaign-operations-actor operator@example.test --yes
expect_invalid "expected exactly one experiment scheduler command" \
  --campaign-operations-production-readiness \
  --campaign-operations-production-status

help_output="$temporary_directory/help"
"$binary" --help >"$help_output"
rg -Fq -- "--campaign-operations-production-readiness" "$help_output"
rg -Fq -- "--campaign-operations-production-status" "$help_output"
rg -Fq -- "Phase H2 mutations are default-off, caller-keyed, and single-request" \
  "$help_output"
rg -Fq -- "--campaign-operations-production-enable" "$help_output"
rg -Fq -- "--campaign-operations-production-disable" "$help_output"
rg -Fq -- "--campaign-operations-dispatch-request" "$help_output"
if rg -q -- '--campaign-operations-manager-run-once|--campaign-operations-production-continuous' \
    "$help_output"; then
    echo "H3/H4 command leaked into H2 help" >&2
    exit 1
fi

rg -q -- 'record_campaign_operations_production_(enable|disable)_v1|transition_campaign_operations_request_dispatch_production_v2' \
  "$repo_root/Sources"

echo "Campaign Operations Phase H1 CLI parser tests passed"
