#!/usr/bin/env bash
set -euo pipefail

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
    if [[ $status -ne 1 ]]; then
        echo "expected exit 1, got $status: $*" >&2
        sed -n '1,30p' "$output_file" >&2
        exit 1
    fi
    if ! rg -Fq -- "$expected" "$output_file"; then
        echo "missing diagnostic '$expected': $*" >&2
        sed -n '1,30p' "$output_file" >&2
        exit 1
    fi
    if rg -q 'SCHEDULER_STATUS|QUEUE_|WORKER_' "$output_file"; then
        echo "Campaign Operations control arguments fell through: $*" >&2
        exit 1
    fi
}

expect_invalid "expected exactly one experiment scheduler command" \
    --campaign-operations-pause 1 \
    --campaign-operations-resume 1 \
    --campaign-operations-expected-control-version 0 \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason --yes

expect_invalid "Campaign Operations mutation requires --yes" \
    --campaign-operations-pause 1 \
    --campaign-operations-expected-control-version 0 \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason

expect_invalid "pause/resume requires --campaign-operations-expected-control-version" \
    --campaign-operations-pause 1 \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason --yes

expect_invalid "cancellation requires --campaign-operations-request-id" \
    --campaign-operations-cancel 1 \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason --yes

expect_invalid "campaign_operations_cancellation_operation_key_invalid" \
    --campaign-operations-cancel 1 \
    --campaign-operations-request-id 2 \
    --campaign-operations-expected-request-version 1 \
    --campaign-operations-operation-key 'bad key' \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason --yes

expect_invalid "reconciliation uses its service identity and accepts no operator actor/reason" \
    --campaign-operations-reconcile-observe restart-run \
    --campaign-operations-actor operator --yes

expect_invalid "campaign_operations_reconciliation_request_invalid" \
    --campaign-operations-reconcile-recover restart-run \
    --campaign-operations-reconcile-limit 1001 --yes

expect_invalid "control version requires pause or resume" \
    --campaign-operations-cancel 1 \
    --campaign-operations-request-id 2 \
    --campaign-operations-expected-request-version 1 \
    --campaign-operations-operation-key cancel-2 \
    --campaign-operations-expected-control-version 0 \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason --yes

expect_invalid "cancellation metadata requires cancellation" \
    --campaign-operations-pause 1 \
    --campaign-operations-expected-control-version 0 \
    --campaign-operations-operation-key cancel-2 \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason --yes

expect_invalid "reconciliation metadata requires reconciliation" \
    --campaign-operations-pause 1 \
    --campaign-operations-expected-control-version 0 \
    --campaign-operations-reconcile-limit 10 \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason --yes

expect_invalid "Campaign Operations status accepts neither --dry-run nor --yes" \
    --campaign-operations-control-status 1 --yes

for command in pause resume cancel control-status; do
    expect_invalid "duplicate --campaign-operations-$command" \
        "--campaign-operations-$command" 1 \
        "--campaign-operations-$command" 1
done

expect_invalid "duplicate Campaign Operations reconciliation command" \
    --campaign-operations-reconcile-observe run-1 \
    --campaign-operations-reconcile-recover run-2 --yes

help_output="$temporary_directory/help"
"$binary" --help >"$help_output"
rg -Fq -- "--campaign-operations-pause CAMPAIGN_ID" "$help_output"
rg -Fq -- "--campaign-operations-cancel CAMPAIGN_ID" "$help_output"
rg -Fq -- "--campaign-operations-reconcile-recover RUN_KEY" "$help_output"
rg -Fq -- "signals workers or controls scheduler processes" "$help_output"

echo "Campaign Operations Phase 4 CLI parser tests passed"
