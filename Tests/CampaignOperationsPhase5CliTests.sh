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
        echo "Campaign Operations completion arguments fell through: $*" >&2
        exit 1
    fi
}

expect_invalid "Campaign Operations mutation requires --yes" \
    --campaign-operations-complete-if-settled 1 \
    --campaign-operations-operation-key complete-1 \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason

expect_invalid "complete-if-settled requires --campaign-operations-operation-key" \
    --campaign-operations-complete-if-settled 1 \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason --yes

expect_invalid "Campaign Operations mutation requires --campaign-operations-actor and --campaign-operations-reason" \
    --campaign-operations-complete-if-settled 1 \
    --campaign-operations-operation-key complete-1 --yes

expect_invalid "campaign_operations_completion_operation_key_invalid" \
    --campaign-operations-complete-if-settled 1 \
    --campaign-operations-operation-key 'bad key' \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason --yes

expect_invalid "Campaign Operations status accepts neither --dry-run nor --yes" \
    --campaign-operations-completion-status 1 --yes

expect_invalid "Campaign Operations status accepts neither --dry-run nor --yes" \
    --campaign-operations-complete-if-settled 1 \
    --campaign-operations-completion-status 1 \
    --campaign-operations-operation-key complete-1 \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason --yes

expect_invalid "duplicate --campaign-operations-complete-if-settled" \
    --campaign-operations-complete-if-settled 1 \
    --campaign-operations-complete-if-settled 1

expect_invalid "duplicate --campaign-operations-completion-status" \
    --campaign-operations-completion-status 1 \
    --campaign-operations-completion-status 1

help_output="$temporary_directory/help"
"$binary" --help >"$help_output"
rg -Fq -- "--campaign-operations-complete-if-settled CAMPAIGN_ID" \
    "$help_output"
rg -Fq -- "--campaign-operations-completion-status CAMPAIGN_ID" \
    "$help_output"
rg -Fq -- "never means scientific success" "$help_output"
rg -Fq -- "no force-complete, reopen, override, or delete command" \
    "$help_output"

echo "Campaign Operations Phase 5 CLI parser tests passed"
