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
        sed -n '1,20p' "$output_file" >&2
        exit 1
    fi
    if ! rg -Fq -- "$expected" "$output_file"; then
        echo "missing diagnostic '$expected': $*" >&2
        sed -n '1,20p' "$output_file" >&2
        exit 1
    fi
    if rg -q 'SCHEDULER_STATUS|QUEUE_|WORKER_' "$output_file"; then
        echo "Campaign Operations arguments fell through: $*" >&2
        exit 1
    fi
}

expect_invalid "expected exactly one experiment scheduler command" \
    --campaign-operations-budget-grant 1 \
    --campaign-operations-budget-amend 1 \
    --campaign-operations-expected-budget-version 0 \
    --campaign-operations-budget-value 1 \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason --yes

expect_invalid "Campaign Operations mutation requires --yes" \
    --campaign-operations-budget-grant 1 \
    --campaign-operations-expected-budget-version 0 \
    --campaign-operations-budget-value 1 \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason
expect_invalid "--dry-run is not valid for durable Campaign Operations mutations" \
    --campaign-operations-budget-amend 1 \
    --campaign-operations-expected-budget-version 1 \
    --campaign-operations-budget-value 1 \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason --dry-run --yes
expect_invalid "Campaign Operations mutation requires --campaign-operations-actor and --campaign-operations-reason" \
    --campaign-operations-accept-request 1 --yes
expect_invalid "Campaign Operations status accepts neither --dry-run nor --yes" \
    --campaign-operations-budget-status 1 --yes
expect_invalid "Campaign Operations status accepts neither --dry-run nor --yes" \
    --campaign-operations-request-status 1 --dry-run
expect_invalid "request acceptance does not accept budget mutation metadata" \
    --campaign-operations-accept-request 1 \
    --campaign-operations-expected-budget-version 1 \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason --yes
expect_invalid "reservation expiry requires request acceptance" \
    --campaign-operations-budget-grant 1 \
    --campaign-operations-expected-budget-version 0 \
    --campaign-operations-budget-value 1 \
    --campaign-operations-reservation-expires-at 2030-01-01T00:00:00.000000Z \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason --yes
expect_invalid "campaign_operations_budget_expected_version_invalid" \
    --campaign-operations-budget-grant 1 \
    --campaign-operations-expected-budget-version 1 \
    --campaign-operations-budget-value 1 \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason --yes
expect_invalid "campaign_operations_budget_value_invalid" \
    --campaign-operations-budget-amend 1 \
    --campaign-operations-expected-budget-version 1 \
    --campaign-operations-budget-value 0 \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason --yes
expect_invalid "campaign_operations_budget_value_invalid" \
    --campaign-operations-budget-revoke 1 \
    --campaign-operations-expected-budget-version 1 \
    --campaign-operations-budget-value 1 \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason --yes
expect_invalid "campaign_operations_budget_value_invalid" \
    --campaign-operations-budget-supersede 1 \
    --campaign-operations-expected-budget-version 1 \
    --campaign-operations-actor operator \
    --campaign-operations-reason reason --yes
expect_invalid "Campaign Operations metadata requires an authorized mutation or status command" \
    --campaign-operations-budget-status 1 \
    --campaign-operations-actor operator
expect_invalid "Campaign Operations metadata requires an authorized mutation or status command" \
    --campaign-operations-request-status 1 \
    --campaign-operations-reason reason

for command in \
    budget-grant budget-amend budget-revoke budget-supersede \
    accept-request budget-status request-status
do
    expect_invalid "duplicate --campaign-operations-$command" \
        "--campaign-operations-$command" 1 \
        "--campaign-operations-$command" 1
done

for option in \
    expected-budget-version budget-value actor reason reservation-expires-at
do
    expect_invalid "duplicate --campaign-operations-$option" \
        --campaign-operations-budget-grant 1 \
        "--campaign-operations-$option" 1 \
        "--campaign-operations-$option" 1
done

help_output="$temporary_directory/help"
"$binary" --help >"$help_output"
rg -Fq -- "--campaign-operations-budget-value UNITS" "$help_output"
rg -Fq -- "--campaign-operations-budget-revoke CAMPAIGN_ID --campaign-operations-expected-budget-version N --campaign-operations-actor ACTOR" "$help_output"

echo "Campaign Operations Phase 2 CLI parser tests passed"
