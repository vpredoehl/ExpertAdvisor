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
        echo "unexpected H3 CLI result ($status): $*" >&2
        sed -n '1,30p' "$output_file" >&2
        exit 1
    fi
}

expect_invalid "invalid --campaign-operations-manager-run-once value '0'" \
    --campaign-operations-manager-run-once 0 --yes
expect_invalid "invalid --campaign-operations-manager-run-once value '-1'" \
    --campaign-operations-manager-run-once -1 --yes
expect_invalid "invalid --campaign-operations-manager-run-once value 'nope'" \
    --campaign-operations-manager-run-once nope --yes
expect_invalid "invalid --campaign-operations-manager-run-once value '999999999999999999999'" \
    --campaign-operations-manager-run-once 999999999999999999999 --yes
expect_invalid "missing required Campaign Operations production principal environment variable CAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER" \
    --campaign-operations-manager-run-once 101 --yes
expect_invalid "duplicate --campaign-operations-manager-run-once" \
    --campaign-operations-manager-run-once 1 \
    --campaign-operations-manager-run-once 1 --yes

help_output="$temporary_directory/help"
"$binary" --help >"$help_output"
rg -Fq -- "--campaign-operations-manager-run-once LIMIT --yes" "$help_output"
if rg -q -- '--campaign-operations-production-continuous|--campaign-operations-manager-daemon|--campaign-operations-manager-poll' \
    "$help_output"; then
    echo "H4 continuous Manager command leaked into H3 help" >&2
    exit 1
fi

# A syntactically valid bounded request must pass the parser. Runtime failure
# is expected in this parser-only test because it does not authorize a live
# production database or a deployable clean Release build.
output_file="$temporary_directory/accepted"
status=0
"$binary" --campaign-operations-manager-run-once 1 --yes \
    >"$output_file" 2>&1 || status=$?
if rg -q -- 'invalid --campaign-operations-manager-run-once|requires --yes|--dry-run' \
    "$output_file"; then
    echo "positive H3 run-once limit was rejected by CLI parser" >&2
    sed -n '1,30p' "$output_file" >&2
    exit 1
fi
rg -q -- 'missing required Campaign Operations production principal environment variable CAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER' \
    "$output_file"

rg -q -- 'DispatchTestHook|DispatchOneRequestForProductionForTest' \
    "$repo_root/Sources/CampaignOperationsManagerService.cpp" && {
    echo "production Manager source references isolated-test hooks" >&2
    exit 1
}

echo "Campaign Operations Phase H3 CLI parser tests passed"
