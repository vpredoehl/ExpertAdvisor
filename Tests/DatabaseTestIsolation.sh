#!/usr/bin/env bash

is_clearly_disposable_database_name() {
    local value="${1:-}"
    local lower
    [[ -n "${value}" ]] || return 1
    lower="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')"
    [[ "${lower}" != "lstm" ]] || return 1
    [[ "${value}" =~ ^[A-Za-z0-9_-]+$ ]] || return 1
    [[ "${lower}" == ea_* || "${lower}" == *test* ||
        "${lower}" == *tmp* || "${lower}" == *disposable* ]]
}

require_disposable_test_database() {
    local configured="${LSTM_TEST_DB_NAME:-}"
    if [[ -z "${configured}" ]]; then
        printf '%s\n' 'LSTM_TEST_DB_NAME_required' >&2
        return 2
    fi
    if ! is_clearly_disposable_database_name "${configured}"; then
        printf '%s\n' 'clearly_disposable_non_LSTM_database_required' >&2
        return 2
    fi
}

require_non_production_maintenance_database() {
    local configured="${1:-}"
    local lower
    lower="$(printf '%s' "${configured}" | tr '[:upper:]' '[:lower:]')"
    if [[ -z "${configured}" || "${lower}" == "lstm" ]]; then
        printf '%s\n' 'non_LSTM_maintenance_database_required' >&2
        return 2
    fi
}
