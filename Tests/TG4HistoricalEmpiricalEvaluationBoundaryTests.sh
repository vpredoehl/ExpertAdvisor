#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
core="${repo_root}/Sources/TG4HistoricalEmpiricalEvaluation.cpp"
repository="${repo_root}/Sources/TG4HistoricalMarketDataRepository.cpp"
cli="${repo_root}/Sources/TG4HistoricalEmpiricalEvaluationCLI.cpp"

rg -q 'pqxx::read_transaction' "${repository}" "${cli}"
rg -q 'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ' "${cli}"
rg -q 'HistoricalFxTimestamp::ParseNewYorkCivilTimestamp' "${repository}"
rg -q 'transaction\.stream' "${repository}"
rg -q 'SupportedSymbols::TrainingSymbols' "${cli}"
rg -q 'confirmation_period_used_for_selection.*false' "${core}"
! rg -q 'Tensor|FeatureLayout|kCurrentModelInputWidth' \
    "${core}" "${repository}" "${cli}"
! rg -q 'INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|TRUNCATE' \
    "${repository}" "${cli}"

echo "TG4HistoricalEmpiricalEvaluationBoundaryTests passed"
