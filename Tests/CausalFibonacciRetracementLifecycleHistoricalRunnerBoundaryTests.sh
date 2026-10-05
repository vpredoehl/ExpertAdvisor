#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cli="${repo_root}/Sources/CausalFibonacciRetracementLifecycleHistoricalEvaluationCLI.cpp"
artifact="${repo_root}/Sources/CausalFibonacciRetracementLifecycleHistoricalArtifact.cpp"
runner="${repo_root}/Scripts/run_causal_fibonacci_retracement_lifecycle_study.sh"

rg -q 'pqxx::read_transaction' "${cli}"
rg -q 'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY' "${cli}"
rg -q 'HistoricalMarketDataRepository::Preflight' "${cli}"
rg -q 'HistoricalMarketDataRepository::StreamCanonicalCandles' "${cli}"
rg -q 'EffectiveFibonacciAbsolutePriceTolerance' "${cli}"
rg -q 'newlyAvailableABStructures' \
    "${repo_root}/Headers/CausalFibonacciRetracementLifecycleHistoricalEvaluation.hpp"
rg -q 'Extension1272' "${artifact}"
rg -q 'Extension1618' \
    "${repo_root}/Headers/CausalFibonacciRetracementLifecycleHistoricalEvaluation.hpp"
rg -q 'read_only_repeatable_read' "${artifact}"
! rg -q 'INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|TRUNCATE' "${cli}" "${artifact}"
! rg -q '20-bar|kPrimaryEvaluationHorizonBars|30-pip' \
    "${repo_root}/Headers/CausalFibonacciRetracementLifecycle.hpp" "${cli}" "${artifact}"
test -x "${runner}" || test -f "${runner}"

echo "CausalFibonacciRetracementLifecycleHistoricalRunnerBoundaryTests passed"
