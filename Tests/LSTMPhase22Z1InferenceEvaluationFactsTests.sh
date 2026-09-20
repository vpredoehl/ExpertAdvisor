#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
facts_header="${repo_root}/Sources/InferenceEvaluationFacts.hpp"
facts_source="${repo_root}/Sources/InferenceEvaluationFacts.cpp"
main_file="${repo_root}/LSTM/main.cpp"
build_dir="${repo_root}/Build/inference_evaluation_facts_tests"
binary="${build_dir}/InferenceEvaluationFactsTests"

# The reusable API is intentionally free of application and persistence types.
! rg -n 'LaunchArgs|pqxx::|transaction_base|pqxx/pqxx|ofstream|Scheduler' \
    "${facts_header}" "${facts_source}"
rg -q 'EvaluationFacts Evaluate\(EA::LSTM& model' "${facts_header}"
rg -q 'InferenceProfitability::Statistics profitability' "${facts_header}"
rg -q 'std::vector<StrategyDecision>' \
    "${facts_header}"

# The main adapter remains responsible for validation and policy, but all
# numerical counts/profitability/acceptance facts come from the new seam.
rg -U -q 'RunInferenceEvaluation\([\s\S]{0,7000}InferenceEvaluationFacts::Evaluate\(' \
    "${main_file}"
rg -U -q 'InferenceEvaluationFacts::Evaluate\([\s\S]{0,1200}HasControlledStrategyEvaluation\(launchArgs\)' \
    "${main_file}"

# Direct, scheduler final/checkpoint, and infer-all still converge through the
# unchanged application adapter, which now invokes the facts boundary.
test "$(rg -c 'RunInferenceEvaluation\(' "${main_file}")" -ge 3

# Acceptance is an owned computational decision, not adapter policy.
mkdir -p "${build_dir}"
clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Sources" \
    "${repo_root}/Tests/InferenceEvaluationFactsTests.cpp" \
    -o "${binary}"
"${binary}"

printf '%s\n' 'LSTMPhase22Z1InferenceEvaluationFactsTests passed'
