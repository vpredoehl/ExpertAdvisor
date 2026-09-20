#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
main_file="${repo_root}/LSTM/main.cpp"
runtime_header="${repo_root}/Sources/InferenceRuntime.hpp"
runtime_source="${repo_root}/Sources/InferenceRuntime.cpp"
facts_source="${repo_root}/Sources/InferenceEvaluationFacts.cpp"

test -f "${runtime_header}"
test -f "${runtime_source}"
rg -q 'struct RuntimeRequest' "${runtime_header}"
rg -q 'struct RuntimeResult' "${runtime_header}"
rg -q 'RunInferenceRuntime\(const RuntimeRequest& request\)' \
    "${runtime_header}" "${runtime_source}"

# The public runtime boundary is typed, independently linkable, and has no
# CLI parser, scheduler lifecycle, result persistence, or worker publication.
! rg -n 'LaunchArgs|ParseLaunch|Scheduler|Persist.*Inference|pqxx::|semantic-worker|worker registry' \
    "${runtime_header}" "${runtime_source}"
rg -q 'InferenceEvaluationFacts::Evaluate' "${runtime_source}"
! rg -n 'EvaluateClassificationBatch|EvaluateRegressionBatch|ComputeAcceptanceSummary' \
    "${runtime_source}"
rg -q 'EvaluateClassificationBatch' "${facts_source}"

# The input composition, direct selected-model path, infer-all path, and the
# scheduler detached branch all converge on the runtime without moving result
# transaction policy out of main.cpp.
rg -q 'EA::Inference::PrepareInferenceInput' "${main_file}"
test "$(rg -c 'EA::Inference::RunInferenceRuntime' "${main_file}")" -ge 2
rg -U -q 'InferAllSummaryRow RunInferAllModel\([\s\S]{0,1800}RunInferenceRuntime\(' \
    "${main_file}"
rg -U -q 'schedulerMaterializationRead\.commit\(\);[\s\S]{0,30000}RunInferenceRuntime\(' \
    "${main_file}"
rg -q 'pqxx::work schedulerPersistenceWork \{ c_LSTM \};' "${main_file}"

printf '%s\n' 'LSTMPhase22Z3InferenceRuntimeCompositionBoundaryTests passed'
