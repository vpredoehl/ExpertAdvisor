#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
application_header="${repo_root}/Sources/ManagedInferenceApplication.hpp"
application_source="${repo_root}/Sources/ManagedInferenceApplication.cpp"
main_file="${repo_root}/LSTM/main.cpp"
project_file="${repo_root}/ExpertAdvisor.xcodeproj/project.pbxproj"

test -f "${application_header}"
test -f "${application_source}"
rg -q 'struct ManagedInferenceRequest' "${application_header}"
rg -q 'struct ManagedInferenceResult' "${application_header}"
rg -q 'RunManagedInference\(const ManagedInferenceRequest& request\)' \
    "${application_header}" "${application_source}"
! rg -n 'LaunchArgs|argv|argc|ParseLaunch' "${application_header}" "${application_source}"

# The operation is managed-only and preserves the distinct final/checkpoint
# identities throughout validation, registration, admission, and persistence.
rg -q 'requires_exactly_one_scheduler_binding' "${application_source}"
rg -q 'invalid_worker_attempt_id' "${application_source}"
rg -q 'RegisterSchedulerWorker' "${application_source}"
rg -q 'checkpoint_infer' "${application_source}"
rg -q 'managed_inference_worker_registration_failed' "${application_source}"
rg -q 'return 125' "${main_file}"

# The RR/RO detached read commits before preparation/runtime.  Result work is
# a fresh RW transaction with scheduler lock/reread/revalidation and atomic
# inference-result plus profitability persistence.
rg -U -q 'REPEATABLE READ, READ ONLY;[\s\S]{0,16000}read\.commit\(\);[\s\S]{0,8000}PrepareInferenceInput\([\s\S]{0,8000}RunInferenceRuntime\(' "${application_source}"
rg -q 'EA::GlobalExperimentControl::AcquireCoordinationLock' "${application_source}"
rg -q 'FOR UPDATE OF ce,e,m' "${application_source}"
rg -q 'scheduler_inference_binding_stale' "${application_source}"
rg -q 'ON CONFLICT \(checkpoint_eval_id\)' "${application_source}"
rg -q 'ON CONFLICT \(model_id,symbol' "${application_source}"
rg -U -q 'PersistResult\([\s\S]{0,1600}PersistProfitability\([\s\S]{0,500}write\.commit\(\)' "${application_source}"
! rg -n 'InferenceEvaluationFacts::Evaluate|ComputeAcceptanceSummary|EvaluateClassificationBatch' "${application_source}"

# main is an adapter consumer; its non-managed direct/infer-all paths remain
# present and no worker target/registry publication is introduced.
rg -U -q 'ManagedInferenceRequest request;[\s\S]{0,1600}RunManagedInference\(request\)' "${main_file}"
rg -q 'return RunInferAllForSymbol\(c_LSTM' "${main_file}"
rg -q 'else if \(gRuntimeInferenceMode && launchArgs\.modelId\.has_value\(\)\)' "${main_file}"
! git -C "${repo_root}" diff -- "${project_file}" \
    "Sources/SchedulerCore/SemanticWorkerRegistry.cpp" | rg -q 'lstm-infer-worker'
rg -q 'ManagedInferenceApplication.cpp in Sources' "${project_file}"

printf '%s\n' 'LSTMPhase22Z4ManagedInferenceApplicationBoundaryTests passed'
