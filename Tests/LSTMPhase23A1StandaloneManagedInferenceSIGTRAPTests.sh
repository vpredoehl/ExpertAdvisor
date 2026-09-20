#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_file="${repo_root}/ExpertAdvisor.xcodeproj/project.pbxproj"
application="${repo_root}/Sources/ManagedInferenceApplication.cpp"
runtime="${repo_root}/Sources/InferenceRuntime.cpp"

# The standalone worker must use the same Release optimization and OpenMP
# composition as the compatibility target, whose managed-inference path is
# known to be production-safe.  This is intentionally target-specific: it
# prevents a future thin-worker-only Release package divergence.
release_target="$(sed -n '/0A1000112F70000100AAA001 \/\* Release \*\//,/name = Release;/p' "${project_file}")"
worker_target="$(sed -n '/0F8000513800000100AAA001 \/\* Release \*\//,/name = Release;/p' "${project_file}")"
for required in \
    'GCC_OPTIMIZATION_LEVEL = 3;' \
    '"-Xpreprocessor", "-fopenmp"' \
    '"-lpq", "-lpqxx", "-lomp"' \
    '/opt/homebrew/opt/libomp/lib' \
    '/opt/homebrew/opt/libomp/include'; do
    grep -Fq "${required}" <<<"${worker_target}"
done
grep -Fq 'GCC_OPTIMIZATION_LEVEL = 3;' <<<"${release_target}"
grep -Fq '"-Xpreprocessor",' <<<"${release_target}"
grep -Fq '"-fopenmp",' <<<"${release_target}"
grep -Fq '"-lomp",' <<<"${release_target}"

# Every durable managed-inference boundary emits a flush-backed stage marker.
for stage in \
    managed_application_entry database_connection_established \
    repeatable_read_transaction_begun detached_materialization_read_begun \
    detached_materialization_read_completed repeatable_read_transaction_committed \
    detached_materialization_train_config_validated \
    managed_request_persisted_config_validated \
    runtime_request_constructed inference_input_tensor_preparation_begun \
    inference_input_tensor_preparation_completed lstm_construction_begun \
    lstm_construction_completed detached_materialization_apply_begun \
    detached_materialization_apply_completed model_config_validation_begun \
    model_config_validation_completed evaluation_begun evaluation_completed \
    fresh_read_write_persistence_transaction_begun \
    scheduler_identity_state_revalidation_completed \
    result_profitability_persistence_completed persistence_transaction_committed \
    managed_application_success_return; do
    rg -Fq "\"${stage}\"" "${application}" "${runtime}"
done

# The diagnostics observe but do not alter the detached RR/RO or fresh RW
# boundaries, registration guard, revalidation, or atomic persistence order.
test "$(rg -c 'RegisterSchedulerWorker' "${application}")" -eq 1
! rg -q 'SIGTRAP|sigaction|LSTM_PHASE23A1_TRAP_PROBE' "${application}"
rg -U -q 'snapshot\.model\.trainSymbol\s*\?\s*\*snapshot\.model\.trainSymbol\s*:\s*\*request\.requestedSymbol' "${application}"
! rg -q 'trainSymbol\.value_or\(\*request\.requestedSymbol\)' "${application}"
rg -U -q 'REPEATABLE READ, READ ONLY;[\s\S]{0,16000}read\.commit\(\);[\s\S]{0,8000}PrepareInferenceInput\([\s\S]{0,8000}RunInferenceRuntime\(' "${application}"
rg -U -q 'RevalidateForWrite\([\s\S]{0,1600}PersistResult\([\s\S]{0,1600}PersistProfitability\([\s\S]{0,800}write\.commit\(\)' "${application}"

printf '%s\n' 'LSTMPhase23A1StandaloneManagedInferenceSIGTRAPTests passed'
