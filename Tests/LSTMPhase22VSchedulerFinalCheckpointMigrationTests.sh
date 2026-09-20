#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
main_file="${repo_root}/LSTM/main.cpp"

# These assertions target the production scheduler final/checkpoint entry
# point.  They protect the transaction boundary rather than a test-only
# materialization path; database-backed detached-value coverage remains in the
# Phase 22T persistence fixture.
rg -U -q 'const bool useDetachedSelectedInference =[\s\S]{0,500}!launchArgs\.inferAll[\s\S]{0,1800}schedulerMaterializationRead\.exec\(\n[[:space:]]*"SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;"\);[\s\S]{0,1000}ReadPersistedModelMaterialization\([\s\S]{0,1600}ResolveSchedulerInferencePersistenceContext\([\s\S]{0,1800}schedulerMaterializationRead\.commit\(\);' "${main_file}"
# Phase 22Z3 preserves the commit boundary while moving Tensor preparation,
# LSTM construction, and detached application into the typed runtime.
rg -U -q 'schedulerMaterializationRead\.commit\(\);[\s\S]*EA::Inference::PrepareInferenceInput\([\s\S]*RunInferenceRuntime\(' "${main_file}"
rg -U -q 'RunInferenceEvaluation\(\n[[:space:]]*schedulerInferenceContext\.has_value\(\)[\s\S]{0,180}? nullptr' "${main_file}"
rg -q 'RevalidateSchedulerInferencePersistenceContextForWrite' "${main_file}"
rg -U -q 'pqxx::work schedulerPersistenceWork \{ c_LSTM \};[\s\S]{0,500}RevalidateSchedulerInferencePersistenceContextForWrite\([\s\S]{0,1600}PersistCompletedCheckpointInferenceResult\([\s\S]{0,900}PersistInferenceProfitabilityObservation\([\s\S]{0,1600}PersistCompletedInferenceResult\([\s\S]{0,900}PersistInferenceProfitabilityObservation\([\s\S]{0,1000}schedulerPersistenceWork\.commit\(\);' "${main_file}"
rg -q 'FOR UPDATE OF ce,e' "${main_file}"
rg -q 'FOR UPDATE;' "${main_file}"
rg -q 'scheduler_inference_binding_stale' "${main_file}"
rg -q 'ON CONFLICT \(checkpoint_eval_id\)' "${main_file}"
rg -q "ON CONFLICT \(" "${main_file}"

# Phase 22W moves infer-all before the scheduler/non-scheduler runtime work.
rg -q 'return RunInferAllForSymbol\(c_LSTM' "${main_file}"

printf '%s\n' 'LSTMPhase22VSchedulerFinalCheckpointMigrationTests passed'
