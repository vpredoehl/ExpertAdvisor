#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
main_file="${repo_root}/LSTM/main.cpp"

# Production infer-all discovery is a short advisory read.  Candidate
# compatibility is intentionally absent from discovery and takes place only
# after the dedicated candidate RR/RO materialization begins.
rg -U -q 'pqxx::work discoveryRead \{ database \};[\s\S]{0,500}SET TRANSACTION READ ONLY;[\s\S]{0,600}LoadInferAllCandidates\([\s\S]{0,500}discoveryRead\.commit\(\);' "${main_file}"
rg -U -q 'LoadInferAllCandidates\([\s\S]{0,1200}candidates\.push_back\(std::move\(candidate\)\)' "${main_file}"

# Every candidate that passes the fresh existing-result check gets its own
# explicit RR/RO snapshot, detached reader, detached compatibility check, and
# commit before LSTM construction/application/evaluation.
rg -U -q 'pqxx::work candidateRead \{ database \};[\s\S]{0,300}SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;[\s\S]{0,500}ReadPersistedModelMaterialization\([\s\S]{0,900}InferAllCandidateCompatible\([\s\S]{0,1300}candidateRead\.commit\(\);[\s\S]{0,900}RunInferAllModel\(' "${main_file}"

# Result-time state and writes use independent short transactions.  Completion
# remains atomic with profitability; failure remains independently durable.
rg -U -q 'pqxx::work resultRead \{ database \};[\s\S]{0,500}LoadCompletedInferenceResult\([\s\S]{0,400}resultRead\.commit\(\);' "${main_file}"
rg -U -q 'pqxx::work resultWrite \{ database \};[\s\S]{0,500}PersistCompletedInferenceResult\([\s\S]{0,600}PersistInferenceProfitabilityObservation\([\s\S]{0,400}resultWrite\.commit\(\);' "${main_file}"
rg -U -q 'pqxx::work failedResultWrite \{ database \};[\s\S]{0,400}PersistFailedInferenceResult\([\s\S]{0,300}failedResultWrite\.commit\(\);' "${main_file}"

# The migrated infer-all model path is query-free after materialization.
rg -U -q 'InferAllSummaryRow RunInferAllModel\([\s\S]{0,4000}ApplyPersistedModelMaterialization\([\s\S]{0,1600}RunInferenceEvaluation\(nullptr' "${main_file}"

printf '%s\n' 'LSTMPhase22WInferAllDetachedMaterializationMigrationTests passed'
