#!/bin/bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
main="$root/LSTM/main.cpp"

for component in RuntimeDatabaseConnection LaunchRuntimeConfig RuntimeEconomicCalendarIdentity SchedulerWorkerOwnershipTestBoundary LstmHotspotProfileFinalizer; do
    test -f "$root/Sources/$component.hpp"
    test -f "$root/Sources/$component.cpp"
    ! rg -q 'TrainingWorkerApplication' "$root/Sources/$component.cpp" "$root/Sources/$component.hpp"
    ! rg -q 'PreparedRuntime|RuntimeContext|evaluationFacts|RunInferenceRuntime|RunInferenceEvaluation' "$root/Sources/$component.cpp" "$root/Sources/$component.hpp"
done

rg -q 'RuntimeDatabaseConnection::ForexConnectionString' "$main"
rg -q 'RuntimeDatabaseConnection::LstmConnectionString' "$main"
! rg -q '^std::string ForexDbConnectionString\(' "$main"
! rg -q '^std::string LstmDbConnectionString\(' "$main"
rg -q 'FOREX_DB_HOST' "$root/Sources/RuntimeDatabaseConnection.cpp"
rg -q 'FOREX_DB_NAME' "$root/Sources/RuntimeDatabaseConnection.cpp"
rg -q 'LSTM_DB_HOST' "$root/Sources/RuntimeDatabaseConnection.cpp"
rg -q 'LSTM_DB_NAME' "$root/Sources/RuntimeDatabaseConnection.cpp"

rg -q 'LaunchRuntimeConfig::Apply\(launchArgs, gRuntimeInferenceMode\)' "$main"
! rg -q '^void ApplyLaunchRuntimeConfig\(' "$main"
for assignment in prediction_horizon c_next_threshold window_size hidden_size n_out num_layers epoch_count core_lr_mult head_weight_lr_mult head_bias_lr_mult; do
    rg -q "$assignment" "$root/Sources/LaunchRuntimeConfig.cpp"
done
rg -q -- '--num-layers currently supports only 1; increasing layers would change the LSTM architecture' "$root/Sources/LaunchRuntimeConfig.cpp"

rg -q 'RuntimeEconomicCalendarIdentity::Resolve' "$main"
rg -q 'RuntimeEconomicCalendarIdentity::FromMaterialization' "$main"
! rg -q '^ResolveRuntimeEconomicCalendarSnapshot\(' "$main"
! rg -q '^EconomicCalendarSnapshotFromMaterialization\(' "$main"
for diagnostic in checkpoint_eval_not_found_for_economic_calendar_snapshot runtime_economic_calendar_snapshot_lineage_mismatch runtime_economic_calendar_snapshot_identity_conflict economic_calendar_snapshot_identity_incomplete; do
    rg -q "$diagnostic" "$root/Sources/RuntimeEconomicCalendarIdentity.cpp"
done

rg -q 'SchedulerWorkerOwnershipTestBoundary::Run' "$main"
! rg -q '^std::optional<int> RunCheckpointStopOwnershipTestBoundary' "$main"
for token in EA_SCHEDULER_OWNERSHIP_TEST_ENABLE EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY EA_SCHEDULER_OWNERSHIP_TEST_CHECKPOINT_MODEL_ID ea_scheduler_process_test_ SIGSTOP 'return 91' 'return 92' 'return 93' 'return 94' 'return 95'; do
    rg -q "$token" "$root/Sources/SchedulerWorkerOwnershipTestBoundary.cpp"
done
rg -q 'CheckpointTrainingControl::LoadCheckpointStopConfig' "$root/Sources/SchedulerWorkerOwnershipTestBoundary.cpp"
rg -q 'CheckpointTrainingControl::RecordCheckpointStopReached' "$root/Sources/SchedulerWorkerOwnershipTestBoundary.cpp"

rg -q 'EA::LstmHotspotProfileFinalizer' "$main"
! rg -q '^struct LSTMHotspotProfileFinalizer' "$main"
rg -q 'LSTM::PrintHotspotProfileSummary' "$root/Sources/LstmHotspotProfileFinalizer.cpp"
rg -q 'LSTM::WriteHotspotProfileReport' "$root/Sources/LstmHotspotProfileFinalizer.cpp"

echo "Runtime foundation structural tests passed"
