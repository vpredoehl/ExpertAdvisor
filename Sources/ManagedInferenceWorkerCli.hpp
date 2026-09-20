#pragma once

#include "ManagedInferenceApplication.hpp"
#include "LstmRuntimeLogging.hpp"

#include <optional>

namespace EA::Inference
{
// Process-boundary grammar shared by the compatibility executable and the
// standalone semantic worker.  This deliberately accepts only scheduler
// managed final/checkpoint inference; the managed application remains the
// authority for lifecycle and persisted binding validation.
struct ManagedInferenceWorkerLaunch
{
    ManagedInferenceRequest request;
    std::optional<EA::RuntimeLogLevel> logLevel;
};

ManagedInferenceWorkerLaunch ParseManagedInferenceWorkerArgs(
    int argc,
    const char* argv[],
    ModelInputPreparation::DatabaseConnectionSettings database);

// Keeps the compatibility process result contract in one place.  In
// particular registration failure remains the scheduler-reserved status 125.
int RunManagedInferenceWorker(const ManagedInferenceWorkerLaunch& launch);

// Standalone entry point adapter.  It intentionally exposes no general LSTM
// CLI modes.
int RunStandaloneManagedInferenceWorkerCli(int argc, const char* argv[]);

} // namespace EA::Inference
