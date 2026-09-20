#pragma once

#include "InferenceRuntime.hpp"

#include <cstddef>
#include <optional>
#include <string>

namespace EA::Inference
{
// This is deliberately not a CLI contract.  Both the compatibility executable
// and a future semantic worker construct it at their process boundary.
struct ManagedInferenceRequest
{
    long long modelId = -1;
    std::optional<long long> finalExperimentId;
    std::optional<long long> checkpointEvalId;
    long long workerAttemptId = -1;
    std::string fromDate;
    std::string toDate;

    // Scheduler launch metadata may redundantly carry these persisted values. They are
    // validated against the RR/RO materialization rather than used as a
    // fallback, so managed work cannot become ordinary direct inference.
    std::optional<std::string> requestedSymbol;
    std::optional<std::size_t> requestedPredictionHorizon;
    std::optional<double> requestedThresholdLogret;
    std::optional<std::size_t> requestedWindowSize;
    std::optional<std::size_t> requestedHiddenSize;
    std::optional<std::size_t> requestedDonchianLookback;
    std::optional<Donchian20Mode> requestedDonchian20Mode;
    std::optional<EA::FeatureWarmupScope> requestedFeatureWarmupScope;

    ModelInputPreparation::DatabaseConnectionSettings database;
};

struct ManagedInferenceResult
{
    long long inferenceEvalResultId = -1;
    bool checkpoint = false;
    bool idempotentExisting = false;
};

// Owns the scheduler-managed final/checkpoint lifecycle: worker registration,
// RR/RO detached admission, query-free runtime computation, and fresh RW
// lock/revalidation/result-plus-profitability persistence.  It has no parser
// dependency.
ManagedInferenceResult RunManagedInference(const ManagedInferenceRequest& request);

} // namespace EA::Inference
