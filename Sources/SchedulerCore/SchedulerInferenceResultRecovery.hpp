#pragma once

#include <optional>

namespace EA::SchedulerCore
{

struct AuthoritativeFinalInferenceResult
{
    long long resultId = 0;
    long long modelId = 0;
    bool forcedFinalInferenceRerun = false;
};

// Historical recovery deliberately uses a separate evidence contract from
// live/orphan reconciliation. A candidate exists only when one exact failed
// inference attempt and one exact durable final result prove the corrupted
// completion relationship.
struct HistoricalFailedInferenceRecoveryEvidence
{
    long long experimentId = 0;
    long long workerAttemptId = 0;
    long long modelId = 0;
    long long inferenceResultId = 0;
};

enum class HistoricalFailedInferenceRecoveryEvidenceStatus
{
    Eligible,
    ExperimentNotEligible,
    NoQualifyingEvidence,
    AmbiguousEvidence
};

struct HistoricalFailedInferenceRecoveryDiscovery
{
    HistoricalFailedInferenceRecoveryEvidenceStatus status =
        HistoricalFailedInferenceRecoveryEvidenceStatus::
            ExperimentNotEligible;
    std::optional<HistoricalFailedInferenceRecoveryEvidence> evidence;
};

enum class HistoricalFailedInferenceRecoveryPersistenceResult
{
    Updated,
    AtomicPreconditionRejected
};

} // namespace EA::SchedulerCore
