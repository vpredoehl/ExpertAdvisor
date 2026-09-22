#pragma once

#include <optional>

namespace EA::CheckpointTrainingControl
{

struct CheckpointStopConfig
{
    int requestedEpoch = 0;
    int effectiveEpoch = 0;
    bool cancellationRequested = false;
};

void UpdateSchedulerExperimentProgress(
    const std::optional<long long>& experimentId,
    int completedEpoch);

void QueueCheckpointInferenceIfEligible(
    const std::optional<long long>& schedulerExperimentId,
    const std::optional<int>& checkpointEvery,
    int checkpointEpoch,
    long long checkpointModelId);

std::optional<CheckpointStopConfig> LoadCheckpointStopConfig(
    const std::optional<long long>& experimentId,
    int checkpointEpoch,
    int targetEpochs,
    const std::optional<int>& checkpointEvery);

bool RecordCheckpointStopReached(
    const std::optional<long long>& experimentId,
    const std::optional<long long>& workerAttemptId,
    int epoch,
    long long modelId);

} // namespace EA::CheckpointTrainingControl
