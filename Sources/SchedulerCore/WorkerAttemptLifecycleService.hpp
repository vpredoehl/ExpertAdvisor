#pragma once

#include "SchedulerRepository.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace EA::SchedulerCore
{

struct WorkerAttemptLifecycleContext
{
    std::string schedulerInvocationId;
    long long schedulerFencingToken = 0;
    std::string selectedWorkerCanonicalExecutablePath;
    int semanticLayoutVersion = 0;
    std::size_t modelInputWidth = 0;
    std::string semanticWorkerRole;
    std::string sourceCommit;
    std::string executableSha256;
    std::string runtimeIdentity;
    std::string canonicalManifestPath;
};

struct ExperimentWorkerAttemptRequest
{
    long long experimentId = -1;
    std::string phase;
    std::string logPath;
    std::string identityNonce;
    bool cancellationOnly = false;
};

struct CheckpointWorkerAttemptRequest
{
    long long experimentId = -1;
    long long checkpointEvalId = -1;
    std::string logPath;
    std::string identityNonce;
};

struct SpawnedWorkerAttemptEvidence
{
    ReservedWorkerAttempt attempt;
    int workerPid = -1;
    std::string processStartIdentity;
    std::string commandLine;
    bool requireSchedulerAuthority = true;
};

struct WorkerAttemptLaunchFailure
{
    ReservedWorkerAttempt attempt;
    int exitCode = 127;
    std::string diagnostic;
};

class WorkerAttemptLifecycleService final
{
public:
    WorkerAttemptLifecycleService(
        SchedulerRepository& repository,
        WorkerAttemptLifecycleContext context);

    std::optional<ReservedWorkerAttempt> reserveExperiment(
        const ExperimentWorkerAttemptRequest& request);
    std::optional<ReservedWorkerAttempt> reserveCheckpoint(
        const CheckpointWorkerAttemptRequest& request);
    void recordSpawned(const SpawnedWorkerAttemptEvidence& evidence);
    void recordLaunchFailure(const WorkerAttemptLaunchFailure& failure);

private:
    std::string launchIdentity(
        std::string_view commandIdentity,
        std::string_view nonce) const;

    SchedulerRepository& repository_;
    WorkerAttemptLifecycleContext context_;
};

} // namespace EA::SchedulerCore
