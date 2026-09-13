#pragma once

#include "SchedulerRepository.hpp"

#include <iosfwd>
#include <optional>
#include <string>

namespace EA::SchedulerCore
{

struct ExperimentTransitionRequest
{
    ExperimentTransitionAction action = ExperimentTransitionAction::Cancel;
    long long experimentId = -1;
    bool dryRun = false;
    bool confirmed = false;
};

struct ExperimentTransitionPlan
{
    bool accepted = false;
    std::string newStatus;
    std::string newPhase;
    std::string rejectionReason;
};

struct RetryTrainingCheckpointSelection
{
    std::optional<long long> previousResumeModelId;
    std::optional<long long> selectedResumeModelId;
    std::optional<int> selectedCompletedEpoch;
    bool promoted = false;
    std::string reason;
};

struct RequeueTrainingCheckpointSelection
{
    std::optional<long long> selectedResumeModelId;
    std::optional<int> selectedCompletedEpoch;
    std::string reason;
};

class ExperimentTransitionCheckpointSelector
{
public:
    virtual ~ExperimentTransitionCheckpointSelector() = default;

    virtual RetryTrainingCheckpointSelection selectRetryTrainingCheckpoint(
        const ExperimentTransitionRecord& experiment) = 0;
    virtual RequeueTrainingCheckpointSelection selectRequeueTrainingCheckpoint(
        const ExperimentTransitionRecord& experiment) = 0;
};

const char* ExperimentTransitionActionName(
    ExperimentTransitionAction action) noexcept;
ExperimentTransitionPlan PlanExperimentTransition(
    ExperimentTransitionAction action,
    const ExperimentTransitionRecord& experiment);

class ExperimentTransitionService final
{
public:
    ExperimentTransitionService(
        SchedulerRepository& repository,
        ExperimentTransitionCheckpointSelector& checkpointSelector,
        std::ostream& output);

    int run(const ExperimentTransitionRequest& request);

private:
    SchedulerRepository& repository_;
    ExperimentTransitionCheckpointSelector& checkpointSelector_;
    std::ostream& output_;
};

} // namespace EA::SchedulerCore
