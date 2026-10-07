#include "SchedulerWorkerOwnershipTestBoundary.hpp"

#include "CheckpointTrainingControl.hpp"
#include "LaunchArguments.hpp"

#include <cstdlib>
#include <iostream>
#include <signal.h>
#include <string>
#include <unistd.h>

namespace EA::SchedulerWorkerOwnershipTestBoundary
{
std::optional<int> Run(const LaunchArgs& launchArgs)
{
    const char* enabled = std::getenv("EA_SCHEDULER_OWNERSHIP_TEST_ENABLE");
    const char* boundary = std::getenv("EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY");
    const char* database = std::getenv("LSTM_DB_NAME");
    if (!enabled || std::string{enabled} != "1" || !boundary ||
        std::string{boundary} != "checkpoint_stop_after_transition_before_exit" ||
        !database || std::string{database}.rfind("ea_scheduler_process_test_", 0) != 0)
        return std::nullopt;
    if (!launchArgs.schedulerExperimentId || !launchArgs.schedulerWorkerAttemptId || !launchArgs.inferenceMode)
        return 91;
    if (*launchArgs.inferenceMode)
    {
        std::cout << "SCHEDULER_TEST_CHECKPOINT_NEXT_PHASE_HELD"
                  << ",experiment_id=" << *launchArgs.schedulerExperimentId
                  << ",worker_attempt_id=" << *launchArgs.schedulerWorkerAttemptId
                  << ",phase=infer" << std::endl;
        std::cout.flush();
        if (::kill(::getpid(), SIGSTOP) != 0) return 92;
        return 0;
    }
    const char* modelText = std::getenv("EA_SCHEDULER_OWNERSHIP_TEST_CHECKPOINT_MODEL_ID");
    if (!modelText || !*modelText || !launchArgs.checkpointEvery || !launchArgs.epochs) return 93;
    const long long modelId = ParseModelIdArg(modelText);
    const int checkpointEpoch = *launchArgs.checkpointEvery;
    const auto stop = CheckpointTrainingControl::LoadCheckpointStopConfig(
        launchArgs.schedulerExperimentId, checkpointEpoch, *launchArgs.epochs,
        launchArgs.checkpointEvery);
    if (!stop || !CheckpointTrainingControl::RecordCheckpointStopReached(
            launchArgs.schedulerExperimentId, launchArgs.schedulerWorkerAttemptId,
            checkpointEpoch, modelId))
        return 94;
    std::cout << "SCHEDULER_TEST_CHECKPOINT_TRAIN_ATTEMPT_TERMINAL"
              << ",experiment_id=" << *launchArgs.schedulerExperimentId
              << ",worker_attempt_id=" << *launchArgs.schedulerWorkerAttemptId
              << ",checkpoint_epoch=" << checkpointEpoch
              << ",checkpoint_model_id=" << modelId << std::endl;
    std::cout.flush();
    if (::kill(::getpid(), SIGSTOP) != 0) return 95;
    return 0;
}
}
