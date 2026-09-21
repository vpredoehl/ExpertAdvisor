#include <cassert>
#include <string>

#include "SchedulerStatusProcessRecognition.hpp"
#include "SchedulerWorkerLimits.hpp"

int main()
{
    using namespace EA::ExperimentScheduler;

    const std::string standalone =
        "/opt/ExpertAdvisor/lstm-scheduler --schedule-experiments "
        "--max-train-procs=2 --max-infer-procs 1 "
        "--max-analyze-procs=2";
    assert(IsSchedulerStatusSchedulerProcessCommand(standalone));
    assert(SchedulerStatusCommandHasExecutableBasename(
        standalone, "lstm-scheduler"));
    assert(IsSchedulerStatusSchedulerProcessCommand(
        "/Volumes/Developer SSD/ExpertAdvisor/lstm-scheduler "
        "--schedule-experiments"));
    const auto train = ExtractSchedulerWorkerLimitFromCommand(
        standalone, "--max-train-procs");
    const auto infer = ExtractSchedulerWorkerLimitFromCommand(
        standalone, "--max-infer-procs");
    const auto analyze = ExtractSchedulerWorkerLimitFromCommand(
        standalone, "--max-analyze-procs");
    assert(train == 2);
    assert(infer == 1);
    assert(analyze == 2);

    assert(IsSchedulerStatusSchedulerProcessCommand(
        "/opt/ExpertAdvisor/LSTM_Release --schedule-experiments "
        "--max-train-procs=1"));
    assert(IsSchedulerStatusSchedulerProcessCommand(
        "LSTM --schedule-experiments"));

    assert(!IsSchedulerStatusSchedulerProcessCommand(
        "/opt/ExpertAdvisor/lstm-infer-worker --infer"));
    assert(!IsSchedulerStatusSchedulerProcessCommand(
        "/opt/ExpertAdvisor/lstm-analyze-worker --analyze-experiment=650"));
    assert(!IsSchedulerStatusSchedulerProcessCommand(
        "/opt/ExpertAdvisor/LSTM_Release --train"));
    assert(!IsSchedulerStatusSchedulerProcessCommand(
        "/usr/bin/printf LSTM_Release --schedule-experiments"));
    assert(!IsSchedulerStatusSchedulerProcessCommand(
        "/usr/bin/printf --worker=/tmp/lstm-scheduler "
        "--schedule-experiments"));
    assert(!IsSchedulerStatusSchedulerProcessCommand(
        "/opt/other-lstm-scheduler --schedule-experiments"));
    assert(!IsSchedulerStatusSchedulerProcessCommand(
        "/opt/ExpertAdvisor/lstm-scheduler --schedule-experiments-extra"));
    assert(!IsSchedulerStatusSchedulerProcessCommand(""));

    return 0;
}
