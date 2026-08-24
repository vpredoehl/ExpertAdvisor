#include <cassert>
#include <climits>
#include <functional>
#include <stdexcept>
#include <string>

#include "SchedulerWorkerLimits.hpp"

namespace
{

void ExpectInvalid(const std::function<void()>& operation)
{
    bool rejected = false;
    try
    {
        operation();
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    assert(rejected);
}

void ParseArguments(
    int argc,
    const char* argv[],
    int& train,
    int& infer,
    int& analyze)
{
    for (int index = 1; index < argc; ++index)
    {
        assert(EA::ExperimentScheduler::
            TryParseSchedulerWorkerLimitArgument(
                argc, argv, index, train, infer, analyze));
    }
}

} // namespace

int main()
{
    using namespace EA::ExperimentScheduler;

    // These constants preserve the defaults verified in SchedulerOptions.
    static_assert(kDefaultMaxTrainProcs == 1);
    static_assert(kDefaultMaxInferProcs == 1);
    static_assert(kDefaultMaxAnalyzeProcs == 1);

    int train = kDefaultMaxTrainProcs;
    int infer = kDefaultMaxInferProcs;
    int analyze = kDefaultMaxAnalyzeProcs;
    {
        const char* argv[] = {
            "LSTM_Release",
            "--max-train-procs=0",
            "--max-infer-procs=0",
            "--max-analyze-procs=0"};
        ParseArguments(4, argv, train, infer, analyze);
        assert(train == 0);
        assert(infer == 0);
        assert(analyze == 0);
    }
    ExpectInvalid([] {
        int localTrain = kDefaultMaxTrainProcs;
        int localInfer = kDefaultMaxInferProcs;
        int localAnalyze = kDefaultMaxAnalyzeProcs;
        const char* argv[] = {
            "LSTM_Release", "--max-train-procs", "-1"};
        ParseArguments(
            3, argv, localTrain, localInfer, localAnalyze);
    });
    ExpectInvalid([] {
        int localTrain = kDefaultMaxTrainProcs;
        int localInfer = kDefaultMaxInferProcs;
        int localAnalyze = kDefaultMaxAnalyzeProcs;
        const char* argv[] = {
            "LSTM_Release", "--max-infer-procs=1x"};
        ParseArguments(
            2, argv, localTrain, localInfer, localAnalyze);
    });
    ExpectInvalid([] {
        int localTrain = kDefaultMaxTrainProcs;
        int localInfer = kDefaultMaxInferProcs;
        int localAnalyze = kDefaultMaxAnalyzeProcs;
        const char* argv[] = {
            "LSTM_Release", "--max-analyze-procs="};
        ParseArguments(
            2, argv, localTrain, localInfer, localAnalyze);
    });
    {
        const char* argv[] = {
            "LSTM_Release",
            "--max-train-procs", "0",
            "--max-infer-procs", "0",
            "--max-analyze-procs", "0"};
        ParseArguments(7, argv, train, infer, analyze);
        assert(train == 0);
        assert(infer == 0);
        assert(analyze == 0);
    }

    assert(ParseNonNegativeSchedulerWorkerLimit(
               "--max-train-procs", "1") == 1);
    assert(ParseNonNegativeSchedulerWorkerLimit(
               "--max-infer-procs", "2") == 2);
    assert(ParseNonNegativeSchedulerWorkerLimit(
               "--max-analyze-procs", std::to_string(INT_MAX)) == INT_MAX);
    for (const std::string invalid : {
             "-1", "-2", "", " ", "not-a-number", "1x",
             "2147483648", "999999999999999999999999999999"})
    {
        ExpectInvalid([&] {
            (void)ParseNonNegativeSchedulerWorkerLimit(
                "--max-train-procs", invalid);
        });
    }

    train = kDefaultMaxTrainProcs;
    infer = kDefaultMaxInferProcs;
    analyze = kDefaultMaxAnalyzeProcs;
    {
        const char* argv[] = {"LSTM_Release", "--help"};
        int index = 1;
        assert(!TryParseSchedulerWorkerLimitArgument(
            2, argv, index, train, infer, analyze));
        assert(train == kDefaultMaxTrainProcs);
        assert(infer == kDefaultMaxInferProcs);
        assert(analyze == kDefaultMaxAnalyzeProcs);
    }

    // Zero is a closed admission class, including when usage already exceeds it.
    assert(!SchedulerWorkerCapacityHasSlot(0, 0));
    assert(!SchedulerWorkerCapacityHasSlot(0, 1));
    assert(AvailableWorkerProcessSlots(0, 0) == 0);
    assert(AvailableWorkerProcessSlots(0, 3) == 0);
    assert(SchedulerWorkerCapacityExceeded(0, 1));
    assert(!SchedulerWorkerCapacityExceeded(0, 0));

    // Positive admission and clamping remain unchanged.
    assert(SchedulerWorkerCapacityHasSlot(2, 0));
    assert(SchedulerWorkerCapacityHasSlot(2, 1));
    assert(!SchedulerWorkerCapacityHasSlot(2, 2));
    assert(!SchedulerWorkerCapacityHasSlot(2, 3));
    assert(AvailableWorkerProcessSlots(2, 0) == 2);
    assert(AvailableWorkerProcessSlots(2, 1) == 1);
    assert(AvailableWorkerProcessSlots(2, 2) == 0);
    assert(AvailableWorkerProcessSlots(2, 3) == 0);

    // Final and checkpoint inference share the same infer class.
    assert(AvailableInferProcessSlots(0, 0, 0) == 0);
    assert(AvailableInferProcessSlots(2, 1, 0) == 1);
    assert(AvailableInferProcessSlots(2, 0, 1) == 1);
    assert(AvailableInferProcessSlots(2, 1, 1) == 0);
    assert(AvailableInferProcessSlots(2, 3, 1) == 0);

    // Status extraction retains explicit zero for both supported forms.
    const auto statusEquals = ExtractSchedulerWorkerLimitFromCommand(
        "LSTM_Release --schedule-experiments --max-infer-procs=0",
        "--max-infer-procs");
    const auto statusSeparate = ExtractSchedulerWorkerLimitFromCommand(
        "LSTM_Release --schedule-experiments --max-analyze-procs 0",
        "--max-analyze-procs");
    assert(statusEquals.has_value() && *statusEquals == 0);
    assert(statusSeparate.has_value() && *statusSeparate == 0);
    assert(!ExtractSchedulerWorkerLimitFromCommand(
                "LSTM_Release --schedule-experiments",
                "--max-train-procs")
                .has_value());

    // Each zero closes only its own independent capacity class.
    assert(AvailableWorkerProcessSlots(0, 0) == 0); // train
    assert(AvailableWorkerProcessSlots(1, 0) == 1); // infer
    assert(AvailableWorkerProcessSlots(1, 0) == 1); // analyze
    assert(AvailableWorkerProcessSlots(1, 0) == 1); // train
    assert(AvailableWorkerProcessSlots(0, 0) == 0); // infer
    assert(AvailableWorkerProcessSlots(1, 0) == 1); // analyze
    assert(AvailableWorkerProcessSlots(1, 0) == 1); // train
    assert(AvailableWorkerProcessSlots(1, 0) == 1); // infer
    assert(AvailableWorkerProcessSlots(0, 0) == 0); // analyze

    return 0;
}
