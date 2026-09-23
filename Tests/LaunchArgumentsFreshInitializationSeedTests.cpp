#include "LaunchArguments.hpp"
#include "StrategyEvaluationCore/Phase19BPostEntryPathMechanismExtractor.hpp"
#include "StrategyEvaluationCore/Phase19CCausalPathPredictability.hpp"
#include "StrategyEvaluationCore/Phase19StateInteractionAnalysis.hpp"
#include "StrategyEvaluationCore/StrategyEvaluation.hpp"

#include <cassert>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace EA::StrategyEvaluation
{
void ValidateControlledOneSidedStopExtensionInvocation(
    const ControlledOneSidedStopExtensionInvocationContext&)
{
}

void ValidatePhase19Invocation(const Phase19InvocationContext&)
{
}

void ValidatePhase19BInvocation(const Phase19BInvocationContext&)
{
}

void ValidatePhase19CInvocation(bool, bool, bool, bool, bool)
{
}
} // namespace EA::StrategyEvaluation

namespace
{
EA::LaunchArgs Parse(const std::vector<std::string>& arguments)
{
    std::vector<const char*> argv;
    argv.reserve(arguments.size());
    for (const std::string& argument : arguments)
        argv.push_back(argument.c_str());
    return EA::ParseLaunchArgs(static_cast<int>(argv.size()), argv.data());
}

void ExpectInvalid(const std::vector<std::string>& arguments,
                   std::string_view expectedMessage)
{
    try
    {
        (void)Parse(arguments);
        assert(false && "expected invalid_argument");
    }
    catch (const std::invalid_argument& error)
    {
        assert(std::string_view{error.what()}.find(expectedMessage) !=
               std::string_view::npos);
    }
}

std::vector<std::string> BasicTrainingArguments()
{
    return {"test", "--train", "2010-01-01", "2010-01-02"};
}
} // namespace

int main()
{
    auto separated = BasicTrainingArguments();
    separated.insert(separated.begin() + 2,
                     {"--fresh-initialization-seed", "42"});
    assert(Parse(separated).freshInitializationSeed == 42U);

    auto equals = BasicTrainingArguments();
    equals.insert(equals.begin() + 2, "--fresh-initialization-seed=42");
    assert(Parse(equals).freshInitializationSeed == 42U);

    // This is the scheduler's --name=value spelling plus the surrounding
    // required managed-training options used by BuildTrainCommand.
    const std::vector<std::string> schedulerEmitted = {
        "test", "--train", "--log-level=summary", "--checkpoint-every=0",
        "--new-model-name=experiment_652", "--donchian20-mode=enabled",
        "--feature-warmup-scope=full_history_warmup", "--donchian-lookback=20",
        "--fresh-initialization-seed=42", "--scheduler-experiment-id=652",
        "--training-objective=legacy", "--symbol=eurusd",
        "--prediction-horizon=4", "--threshold=0.0008", "--epochs=1",
        "2010-01-01", "2010-01-02"};
    assert(Parse(schedulerEmitted).freshInitializationSeed == 42U);

    ExpectInvalid({"test", "--train", "--fresh-initialization-seed", "42",
                   "--fresh-initialization-seed=43", "2010-01-01", "2010-01-02"},
                  "--fresh-initialization-seed specified more than once");
    ExpectInvalid({"test", "--train", "--fresh-initialization-seed=42",
                   "--fresh-initialization-seed", "43", "2010-01-01", "2010-01-02"},
                  "--fresh-initialization-seed specified more than once");
    ExpectInvalid({"test", "--train", "--fresh-initialization-seed=0",
                   "2010-01-01", "2010-01-02"}, "expected a positive integer");
    ExpectInvalid({"test", "--train", "--fresh-initialization-seed=4294967296",
                   "2010-01-01", "2010-01-02"}, "exceeds uint32 range");
    ExpectInvalid({"test", "--train", "--fresh-initialization-seed=not-a-number",
                   "2010-01-01", "2010-01-02"}, "expected a positive integer");
    ExpectInvalid({"test", "--resume-model-id=1", "--fresh-initialization-seed=42"},
                  "cannot be combined with --resume-model-id");
    return 0;
}
