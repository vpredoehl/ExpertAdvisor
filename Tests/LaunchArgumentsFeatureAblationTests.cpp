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

std::vector<std::string> SchedulerTrainArguments()
{
    return {"test", "--train", "--scheduler-experiment-id=669",
            "2010-01-01", "2010-01-02"};
}
} // namespace

int main()
{
    const std::string tg4Mask =
        "tg4_inner_break_any,tg4_source_tg3_structurally_eligible,"
        "tg4_source_tg3_confluent";

    auto accepted = SchedulerTrainArguments();
    accepted.insert(accepted.begin() + 3,
                    "--ablate-features=tg4_source_tg3_confluent,"
                    "tg4_inner_break_any,"
                    "tg4_source_tg3_structurally_eligible");
    const EA::LaunchArgs parsed = Parse(accepted);
    assert(parsed.featureAblationMask.has_value());
    assert(parsed.featureAblationMask->CanonicalText() == tg4Mask);

    const EA::LaunchArgs control = Parse(SchedulerTrainArguments());
    assert(!control.featureAblationMask.has_value());

    ExpectInvalid({"test", "--train", "--scheduler-experiment-id=669",
                   "--ablate-features=", "2010-01-01", "2010-01-02"},
                  "requires a non-empty feature mask");
    ExpectInvalid({"test", "--train", "--scheduler-experiment-id=669",
                   "--ablate-features=tg4_inner_break_any,,tg4_source_tg3_confluent",
                   "2010-01-01", "2010-01-02"},
                  "FEATURE_ABLATION_MASK_INVALID");
    ExpectInvalid({"test", "--train", "--scheduler-experiment-id=669",
                   "--ablate-features=not_a_feature", "2010-01-01",
                   "2010-01-02"}, "FEATURE_ABLATION_MASK_UNKNOWN_FEATURE");
    ExpectInvalid({"test", "--train", "--scheduler-experiment-id=669",
                   "--ablate-features=tg4_inner_break_any,tg4_inner_break_any",
                   "2010-01-01", "2010-01-02"},
                  "FEATURE_ABLATION_MASK_DUPLICATE_FEATURE");
    ExpectInvalid({"test", "--train", "--scheduler-experiment-id=669",
                   "--ablate-features=tg4_inner_break_any",
                   "--ablate-features=tg4_source_tg3_confluent",
                   "2010-01-01", "2010-01-02"},
                  "--ablate-features specified more than once");
    ExpectInvalid({"test", "--train", "--ablate-features=tg4_inner_break_any",
                   "2010-01-01", "2010-01-02"},
                  "requires --scheduler-experiment-id");
    ExpectInvalid({"test", "--infer", "--model=1",
                   "--scheduler-experiment-id=669",
                   "--ablate-features=tg4_inner_break_any", "2010-01-01",
                   "2010-01-02"}, "requires explicit scheduler-managed --train");
    ExpectInvalid({"test", "--run-frozen-model-outcome-inference="
                   "cohort,1,1,2010-01-01,2010-01-02,job",
                   "--ablate-features=tg4_inner_break_any"},
                  "rejects training, model, feature, scheduler, and inference");
    return 0;
}
