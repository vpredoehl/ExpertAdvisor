#include "LaunchArguments.hpp"

#include "CanonicalSymbol.hpp"
#include "DonchianLookback.hpp"
#include "../Sources/StrategyEvaluationCore/StrategyEvaluation.hpp"
#include "../Sources/StrategyEvaluationCore/Phase19StateInteractionAnalysis.hpp"
#include "../Sources/StrategyEvaluationCore/Phase19BPostEntryPathMechanismExtractor.hpp"
#include "../Sources/StrategyEvaluationCore/Phase19CCausalPathPredictability.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace EA
{
RuntimeLogLevel ParseRuntimeLogLevel(const std::string& value)
{
    if (value == "quiet")
        return RuntimeLogLevel::Quiet;
    if (value == "summary")
        return RuntimeLogLevel::Summary;
    if (value == "diagnostic")
        return RuntimeLogLevel::Diagnostic;
    throw std::invalid_argument("invalid --log-level value '" + value + "'; expected quiet, summary, or diagnostic");
}

bool HasControlledStrategyEvaluation(const LaunchArgs& launchArgs)
{
    return launchArgs.controlledFixedStopEvaluationPath.has_value() ||
        launchArgs.probabilityConditionedStopEvaluationPath.has_value() ||
        launchArgs.probabilityConditionedStopExtensionEvaluationPath.
            has_value() ||
        launchArgs.probabilityStopExtensionStateAnalysisPath.has_value() ||
        launchArgs.probabilityStopExtensionPathMechanismPath.has_value() ||
        launchArgs.phase19CCausalPathPredictabilityDirectory.has_value();
}

long long ParseModelIdArg(const std::string& value)
{
    if (value.empty())
        throw std::invalid_argument("--model requires a non-empty model_id");

    size_t consumed = 0;
    long long modelId = 0;
    try
    {
        modelId = std::stoll(value, &consumed, 10);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("invalid --model value '" + value + "'; expected a positive integer model_id");
    }

    if (consumed != value.size() || modelId <= 0)
        throw std::invalid_argument("invalid --model value '" + value + "'; expected a positive integer model_id");

    return modelId;
}

LaunchArgs::FrozenOutcomeSpec ParseFrozenOutcomeSpec(const std::string& value)
{
    std::vector<std::string> fields;
    std::size_t begin = 0;
    for (;;)
    {
        const std::size_t comma = value.find(',', begin);
        fields.push_back(value.substr(begin, comma - begin));
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    if (fields.size() != 6 ||
        std::any_of(fields.begin(), fields.end(),
                    [](const std::string& field) { return field.empty(); }))
        throw std::invalid_argument(
            "--run-frozen-model-outcome-inference requires "
            "COHORT_HASH,SOURCE_EXPERIMENT_ID,SOURCE_MODEL_ID,FROM_DATE,"
            "TO_DATE,JOB_HASH");
    LaunchArgs::FrozenOutcomeSpec spec;
    spec.cohortHash = fields[0];
    spec.sourceExperimentId = ParseModelIdArg(fields[1]);
    spec.sourceModelId = ParseModelIdArg(fields[2]);
    spec.outcomeStart = fields[3];
    spec.outcomeEnd = fields[4];
    spec.jobHash = fields[5];
    return spec;
}

size_t ParsePositiveSizeArg(const std::string& optionName, const std::string& value)
{
    if (value.empty())
        throw std::invalid_argument(optionName + " requires a non-empty integer value");

    size_t consumed = 0;
    unsigned long long parsed = 0;
    try
    {
        parsed = std::stoull(value, &consumed, 10);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'; expected a positive integer");
    }

    if (consumed != value.size() || parsed == 0)
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'; expected a positive integer");

    return static_cast<size_t>(parsed);
}

int ParsePositiveIntArg(const std::string& optionName, const std::string& value)
{
    const size_t parsed = ParsePositiveSizeArg(optionName, value);
    if (parsed > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'; exceeds int range");
    return static_cast<int>(parsed);
}

unsigned int ParseFreshInitializationSeedArg(const std::string& value)
{
    const size_t parsed = ParsePositiveSizeArg("--fresh-initialization-seed", value);
    if (parsed > std::numeric_limits<unsigned int>::max())
        throw std::invalid_argument("--fresh-initialization-seed exceeds uint32 range");
    return static_cast<unsigned int>(parsed);
}

int ParseNonNegativeIntArg(const std::string& optionName, const std::string& value)
{
    if (value.empty())
        throw std::invalid_argument(optionName + " requires a non-empty integer value");

    size_t consumed = 0;
    long long parsed = 0;
    try
    {
        parsed = std::stoll(value, &consumed, 10);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'; expected a non-negative integer");
    }

    if (consumed != value.size() || parsed < 0 || parsed > std::numeric_limits<int>::max())
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'; expected a non-negative integer");

    return static_cast<int>(parsed);
}

double ParsePositiveDoubleArg(const std::string& optionName, const std::string& value)
{
    if (value.empty())
        throw std::invalid_argument(optionName + " requires a non-empty numeric value");

    size_t consumed = 0;
    double parsed = 0.0;
    try
    {
        parsed = std::stod(value, &consumed);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'; expected a positive number");
    }

    if (consumed != value.size() || !std::isfinite(parsed) || parsed <= 0.0)
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'; expected a positive number");

    return parsed;
}

bool SplitOptionWithValue(const std::string& arg,
                          const char* optionName,
                          std::string& value)
{
    const std::string prefix = std::string(optionName) + "=";
    if (arg.rfind(prefix, 0) != 0)
        return false;

    value = arg.substr(prefix.size());
    return true;
}

LaunchArgs ParseLaunchArgs(int argc, const char* argv[])
{
    constexpr const char* kModelPrefix = "--model=";
    constexpr size_t kModelPrefixLen = 8;
    constexpr const char* kResumeModelPrefix = "--resume-model-id=";
    constexpr size_t kResumeModelPrefixLen = 18;

    LaunchArgs parsed;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg{ argv[i] };

        if (arg.rfind("--run-frozen-model-outcome-inference=", 0) == 0)
        {
            if (parsed.frozenOutcome)
                throw std::invalid_argument(
                    "--run-frozen-model-outcome-inference specified more than once");
            parsed.frozenOutcome = ParseFrozenOutcomeSpec(arg.substr(
                std::string{"--run-frozen-model-outcome-inference="}.size()));
        }
        else if (arg.rfind(kModelPrefix, 0) == 0)
        {
            if (parsed.modelId.has_value())
                throw std::invalid_argument("--model specified more than once");

            parsed.modelId = ParseModelIdArg(arg.substr(kModelPrefixLen));
        }
        else if (arg.rfind(kResumeModelPrefix, 0) == 0)
        {
            if (parsed.resumeModelId.has_value())
                throw std::invalid_argument("--resume-model-id specified more than once");
            parsed.resumeModelId = ParseModelIdArg(arg.substr(kResumeModelPrefixLen));
        }
        else if (arg == "--resume-model-id")
        {
            if (parsed.resumeModelId.has_value())
                throw std::invalid_argument("--resume-model-id specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--resume-model-id requires a model_id value");
            parsed.resumeModelId = ParseModelIdArg(argv[++i]);
        }
        else if (arg == "--resume-expand-input-width")
        {
            if (parsed.resumeExpandInputWidth)
                throw std::invalid_argument(
                    "--resume-expand-input-width specified more than once");
            parsed.resumeExpandInputWidth = true;
        }
        else if (std::string seedValue;
                 SplitOptionWithValue(arg, "--fresh-initialization-seed", seedValue))
        {
            if (parsed.freshInitializationSeed)
                throw std::invalid_argument("--fresh-initialization-seed specified more than once");
            parsed.freshInitializationSeed = ParseFreshInitializationSeedArg(seedValue);
        }
        else if (arg == "--fresh-initialization-seed")
        {
            if (parsed.freshInitializationSeed)
                throw std::invalid_argument("--fresh-initialization-seed specified more than once");
            if (++i >= argc)
                throw std::invalid_argument("--fresh-initialization-seed requires a value");
            parsed.freshInitializationSeed = ParseFreshInitializationSeedArg(argv[i]);
        }
        else if (arg == "--target-epochs")
        {
            if (parsed.targetEpochs.has_value())
                throw std::invalid_argument("--target-epochs specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--target-epochs requires a value");
            parsed.targetEpochs = ParsePositiveIntArg("--target-epochs", argv[++i]);
        }
        else if (arg == "--new-model-name")
        {
            if (parsed.newModelName.has_value())
                throw std::invalid_argument("--new-model-name specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--new-model-name requires a value");
            std::string value{ argv[++i] };
            if (value.empty())
                throw std::invalid_argument("--new-model-name requires a non-empty value");
            parsed.newModelName = value;
        }
        else if (arg == "--checkpoint-every")
        {
            if (parsed.checkpointEvery.has_value())
                throw std::invalid_argument("--checkpoint-every specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--checkpoint-every requires a value");
            parsed.checkpointEvery = ParseNonNegativeIntArg("--checkpoint-every", argv[++i]);
        }
        else if (arg == "--donchian20-mode")
        {
            if (parsed.donchian20Mode.has_value())
                throw std::invalid_argument("--donchian20-mode specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--donchian20-mode requires enabled or zero_ablation");
            parsed.donchian20Mode = ParseDonchian20Mode(argv[++i]);
        }
        else if (arg == "--feature-warmup-scope")
        {
            if (parsed.featureWarmupScope.has_value())
                throw std::invalid_argument("--feature-warmup-scope specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--feature-warmup-scope requires a value");
            parsed.featureWarmupScope = EA::ParseFeatureWarmupScope(argv[++i]);
        }
        else if (arg == "--donchian-lookback")
        {
            if (parsed.donchianLookback.has_value())
                throw std::invalid_argument("--donchian-lookback specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--donchian-lookback requires a positive integer");
            parsed.donchianLookback = ParseDonchianLookback(argv[++i]);
        }
        else if (arg == "--infer-start-after-model-id")
        {
            if (parsed.inferStartAfterModelId.has_value())
                throw std::invalid_argument("--infer-start-after-model-id specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--infer-start-after-model-id requires a model_id value");
            parsed.inferStartAfterModelId = ParseModelIdArg(argv[++i]);
        }
        else if (arg == "--scheduler-experiment-id")
        {
            if (parsed.schedulerExperimentId.has_value())
                throw std::invalid_argument("--scheduler-experiment-id specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--scheduler-experiment-id requires an experiment_id value");
            parsed.schedulerExperimentId = ParseModelIdArg(argv[++i]);
        }
        else if (arg == "--scheduler-checkpoint-eval-id")
        {
            if (parsed.schedulerCheckpointEvalId.has_value())
                throw std::invalid_argument("--scheduler-checkpoint-eval-id specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--scheduler-checkpoint-eval-id requires a checkpoint_eval_id value");
            parsed.schedulerCheckpointEvalId = ParseModelIdArg(argv[++i]);
        }
        else if (arg == "--scheduler-worker-attempt-id")
        {
            if (parsed.schedulerWorkerAttemptId.has_value())
                throw std::invalid_argument(
                    "--scheduler-worker-attempt-id specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument(
                    "--scheduler-worker-attempt-id requires an attempt_id value");
            parsed.schedulerWorkerAttemptId =
                ParseModelIdArg(argv[++i]);
        }
        else if (arg == "--training-objective")
        {
            if (parsed.trainingObjective.has_value())
                throw std::invalid_argument(
                    "--training-objective specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument(
                    "--training-objective requires a value");
            parsed.trainingObjective =
                EA::TrainingObjective::ParseCliSelection(argv[++i]);
        }
        else if (arg == "--log-level")
        {
            if (parsed.logLevel.has_value())
                throw std::invalid_argument("--log-level specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--log-level requires quiet, summary, or diagnostic");
            parsed.logLevel = ParseRuntimeLogLevel(argv[++i]);
        }
        else if (arg == "--model")
        {
            if (parsed.modelId.has_value())
                throw std::invalid_argument("--model specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--model requires a model_id value");

            parsed.modelId = ParseModelIdArg(argv[++i]);
        }
        else if (arg == "--train")
        {
            if (parsed.inferenceMode.has_value())
                throw std::invalid_argument("--train and --infer are mutually exclusive");
            parsed.inferenceMode = false;
        }
        else if (arg == "--infer")
        {
            if (parsed.inferenceMode.has_value())
                throw std::invalid_argument("--train and --infer are mutually exclusive");
            parsed.inferenceMode = true;
        }
        else if (arg == "--eval-trading")
        {
            parsed.evalTrading = true;
        }
        else if (arg == "--controlled-fixed-stop-evaluation")
        {
            if (HasControlledStrategyEvaluation(parsed))
                throw std::invalid_argument(
                    "strategy evaluation artifact option specified more than once");
            if (i + 1 >= argc || std::string{argv[i + 1]}.empty())
                throw std::invalid_argument(
                    "--controlled-fixed-stop-evaluation requires an output path");
            parsed.controlledFixedStopEvaluationPath = argv[++i];
        }
        else if (arg == "--probability-conditioned-stop-evaluation")
        {
            if (HasControlledStrategyEvaluation(parsed))
                throw std::invalid_argument(
                    "strategy evaluation artifact option specified more than once");
            if (i + 1 >= argc || std::string{argv[i + 1]}.empty())
                throw std::invalid_argument(
                    "--probability-conditioned-stop-evaluation requires an output path");
            parsed.probabilityConditionedStopEvaluationPath = argv[++i];
        }
        else if (arg ==
                 "--probability-conditioned-stop-extension-evaluation")
        {
            if (HasControlledStrategyEvaluation(parsed))
                throw std::invalid_argument(
                    "strategy evaluation artifact option specified more than once");
            if (i + 1 >= argc || std::string{argv[i + 1]}.empty())
                throw std::invalid_argument(
                    "--probability-conditioned-stop-extension-evaluation requires an output path");
            parsed.probabilityConditionedStopExtensionEvaluationPath =
                argv[++i];
        }
        else if (arg ==
                 "--probability-stop-extension-state-analysis")
        {
            if (HasControlledStrategyEvaluation(parsed))
                throw std::invalid_argument(
                    "strategy evaluation artifact option specified more than once");
            if (i + 1 >= argc || std::string{argv[i + 1]}.empty())
                throw std::invalid_argument(
                    "--probability-stop-extension-state-analysis requires an output path");
            parsed.probabilityStopExtensionStateAnalysisPath = argv[++i];
        }
        else if (arg ==
                 "--probability-stop-extension-path-mechanism")
        {
            if (HasControlledStrategyEvaluation(parsed))
                throw std::invalid_argument(
                    "strategy evaluation artifact option specified more than once");
            if (i + 1 >= argc || std::string{argv[i + 1]}.empty())
                throw std::invalid_argument(
                    "--probability-stop-extension-path-mechanism requires an output path");
            parsed.probabilityStopExtensionPathMechanismPath = argv[++i];
        }
        else if (arg == "--phase19c-causal-path-predictability")
        {
            if (HasControlledStrategyEvaluation(parsed))
                throw std::invalid_argument(
                    "strategy evaluation artifact option specified more than once");
            if (i + 1 >= argc || std::string{argv[i + 1]}.empty())
                throw std::invalid_argument(
                    "--phase19c-causal-path-predictability requires an output directory");
            parsed.phase19CCausalPathPredictabilityDirectory = argv[++i];
        }
        else if (arg == "--phase19b-path-artifact-dir")
        {
            if (parsed.phase19BPathArtifactDirectory || i + 1 >= argc ||
                std::string{argv[i + 1]}.empty())
                throw std::invalid_argument(
                    "--phase19b-path-artifact-dir requires one directory");
            parsed.phase19BPathArtifactDirectory = argv[++i];
        }
        else if (arg == "--infer-all")
        {
            parsed.inferAll = true;
        }
        else if (arg == "--force-infer")
        {
            parsed.forceInfer = true;
        }
        else if (arg == "--lstm-profile-hotspots")
        {
            parsed.lstmProfileHotspots = true;
        }
        else
        {
            std::string value;
            if (SplitOptionWithValue(arg, "--prediction-horizon", value))
            {
                parsed.predictionHorizon = ParsePositiveSizeArg("--prediction-horizon", value);
            }
            else if (SplitOptionWithValue(arg, "--threshold", value))
            {
                parsed.thresholdLogret = ParsePositiveDoubleArg("--threshold", value);
            }
            else if (SplitOptionWithValue(arg, "--window-size", value))
            {
                parsed.windowSize = ParsePositiveSizeArg("--window-size", value);
            }
            else if (SplitOptionWithValue(arg, "--hidden-size", value))
            {
                parsed.hiddenSize = ParsePositiveSizeArg("--hidden-size", value);
            }
            else if (SplitOptionWithValue(arg, "--num-layers", value))
            {
                parsed.numLayers = ParsePositiveSizeArg("--num-layers", value);
            }
            else if (SplitOptionWithValue(arg, "--epochs", value))
            {
                parsed.epochs = ParsePositiveIntArg("--epochs", value);
            }
            else if (SplitOptionWithValue(
                         arg, "--controlled-fixed-stop-evaluation", value))
            {
                if (HasControlledStrategyEvaluation(parsed) || value.empty())
                    throw std::invalid_argument(
                        "invalid or duplicate --controlled-fixed-stop-evaluation");
                parsed.controlledFixedStopEvaluationPath = value;
            }
            else if (SplitOptionWithValue(
                         arg, "--probability-conditioned-stop-evaluation",
                         value))
            {
                if (HasControlledStrategyEvaluation(parsed) || value.empty())
                    throw std::invalid_argument(
                        "invalid or duplicate --probability-conditioned-stop-evaluation");
                parsed.probabilityConditionedStopEvaluationPath = value;
            }
            else if (SplitOptionWithValue(
                         arg,
                         "--probability-conditioned-stop-extension-evaluation",
                         value))
            {
                if (HasControlledStrategyEvaluation(parsed) || value.empty())
                    throw std::invalid_argument(
                        "invalid or duplicate --probability-conditioned-stop-extension-evaluation");
                parsed.probabilityConditionedStopExtensionEvaluationPath =
                    value;
            }
            else if (SplitOptionWithValue(
                         arg,
                         "--probability-stop-extension-state-analysis",
                         value))
            {
                if (HasControlledStrategyEvaluation(parsed) || value.empty())
                    throw std::invalid_argument(
                        "invalid or duplicate --probability-stop-extension-state-analysis");
                parsed.probabilityStopExtensionStateAnalysisPath = value;
            }
            else if (SplitOptionWithValue(
                         arg,
                         "--probability-stop-extension-path-mechanism",
                         value))
            {
                if (HasControlledStrategyEvaluation(parsed) || value.empty())
                    throw std::invalid_argument(
                        "invalid or duplicate --probability-stop-extension-path-mechanism");
                parsed.probabilityStopExtensionPathMechanismPath = value;
            }
            else if (SplitOptionWithValue(
                         arg, "--phase19c-causal-path-predictability", value))
            {
                if (HasControlledStrategyEvaluation(parsed) || value.empty())
                    throw std::invalid_argument(
                        "invalid or duplicate --phase19c-causal-path-predictability");
                parsed.phase19CCausalPathPredictabilityDirectory = value;
            }
            else if (SplitOptionWithValue(
                         arg, "--phase19b-path-artifact-dir", value))
            {
                if (parsed.phase19BPathArtifactDirectory || value.empty())
                    throw std::invalid_argument(
                        "invalid or duplicate --phase19b-path-artifact-dir");
                parsed.phase19BPathArtifactDirectory = value;
            }
            else if (SplitOptionWithValue(arg, "--core-lr-mult", value))
            {
                parsed.coreLrMult = ParsePositiveDoubleArg("--core-lr-mult", value);
            }
            else if (SplitOptionWithValue(arg, "--head-weight-lr-mult", value))
            {
                parsed.headWeightLrMult = ParsePositiveDoubleArg("--head-weight-lr-mult", value);
            }
            else if (SplitOptionWithValue(arg, "--head-bias-lr-mult", value))
            {
                parsed.headBiasLrMult = ParsePositiveDoubleArg("--head-bias-lr-mult", value);
            }
            else if (SplitOptionWithValue(arg, "--checkpoint-every", value))
            {
                if (parsed.checkpointEvery.has_value())
                    throw std::invalid_argument("--checkpoint-every specified more than once");
                parsed.checkpointEvery = ParseNonNegativeIntArg("--checkpoint-every", value);
            }
            else if (SplitOptionWithValue(arg, "--donchian20-mode", value))
            {
                if (parsed.donchian20Mode.has_value())
                    throw std::invalid_argument("--donchian20-mode specified more than once");
                parsed.donchian20Mode = ParseDonchian20Mode(value);
            }
            else if (SplitOptionWithValue(arg, "--feature-warmup-scope", value))
            {
                if (parsed.featureWarmupScope.has_value())
                    throw std::invalid_argument("--feature-warmup-scope specified more than once");
                parsed.featureWarmupScope = EA::ParseFeatureWarmupScope(value);
            }
            else if (SplitOptionWithValue(arg, "--donchian-lookback", value))
            {
                if (parsed.donchianLookback.has_value())
                    throw std::invalid_argument("--donchian-lookback specified more than once");
                parsed.donchianLookback = ParseDonchianLookback(value);
            }
            else if (SplitOptionWithValue(arg, "--infer-start-after-model-id", value))
            {
                if (parsed.inferStartAfterModelId.has_value())
                    throw std::invalid_argument("--infer-start-after-model-id specified more than once");
                parsed.inferStartAfterModelId = ParseModelIdArg(value);
            }
            else if (SplitOptionWithValue(arg, "--scheduler-experiment-id", value))
            {
                if (parsed.schedulerExperimentId.has_value())
                    throw std::invalid_argument("--scheduler-experiment-id specified more than once");
                parsed.schedulerExperimentId = ParseModelIdArg(value);
            }
            else if (SplitOptionWithValue(arg, "--scheduler-checkpoint-eval-id", value))
            {
                if (parsed.schedulerCheckpointEvalId.has_value())
                    throw std::invalid_argument("--scheduler-checkpoint-eval-id specified more than once");
                parsed.schedulerCheckpointEvalId = ParseModelIdArg(value);
            }
            else if (SplitOptionWithValue(
                         arg, "--scheduler-worker-attempt-id", value))
            {
                if (parsed.schedulerWorkerAttemptId.has_value())
                    throw std::invalid_argument(
                        "--scheduler-worker-attempt-id specified more than once");
                parsed.schedulerWorkerAttemptId =
                    ParseModelIdArg(value);
            }
            else if (SplitOptionWithValue(
                         arg, "--training-objective", value))
            {
                if (parsed.trainingObjective.has_value())
                    throw std::invalid_argument(
                        "--training-objective specified more than once");
                parsed.trainingObjective =
                    EA::TrainingObjective::ParseCliSelection(value);
            }
            else if (SplitOptionWithValue(arg, "--log-level", value))
            {
                if (parsed.logLevel.has_value())
                    throw std::invalid_argument("--log-level specified more than once");
                parsed.logLevel = ParseRuntimeLogLevel(value);
            }
            else if (SplitOptionWithValue(arg, "--lstm-profile-output", value))
            {
                if (parsed.lstmProfileOutputPath.has_value())
                    throw std::invalid_argument("--lstm-profile-output specified more than once");
                if (value.empty())
                    throw std::invalid_argument("--lstm-profile-output requires a non-empty path");
                parsed.lstmProfileOutputPath = value;
            }
            else if (SplitOptionWithValue(arg, "--symbol", value))
            {
                if (parsed.symbol.has_value())
                    throw std::invalid_argument("--symbol specified more than once");
                if (value.empty())
                    throw std::invalid_argument("--symbol requires a non-empty table_name");
                parsed.symbol = EA::CanonicalSymbol::Normalize(value);
            }
            else if (SplitOptionWithValue(arg, "--target-epochs", value))
            {
                if (parsed.targetEpochs.has_value())
                    throw std::invalid_argument("--target-epochs specified more than once");
                parsed.targetEpochs = ParsePositiveIntArg("--target-epochs", value);
            }
            else if (SplitOptionWithValue(arg, "--new-model-name", value))
            {
                if (parsed.newModelName.has_value())
                    throw std::invalid_argument("--new-model-name specified more than once");
                if (value.empty())
                    throw std::invalid_argument("--new-model-name requires a non-empty value");
                parsed.newModelName = value;
            }
            else if (arg.rfind("--", 0) == 0)
            {
                throw std::invalid_argument("unknown option '" + arg + "'");
            }
            else
            {
                positional.push_back(arg);
            }
        }
    }

    if (parsed.frozenOutcome)
    {
        const bool hasOverride = parsed.inferenceMode.has_value() ||
            parsed.symbol.has_value() || parsed.modelId.has_value() ||
            parsed.resumeModelId.has_value() || parsed.targetEpochs.has_value() ||
            parsed.newModelName.has_value() || parsed.predictionHorizon.has_value() ||
            parsed.thresholdLogret.has_value() || parsed.windowSize.has_value() ||
            parsed.hiddenSize.has_value() || parsed.numLayers.has_value() ||
            parsed.epochs.has_value() || parsed.coreLrMult.has_value() ||
            parsed.headWeightLrMult.has_value() ||
            parsed.headBiasLrMult.has_value() || parsed.checkpointEvery.has_value() ||
            parsed.schedulerExperimentId.has_value() ||
            parsed.schedulerCheckpointEvalId.has_value() ||
            parsed.schedulerWorkerAttemptId.has_value() ||
            parsed.trainingObjective.has_value() || parsed.inferAll ||
            parsed.forceInfer || parsed.evalTrading ||
            parsed.resumeExpandInputWidth || parsed.featureWarmupScope.has_value() ||
            parsed.donchian20Mode.has_value() ||
            parsed.donchianLookback.has_value() ||
            parsed.controlledFixedStopEvaluationPath.has_value() ||
            parsed.probabilityConditionedStopEvaluationPath.has_value() ||
            parsed.probabilityConditionedStopExtensionEvaluationPath.
                has_value() ||
            !positional.empty();
        if (hasOverride)
            throw std::invalid_argument(
                "--run-frozen-model-outcome-inference rejects training, model, "
                "feature, scheduler, and inference semantic overrides");
        parsed.inferenceMode = true;
        parsed.modelId = parsed.frozenOutcome->sourceModelId;
        parsed.fromDate = parsed.frozenOutcome->outcomeStart;
        parsed.toDate = parsed.frozenOutcome->outcomeEnd;
        return parsed;
    }

    if (parsed.trainingObjective.has_value())
    {
        if (!parsed.schedulerExperimentId.has_value())
            throw std::invalid_argument(
                "--training-objective requires --scheduler-experiment-id");
        if (!parsed.inferenceMode.has_value() || *parsed.inferenceMode)
            throw std::invalid_argument(
                "--training-objective is valid only for scheduler-managed training");
    }

    if (parsed.resumeModelId.has_value())
    {
        if (parsed.freshInitializationSeed)
            throw std::invalid_argument("--fresh-initialization-seed cannot be combined with --resume-model-id");
        if (parsed.schedulerCheckpointEvalId.has_value())
            throw std::invalid_argument("--scheduler-checkpoint-eval-id cannot be combined with --resume-model-id");
        if (parsed.inferAll)
            throw std::invalid_argument("--infer-all cannot be combined with --resume-model-id");
        if (parsed.inferStartAfterModelId.has_value())
            throw std::invalid_argument("--infer-start-after-model-id cannot be combined with --resume-model-id");
        if (positional.size() == 2)
        {
            parsed.fromDate = positional[0];
            parsed.toDate = positional[1];
            parsed.positionalDateRangeSupplied = true;
        }
        else if (!positional.empty())
        {
            throw std::invalid_argument("resume mode accepts no positional date arguments");
        }
        return parsed;
    }

    if (parsed.resumeExpandInputWidth)
        throw std::invalid_argument(
            "--resume-expand-input-width requires --resume-model-id");

    if (parsed.inferStartAfterModelId.has_value() && !parsed.inferAll)
        throw std::invalid_argument("--infer-start-after-model-id requires --infer-all");
    if (parsed.forceInfer && !parsed.inferAll)
        throw std::invalid_argument("--force-infer requires --infer-all");
    if (parsed.schedulerCheckpointEvalId.has_value())
    {
        if (!parsed.inferenceMode.has_value() || !*parsed.inferenceMode)
            throw std::invalid_argument("--scheduler-checkpoint-eval-id requires explicit --infer");
        if (!parsed.modelId.has_value())
            throw std::invalid_argument("--scheduler-checkpoint-eval-id requires --model=<checkpoint_model_id>");
        if (parsed.inferAll)
            throw std::invalid_argument("--scheduler-checkpoint-eval-id cannot be combined with --infer-all");
        if (parsed.schedulerExperimentId.has_value())
            throw std::invalid_argument("--scheduler-checkpoint-eval-id cannot be combined with --scheduler-experiment-id");
    }
    if (parsed.schedulerExperimentId.has_value() &&
        parsed.inferenceMode.has_value() && *parsed.inferenceMode &&
        (!parsed.modelId.has_value() || parsed.inferAll))
    {
        throw std::invalid_argument("scheduler-managed final inference requires one explicit --model");
    }
    if (parsed.inferAll)
    {
        if (!parsed.inferenceMode.has_value() || !*parsed.inferenceMode)
            throw std::invalid_argument("--infer-all requires explicit --infer");
        if (!parsed.symbol.has_value() && !parsed.modelId.has_value())
            throw std::invalid_argument("--infer-all requires --model=<anchor_model_id> or --symbol=<table_name>");
        if (parsed.modelId.has_value() && parsed.inferStartAfterModelId.has_value())
            throw std::invalid_argument("--infer-all cannot combine --model anchor with --infer-start-after-model-id");
    }
    if (parsed.controlledFixedStopEvaluationPath)
    {
        if (!parsed.inferenceMode.has_value() || !*parsed.inferenceMode ||
            !parsed.modelId.has_value() || parsed.inferAll ||
            parsed.schedulerExperimentId || parsed.schedulerCheckpointEvalId ||
            parsed.schedulerWorkerAttemptId || parsed.frozenOutcome)
        {
            throw std::invalid_argument(
                "--controlled-fixed-stop-evaluation requires standalone "
                "--infer with one explicit --model and no scheduler context");
        }
    }
    if (parsed.probabilityConditionedStopEvaluationPath)
    {
        if (!parsed.inferenceMode.has_value() || !*parsed.inferenceMode ||
            !parsed.modelId.has_value() || parsed.inferAll ||
            parsed.schedulerExperimentId || parsed.schedulerCheckpointEvalId ||
            parsed.schedulerWorkerAttemptId || parsed.frozenOutcome)
        {
            throw std::invalid_argument(
                "--probability-conditioned-stop-evaluation requires standalone "
                "--infer with one explicit --model and no scheduler context");
        }
    }
    if (parsed.probabilityConditionedStopExtensionEvaluationPath)
    {
        EA::StrategyEvaluation::
            ValidateControlledOneSidedStopExtensionInvocation({
                parsed.inferenceMode.has_value() && *parsed.inferenceMode,
                parsed.modelId.has_value(),
                parsed.inferAll,
                parsed.schedulerExperimentId.has_value(),
                parsed.schedulerCheckpointEvalId.has_value(),
                parsed.schedulerWorkerAttemptId.has_value(),
                parsed.frozenOutcome.has_value()});
    }
    if (parsed.probabilityStopExtensionStateAnalysisPath)
    {
        EA::StrategyEvaluation::ValidatePhase19Invocation({
            parsed.inferenceMode.has_value() && *parsed.inferenceMode,
            parsed.modelId.has_value(),
            parsed.inferAll,
            parsed.schedulerExperimentId.has_value(),
            parsed.schedulerCheckpointEvalId.has_value(),
            parsed.schedulerWorkerAttemptId.has_value(),
            parsed.frozenOutcome.has_value()});
    }
    if (parsed.probabilityStopExtensionPathMechanismPath)
    {
        EA::StrategyEvaluation::ValidatePhase19BInvocation({
            parsed.inferenceMode.has_value() && *parsed.inferenceMode,
            parsed.modelId.has_value(),
            parsed.inferAll,
            parsed.schedulerExperimentId.has_value(),
            parsed.schedulerCheckpointEvalId.has_value(),
            parsed.schedulerWorkerAttemptId.has_value(),
            parsed.frozenOutcome.has_value()});
    }
    if (parsed.phase19CCausalPathPredictabilityDirectory)
    {
        EA::StrategyEvaluation::ValidatePhase19CInvocation(
            parsed.inferenceMode.has_value() && *parsed.inferenceMode,
            parsed.modelId.has_value(), parsed.inferAll,
            parsed.schedulerExperimentId.has_value() ||
                parsed.schedulerCheckpointEvalId.has_value() ||
                parsed.schedulerWorkerAttemptId.has_value() ||
                parsed.frozenOutcome.has_value(),
            parsed.phase19BPathArtifactDirectory.has_value());
    }
    else if (parsed.phase19BPathArtifactDirectory)
        throw std::invalid_argument(
            "--phase19b-path-artifact-dir requires --phase19c-causal-path-predictability");

    if (positional.size() != 2)
        throw std::invalid_argument("expected arguments: [--train|--infer] [--infer-all] [--force-infer] [--infer-start-after-model-id <model_id>] [--eval-trading] [--controlled-fixed-stop-evaluation=<artifact_path>] [--probability-conditioned-stop-evaluation=<artifact_path>] [--probability-conditioned-stop-extension-evaluation=<artifact_path>] [--probability-stop-extension-state-analysis=<artifact_path>] [--probability-stop-extension-path-mechanism=<artifact_path>] [--phase19c-causal-path-predictability=<output_dir> --phase19b-path-artifact-dir=<dir>] [--log-level quiet|summary|diagnostic] [--lstm-profile-hotspots] [--lstm-profile-output=<path>] [--resume-model-id=<model_id>] [--resume-expand-input-width] [--fresh-initialization-seed=<positive_uint32>] [--target-epochs=<absolute_final_epoch>] [--new-model-name=<name>] [--checkpoint-every <N>] [--symbol=<table_name>] [--model=<model_id>] [--prediction-horizon=<int>] [--threshold=<double>] [--window-size=<int>] [--hidden-size=<int>] [--num-layers=<int>] [--epochs=<int>] [--core-lr-mult=<float>] [--head-weight-lr-mult=<float>] [--head-bias-lr-mult=<float>] <fromDate> <toDate>; preferred inference: --infer --model=<model_id> <fromDate> <toDate>; preferred Phase 19C extraction: --infer --model=<model_id> --phase19c-causal-path-predictability=<output_dir> --phase19b-path-artifact-dir=<dir> <fromDate> <toDate>; preferred infer-all: --infer --infer-all --model=<anchor_model_id> <fromDate> <toDate>");

    parsed.fromDate = positional[0];
    parsed.toDate = positional[1];
    return parsed;
}


} // namespace EA
