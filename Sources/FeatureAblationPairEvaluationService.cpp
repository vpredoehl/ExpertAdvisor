#include "FeatureAblationPairEvaluationService.hpp"

#include "FeatureAblationPairEvaluationRepository.hpp"
#include "FeatureAblation.hpp"
#include "PairedTrainingObjectiveEvaluationRepository.hpp"
#include "TrainingObjective.hpp"

#include <iomanip>
#include <optional>
#include <sstream>
#include <string_view>

#include <pqxx/pqxx>

namespace EA::FeatureAblationPairEvaluation
{
namespace
{

std::string MachineText(std::string value)
{
    for (char& character : value)
    {
        const bool safe =
            (character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') ||
            character == '_' || character == '-' || character == '.';
        if (!safe) character = '_';
    }
    return value.empty() ? "EMPTY" : value;
}

template <typename Value>
std::string OptionalNumber(const std::optional<Value>& value)
{
    if (!value) return "NULL";
    std::ostringstream output;
    output << std::setprecision(17) << *value;
    return output.str();
}

std::string Number(double value)
{
    std::ostringstream output;
    output << std::setprecision(17) << value;
    return output.str();
}

std::string OptionalId(const std::optional<long long>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

std::string Reasons(const std::vector<std::string>& values)
{
    if (values.empty()) return "NONE";
    std::ostringstream output;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index != 0) output << '|';
        output << values[index];
    }
    return output.str();
}

std::string ObjectiveId(const std::string& canonical)
{
    constexpr std::string_view key = "objective_id=";
    const std::size_t start = canonical.find(key);
    if (start == std::string::npos) return "UNKNOWN";
    const std::size_t valueStart = start + key.size();
    const std::size_t end = canonical.find(';', valueStart);
    return MachineText(canonical.substr(valueStart, end - valueStart));
}

void PrintMetric(std::ostringstream& output,
                 std::string_view name,
                 const MetricDelta& metric)
{
    output << "FEATURE_ABLATION_PAIR_DELTA"
           << ",metric=" << name
           << ",control=" << OptionalNumber(metric.control)
           << ",ablation=" << OptionalNumber(metric.ablation)
           << ",control_minus_ablation="
           << OptionalNumber(metric.controlMinusAblation) << '\n';
}

std::optional<std::uint64_t> PredictionCount(
    const SharedEvidence::ClassificationEvidence& value)
{
    if (!value.predictedDownCount || !value.predictedNeutralCount ||
        !value.predictedUpCount)
        return std::nullopt;
    return *value.predictedDownCount + *value.predictedNeutralCount +
        *value.predictedUpCount;
}

void PrintArm(std::ostringstream& output,
              std::string_view role,
              const ArmEvidence& arm)
{
    const auto& shared = arm.authoritative;
    const auto& configuration = shared.configuration;
    const std::optional<int> reportedWidth =
        arm.extended.configuredModelInputWidth
            ? arm.extended.configuredModelInputWidth
            : (shared.finalModelId
                   ? std::optional<int>{configuration.inputWidth}
                   : std::nullopt);
    const std::optional<int> reportedLayout =
        arm.extended.configuredModelInputLayoutVersion
            ? arm.extended.configuredModelInputLayoutVersion
            : (shared.finalModelId
                   ? std::optional<int>{configuration.modelInputLayoutVersion}
                   : std::nullopt);
    output << "FEATURE_ABLATION_PAIR_ARM"
           << ",role=" << role
           << ",experiment_id=" << configuration.experimentId
           << ",experiment_status=" << MachineText(shared.experimentStatus)
           << ",experiment_phase=" << MachineText(shared.experimentPhase)
           << ",feature_ablation_mask="
           << (configuration.featureAblationMask.empty()
                   ? "EMPTY" : configuration.featureAblationMask)
           << ",final_model_id=" << OptionalId(shared.finalModelId)
           << ",resume_model_id="
           << OptionalId(configuration.resumeModelId)
           << ",resume_checkpoint_epoch="
           << (arm.resumeCheckpointProvenance &&
                       arm.resumeCheckpointProvenance->checkpointEpoch
                   ? std::to_string(
                         *arm.resumeCheckpointProvenance->checkpointEpoch)
                   : "NULL")
           << ",model_input_width="
           << (reportedWidth ? std::to_string(*reportedWidth) : "NULL")
           << ",model_input_semantic_layout_version="
           << (reportedLayout ? std::to_string(*reportedLayout) : "NULL")
           << ",economic_calendar_snapshot_id="
           << OptionalId(arm.extended.economicCalendarSnapshotId)
           << ",economic_calendar_snapshot_hash="
           << arm.extended.economicCalendarSnapshotHash.value_or("NULL")
           << ",symbol=" << MachineText(configuration.symbol)
           << ",prediction_horizon=" << configuration.predictionHorizon
           << ",train_start=" << MachineText(configuration.trainStart)
           << ",train_end=" << MachineText(configuration.trainEnd)
           << ",configured_inference_start="
           << MachineText(configuration.inferenceStart)
           << ",configured_inference_end="
           << MachineText(configuration.inferenceEnd)
           << ",target_epochs=" << configuration.targetEpochs
           << ",training_objective_id="
           << ObjectiveId(configuration.experimentObjective.canonical)
           << ",training_objective_hash="
           << configuration.experimentObjective.hash
           << ",scheduler_priority="
           << (arm.operational.schedulerPriority.empty()
                   ? "NULL" : MachineText(arm.operational.schedulerPriority))
           << ",worker_pid="
           << (arm.operational.workerPid
                   ? std::to_string(*arm.operational.workerPid) : "NULL")
           << ",operational_metadata_in_scientific_identity=false\n";

    output << "FEATURE_ABLATION_PAIR_INFERENCE"
           << ",role=" << role
           << ",inference_result_id="
           << OptionalId(arm.exactFinalInferenceResultId)
           << ",analysis_id="
           << (shared.classification
                   ? std::to_string(shared.classification->analysisId)
                   : "NULL")
           << ",final_epoch="
           << (shared.finalModelId
                   ? std::to_string(configuration.targetEpochs)
                   : "NULL")
           << ",inference_start="
           << (shared.classification
                   ? shared.classification->inferenceStart : "NULL")
           << ",inference_end="
           << (shared.classification
                   ? shared.classification->inferenceEnd : "NULL")
           << ",scope="
           << (shared.classification
                   ? shared.classification->inferenceScope : "NULL")
           << ",status="
           << (shared.classification
                   ? shared.classification->status : "NULL")
           << ",prediction_count="
           << (shared.classification &&
                       PredictionCount(*shared.classification)
                   ? std::to_string(
                         *PredictionCount(*shared.classification))
                   : "NULL")
           << ",predicted_down_count="
           << (shared.classification &&
                       shared.classification->predictedDownCount
                   ? std::to_string(
                         *shared.classification->predictedDownCount) : "NULL")
           << ",predicted_neutral_count="
           << (shared.classification &&
                       shared.classification->predictedNeutralCount
                   ? std::to_string(
                         *shared.classification->predictedNeutralCount) : "NULL")
           << ",predicted_up_count="
           << (shared.classification &&
                       shared.classification->predictedUpCount
                   ? std::to_string(
                         *shared.classification->predictedUpCount) : "NULL")
           << ",accepted_prediction_count="
           << (shared.classification &&
                       shared.classification->acceptedPredictionCount
                   ? std::to_string(
                         *shared.classification->acceptedPredictionCount)
                   : "NULL")
           << ",inference_accuracy="
           << (shared.classification
                   ? OptionalNumber(shared.classification->inferenceAccuracy)
                   : "NULL")
           << ",accept_accuracy="
           << (shared.classification
                   ? OptionalNumber(shared.classification->acceptAccuracy)
                   : "NULL")
           << ",accept_rate="
           << (shared.classification
                   ? OptionalNumber(shared.classification->acceptRate)
                   : "NULL")
           << ",neutral_proportion="
           << (shared.classification
                   ? OptionalNumber(
                         shared.classification->predictedNeutralProportion)
                   : "NULL")
           << ",down_proportion="
           << (shared.classification
                   ? OptionalNumber(
                         shared.classification->predictedDownProportion)
                   : "NULL")
           << ",up_proportion="
           << (shared.classification
                   ? OptionalNumber(
                         shared.classification->predictedUpProportion)
                   : "NULL")
           << ",leader_score="
           << (shared.classification
                   ? OptionalNumber(shared.classification->leaderScore)
                   : "NULL") << '\n';

    output << "FEATURE_ABLATION_PAIR_PROFITABILITY"
           << ",role=" << role
           << ",evidence_available="
           << (shared.profitability ? "true" : "false")
           << ",observation_id="
           << (shared.profitability
                   ? std::to_string(shared.profitability->observationId)
                   : "NULL")
           << ",model_id="
           << (shared.profitability
                   ? std::to_string(shared.profitability->modelId)
                   : "NULL")
           << ",inference_result_id="
           << (shared.profitability
                   ? std::to_string(shared.profitability->inferenceResultId)
                   : "NULL")
           << ",checkpoint_eval_id="
           << (shared.profitability &&
                       shared.profitability->checkpointEvalId
                   ? std::to_string(
                         *shared.profitability->checkpointEvalId)
                   : "NULL")
           << ",scope="
           << (shared.profitability
                   ? shared.profitability->inferenceScope : "NULL")
           << ",actionable_count="
           << (shared.profitability
                   ? std::to_string(shared.profitability->actionableCount)
                   : "NULL")
           << ",aggregate_terminal_horizon_log_return_sum="
           << (shared.profitability
                   ? Number(shared.profitability
                                ->aggregateTerminalHorizonLogReturnSum)
                   : "NULL")
           << ",average_terminal_horizon_log_return_per_actionable_prediction="
           << (shared.profitability
                   ? OptionalNumber(shared.profitability
                         ->averageTerminalHorizonLogReturnPerActionablePrediction)
                   : "NULL") << '\n';
}

} // namespace

std::string RenderComparisonOutput(const ArmEvidence& control,
                                   const ArmEvidence& ablation,
                                   const ComparisonResult& result)
{
    std::ostringstream output;
    output << "FEATURE_ABLATION_PAIR_COMPARISON"
           << ",version=2"
           << ",control_experiment_id="
           << control.authoritative.configuration.experimentId
           << ",ablation_experiment_id="
           << ablation.authoritative.configuration.experimentId
           << ",read_only=true\n";
    output << "FEATURE_ABLATION_PAIR_IDENTITY"
           << ",expected_ablation_mask="
           << (result.canonicalAblatedFeatureSet.empty()
                   ? "NULL" : result.canonicalAblatedFeatureSet)
           << ",ablation_identity_hash="
           << (result.ablationIdentityHash.empty()
                   ? "NULL" : result.ablationIdentityHash) << '\n';
    PrintArm(output, "control", control);
    PrintArm(output, "ablation", ablation);
    PrintMetric(output, "prediction_count", result.predictionCount);
    PrintMetric(output, "predicted_down_count", result.predictedDownCount);
    PrintMetric(output, "predicted_neutral_count", result.predictedNeutralCount);
    PrintMetric(output, "predicted_up_count", result.predictedUpCount);
    PrintMetric(output, "accepted_prediction_count",
                result.acceptedPredictionCount);
    PrintMetric(output, "actionable_count", result.actionableCount);
    PrintMetric(output, "aggregate_terminal_horizon_log_return_sum",
                result.aggregateProfitability);
    PrintMetric(output,
                "average_terminal_horizon_log_return_per_actionable_prediction",
                result.averageProfitability);
    PrintMetric(output, "inference_accuracy", result.inferenceAccuracy);
    PrintMetric(output, "accept_accuracy", result.acceptAccuracy);
    PrintMetric(output, "accept_rate", result.acceptRate);
    PrintMetric(output, "down_proportion", result.downProportion);
    PrintMetric(output, "neutral_proportion", result.neutralProportion);
    PrintMetric(output, "up_proportion", result.upProportion);
    PrintMetric(output, "leader_score", result.leaderScore);
    output << "FEATURE_ABLATION_PAIR_RESULT"
           << ",disposition=" << DispositionText(result.disposition)
           << ",pair_validity_state="
           << (result.disposition == Disposition::IncompatibleConfiguration ||
                       result.disposition == Disposition::AmbiguousFinalInference ||
                       result.disposition == Disposition::InvalidAblationPair
                   ? "invalid" : "valid")
           << ",readiness_state="
           << (result.disposition == Disposition::ComparableComplete
                   ? "ready"
                   : result.disposition ==
                             Disposition::ProfitabilityEvidenceUnavailable
                         ? "final_inference_ready_profitability_unavailable"
                         : "not_ready")
           << ",delta_sign_convention=control_minus_ablation"
           << ",invalid_reasons=" << Reasons(result.invalidReasons)
           << ",incomplete_reasons=" << Reasons(result.incompleteReasons)
           << ",evaluation_identity_hash="
           << EvaluationIdentityHash(control, ablation, result)
           << ",exit_code=" << ExitCode(result.disposition)
           << ",software_success=true\n";
    return output.str();
}

int RunComparisonCommand(const std::string& connectionString,
                         const ComparisonCommand& command,
                         std::ostream& output,
                         std::ostream& errors)
{
    const bool legacyCompatibility = !command.expectedAblationMask;
    const long long controlExperimentId = legacyCompatibility
        ? command.experimentIds.second : command.experimentIds.first;
    const long long ablationExperimentId = legacyCompatibility
        ? command.experimentIds.first : command.experimentIds.second;
    try
    {
        const std::string expectedMask = FeatureAblationMask::Parse(
            command.expectedAblationMask.value_or(
                std::string{kEconomicEventConsensusAblationMaskText}))
                .CanonicalText();
        if (command.expectedAblationMask && expectedMask.empty())
            throw std::invalid_argument(
                "--expected-ablation-mask must not be empty");
        pqxx::connection connection{connectionString};
        pqxx::read_transaction transaction{connection};
        transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
        const ArmEvidence first = LoadAuthoritativeArmEvidence(
            transaction, command.experimentIds.first);
        const ArmEvidence second = LoadAuthoritativeArmEvidence(
            transaction, command.experimentIds.second);
        const ArmEvidence& control = legacyCompatibility ? second : first;
        const ArmEvidence& ablation = legacyCompatibility ? first : second;
        const ComparisonResult result = Compare(
            control, ablation, expectedMask);
        output << RenderComparisonOutput(control, ablation, result);
        return ExitCode(result.disposition);
    }
    catch (const PairedTrainingObjectiveEvaluation::EvidenceLoadError& error)
    {
        const bool ambiguous = error.kind() ==
            PairedTrainingObjectiveEvaluation::EvidenceLoadErrorKind::
                AmbiguousEvidence;
        const Disposition disposition = ambiguous
            ? Disposition::AmbiguousFinalInference
            : Disposition::IncompatibleConfiguration;
        errors << "FEATURE_ABLATION_PAIR_LOAD_FAILED"
               << ",control_experiment_id=" << controlExperimentId
               << ",ablation_experiment_id=" << ablationExperimentId
               << ",disposition=" << DispositionText(disposition)
               << ",reason=" << MachineText(error.reason())
               << ",exit_code=" << ExitCode(disposition)
               << ",read_only=true\n";
        return ExitCode(disposition);
    }
    catch (const std::invalid_argument& error)
    {
        errors << "FEATURE_ABLATION_PAIR_LOAD_FAILED"
               << ",control_experiment_id=" << controlExperimentId
               << ",ablation_experiment_id=" << ablationExperimentId
               << ",disposition=incompatible_configuration"
               << ",reason=" << MachineText(error.what())
               << ",exit_code=3,read_only=true\n";
        return 3;
    }
    catch (const pqxx::failure&)
    {
        throw;
    }
    catch (const std::runtime_error& error)
    {
        errors << "FEATURE_ABLATION_PAIR_LOAD_FAILED"
               << ",control_experiment_id=" << controlExperimentId
               << ",ablation_experiment_id=" << ablationExperimentId
               << ",disposition=incompatible_configuration"
               << ",reason=" << MachineText(error.what())
               << ",exit_code=3,read_only=true\n";
        return 3;
    }
}

} // namespace EA::FeatureAblationPairEvaluation
