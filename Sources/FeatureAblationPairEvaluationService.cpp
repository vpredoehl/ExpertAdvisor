#include "FeatureAblationPairEvaluationService.hpp"

#include "FeatureAblationPairEvaluationRepository.hpp"
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

void PrintMetric(std::ostringstream& output,
                 std::string_view name,
                 const MetricDelta& metric)
{
    output << "FEATURE_ABLATION_PAIR_DELTA"
           << ",metric=" << name
           << ",control=" << OptionalNumber(metric.control)
           << ",treatment=" << OptionalNumber(metric.treatment)
           << ",treatment_minus_control="
           << OptionalNumber(metric.treatmentMinusControl) << '\n';
}

void PrintArm(std::ostringstream& output,
              std::string_view role,
              const ArmEvidence& arm)
{
    const auto& shared = arm.authoritative;
    const auto& configuration = shared.configuration;
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
           << (shared.finalModelId ? std::to_string(configuration.inputWidth)
                                   : "NULL")
           << ",semantic_layout_version="
           << (shared.finalModelId
                   ? std::to_string(configuration.modelInputLayoutVersion)
                   : "NULL") << '\n';

    output << "FEATURE_ABLATION_PAIR_INFERENCE"
           << ",role=" << role
           << ",inference_result_id="
           << OptionalId(arm.exactFinalInferenceResultId)
           << ",analysis_id="
           << (shared.classification
                   ? std::to_string(shared.classification->analysisId)
                   : "NULL")
           << ",scope="
           << (shared.classification
                   ? shared.classification->inferenceScope : "NULL")
           << ",status="
           << (shared.classification
                   ? shared.classification->status : "NULL")
           << ",prediction_count="
           << (shared.profitability
                   ? std::to_string(shared.profitability->predictionCount)
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
           << ",leader_score="
           << (shared.classification
                   ? OptionalNumber(shared.classification->leaderScore)
                   : "NULL") << '\n';

    output << "FEATURE_ABLATION_PAIR_PROFITABILITY"
           << ",role=" << role
           << ",observation_id="
           << (shared.profitability
                   ? std::to_string(shared.profitability->observationId)
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
                                   const ArmEvidence& treatment,
                                   const ComparisonResult& result)
{
    std::ostringstream output;
    output << "FEATURE_ABLATION_PAIR_COMPARISON"
           << ",version=1"
           << ",control_experiment_id="
           << control.authoritative.configuration.experimentId
           << ",treatment_experiment_id="
           << treatment.authoritative.configuration.experimentId
           << ",read_only=true\n";
    output << "FEATURE_ABLATION_PAIR_IDENTITY"
           << ",ablated_features="
           << (result.canonicalAblatedFeatureSet.empty()
                   ? "NULL" : result.canonicalAblatedFeatureSet)
           << ",ablation_identity_hash="
           << (result.ablationIdentityHash.empty()
                   ? "NULL" : result.ablationIdentityHash) << '\n';
    PrintArm(output, "control", control);
    PrintArm(output, "treatment", treatment);
    PrintMetric(output, "prediction_count", result.predictionCount);
    PrintMetric(output, "actionable_count", result.actionableCount);
    PrintMetric(output, "aggregate_terminal_horizon_log_return_sum",
                result.aggregateProfitability);
    PrintMetric(output,
                "average_terminal_horizon_log_return_per_actionable_prediction",
                result.averageProfitability);
    PrintMetric(output, "inference_accuracy", result.inferenceAccuracy);
    PrintMetric(output, "accept_accuracy", result.acceptAccuracy);
    PrintMetric(output, "accept_rate", result.acceptRate);
    PrintMetric(output, "neutral_proportion", result.neutralProportion);
    PrintMetric(output, "leader_score", result.leaderScore);
    output << "FEATURE_ABLATION_PAIR_RESULT"
           << ",disposition=" << DispositionText(result.disposition)
           << ",invalid_reasons=" << Reasons(result.invalidReasons)
           << ",incomplete_reasons=" << Reasons(result.incompleteReasons)
           << ",evaluation_identity_hash="
           << EvaluationIdentityHash(control, treatment, result)
           << ",exit_code=" << ExitCode(result.disposition)
           << ",software_success=true\n";
    return output.str();
}

int RunComparisonCommand(const std::string& connectionString,
                         const ComparisonCommand& command,
                         std::ostream& output,
                         std::ostream& errors)
{
    try
    {
        pqxx::connection connection{connectionString};
        pqxx::read_transaction transaction{connection};
        transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
        const ArmEvidence control = LoadAuthoritativeArmEvidence(
            transaction, command.experimentIds.first);
        const ArmEvidence treatment = LoadAuthoritativeArmEvidence(
            transaction, command.experimentIds.second);
        const ComparisonResult result = Compare(control, treatment);
        output << RenderComparisonOutput(control, treatment, result);
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
               << ",control_experiment_id=" << command.experimentIds.first
               << ",treatment_experiment_id=" << command.experimentIds.second
               << ",disposition=" << DispositionText(disposition)
               << ",reason=" << MachineText(error.reason())
               << ",exit_code=" << ExitCode(disposition)
               << ",read_only=true\n";
        return ExitCode(disposition);
    }
    catch (const std::invalid_argument& error)
    {
        errors << "FEATURE_ABLATION_PAIR_LOAD_FAILED"
               << ",control_experiment_id=" << command.experimentIds.first
               << ",treatment_experiment_id=" << command.experimentIds.second
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
               << ",control_experiment_id=" << command.experimentIds.first
               << ",treatment_experiment_id=" << command.experimentIds.second
               << ",disposition=incompatible_configuration"
               << ",reason=" << MachineText(error.what())
               << ",exit_code=3,read_only=true\n";
        return 3;
    }
}

} // namespace EA::FeatureAblationPairEvaluation
