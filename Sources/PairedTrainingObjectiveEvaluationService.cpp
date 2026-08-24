#include "PairedTrainingObjectiveEvaluationService.hpp"

#include "PairedTrainingObjectiveEvaluationRepository.hpp"
#include "TrainingObjective.hpp"

#include <iomanip>
#include <optional>
#include <sstream>
#include <string_view>

#include <pqxx/pqxx>

namespace EA::PairedTrainingObjectiveEvaluation
{
namespace
{

std::string Boolean(bool value)
{
    return value ? "true" : "false";
}

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
    std::ostringstream out;
    out << std::setprecision(17) << *value;
    return out.str();
}

std::string Number(double value)
{
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

std::string OptionalId(const std::optional<long long>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

std::string Reasons(const std::vector<std::string>& values)
{
    if (values.empty()) return "NONE";
    std::ostringstream out;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index != 0) out << '|';
        out << values[index];
    }
    return out.str();
}

std::string ObjectiveIdentifier(const ObjectiveProvenance& provenance)
{
    try
    {
        return TrainingObjective::ParseSupportedCanonicalText(
            provenance.canonical).objectiveIdentifier;
    }
    catch (const std::exception&)
    {
        return "INVALID";
    }
}

std::string PrimaryMetricText(ProfitabilityPrimaryMetric metric)
{
    switch (metric)
    {
        case ProfitabilityPrimaryMetric::
                AggregateTerminalHorizonLogReturnSum:
            return "aggregate_terminal_horizon_log_return_sum";
        case ProfitabilityPrimaryMetric::
                AverageTerminalHorizonLogReturnPerActionablePrediction:
            return "average_terminal_horizon_log_return_per_actionable_prediction";
    }
    return "INVALID";
}

void PrintMetric(std::ostringstream& output,
                 std::string_view name,
                 const MetricDelta& metric)
{
    output << "TRAINING_OBJECTIVE_PAIR_DELTA"
           << ",metric=" << name
           << ",control=" << OptionalNumber(metric.control)
           << ",treatment=" << OptionalNumber(metric.treatment)
           << ",treatment_minus_control="
           << OptionalNumber(metric.treatmentMinusControl)
           << ",relative_to_absolute_control="
           << OptionalNumber(metric.relativeToAbsoluteControl) << '\n';
}

void PrintArm(std::ostringstream& output,
              std::string_view role,
              const ArmEvidence& arm)
{
    output << "TRAINING_OBJECTIVE_PAIR_OBJECTIVE"
           << ",role=" << role
           << ",experiment_id=" << arm.configuration.experimentId
           << ",objective_id="
           << ObjectiveIdentifier(arm.configuration.experimentObjective)
           << ",objective_hash=" << arm.configuration.experimentObjective.hash
           << ",final_model_id=" << OptionalId(arm.finalModelId) << '\n';

    output << "TRAINING_OBJECTIVE_PAIR_INFERENCE"
           << ",role=" << role
           << ",inference_result_id="
           << (arm.classification
                   ? std::to_string(arm.classification->inferenceResultId)
                   : "NULL")
           << ",model_id="
           << (arm.classification
                   ? std::to_string(arm.classification->modelId)
                   : "NULL")
           << ",scope="
           << (arm.classification ? arm.classification->inferenceScope : "NULL")
           << ",status="
           << (arm.classification ? arm.classification->status : "NULL")
           << '\n';

    output << "TRAINING_OBJECTIVE_PAIR_PROFITABILITY"
           << ",role=" << role
           << ",observation_id="
           << (arm.profitability
                   ? std::to_string(arm.profitability->observationId)
                   : "NULL")
           << ",inference_result_id="
           << (arm.profitability
                   ? std::to_string(arm.profitability->inferenceResultId)
                   : "NULL")
           << ",actionable_count="
           << (arm.profitability
                   ? std::to_string(arm.profitability->actionableCount)
                   : "NULL")
           << ",aggregate_terminal_horizon_log_return_sum="
           << (arm.profitability
                   ? Number(arm.profitability
                                ->aggregateTerminalHorizonLogReturnSum)
                   : "NULL")
           << ",average_terminal_horizon_log_return_per_actionable_prediction="
           << (arm.profitability
                   ? OptionalNumber(arm.profitability
                         ->averageTerminalHorizonLogReturnPerActionablePrediction)
                   : "NULL") << '\n';

    output << "TRAINING_OBJECTIVE_PAIR_CLASSIFICATION"
           << ",role=" << role
           << ",inference_accuracy="
           << (arm.classification
                   ? OptionalNumber(arm.classification->inferenceAccuracy)
                   : "NULL")
           << ",accept_accuracy="
           << (arm.classification
                   ? OptionalNumber(arm.classification->acceptAccuracy)
                   : "NULL")
           << ",accept_rate="
           << (arm.classification
                   ? OptionalNumber(arm.classification->acceptRate)
                   : "NULL")
           << ",neutral_proportion="
           << (arm.classification
                   ? OptionalNumber(
                         arm.classification->predictedNeutralProportion)
                   : "NULL")
           << ",leader_score="
           << (arm.classification
                   ? OptionalNumber(arm.classification->leaderScore)
                   : "NULL") << '\n';
}

} // namespace

std::string RenderComparisonOutput(
    const ArmEvidence& control,
    const ArmEvidence& treatment,
    const MaterialityPolicy& policy,
    const ComparisonResult& result)
{
    std::ostringstream output;
    output << "TRAINING_OBJECTIVE_PAIR_COMPARISON"
           << ",version=1"
           << ",control_experiment_id=" << control.configuration.experimentId
           << ",treatment_experiment_id="
           << treatment.configuration.experimentId
           << ",read_only=true\n";
    output << "TRAINING_OBJECTIVE_PAIR_CONFIGURATION"
           << ",primary_profitability_metric="
           << PrimaryMetricText(policy.primaryProfitabilityMetric)
           << ",minimum_profitability_improvement="
           << Number(policy.minimumProfitabilityImprovement)
           << ",maximum_profitability_worsening="
           << Number(policy.maximumProfitabilityWorsening)
           << ",classification_policy_configured="
           << Boolean(policy.classification.has_value())
           << ",maximum_inference_accuracy_decrease="
           << (policy.classification
                   ? Number(policy.classification
                                ->maximumInferenceAccuracyDecrease)
                   : "NULL")
           << ",maximum_accept_accuracy_decrease="
           << (policy.classification
                   ? Number(policy.classification
                                ->maximumAcceptAccuracyDecrease)
                   : "NULL")
           << ",maximum_accept_rate_decrease="
           << (policy.classification
                   ? Number(policy.classification
                                ->maximumAcceptRateDecrease)
                   : "NULL")
           << ",maximum_leader_score_decrease="
           << (policy.classification
                   ? Number(policy.classification
                                ->maximumLeaderScoreDecrease)
                   : "NULL")
           << ",maximum_neutral_proportion_increase="
           << (policy.classification
                   ? Number(policy.classification
                                ->maximumNeutralProportionIncrease)
                   : "NULL") << '\n';
    PrintArm(output, "control", control);
    PrintArm(output, "treatment", treatment);
    PrintMetric(output, "actionable_count", result.actionableCount);
    PrintMetric(output, "aggregate_profitability", result.aggregateProfitability);
    PrintMetric(output, "average_profitability", result.averageProfitability);
    PrintMetric(output, "inference_accuracy", result.inferenceAccuracy);
    PrintMetric(output, "accept_accuracy", result.acceptAccuracy);
    PrintMetric(output, "accept_rate", result.acceptRate);
    PrintMetric(output, "neutral_proportion",
                result.predictedNeutralProportion);
    PrintMetric(output, "leader_score", result.leaderScore);
    output << "TRAINING_OBJECTIVE_PAIR_RESULT"
           << ",disposition=" << DispositionText(result.disposition)
           << ",invalid_reasons=" << Reasons(result.invalidReasons)
           << ",incomplete_reasons=" << Reasons(result.incompleteReasons)
           << ",interpretation_reasons="
           << Reasons(result.interpretationReasons)
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
        transaction.exec(
            "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
        const ArmEvidence control = LoadAuthoritativeArmEvidence(
            transaction, command.experimentIds.first);
        const ArmEvidence treatment = LoadAuthoritativeArmEvidence(
            transaction, command.experimentIds.second);
        const ComparisonResult result = Compare(
            control, treatment, command.policy);
        output << RenderComparisonOutput(
            control, treatment, command.policy, result);
        return 0;
    }
    catch (const EvidenceLoadError& error)
    {
        errors << "TRAINING_OBJECTIVE_PAIR_LOAD_FAILED"
               << ",control_experiment_id=" << command.experimentIds.first
               << ",treatment_experiment_id=" << command.experimentIds.second
               << ",reason=" << MachineText(error.reason())
               << ",read_only=true\n";
        return 3;
    }
    catch (const std::invalid_argument& error)
    {
        errors << "TRAINING_OBJECTIVE_PAIR_LOAD_FAILED"
               << ",control_experiment_id=" << command.experimentIds.first
               << ",treatment_experiment_id=" << command.experimentIds.second
               << ",reason=" << MachineText(error.what())
               << ",read_only=true\n";
        return 3;
    }
    catch (const pqxx::failure&)
    {
        throw;
    }
    catch (const std::runtime_error& error)
    {
        errors << "TRAINING_OBJECTIVE_PAIR_LOAD_FAILED"
               << ",control_experiment_id=" << command.experimentIds.first
               << ",treatment_experiment_id=" << command.experimentIds.second
               << ",reason=" << MachineText(error.what())
               << ",read_only=true\n";
        return 3;
    }
}

} // namespace EA::PairedTrainingObjectiveEvaluation
