#include "ExperimentRecommendationConversionActivationService.hpp"

#include "ExperimentRecommendationConversionActivationRepository.hpp"
#include "ExperimentRecommendationService.hpp"

#include <ostream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

void RequireSchema(pqxx::connection& connection)
{
    if (!RecommendationConversionActivationSchemaExists(connection))
        throw std::runtime_error(
            "conversion activation schema required; run ./migrate_lstm_db.sh");
}

void PrintActivation(
    std::ostream& output,
    const char* event,
    const char* outcome,
    const PersistedRecommendationConversionActivation& activation)
{
    output << event
           << ",outcome=" << outcome
           << ",activation_id=" << activation.activationId
           << ",conversion_execution_id=" << activation.executionId
           << ",proposal_id=" << activation.proposalId
           << ",review_decision_id=" << activation.reviewDecisionId
           << ",experiment_id=" << activation.experimentId
           << ",previous_status=" << activation.previousStatus
           << ",previous_phase=" << activation.previousPhase
           << ",resulting_status=" << activation.resultingStatus
           << ",resulting_phase=" << activation.resultingPhase
           << ",activation_identity_hash="
           << RecommendationMachineText(activation.activationIdentityHash)
           << ",created_at=" << RecommendationMachineText(activation.createdAt)
           << ",experiment_created=false,worker_started=false,"
              "scheduler_started=false\n";
}

} // namespace

int RunActivateRecommendationConversionExecutionCommand(
    const std::string& connectionString,
    long long executionId,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        pqxx::connection connection{connectionString};
        RequireSchema(connection);
        const auto result = ActivateRecommendationConversionExecution(
            connection, executionId);
        if (result.outcome ==
            RecommendationConversionActivationOutcome::notFound)
        {
            errors << "RECOMMENDATION_CONVERSION_EXECUTION_NOT_FOUND,"
                   << "conversion_execution_id=" << executionId
                   << ",experiment_created=false,worker_started=false,"
                      "scheduler_started=false\n";
            return 1;
        }
        if (result.outcome ==
            RecommendationConversionActivationOutcome::invalidState)
        {
            errors << "RECOMMENDATION_CONVERSION_ACTIVATION_INVALID_STATE,"
                   << "conversion_execution_id=" << executionId
                   << ",reason=" << RecommendationMachineText(result.reason)
                   << ",experiment_created=false,worker_started=false,"
                      "scheduler_started=false\n";
            return 2;
        }
        if (result.outcome ==
            RecommendationConversionActivationOutcome::conflict)
        {
            errors << "RECOMMENDATION_CONVERSION_ACTIVATION_CONFLICT,"
                   << "conversion_execution_id=" << executionId
                   << ",reason=" << RecommendationMachineText(result.reason)
                   << ",experiment_created=false,worker_started=false,"
                      "scheduler_started=false\n";
            return 3;
        }
        const bool activated = result.outcome ==
            RecommendationConversionActivationOutcome::activated;
        PrintActivation(
            output,
            activated ? "RECOMMENDATION_CONVERSION_ACTIVATED"
                      : "RECOMMENDATION_CONVERSION_ALREADY_ACTIVATED",
            activated ? "activated" : "existing_identical",
            *result.activation);
        output << "Experiment " << result.activation->experimentId
               << (activated ? " is now" : " remains")
               << " pending/train and eligible for the existing scheduler.\n"
               << "No worker was started by this command.\n";
        return 0;
    }
    catch (const std::invalid_argument& error)
    {
        errors << "RECOMMENDATION_CONVERSION_ACTIVATION_INVALID,"
               << "conversion_execution_id=" << executionId
               << ",error=" << RecommendationMachineText(error.what())
               << ",experiment_created=false,worker_started=false,"
                  "scheduler_started=false\n";
        return 1;
    }
    catch (const std::exception& error)
    {
        errors << "RECOMMENDATION_CONVERSION_ACTIVATION_FAILED,"
               << "conversion_execution_id=" << executionId
               << ",error=" << RecommendationMachineText(error.what())
               << ",experiment_created=false,worker_started=false,"
                  "scheduler_started=false\n";
        return 1;
    }
}

int RunRecommendationConversionActivationStatusCommand(
    const std::string& connectionString,
    long long activationId,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    RequireSchema(connection);
    const auto activation = FindRecommendationConversionActivation(
        connection, activationId);
    if (!activation)
    {
        output << "RECOMMENDATION_CONVERSION_ACTIVATION_NOT_FOUND,"
               << "activation_id=" << activationId
               << ",experiment_created=false,worker_started=false,"
                  "scheduler_started=false\n";
        return 1;
    }
    PrintActivation(
        output,
        "RECOMMENDATION_CONVERSION_ACTIVATION",
        "activated",
        *activation);
    return 0;
}

} // namespace EA::ExperimentRecommendation
