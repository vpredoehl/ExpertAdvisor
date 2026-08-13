#include "ExperimentRecommendationConversionExecutionService.hpp"

#include "ExperimentRecommendationConversionExecutionRepository.hpp"
#include "ExperimentRecommendationService.hpp"

#include <ostream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

void RequireSchema(pqxx::connection& connection)
{
    if (!RecommendationConversionExecutionSchemaExists(connection))
        throw std::runtime_error(
            "conversion execution schema required; run ./migrate_lstm_db.sh");
}

void PrintExecution(
    std::ostream& output,
    const char* event,
    const PersistedRecommendationConversionExecution& execution,
    bool createdByCommand)
{
    output << event
           << ",conversion_execution_id=" << execution.executionId
           << ",proposal_id=" << execution.proposalId
           << ",review_decision_id=" << execution.reviewDecisionId
           << ",review_decision="
           << RecommendationConversionProposalReviewDecisionText(
                  execution.authorizationDecision)
           << ",experiment_id=" << execution.experimentId
           << ",execution_contract_version="
           << execution.executionContractVersion
           << ",execution_identity_hash="
           << RecommendationMachineText(execution.executionIdentityHash)
           << ",created_at=" << RecommendationMachineText(execution.createdAt)
           << ",experiment_status=paused,experiment_exists=true,"
              "experiment_created="
           << (createdByCommand ? "true" : "false")
           << ",experiment_queued=false,scheduler_modified=false\n";
}

} // namespace

int RunExecuteApprovedRecommendationConversionProposalCommand(
    const std::string& connectionString,
    long long proposalId,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        pqxx::connection connection{connectionString};
        RequireSchema(connection);
        const auto result = ExecuteApprovedRecommendationConversionProposal(
            connection, proposalId);
        if (result.outcome ==
            RecommendationConversionExecutionOutcome::proposalNotFound)
        {
            errors << "CONVERSION_PROPOSAL_NOT_FOUND,proposal_id=" << proposalId
                   << ",experiment_created=false,experiment_queued=false,"
                      "scheduler_modified=false\n";
            return 1;
        }
        if (result.outcome ==
                RecommendationConversionExecutionOutcome::pendingReview ||
            result.outcome ==
                RecommendationConversionExecutionOutcome::rejected)
        {
            errors << "CONVERSION_PROPOSAL_NOT_APPROVED,proposal_id="
                   << proposalId << ",review_disposition="
                   << RecommendationConversionExecutionOutcomeText(
                          result.outcome)
                   << ",experiment_created=false,experiment_queued=false,"
                      "scheduler_modified=false\n";
            return 2;
        }
        const bool created = result.outcome ==
            RecommendationConversionExecutionOutcome::created;
        PrintExecution(
            output,
            created ? "CONVERSION_PROPOSAL_EXPERIMENT_CREATED"
                    : "CONVERSION_PROPOSAL_EXPERIMENT_ALREADY_CREATED",
            *result.execution,
            created);
        output << "Conversion proposal " << proposalId
               << (created ? " created" : " already created")
               << " paused experiment " << result.execution->experimentId
               << ".\nThe experiment was not queued or started.\n";
        return 0;
    }
    catch (const std::invalid_argument& error)
    {
        errors << "CONVERSION_PROPOSAL_EXECUTION_INVALID,proposal_id="
               << proposalId << ",error="
               << RecommendationMachineText(error.what())
               << ",experiment_created=false,experiment_queued=false,"
                  "scheduler_modified=false\n";
        return 1;
    }
    catch (const std::exception& error)
    {
        errors << "CONVERSION_PROPOSAL_EXECUTION_FAILED,proposal_id="
               << proposalId << ",error="
               << RecommendationMachineText(error.what())
               << ",experiment_created=false,experiment_queued=false,"
                  "scheduler_modified=false\n";
        return 1;
    }
}

int RunRecommendationConversionExecutionStatusCommand(
    const std::string& connectionString,
    long long proposalId,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    RequireSchema(connection);
    const auto execution = FindRecommendationConversionExecutionByProposal(
        connection, proposalId);
    if (!execution)
    {
        output << "CONVERSION_PROPOSAL_EXECUTION_NOT_FOUND,proposal_id="
               << proposalId
               << ",experiment_created=false,experiment_queued=false,"
                  "scheduler_modified=false\n";
        return 1;
    }
    PrintExecution(
        output, "CONVERSION_PROPOSAL_EXECUTION", *execution, false);
    return 0;
}

} // namespace EA::ExperimentRecommendation
