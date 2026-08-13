#include "ExperimentRecommendationConversionWorkflowService.hpp"

#include "ExperimentRecommendationConversionWorkflowRepository.hpp"
#include "ExperimentRecommendationService.hpp"

#include <sstream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

void RequireSchemas(pqxx::connection& connection)
{
    if (!RecommendationConversionWorkflowSchemasExist(connection))
        throw std::runtime_error(
            "conversion workflow schemas required; run ./migrate_lstm_db.sh");
}

template <typename Value>
std::string OptionalNumber(const std::optional<Value>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

std::string OptionalText(const std::optional<std::string>& value)
{
    return value ? RecommendationMachineText(*value) : "NULL";
}

std::string DiagnosticsText(const std::vector<std::string>& diagnostics)
{
    if (diagnostics.empty()) return "NULL";
    std::ostringstream out;
    for (std::size_t i = 0; i < diagnostics.size(); ++i)
    {
        if (i != 0) out << ':';
        out << diagnostics[i];
    }
    return RecommendationMachineText(out.str());
}

std::string LatestDisposition(
    const RecommendationConversionWorkflowView& view)
{
    if (!view.latestReview) return "pending_review";
    if (view.latestReview->decision == "approve") return "approved";
    if (view.latestReview->decision == "reject") return "rejected";
    return "invalid";
}

void PrintWorkflow(
    std::ostream& output,
    const RecommendationConversionWorkflowView& view)
{
    const auto executionId = view.execution
        ? std::optional<long long>{view.execution->executionId}
        : std::nullopt;
    const auto approvingReviewId = view.execution
        ? std::optional<long long>{view.execution->reviewDecisionId}
        : std::nullopt;
    const auto activationId = view.activation
        ? std::optional<long long>{view.activation->activationId}
        : std::nullopt;
    const auto experimentId = view.execution
        ? std::optional<long long>{view.execution->experimentId}
        : std::nullopt;
    const auto latestReviewId = view.latestReview
        ? std::optional<long long>{view.latestReview->reviewDecisionId}
        : std::nullopt;
    const auto workerPid = view.experiment
        ? view.experiment->workerPid
        : std::optional<int>{};
    const auto currentOperation = view.experiment
        ? view.experiment->currentOperation
        : std::optional<std::string>{};
    const auto currentEpoch = view.experiment
        ? view.experiment->currentEpoch
        : std::optional<int>{};
    output
        << "RECOMMENDATION_CONVERSION_WORKFLOW"
        << ",workflow_state="
        << RecommendationConversionWorkflowStateText(view.derivation.state)
        << ",integrity_status="
        << RecommendationConversionWorkflowIntegrityText(
               view.derivation.integrity)
        << ",proposal_id=" << view.proposalId
        << ",recommendation_id=" << view.recommendationId
        << ",source_experiment_id=" << view.sourceExperimentId
        << ",proposal_created_at="
        << RecommendationMachineText(view.proposalCreatedAt)
        << ",latest_review_decision_id=" << OptionalNumber(latestReviewId)
        << ",latest_review_disposition=" << LatestDisposition(view)
        << ",latest_review_decided_at="
        << OptionalText(view.latestReviewDecidedAt)
        << ",approving_review_decision_id="
        << OptionalNumber(approvingReviewId)
        << ",execution_id=" << OptionalNumber(executionId)
        << ",execution_created_at=" << OptionalText(view.executionCreatedAt)
        << ",activation_id=" << OptionalNumber(activationId)
        << ",activation_created_at=" << OptionalText(view.activationCreatedAt)
        << ",experiment_id=" << OptionalNumber(experimentId)
        << ",experiment_status="
        << (view.experiment
                ? RecommendationMachineText(view.experiment->status)
                : "NULL")
        << ",experiment_phase="
        << (view.experiment
                ? RecommendationMachineText(view.experiment->phase)
                : "NULL")
        << ",worker_pid=" << OptionalNumber(workerPid)
        << ",current_operation=" << OptionalText(currentOperation)
        << ",current_epoch=" << OptionalNumber(currentEpoch)
        << ",experiment_updated_at=" << OptionalText(view.experimentUpdatedAt)
        << ",activation_previous_status="
        << (view.activation
                ? RecommendationMachineText(view.activation->previousStatus)
                : "NULL")
        << ",activation_previous_phase="
        << (view.activation
                ? RecommendationMachineText(view.activation->previousPhase)
                : "NULL")
        << ",activation_resulting_status="
        << (view.activation
                ? RecommendationMachineText(view.activation->resultingStatus)
                : "NULL")
        << ",activation_resulting_phase="
        << (view.activation
                ? RecommendationMachineText(view.activation->resultingPhase)
                : "NULL")
        << ",proposal_identity_hash="
        << RecommendationMachineText(view.proposalIdentityHash)
        << ",execution_identity_hash="
        << (view.execution
                ? RecommendationMachineText(view.execution->identityHash)
                : "NULL")
        << ",activation_identity_hash="
        << (view.activation
                ? RecommendationMachineText(view.activation->identityHash)
                : "NULL")
        << ",diagnostic_count=" << view.derivation.diagnosticCodes.size()
        << ",diagnostic_codes="
        << DiagnosticsText(view.derivation.diagnosticCodes)
        << ",read_only=true,experiment_created=false,experiment_modified=false,"
           "worker_started=false,scheduler_started=false\n";
}

} // namespace

int RunRecommendationConversionWorkflowCommand(
    const std::string& connectionString,
    long long proposalId,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        pqxx::connection connection{connectionString};
        RequireSchemas(connection);
        const auto workflow = FindRecommendationConversionWorkflow(
            connection, proposalId);
        if (!workflow)
        {
            errors << "RECOMMENDATION_CONVERSION_WORKFLOW_NOT_FOUND,proposal_id="
                   << proposalId
                   << ",read_only=true,experiment_modified=false,"
                      "worker_started=false,scheduler_started=false\n";
            return 1;
        }
        PrintWorkflow(output, *workflow);
        output << "Proposal " << workflow->proposalId << " is "
               << RecommendationConversionWorkflowStateText(
                      workflow->derivation.state)
               << "; integrity is "
               << RecommendationConversionWorkflowIntegrityText(
                      workflow->derivation.integrity)
               << ".\nObservation is read-only; no experiment or scheduler "
                  "state was changed.\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        errors << "RECOMMENDATION_CONVERSION_WORKFLOW_FAILED,proposal_id="
               << proposalId << ",error="
               << RecommendationMachineText(error.what())
               << ",read_only=true,experiment_modified=false,"
                  "worker_started=false,scheduler_started=false\n";
        return 2;
    }
}

int RunListRecommendationConversionWorkflowsCommand(
    const std::string& connectionString,
    std::optional<RecommendationConversionWorkflowState> state,
    int limit,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        if (limit <= 0 ||
            limit > kMaximumRecommendationConversionWorkflowListLimit)
            throw std::invalid_argument(
                "recommendation_conversion_workflow_limit_invalid");
        pqxx::connection connection{connectionString};
        RequireSchemas(connection);
        const int candidateLimit = state
            ? kMaximumRecommendationConversionWorkflowListLimit
            : limit;
        const auto workflows = ListRecommendationConversionWorkflows(
            connection, candidateLimit);
        int emitted = 0;
        for (const auto& workflow : workflows)
        {
            if (state && workflow.derivation.state != *state) continue;
            if (emitted == limit) break;
            PrintWorkflow(output, workflow);
            ++emitted;
        }
        output << "RECOMMENDATION_CONVERSION_WORKFLOW_LIST_COMPLETE,count="
               << emitted << ",candidate_count=" << workflows.size()
               << ",state="
               << (state ? RecommendationConversionWorkflowStateText(*state)
                         : "NULL")
               << ",limit=" << limit
               << ",read_only=true,experiment_modified=false,"
                  "worker_started=false,scheduler_started=false\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        errors << "RECOMMENDATION_CONVERSION_WORKFLOW_LIST_FAILED,error="
               << RecommendationMachineText(error.what())
               << ",read_only=true,experiment_modified=false,"
                  "worker_started=false,scheduler_started=false\n";
        return 2;
    }
}

} // namespace EA::ExperimentRecommendation
