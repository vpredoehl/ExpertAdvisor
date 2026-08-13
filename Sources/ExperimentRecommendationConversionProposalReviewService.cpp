#include "ExperimentRecommendationConversionProposalReviewService.hpp"

#include "ExperimentRecommendationConversionRepository.hpp"
#include "ExperimentRecommendationService.hpp"

#include <ostream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

std::string OptionalText(const std::optional<std::string>& value)
{
    return value ? RecommendationMachineText(*value) : "NULL";
}

std::string OptionalNumber(const std::optional<long long>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

void RequireSchema(pqxx::connection& connection)
{
    if (!RecommendationConversionProposalSchemaExists(connection) ||
        !RecommendationConversionProposalReviewSchemaExists(connection))
        throw std::runtime_error(
            "conversion proposal review schema required; run "
            "./migrate_lstm_db.sh");
}

void PrintDecision(
    std::ostream& output,
    const PersistedRecommendationConversionProposalReviewDecision& decision)
{
    output << "CONVERSION_PROPOSAL_REVIEW"
           << ",review_decision_id=" << decision.reviewDecisionId
           << ",proposal_id=" << decision.proposalId
           << ",decision="
           << RecommendationConversionProposalReviewDecisionText(
                  decision.decision)
           << ",request_id=" << RecommendationMachineText(decision.requestId)
           << ",operator=" << OptionalText(decision.operatorIdentity)
           << ",reason=" << OptionalText(decision.reasonText)
           << ",decided_at=" << RecommendationMachineText(decision.decidedAt)
           << ",created_at=" << RecommendationMachineText(decision.createdAt)
           << ",experiment_created=false,experiment_queued=false,"
              "scheduler_modified=false\n";
}

} // namespace

int RunRecommendationConversionProposalReviewCommand(
    const std::string& connectionString,
    const RecommendationConversionProposalReviewRequest& request,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        const auto normalized =
            NormalizeRecommendationConversionProposalReviewRequest(request);
        pqxx::connection connection{connectionString};
        RequireSchema(connection);
        const auto result =
            RecordRecommendationConversionProposalReviewDecision(
                connection, normalized);
        if (result.outcome ==
            RecommendationConversionProposalReviewPersistOutcome::proposalNotFound)
        {
            errors << "CONVERSION_PROPOSAL_NOT_FOUND,proposal_id="
                   << normalized.proposalId
                   << ",experiment_created=false,experiment_queued=false,"
                      "scheduler_modified=false\n";
            return 1;
        }
        const auto& decision = *result.decision;
        output << (result.outcome ==
                       RecommendationConversionProposalReviewPersistOutcome::
                           recorded
                   ? "CONVERSION_PROPOSAL_REVIEW_RECORDED"
                   : "CONVERSION_PROPOSAL_REVIEW_ALREADY_RECORDED")
               << ",review_decision_id=" << decision.reviewDecisionId
               << ",proposal_id=" << decision.proposalId
               << ",decision="
               << RecommendationConversionProposalReviewDecisionText(
                      decision.decision)
               << ",request_id="
               << RecommendationMachineText(decision.requestId)
               << ",operator=" << OptionalText(decision.operatorIdentity)
               << ",reason=" << OptionalText(decision.reasonText)
               << ",experiment_created=false,experiment_queued=false,"
                  "scheduler_modified=false\n";
        output << "Conversion proposal " << decision.proposalId << " was "
               << (decision.decision ==
                           RecommendationConversionProposalReviewDecision::approve
                       ? "approved"
                       : "rejected")
               << " for possible later conversion.\n";
        if (decision.operatorIdentity)
            output << "Operator: "
                   << RecommendationHumanText(*decision.operatorIdentity) << '\n';
        if (decision.reasonText)
            output << "Reason: "
                   << RecommendationHumanText(*decision.reasonText) << '\n';
        output << "Review is administrative only. No experiment was created "
                  "or queued.\n";
        return 0;
    }
    catch (const std::invalid_argument& error)
    {
        errors << "CONVERSION_PROPOSAL_REVIEW_INVALID,proposal_id="
               << request.proposalId
               << ",error=" << RecommendationMachineText(error.what())
               << ",experiment_created=false,experiment_queued=false,"
                  "scheduler_modified=false\n";
        return 1;
    }
    catch (const std::runtime_error& error)
    {
        const bool conflict = std::string{error.what()} ==
            "recommendation_conversion_proposal_review_request_conflict";
        errors << (conflict ? "CONVERSION_PROPOSAL_REVIEW_CONFLICT"
                            : "CONVERSION_PROPOSAL_REVIEW_FAILED")
               << ",proposal_id=" << request.proposalId
               << ",error=" << RecommendationMachineText(error.what())
               << ",experiment_created=false,experiment_queued=false,"
                  "scheduler_modified=false\n";
        return conflict ? 2 : 1;
    }
}

int RunShowRecommendationConversionProposalCommand(
    const std::string& connectionString,
    long long proposalId,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    RequireSchema(connection);
    const auto proposal = FindRecommendationConversionProposal(
        connection, proposalId);
    if (!proposal)
    {
        output << "CONVERSION_PROPOSAL_NOT_FOUND,proposal_id=" << proposalId
               << ",experiment_created=false,experiment_queued=false,"
                  "scheduler_modified=false\n";
        return 1;
    }
    const auto current = GetRecommendationConversionProposalCurrentReview(
        connection, proposalId);
    if (!current)
        throw std::runtime_error(
            "conversion_proposal_review_current_state_missing");
    const std::optional<long long> latestId = current->latestDecision
        ? std::optional<long long>{current->latestDecision->reviewDecisionId}
        : std::nullopt;
    output << "CONVERSION_PROPOSAL,proposal_id=" << proposal->proposalId
           << ",recommendation_id=" << proposal->proposal.recommendationId
           << ",source_experiment_id="
           << proposal->proposal.sourceExperimentId
           << ",conversion_identity_hash="
           << RecommendationMachineText(
                  proposal->proposal.conversionIdentityHash)
           << ",changed_parameter="
           << RecommendationMutationParameterText(
                  proposal->proposal.changedParameter)
           << ",source_value="
           << RecommendationMachineText(
                  proposal->proposal.sourceValueCanonical)
           << ",proposed_value="
           << RecommendationMachineText(
                  proposal->proposal.proposedValueCanonical)
           << ",review_disposition="
           << RecommendationConversionProposalReviewDispositionText(
                  current->disposition)
           << ",latest_review_decision_id=" << OptionalNumber(latestId)
           << ",experiment_created=false,experiment_queued=false,"
              "scheduler_modified=false\n";
    output << "Conversion proposal " << proposal->proposalId << "\n"
           << "Current review disposition: "
           << RecommendationConversionProposalReviewDispositionText(
                  current->disposition)
           << "\nReview disposition is administrative only. No experiment was created "
              "or queued.\n";
    return 0;
}

int RunListRecommendationConversionProposalReviewsCommand(
    const std::string& connectionString,
    long long proposalId,
    int limit,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    RequireSchema(connection);
    const auto current = GetRecommendationConversionProposalCurrentReview(
        connection, proposalId);
    if (!current)
    {
        output << "CONVERSION_PROPOSAL_NOT_FOUND,proposal_id=" << proposalId
               << ",experiment_created=false,experiment_queued=false,"
                  "scheduler_modified=false\n";
        return 1;
    }
    const auto decisions = ListRecommendationConversionProposalReviewDecisions(
        connection, proposalId, limit);
    for (const auto& decision : decisions) PrintDecision(output, decision);
    output << "CONVERSION_PROPOSAL_REVIEW_HISTORY_COMPLETE,proposal_id="
           << proposalId << ",count=" << decisions.size()
           << ",current_disposition="
           << RecommendationConversionProposalReviewDispositionText(
                  current->disposition)
           << ",experiment_created=false,experiment_queued=false,"
              "scheduler_modified=false\n";
    return 0;
}

int RunListRecommendationConversionProposalsByReviewDispositionCommand(
    const std::string& connectionString,
    const RecommendationConversionProposalReviewListRequest& request,
    std::ostream& output)
{
    pqxx::connection connection{connectionString};
    RequireSchema(connection);
    const auto proposals =
        ListRecommendationConversionProposalsByReviewDisposition(
            connection, request.disposition, request.limit);
    for (const auto& proposal : proposals)
    {
        const std::optional<long long> latestId =
            proposal.currentReview.latestDecision
            ? std::optional<long long>{
                  proposal.currentReview.latestDecision->reviewDecisionId}
            : std::nullopt;
        output << "CONVERSION_PROPOSAL_REVIEW_SUMMARY,proposal_id="
               << proposal.proposalId
               << ",recommendation_id=" << proposal.recommendationId
               << ",source_experiment_id=" << proposal.sourceExperimentId
               << ",conversion_identity_hash="
               << RecommendationMachineText(proposal.conversionIdentityHash)
               << ",review_disposition="
               << RecommendationConversionProposalReviewDispositionText(
                      proposal.currentReview.disposition)
               << ",latest_review_decision_id=" << OptionalNumber(latestId)
               << ",experiment_created=false,experiment_queued=false,"
                  "scheduler_modified=false\n";
    }
    output << "CONVERSION_PROPOSAL_REVIEW_LIST_COMPLETE,count="
           << proposals.size()
           << ",filter="
           << (request.disposition
                   ? RecommendationConversionProposalReviewDispositionText(
                         *request.disposition)
                   : "NULL")
           << ",experiment_created=false,experiment_queued=false,"
              "scheduler_modified=false\n";
    return 0;
}

} // namespace EA::ExperimentRecommendation
