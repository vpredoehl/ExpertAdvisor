#pragma once

#include "ExperimentRecommendationConversionRepository.hpp"
#include "ExperimentRecommendationConversionProposalReview.hpp"

#include <optional>
#include <string>

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

inline constexpr int kRecommendationConversionExecutionContractVersion = 1;

enum class RecommendationConversionExecutionOutcome
{
    created,
    existingIdentical,
    proposalNotFound,
    pendingReview,
    rejected
};

struct PersistedRecommendationConversionExecution
{
    long long executionId = -1;
    long long proposalId = -1;
    long long reviewDecisionId = -1;
    long long experimentId = -1;
    int executionContractVersion = 0;
    RecommendationConversionProposalReviewDecision authorizationDecision =
        RecommendationConversionProposalReviewDecision::approve;
    std::string executionIdentityCanonical;
    std::string executionIdentityHash;
    std::string createdAt;
};

struct RecommendationConversionExecutionResult
{
    RecommendationConversionExecutionOutcome outcome =
        RecommendationConversionExecutionOutcome::created;
    std::optional<PersistedRecommendationConversionExecution> execution;
};

std::string RecommendationConversionExecutionOutcomeText(
    RecommendationConversionExecutionOutcome outcome);

bool RecommendationConversionExecutionSchemaExists(
    pqxx::connection& connection);

RecommendationConversionExecutionResult
ExecuteApprovedRecommendationConversionProposal(
    pqxx::connection& connection,
    long long proposalId);

std::optional<PersistedRecommendationConversionExecution>
FindRecommendationConversionExecution(
    pqxx::connection& connection,
    long long executionId);

std::optional<PersistedRecommendationConversionExecution>
FindRecommendationConversionExecution(
    pqxx::transaction_base& transaction,
    long long executionId);

void ValidatePersistedRecommendationConversionExecution(
    pqxx::transaction_base& transaction,
    const PersistedRecommendationConversionExecution& execution);

std::optional<PersistedRecommendationConversionExecution>
FindRecommendationConversionExecutionByProposal(
    pqxx::connection& connection,
    long long proposalId);

} // namespace EA::ExperimentRecommendation
