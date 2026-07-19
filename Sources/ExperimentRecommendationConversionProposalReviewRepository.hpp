#pragma once

#include "ExperimentRecommendationConversionProposalReview.hpp"

#include <optional>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

inline constexpr int
    kMaximumRecommendationConversionProposalReviewListLimit = 1000;

enum class RecommendationConversionProposalReviewPersistOutcome
{
    recorded,
    existingIdentical,
    proposalNotFound
};

struct PersistedRecommendationConversionProposalReviewDecision
{
    long long reviewDecisionId = -1;
    long long proposalId = -1;
    RecommendationConversionProposalReviewDecision decision =
        RecommendationConversionProposalReviewDecision::approve;
    std::string requestId;
    std::optional<std::string> operatorIdentity;
    std::optional<std::string> reasonText;
    std::string decidedAt;
    std::string createdAt;
};

struct RecommendationConversionProposalReviewPersistResult
{
    RecommendationConversionProposalReviewPersistOutcome outcome =
        RecommendationConversionProposalReviewPersistOutcome::recorded;
    std::optional<PersistedRecommendationConversionProposalReviewDecision>
        decision;
};

struct RecommendationConversionProposalCurrentReview
{
    RecommendationConversionProposalReviewDisposition disposition =
        RecommendationConversionProposalReviewDisposition::pendingReview;
    std::optional<PersistedRecommendationConversionProposalReviewDecision>
        latestDecision;
};

struct RecommendationConversionProposalReviewSummary
{
    long long proposalId = -1;
    long long recommendationId = -1;
    long long sourceExperimentId = -1;
    std::string conversionIdentityHash;
    RecommendationConversionProposalCurrentReview currentReview;
};

std::string RecommendationConversionProposalReviewPersistOutcomeText(
    RecommendationConversionProposalReviewPersistOutcome outcome);

bool RecommendationConversionProposalReviewSchemaExists(
    pqxx::connection& connection);

RecommendationConversionProposalReviewPersistResult
RecordRecommendationConversionProposalReviewDecision(
    pqxx::connection& connection,
    const RecommendationConversionProposalReviewRequest& request);

std::optional<PersistedRecommendationConversionProposalReviewDecision>
FindRecommendationConversionProposalReviewDecision(
    pqxx::connection& connection,
    long long reviewDecisionId);

std::vector<PersistedRecommendationConversionProposalReviewDecision>
ListRecommendationConversionProposalReviewDecisions(
    pqxx::connection& connection,
    long long proposalId,
    int limit = 100);

std::optional<RecommendationConversionProposalCurrentReview>
GetRecommendationConversionProposalCurrentReview(
    pqxx::connection& connection,
    long long proposalId);

std::vector<RecommendationConversionProposalReviewSummary>
ListRecommendationConversionProposalsByReviewDisposition(
    pqxx::connection& connection,
    std::optional<RecommendationConversionProposalReviewDisposition> disposition,
    int limit = 100);

} // namespace EA::ExperimentRecommendation
