#pragma once

#include <cstddef>
#include <optional>
#include <string>

namespace EA::ExperimentRecommendation
{

inline constexpr std::size_t
    kRecommendationConversionProposalReviewRequestIdMaximum = 128;
inline constexpr std::size_t
    kRecommendationConversionProposalReviewOperatorMaximum = 200;
inline constexpr std::size_t
    kRecommendationConversionProposalReviewReasonMaximum = 2000;

enum class RecommendationConversionProposalReviewDecision
{
    approve,
    reject
};

enum class RecommendationConversionProposalReviewDisposition
{
    pendingReview,
    approved,
    rejected
};

struct RecommendationConversionProposalReviewRequest
{
    long long proposalId = -1;
    RecommendationConversionProposalReviewDecision decision =
        RecommendationConversionProposalReviewDecision::approve;
    std::string requestId;
    std::optional<std::string> operatorIdentity;
    std::optional<std::string> reasonText;
};

std::string RecommendationConversionProposalReviewDecisionText(
    RecommendationConversionProposalReviewDecision decision);
std::optional<RecommendationConversionProposalReviewDecision>
ParseRecommendationConversionProposalReviewDecision(const std::string& text);

std::string RecommendationConversionProposalReviewDispositionText(
    RecommendationConversionProposalReviewDisposition disposition);
std::optional<RecommendationConversionProposalReviewDisposition>
ParseRecommendationConversionProposalReviewDisposition(const std::string& text);

RecommendationConversionProposalReviewRequest
NormalizeRecommendationConversionProposalReviewRequest(
    const RecommendationConversionProposalReviewRequest& request);

} // namespace EA::ExperimentRecommendation
