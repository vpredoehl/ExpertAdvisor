#pragma once

#include "ExperimentRecommendationConversionProposalReviewRepository.hpp"

#include <iosfwd>
#include <optional>
#include <string>

namespace EA::ExperimentRecommendation
{

struct RecommendationConversionProposalReviewListRequest
{
    std::optional<RecommendationConversionProposalReviewDisposition> disposition;
    int limit = 100;
};

int RunRecommendationConversionProposalReviewCommand(
    const std::string& connectionString,
    const RecommendationConversionProposalReviewRequest& request,
    std::ostream& output,
    std::ostream& errors);

int RunShowRecommendationConversionProposalCommand(
    const std::string& connectionString,
    long long proposalId,
    std::ostream& output);

int RunListRecommendationConversionProposalReviewsCommand(
    const std::string& connectionString,
    long long proposalId,
    int limit,
    std::ostream& output);

int RunListRecommendationConversionProposalsByReviewDispositionCommand(
    const std::string& connectionString,
    const RecommendationConversionProposalReviewListRequest& request,
    std::ostream& output);

} // namespace EA::ExperimentRecommendation
