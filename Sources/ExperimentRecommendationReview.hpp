#pragma once

#include "ExperimentRecommendation.hpp"

#include <cstddef>
#include <optional>
#include <string>

namespace EA::ExperimentRecommendation
{

inline constexpr std::size_t kRecommendationReviewReasonCodeMaximum = 64;
inline constexpr std::size_t kRecommendationReviewReasonMaximum = 2000;
inline constexpr std::size_t kRecommendationReviewerMaximum = 200;
inline constexpr std::size_t kRecommendationReviewNoteMaximum = 2000;

enum class RecommendationReviewAction
{
    approve,
    reject,
    expire
};

struct RecommendationReviewRequest
{
    RecommendationReviewAction action = RecommendationReviewAction::approve;
    std::string reasonCode;
    std::optional<std::string> reasonText;
    std::optional<std::string> reviewer;
    std::optional<std::string> note;
    std::optional<long long> recommendationScoreId;
};

std::string RecommendationReviewActionText(RecommendationReviewAction action);
std::optional<RecommendationReviewAction> ParseRecommendationReviewAction(
    const std::string& text);
RecommendationStatus ResultingRecommendationStatus(
    RecommendationReviewAction action);
std::optional<std::string> ValidateRecommendationStatusTransition(
    RecommendationStatus current,
    RecommendationReviewAction action);
std::optional<std::string> ValidateRecommendationReviewRequest(
    const RecommendationReviewRequest& request);
RecommendationReviewRequest NormalizeRecommendationReviewRequest(
    const RecommendationReviewRequest& request);
bool RecommendationReviewReasonCodeAllowed(
    RecommendationReviewAction action,
    const std::string& reasonCode);

} // namespace EA::ExperimentRecommendation
