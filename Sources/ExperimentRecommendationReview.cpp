#include "ExperimentRecommendationReview.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

std::string TrimAscii(const std::string& value)
{
    const auto whitespace = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
               c == '\f' || c == '\v';
    };
    std::size_t first = 0;
    while (first < value.size() && whitespace(value[first])) ++first;
    std::size_t last = value.size();
    while (last > first && whitespace(value[last - 1])) --last;
    return value.substr(first, last - first);
}

std::optional<std::string> NormalizeOptional(
    const std::optional<std::string>& value,
    std::size_t maximum,
    const char* emptyError,
    const char* longError)
{
    if (!value) return std::nullopt;
    const std::string normalized = TrimAscii(*value);
    if (normalized.empty()) throw std::invalid_argument(emptyError);
    if (normalized.size() > maximum) throw std::invalid_argument(longError);
    return normalized;
}

template <std::size_t Size>
bool Contains(const std::array<const char*, Size>& values,
              const std::string& value)
{
    return std::any_of(values.begin(), values.end(), [&](const char* item) {
        return value == item;
    });
}

} // namespace

std::string RecommendationReviewActionText(RecommendationReviewAction action)
{
    switch (action)
    {
        case RecommendationReviewAction::approve: return "approve";
        case RecommendationReviewAction::reject: return "reject";
        case RecommendationReviewAction::expire: return "expire";
    }
    throw std::invalid_argument("invalid_recommendation_review_action");
}

std::optional<RecommendationReviewAction> ParseRecommendationReviewAction(
    const std::string& text)
{
    if (text == "approve") return RecommendationReviewAction::approve;
    if (text == "reject") return RecommendationReviewAction::reject;
    if (text == "expire") return RecommendationReviewAction::expire;
    return std::nullopt;
}

RecommendationStatus ResultingRecommendationStatus(
    RecommendationReviewAction action)
{
    switch (action)
    {
        case RecommendationReviewAction::approve:
            return RecommendationStatus::approved;
        case RecommendationReviewAction::reject:
            return RecommendationStatus::rejected;
        case RecommendationReviewAction::expire:
            return RecommendationStatus::expired;
    }
    throw std::invalid_argument("invalid_recommendation_review_action");
}

std::optional<std::string> ValidateRecommendationStatusTransition(
    RecommendationStatus current,
    RecommendationReviewAction action)
{
    (void)action;
    if (current != RecommendationStatus::proposed)
        return "recommendation_review_status_conflict";
    return std::nullopt;
}

bool RecommendationReviewReasonCodeAllowed(
    RecommendationReviewAction action,
    const std::string& reasonCode)
{
    constexpr std::array rejectCodes{
        "operator_rejected", "duplicate_research_direction",
        "unsupported_research_priority", "insufficient_evidence",
        "excessive_scope_change", "superseded_by_manual_plan", "other"};
    constexpr std::array expireCodes{
        "operator_expired", "stale_recommendation", "policy_obsolete",
        "source_evidence_obsolete", "research_window_closed", "other"};
    switch (action)
    {
        case RecommendationReviewAction::approve:
            return reasonCode == "operator_approved";
        case RecommendationReviewAction::reject:
            return Contains(rejectCodes, reasonCode);
        case RecommendationReviewAction::expire:
            return Contains(expireCodes, reasonCode);
    }
    return false;
}

RecommendationReviewRequest NormalizeRecommendationReviewRequest(
    const RecommendationReviewRequest& request)
{
    RecommendationReviewRequest normalized = request;
    normalized.reasonCode = TrimAscii(request.reasonCode);
    if (normalized.reasonCode.empty() &&
        request.action == RecommendationReviewAction::approve)
        normalized.reasonCode = "operator_approved";
    if (normalized.reasonCode.empty())
        throw std::invalid_argument("recommendation_review_reason_code_required");
    if (normalized.reasonCode.size() > kRecommendationReviewReasonCodeMaximum)
        throw std::invalid_argument("recommendation_review_reason_code_too_long");
    if (!RecommendationReviewReasonCodeAllowed(
            normalized.action, normalized.reasonCode))
        throw std::invalid_argument("recommendation_review_reason_code_invalid");

    normalized.reasonText = NormalizeOptional(
        request.reasonText, kRecommendationReviewReasonMaximum,
        "recommendation_review_reason_required",
        "recommendation_review_reason_too_long");
    if ((normalized.action == RecommendationReviewAction::reject ||
         normalized.action == RecommendationReviewAction::expire) &&
        !normalized.reasonText)
        throw std::invalid_argument("recommendation_review_reason_required");
    normalized.reviewer = NormalizeOptional(
        request.reviewer, kRecommendationReviewerMaximum,
        "recommendation_reviewer_empty", "recommendation_reviewer_too_long");
    normalized.note = NormalizeOptional(
        request.note, kRecommendationReviewNoteMaximum,
        "recommendation_review_note_empty",
        "recommendation_review_note_too_long");
    if (normalized.recommendationScoreId &&
        *normalized.recommendationScoreId <= 0)
        throw std::invalid_argument("recommendation_review_score_id_invalid");
    return normalized;
}

std::optional<std::string> ValidateRecommendationReviewRequest(
    const RecommendationReviewRequest& request)
{
    try
    {
        (void)NormalizeRecommendationReviewRequest(request);
        return std::nullopt;
    }
    catch (const std::invalid_argument& error)
    {
        return error.what();
    }
}

} // namespace EA::ExperimentRecommendation
