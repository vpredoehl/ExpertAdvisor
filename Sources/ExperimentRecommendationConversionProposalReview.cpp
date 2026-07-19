#include "ExperimentRecommendationConversionProposalReview.hpp"

#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

bool IsAsciiWhitespace(unsigned char value)
{
    return value == ' ' || value == '\t' || value == '\n' || value == '\r' ||
           value == '\f' || value == '\v';
}

std::string TrimAscii(const std::string& value)
{
    std::size_t first = 0;
    while (first < value.size() && IsAsciiWhitespace(value[first])) ++first;
    std::size_t last = value.size();
    while (last > first && IsAsciiWhitespace(value[last - 1])) --last;
    return value.substr(first, last - first);
}

std::optional<std::string> NormalizeOptional(
    const std::optional<std::string>& value,
    std::size_t maximum,
    const char* emptyError,
    const char* invalidError,
    const char* longError)
{
    if (!value) return std::nullopt;
    const std::string normalized = TrimAscii(*value);
    if (normalized.empty()) throw std::invalid_argument(emptyError);
    if (normalized.find('\0') != std::string::npos)
        throw std::invalid_argument(invalidError);
    if (normalized.size() > maximum) throw std::invalid_argument(longError);
    return normalized;
}

bool ValidRequestId(const std::string& value)
{
    if (value.empty()) return false;
    const auto isAlphaNumeric = [](unsigned char character) {
        return (character >= 'a' && character <= 'z') ||
               (character >= 'A' && character <= 'Z') ||
               (character >= '0' && character <= '9');
    };
    if (!isAlphaNumeric(value.front())) return false;
    for (const unsigned char character : value)
    {
        if (!isAlphaNumeric(character) && character != '.' && character != '_' &&
            character != ':' && character != '-')
            return false;
    }
    return true;
}

} // namespace

std::string RecommendationConversionProposalReviewDecisionText(
    RecommendationConversionProposalReviewDecision decision)
{
    switch (decision)
    {
        case RecommendationConversionProposalReviewDecision::approve:
            return "approve";
        case RecommendationConversionProposalReviewDecision::reject:
            return "reject";
    }
    throw std::invalid_argument(
        "invalid_recommendation_conversion_proposal_review_decision");
}

std::optional<RecommendationConversionProposalReviewDecision>
ParseRecommendationConversionProposalReviewDecision(const std::string& text)
{
    if (text == "approve")
        return RecommendationConversionProposalReviewDecision::approve;
    if (text == "reject")
        return RecommendationConversionProposalReviewDecision::reject;
    return std::nullopt;
}

std::string RecommendationConversionProposalReviewDispositionText(
    RecommendationConversionProposalReviewDisposition disposition)
{
    switch (disposition)
    {
        case RecommendationConversionProposalReviewDisposition::pendingReview:
            return "pending_review";
        case RecommendationConversionProposalReviewDisposition::approved:
            return "approved";
        case RecommendationConversionProposalReviewDisposition::rejected:
            return "rejected";
    }
    throw std::invalid_argument(
        "invalid_recommendation_conversion_proposal_review_disposition");
}

std::optional<RecommendationConversionProposalReviewDisposition>
ParseRecommendationConversionProposalReviewDisposition(const std::string& text)
{
    if (text == "pending_review")
        return RecommendationConversionProposalReviewDisposition::pendingReview;
    if (text == "approved")
        return RecommendationConversionProposalReviewDisposition::approved;
    if (text == "rejected")
        return RecommendationConversionProposalReviewDisposition::rejected;
    return std::nullopt;
}

RecommendationConversionProposalReviewRequest
NormalizeRecommendationConversionProposalReviewRequest(
    const RecommendationConversionProposalReviewRequest& request)
{
    RecommendationConversionProposalReviewRequest normalized = request;
    if (normalized.proposalId <= 0)
        throw std::invalid_argument(
            "recommendation_conversion_proposal_review_id_invalid");
    normalized.requestId = TrimAscii(request.requestId);
    if (normalized.requestId.empty())
        throw std::invalid_argument(
            "recommendation_conversion_proposal_review_request_id_required");
    if (normalized.requestId.size() >
        kRecommendationConversionProposalReviewRequestIdMaximum)
        throw std::invalid_argument(
            "recommendation_conversion_proposal_review_request_id_too_long");
    if (!ValidRequestId(normalized.requestId))
        throw std::invalid_argument(
            "recommendation_conversion_proposal_review_request_id_invalid");
    normalized.operatorIdentity = NormalizeOptional(
        request.operatorIdentity,
        kRecommendationConversionProposalReviewOperatorMaximum,
        "recommendation_conversion_proposal_review_operator_empty",
        "recommendation_conversion_proposal_review_operator_invalid",
        "recommendation_conversion_proposal_review_operator_too_long");
    normalized.reasonText = NormalizeOptional(
        request.reasonText,
        kRecommendationConversionProposalReviewReasonMaximum,
        "recommendation_conversion_proposal_review_reason_empty",
        "recommendation_conversion_proposal_review_reason_invalid",
        "recommendation_conversion_proposal_review_reason_too_long");
    (void)RecommendationConversionProposalReviewDecisionText(
        normalized.decision);
    return normalized;
}

} // namespace EA::ExperimentRecommendation
