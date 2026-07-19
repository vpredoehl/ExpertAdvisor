#include "../Sources/ExperimentRecommendationConversionProposalReview.hpp"

#include <cassert>
#include <string>

using namespace EA::ExperimentRecommendation;

int main()
{
    assert(ParseRecommendationConversionProposalReviewDecision("approve") ==
           RecommendationConversionProposalReviewDecision::approve);
    assert(ParseRecommendationConversionProposalReviewDecision("reject") ==
           RecommendationConversionProposalReviewDecision::reject);
    assert(!ParseRecommendationConversionProposalReviewDecision("approved"));
    assert(RecommendationConversionProposalReviewDecisionText(
               RecommendationConversionProposalReviewDecision::approve) ==
           "approve");
    assert(RecommendationConversionProposalReviewDecisionText(
               RecommendationConversionProposalReviewDecision::reject) ==
           "reject");

    assert(ParseRecommendationConversionProposalReviewDisposition(
               "pending_review") ==
           RecommendationConversionProposalReviewDisposition::pendingReview);
    assert(ParseRecommendationConversionProposalReviewDisposition("approved") ==
           RecommendationConversionProposalReviewDisposition::approved);
    assert(ParseRecommendationConversionProposalReviewDisposition("rejected") ==
           RecommendationConversionProposalReviewDisposition::rejected);
    assert(!ParseRecommendationConversionProposalReviewDisposition("queued"));

    RecommendationConversionProposalReviewRequest request;
    request.proposalId = 42;
    request.decision =
        RecommendationConversionProposalReviewDecision::approve;
    request.requestId = "  review-2026.07.18:0001  ";
    request.operatorIdentity = "  Operator Name  ";
    request.reasonText = "  Exact proposal reviewed manually.  ";
    const auto normalized =
        NormalizeRecommendationConversionProposalReviewRequest(request);
    assert(normalized.requestId == "review-2026.07.18:0001");
    assert(normalized.operatorIdentity ==
           std::optional<std::string>{"Operator Name"});
    assert(normalized.reasonText ==
           std::optional<std::string>{"Exact proposal reviewed manually."});

    request.operatorIdentity.reset();
    request.reasonText.reset();
    assert(!NormalizeRecommendationConversionProposalReviewRequest(request)
                .operatorIdentity);
    assert(!NormalizeRecommendationConversionProposalReviewRequest(request)
                .reasonText);

    const auto rejected = [](RecommendationConversionProposalReviewRequest value,
                             const std::string& expected) {
        try
        {
            (void)NormalizeRecommendationConversionProposalReviewRequest(value);
        }
        catch (const std::invalid_argument& error)
        {
            return error.what() == expected;
        }
        return false;
    };
    request.proposalId = 0;
    assert(rejected(
        request, "recommendation_conversion_proposal_review_id_invalid"));
    request.proposalId = 42;
    request.requestId = " ";
    assert(rejected(
        request,
        "recommendation_conversion_proposal_review_request_id_required"));
    request.requestId = "contains space";
    assert(rejected(
        request,
        "recommendation_conversion_proposal_review_request_id_invalid"));
    request.requestId = "-starts-with-punctuation";
    assert(rejected(
        request,
        "recommendation_conversion_proposal_review_request_id_invalid"));
    request.requestId = std::string(
        kRecommendationConversionProposalReviewRequestIdMaximum + 1, 'x');
    assert(rejected(
        request,
        "recommendation_conversion_proposal_review_request_id_too_long"));
    request.requestId = "valid-token";
    request.operatorIdentity = "   ";
    assert(rejected(
        request,
        "recommendation_conversion_proposal_review_operator_empty"));
    request.operatorIdentity = std::string(
        kRecommendationConversionProposalReviewOperatorMaximum + 1, 'x');
    assert(rejected(
        request,
        "recommendation_conversion_proposal_review_operator_too_long"));
    request.operatorIdentity = std::string{"operator\0name", 13};
    assert(rejected(
        request,
        "recommendation_conversion_proposal_review_operator_invalid"));
    request.operatorIdentity.reset();
    request.reasonText = "\n\t";
    assert(rejected(
        request,
        "recommendation_conversion_proposal_review_reason_empty"));
    request.reasonText = std::string(
        kRecommendationConversionProposalReviewReasonMaximum + 1, 'x');
    assert(rejected(
        request,
        "recommendation_conversion_proposal_review_reason_too_long"));
    request.reasonText = std::string{"reason\0text", 11};
    assert(rejected(
        request,
        "recommendation_conversion_proposal_review_reason_invalid"));

    return 0;
}
