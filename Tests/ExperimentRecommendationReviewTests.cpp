#include <cassert>
#include <string>

#include "../Sources/ExperimentRecommendationReview.hpp"

using namespace EA::ExperimentRecommendation;

int main()
{
    assert(ParseRecommendationReviewAction("approve") ==
           RecommendationReviewAction::approve);
    assert(ParseRecommendationReviewAction("reject") ==
           RecommendationReviewAction::reject);
    assert(ParseRecommendationReviewAction("expire") ==
           RecommendationReviewAction::expire);
    assert(!ParseRecommendationReviewAction("auto_approved"));
    assert(RecommendationReviewActionText(RecommendationReviewAction::approve) ==
           "approve");
    assert(RecommendationReviewActionText(RecommendationReviewAction::reject) ==
           "reject");
    assert(RecommendationReviewActionText(RecommendationReviewAction::expire) ==
           "expire");

    assert(ResultingRecommendationStatus(RecommendationReviewAction::approve) ==
           RecommendationStatus::approved);
    assert(ResultingRecommendationStatus(RecommendationReviewAction::reject) ==
           RecommendationStatus::rejected);
    assert(ResultingRecommendationStatus(RecommendationReviewAction::expire) ==
           RecommendationStatus::expired);
    for (const auto action : {RecommendationReviewAction::approve,
                              RecommendationReviewAction::reject,
                              RecommendationReviewAction::expire})
    {
        assert(!ValidateRecommendationStatusTransition(
            RecommendationStatus::proposed, action));
        assert(ValidateRecommendationStatusTransition(
            RecommendationStatus::approved, action) ==
            std::optional<std::string>{"recommendation_review_status_conflict"});
        assert(ValidateRecommendationStatusTransition(
            RecommendationStatus::rejected, action));
        assert(ValidateRecommendationStatusTransition(
            RecommendationStatus::expired, action));
    }

    RecommendationReviewRequest approve;
    approve.action = RecommendationReviewAction::approve;
    approve.note = "  reviewed manually  ";
    approve.reviewer = " operator ";
    const auto normalizedApproval = NormalizeRecommendationReviewRequest(approve);
    assert(normalizedApproval.reasonCode == "operator_approved");
    assert(normalizedApproval.note == std::optional<std::string>{"reviewed manually"});
    assert(normalizedApproval.reviewer == std::optional<std::string>{"operator"});

    RecommendationReviewRequest reject;
    reject.action = RecommendationReviewAction::reject;
    reject.reasonCode = "insufficient_evidence";
    assert(ValidateRecommendationReviewRequest(reject) ==
           std::optional<std::string>{"recommendation_review_reason_required"});
    reject.reasonText = "  Evidence is incomplete.  ";
    assert(NormalizeRecommendationReviewRequest(reject).reasonText ==
           std::optional<std::string>{"Evidence is incomplete."});

    RecommendationReviewRequest expire;
    expire.action = RecommendationReviewAction::expire;
    expire.reasonCode = "stale_recommendation";
    assert(ValidateRecommendationReviewRequest(expire));
    expire.reasonText = "Research window closed.";
    assert(!ValidateRecommendationReviewRequest(expire));
    expire.reasonCode = "insufficient_evidence";
    assert(ValidateRecommendationReviewRequest(expire) ==
           std::optional<std::string>{"recommendation_review_reason_code_invalid"});

    approve.reviewer = std::string(kRecommendationReviewerMaximum + 1, 'x');
    assert(ValidateRecommendationReviewRequest(approve) ==
           std::optional<std::string>{"recommendation_reviewer_too_long"});
    approve.reviewer = "   ";
    assert(ValidateRecommendationReviewRequest(approve) ==
           std::optional<std::string>{"recommendation_reviewer_empty"});
    approve.reviewer.reset();
    approve.note = std::string(kRecommendationReviewNoteMaximum + 1, 'x');
    assert(ValidateRecommendationReviewRequest(approve) ==
           std::optional<std::string>{"recommendation_review_note_too_long"});
    approve.note.reset();
    approve.recommendationScoreId = 0;
    assert(ValidateRecommendationReviewRequest(approve) ==
           std::optional<std::string>{"recommendation_review_score_id_invalid"});
    return 0;
}
