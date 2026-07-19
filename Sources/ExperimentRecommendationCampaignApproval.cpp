#include "ExperimentRecommendationCampaignApproval.hpp"

#include "ExperimentRecommendation.hpp"

#include <algorithm>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace EA::ExperimentRecommendation
{
namespace
{

std::string TrimAsciiWhitespace(const std::string& value)
{
    constexpr const char* whitespace = " \t\n\r\f\v";
    const std::size_t first = value.find_first_not_of(whitespace);
    if (first == std::string::npos) return {};
    const std::size_t last = value.find_last_not_of(whitespace);
    return value.substr(first, last - first + 1);
}

bool ContainsNul(const std::string& value)
{
    return value.find('\0') != std::string::npos;
}

bool ContainsAsciiControl(const std::string& value)
{
    return std::any_of(
        value.begin(), value.end(),
        [](unsigned char value)
        { return value < 0x20 || value == 0x7f; });
}

bool ValidCanonicalHashText(const std::string& value)
{
    constexpr std::string_view prefix = "fnv1a64:";
    if (value.size() != prefix.size() + 16 ||
        value.compare(0, prefix.size(), prefix) != 0)
        return false;
    return std::all_of(
        value.begin() + static_cast<std::ptrdiff_t>(prefix.size()),
        value.end(),
        [](unsigned char character)
        {
            return (character >= '0' && character <= '9') ||
                (character >= 'a' && character <= 'f');
        });
}

bool ValidText(
    const std::string& value,
    std::size_t maximum,
    bool allowAsciiControl = false)
{
    return !value.empty() && value.size() <= maximum && !ContainsNul(value) &&
           (allowAsciiControl || !ContainsAsciiControl(value));
}

std::string LengthText(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

void RequireIdentity(
    const std::string& canonical,
    const std::string& hash,
    std::size_t canonicalMaximum,
    const char* error)
{
    if (!ValidText(canonical, canonicalMaximum, true) ||
        !ValidText(hash, kRecommendationCampaignApprovalHashMaximum) ||
        hash != RecommendationCanonicalHash(canonical))
        throw std::invalid_argument(error);
}

std::string ApprovalCanonicalText(
    const RecommendationCampaignApprovalEvidence& value)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_campaign_approval_v1"
        << ";approval_contract_version=" << value.approvalContractVersion
        << ";ranking_snapshot_id=" << value.rankingSnapshotId
        << ";ranking_snapshot_identity="
        << LengthText(value.rankingSnapshotIdentityCanonical)
        << ";ranking_snapshot_identity_hash="
        << LengthText(value.rankingSnapshotIdentityHash)
        << ";planning_policy=" << LengthText(value.planningPolicyCanonical)
        << ";planning_policy_hash=" << LengthText(value.planningPolicyHash)
        << ";planning_scope=" << LengthText(value.planningScopeCanonical)
        << ";campaign_plan_identity="
        << LengthText(value.campaignPlanIdentityCanonical)
        << ";campaign_plan_identity_hash="
        << LengthText(value.campaignPlanIdentityHash)
        << ";review_contract_version=" << value.reviewContractVersion
        << ";campaign_review_identity="
        << LengthText(value.campaignReviewIdentityCanonical)
        << ";campaign_review_identity_hash="
        << LengthText(value.campaignReviewIdentityHash)
        << ";candidate_count=" << value.summary.candidateCount
        << ";selected_count=" << value.summary.selectedCount
        << ";excluded_count=" << value.summary.excludedCount
        << ";duplicate_group_count=" << value.summary.duplicateGroupCount
        << ";duplicate_candidate_count="
        << value.summary.duplicateCandidateCount
        << ";considered_family_count="
        << value.summary.consideredFamilyCount
        << ";selected_family_count=" << value.summary.selectedFamilyCount
        << ";considered_symbol_count="
        << value.summary.consideredSymbolCount
        << ";selected_symbol_count=" << value.summary.selectedSymbolCount
        << ";considered_horizon_count="
        << value.summary.consideredHorizonCount
        << ";selected_horizon_count=" << value.summary.selectedHorizonCount
        << ";deterministic_ordering_verified="
        << (value.summary.deterministicOrderingVerified ? 1 : 0)
        << ";decision="
        << RecommendationCampaignApprovalDecisionText(value.decision)
        << ";reviewer=" << LengthText(value.reviewerIdentity)
        << ";reason=" << LengthText(value.reasonText);
    return out.str();
}

void ValidateSummary(const RecommendationCampaignReviewSummary& summary)
{
    if (summary.candidateCount < 0 || summary.selectedCount < 0 ||
        summary.excludedCount < 0 ||
        summary.selectedCount + summary.excludedCount !=
            summary.candidateCount ||
        summary.duplicateGroupCount < 0 ||
        summary.duplicateCandidateCount < 0 ||
        summary.duplicateCandidateCount > summary.candidateCount ||
        summary.consideredFamilyCount < 0 ||
        summary.selectedFamilyCount < 0 ||
        summary.selectedFamilyCount > summary.consideredFamilyCount ||
        summary.consideredSymbolCount < 0 ||
        summary.selectedSymbolCount < 0 ||
        summary.selectedSymbolCount > summary.consideredSymbolCount ||
        summary.consideredHorizonCount < 0 ||
        summary.selectedHorizonCount < 0 ||
        summary.selectedHorizonCount > summary.consideredHorizonCount ||
        !summary.deterministicOrderingVerified)
        throw std::invalid_argument(
            "recommendation_campaign_approval_summary_invalid");
}

} // namespace

std::string RecommendationCampaignApprovalDecisionText(
    RecommendationCampaignApprovalDecision decision)
{
    switch (decision)
    {
        case RecommendationCampaignApprovalDecision::approved:
            return "approved";
        case RecommendationCampaignApprovalDecision::rejected:
            return "rejected";
    }
    throw std::invalid_argument(
        "recommendation_campaign_approval_decision_invalid");
}

std::optional<RecommendationCampaignApprovalDecision>
ParseRecommendationCampaignApprovalDecision(const std::string& text)
{
    if (text == "approved")
        return RecommendationCampaignApprovalDecision::approved;
    if (text == "rejected")
        return RecommendationCampaignApprovalDecision::rejected;
    return std::nullopt;
}

RecommendationCampaignApprovalRequest NormalizeRecommendationCampaignApprovalRequest(
    const RecommendationCampaignApprovalRequest& request)
{
    RecommendationCampaignApprovalRequest normalized = request;
    normalized.reviewerIdentity = TrimAsciiWhitespace(request.reviewerIdentity);
    normalized.reasonText = TrimAsciiWhitespace(request.reasonText);
    if (!ValidText(
            normalized.expectedCampaignReviewIdentityHash,
            kRecommendationCampaignApprovalHashMaximum) ||
        !ValidCanonicalHashText(
            normalized.expectedCampaignReviewIdentityHash))
        throw std::invalid_argument(
            "recommendation_campaign_approval_expected_review_hash_invalid");
    if (!ValidText(
            normalized.reviewerIdentity,
            kRecommendationCampaignApprovalReviewerMaximum))
        throw std::invalid_argument(
            "recommendation_campaign_approval_reviewer_invalid");
    if (!ValidText(
            normalized.reasonText,
            kRecommendationCampaignApprovalReasonMaximum,
            true))
        throw std::invalid_argument(
            "recommendation_campaign_approval_reason_invalid");
    return normalized;
}

RecommendationCampaignApprovalEvidence BuildRecommendationCampaignApprovalEvidence(
    long long rankingSnapshotId,
    const std::string& rankingSnapshotIdentityCanonical,
    const std::string& rankingSnapshotIdentityHash,
    const RecommendationCampaignPlan& plan,
    const RecommendationCampaignReview& review,
    const RecommendationCampaignApprovalRequest& request)
{
    const RecommendationCampaignApprovalRequest normalized =
        NormalizeRecommendationCampaignApprovalRequest(request);
    if (rankingSnapshotId <= 0 || plan.scope.rankingSnapshotId != rankingSnapshotId)
        throw std::invalid_argument(
            "recommendation_campaign_approval_ranking_snapshot_invalid");
    RequireIdentity(
        rankingSnapshotIdentityCanonical,
        rankingSnapshotIdentityHash,
        kRecommendationCampaignApprovalSnapshotCanonicalMaximum,
        "recommendation_campaign_approval_snapshot_identity_invalid");
    RequireIdentity(
        plan.identityCanonical,
        plan.identityHash,
        kRecommendationCampaignApprovalPlanCanonicalMaximum,
        "recommendation_campaign_approval_plan_identity_invalid");
    if (review.contractVersion != kRecommendationCampaignReviewContractVersion)
        throw std::invalid_argument(
            "recommendation_campaign_approval_review_contract_unsupported");
    RequireIdentity(
        review.identityCanonical,
        review.identityHash,
        kRecommendationCampaignApprovalReviewCanonicalMaximum,
        "recommendation_campaign_approval_review_identity_invalid");
    const RecommendationCampaignReview reconstructedReview =
        ReviewRecommendationCampaignPlan(plan);
    if (review != reconstructedReview)
        throw std::invalid_argument(
            "recommendation_campaign_approval_review_reconstruction_mismatch");
    if (normalized.expectedCampaignReviewIdentityHash != review.identityHash)
        throw std::invalid_argument(
            "recommendation_campaign_approval_review_identity_mismatch");
    if (review.campaignPlanIdentityCanonical != plan.identityCanonical ||
        review.campaignPlanIdentityHash != plan.identityHash ||
        review.policyIdentityHash != plan.policyHash ||
        review.scopeCanonical != plan.scopeCanonical)
        throw std::invalid_argument(
            "recommendation_campaign_approval_review_provenance_invalid");
    ValidateSummary(review.summary);
    if (normalized.decision == RecommendationCampaignApprovalDecision::approved &&
        review.summary.selectedCount == 0)
        throw std::invalid_argument(
            "recommendation_campaign_approval_zero_selection_forbidden");

    RecommendationCampaignApprovalEvidence evidence;
    evidence.rankingSnapshotId = rankingSnapshotId;
    evidence.rankingSnapshotIdentityCanonical =
        rankingSnapshotIdentityCanonical;
    evidence.rankingSnapshotIdentityHash = rankingSnapshotIdentityHash;
    evidence.planningPolicyCanonical = plan.policyCanonical;
    evidence.planningPolicyHash = plan.policyHash;
    evidence.planningScopeCanonical = plan.scopeCanonical;
    evidence.campaignPlanIdentityCanonical = plan.identityCanonical;
    evidence.campaignPlanIdentityHash = plan.identityHash;
    evidence.reviewContractVersion = review.contractVersion;
    evidence.campaignReviewIdentityCanonical = review.identityCanonical;
    evidence.campaignReviewIdentityHash = review.identityHash;
    evidence.summary = review.summary;
    evidence.decision = normalized.decision;
    evidence.reviewerIdentity = normalized.reviewerIdentity;
    evidence.reasonText = normalized.reasonText;
    evidence.approvalIdentityCanonical = ApprovalCanonicalText(evidence);
    evidence.approvalIdentityHash = RecommendationCanonicalHash(
        evidence.approvalIdentityCanonical);
    ValidateRecommendationCampaignApprovalEvidence(evidence);
    return evidence;
}

void ValidateRecommendationCampaignApprovalEvidence(
    const RecommendationCampaignApprovalEvidence& evidence)
{
    if (evidence.approvalContractVersion !=
            kRecommendationCampaignApprovalContractVersion ||
        evidence.rankingSnapshotId <= 0 ||
        evidence.reviewContractVersion !=
            kRecommendationCampaignReviewContractVersion)
        throw std::invalid_argument(
            "recommendation_campaign_approval_contract_invalid");
    RequireIdentity(
        evidence.rankingSnapshotIdentityCanonical,
        evidence.rankingSnapshotIdentityHash,
        kRecommendationCampaignApprovalSnapshotCanonicalMaximum,
        "recommendation_campaign_approval_snapshot_identity_invalid");
    RequireIdentity(
        evidence.planningPolicyCanonical,
        evidence.planningPolicyHash,
        kRecommendationCampaignApprovalPolicyCanonicalMaximum,
        "recommendation_campaign_approval_policy_identity_invalid");
    if (!ValidText(
            evidence.planningScopeCanonical,
            kRecommendationCampaignApprovalScopeCanonicalMaximum,
            true))
        throw std::invalid_argument(
            "recommendation_campaign_approval_scope_identity_invalid");
    RequireIdentity(
        evidence.campaignPlanIdentityCanonical,
        evidence.campaignPlanIdentityHash,
        kRecommendationCampaignApprovalPlanCanonicalMaximum,
        "recommendation_campaign_approval_plan_identity_invalid");
    RequireIdentity(
        evidence.campaignReviewIdentityCanonical,
        evidence.campaignReviewIdentityHash,
        kRecommendationCampaignApprovalReviewCanonicalMaximum,
        "recommendation_campaign_approval_review_identity_invalid");
    ValidateSummary(evidence.summary);
    if (!ValidText(
            evidence.reviewerIdentity,
            kRecommendationCampaignApprovalReviewerMaximum) ||
        evidence.reviewerIdentity != TrimAsciiWhitespace(
            evidence.reviewerIdentity) ||
        !ValidText(
            evidence.reasonText,
            kRecommendationCampaignApprovalReasonMaximum,
            true) ||
        evidence.reasonText != TrimAsciiWhitespace(evidence.reasonText))
        throw std::invalid_argument(
            "recommendation_campaign_approval_operator_text_invalid");
    if (evidence.decision == RecommendationCampaignApprovalDecision::approved &&
        evidence.summary.selectedCount == 0)
        throw std::invalid_argument(
            "recommendation_campaign_approval_zero_selection_forbidden");
    if (!ValidText(
            evidence.approvalIdentityCanonical,
            kRecommendationCampaignApprovalIdentityCanonicalMaximum,
            true) ||
        !ValidText(
            evidence.approvalIdentityHash,
            kRecommendationCampaignApprovalHashMaximum) ||
        evidence.approvalIdentityCanonical != ApprovalCanonicalText(evidence) ||
        evidence.approvalIdentityHash != RecommendationCanonicalHash(
            evidence.approvalIdentityCanonical))
        throw std::invalid_argument(
            "recommendation_campaign_approval_identity_invalid");
}

} // namespace EA::ExperimentRecommendation
