#pragma once

#include "ExperimentRecommendationCampaignReview.hpp"

#include <cstddef>
#include <optional>
#include <string>

namespace EA::ExperimentRecommendation
{

inline constexpr int kRecommendationCampaignApprovalContractVersion = 1;
inline constexpr std::size_t kRecommendationCampaignApprovalReviewerMaximum = 200;
inline constexpr std::size_t kRecommendationCampaignApprovalReasonMaximum = 2000;
inline constexpr std::size_t kRecommendationCampaignApprovalHashMaximum = 256;
inline constexpr std::size_t
    kRecommendationCampaignApprovalSnapshotCanonicalMaximum = 4 * 1024 * 1024;
inline constexpr std::size_t
    kRecommendationCampaignApprovalPolicyCanonicalMaximum = 1024 * 1024;
inline constexpr std::size_t
    kRecommendationCampaignApprovalScopeCanonicalMaximum = 64 * 1024;
inline constexpr std::size_t
    kRecommendationCampaignApprovalPlanCanonicalMaximum = 16 * 1024 * 1024;
inline constexpr std::size_t
    kRecommendationCampaignApprovalReviewCanonicalMaximum = 32 * 1024 * 1024;
inline constexpr std::size_t
    kRecommendationCampaignApprovalIdentityCanonicalMaximum = 64 * 1024 * 1024;

enum class RecommendationCampaignApprovalDecision
{
    approved,
    rejected
};

std::string RecommendationCampaignApprovalDecisionText(
    RecommendationCampaignApprovalDecision decision);
std::optional<RecommendationCampaignApprovalDecision>
ParseRecommendationCampaignApprovalDecision(const std::string& text);

struct RecommendationCampaignApprovalRequest
{
    RecommendationCampaignApprovalDecision decision =
        RecommendationCampaignApprovalDecision::approved;
    std::string expectedCampaignReviewIdentityHash;
    std::string reviewerIdentity;
    std::string reasonText;
};

struct RecommendationCampaignApprovalEvidence
{
    int approvalContractVersion =
        kRecommendationCampaignApprovalContractVersion;
    long long rankingSnapshotId = -1;
    std::string rankingSnapshotIdentityCanonical;
    std::string rankingSnapshotIdentityHash;
    std::string planningPolicyCanonical;
    std::string planningPolicyHash;
    std::string planningScopeCanonical;
    std::string campaignPlanIdentityCanonical;
    std::string campaignPlanIdentityHash;
    int reviewContractVersion = 0;
    std::string campaignReviewIdentityCanonical;
    std::string campaignReviewIdentityHash;
    RecommendationCampaignReviewSummary summary;
    RecommendationCampaignApprovalDecision decision =
        RecommendationCampaignApprovalDecision::approved;
    std::string reviewerIdentity;
    std::string reasonText;
    std::string approvalIdentityCanonical;
    std::string approvalIdentityHash;
};

RecommendationCampaignApprovalRequest NormalizeRecommendationCampaignApprovalRequest(
    const RecommendationCampaignApprovalRequest& request);

RecommendationCampaignApprovalEvidence BuildRecommendationCampaignApprovalEvidence(
    long long rankingSnapshotId,
    const std::string& rankingSnapshotIdentityCanonical,
    const std::string& rankingSnapshotIdentityHash,
    const RecommendationCampaignPlan& plan,
    const RecommendationCampaignReview& review,
    const RecommendationCampaignApprovalRequest& request);

void ValidateRecommendationCampaignApprovalEvidence(
    const RecommendationCampaignApprovalEvidence& evidence);

} // namespace EA::ExperimentRecommendation
