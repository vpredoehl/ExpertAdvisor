#pragma once

#include "ExperimentRecommendationCampaignPlanning.hpp"

#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentRecommendation
{

inline constexpr int kRecommendationCampaignReviewContractVersion = 1;

struct RecommendationCampaignReviewCandidate
{
    int ordinal = 0;
    RecommendationCampaignDecision decision =
        RecommendationCampaignDecision::exclude;
    long long recommendationId = -1;
    long long sourceExperimentId = -1;
    long long rankingMemberId = -1;
    int rankingPosition = 0;
    std::string symbol;
    int predictionHorizon = 0;
    std::string family;
    std::vector<RecommendationCampaignReason> reasons;
    bool duplicate = false;
    std::optional<std::string> duplicateIdentityHash;
};

struct RecommendationCampaignReviewTextCoverage
{
    std::string value;
    int consideredCount = 0;
    int selectedCount = 0;
    int excludedCount = 0;
};

struct RecommendationCampaignReviewHorizonCoverage
{
    int horizon = 0;
    int consideredCount = 0;
    int selectedCount = 0;
    int excludedCount = 0;
};

struct RecommendationCampaignReviewDuplicateGroup
{
    std::string recommendationInvocationIdentityHash;
    std::vector<long long> recommendationIds;
};

struct RecommendationCampaignReviewSummary
{
    int candidateCount = 0;
    int selectedCount = 0;
    int excludedCount = 0;
    int duplicateGroupCount = 0;
    int duplicateCandidateCount = 0;
    int consideredFamilyCount = 0;
    int selectedFamilyCount = 0;
    int consideredSymbolCount = 0;
    int selectedSymbolCount = 0;
    int consideredHorizonCount = 0;
    int selectedHorizonCount = 0;
    bool deterministicOrderingVerified = false;
};

struct RecommendationCampaignReview
{
    int contractVersion = kRecommendationCampaignReviewContractVersion;
    std::string campaignPlanIdentityCanonical;
    std::string campaignPlanIdentityHash;
    std::string policyIdentityHash;
    std::string scopeCanonical;
    std::string generatedAt;
    RecommendationCampaignReviewSummary summary;
    std::vector<RecommendationCampaignReviewCandidate> selected;
    std::vector<RecommendationCampaignReviewCandidate> excluded;
    std::vector<RecommendationCampaignReviewDuplicateGroup> duplicateGroups;
    std::vector<RecommendationCampaignReviewTextCoverage> familyCoverage;
    std::vector<RecommendationCampaignReviewTextCoverage> symbolCoverage;
    std::vector<RecommendationCampaignReviewHorizonCoverage> horizonCoverage;
    std::string identityCanonical;
    std::string identityHash;
};

RecommendationCampaignReview ReviewRecommendationCampaignPlan(
    const RecommendationCampaignPlan& plan);

} // namespace EA::ExperimentRecommendation
