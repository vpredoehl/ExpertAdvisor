#pragma once

#include "ExperimentRecommendationCampaignFollowUpProposalReview.hpp"

#include <cstddef>
#include <string>

namespace EA::ExperimentRecommendation
{

inline constexpr int
    kRecommendationCampaignFollowUpProposalRatificationContractVersion = 1;
inline constexpr char
    kRecommendationCampaignFollowUpProposalRatificationAuthorityRole[] =
        "follow_up_governance_ratifier";
inline constexpr std::size_t
    kMaximumRecommendationCampaignFollowUpProposalRatifierIdentityBytes = 128U;
inline constexpr std::size_t
    kMaximumRecommendationCampaignFollowUpProposalRatificationBasisBytes =
        4096U;
inline constexpr std::size_t
    kMaximumRecommendationCampaignFollowUpProposalRatificationCanonicalTextBytes =
        kMaximumRecommendationCampaignFollowUpProposalReviewCanonicalTextBytes +
        kMaximumRecommendationCampaignFollowUpProposalCanonicalTextBytes +
        8192U;

enum class RecommendationCampaignFollowUpProposalRatificationDecision
{
    ratified
};

struct RecommendationCampaignFollowUpProposalRatificationIdentity
{
    const int contractVersion;
    const std::string canonicalText;
    const std::string hash;

    RecommendationCampaignFollowUpProposalRatificationIdentity(
        const RecommendationCampaignFollowUpProposalRatificationIdentity&) =
        default;
    RecommendationCampaignFollowUpProposalRatificationIdentity(
        RecommendationCampaignFollowUpProposalRatificationIdentity&&) = default;
    bool operator==(
        const RecommendationCampaignFollowUpProposalRatificationIdentity&) const =
        default;

private:
    RecommendationCampaignFollowUpProposalRatificationIdentity(
        int contractVersion,
        std::string canonicalText,
        std::string hash);
    friend struct RecommendationCampaignFollowUpProposalRatificationBuilder;
};

// Phase 6D is immutable governance ratification evidence. It ratifies only
// advancement into the next separately controlled phase and grants no
// capability in that phase.
struct RecommendationCampaignFollowUpProposalRatification
{
    static constexpr bool readOnly = true;
    static constexpr bool databaseFree = true;
    static constexpr bool persistent = false;
    static constexpr bool governanceRatification = true;
    static constexpr bool ratifiesAdvancementToNextSeparatelyControlledPhase =
        true;
    static constexpr bool eligibleReviewRequired = true;
    static constexpr bool separationOfDutiesRequired = true;
    static constexpr bool phase6ECapabilityGranted = false;
    static constexpr bool activated = false;
    static constexpr bool executionAuthorized = false;
    static constexpr bool followUpAuthorized = false;
    static constexpr bool queued = false;
    static constexpr bool scheduled = false;
    static constexpr bool schedulerStarted = false;
    static constexpr bool schedulerSignaled = false;
    static constexpr bool workersStarted = false;
    static constexpr bool experimentsCreated = false;
    static constexpr bool experimentsModified = false;
    static constexpr bool campaignSuccessDeclared = false;

    const RecommendationCampaignFollowUpProposalRatificationIdentity identity;
    const long long reviewEventId;
    const int reviewContractVersion;
    const std::string reviewCanonicalText;
    const std::string reviewIdentityHash;
    const long long followUpProposalId;
    const int proposalContractVersion;
    const std::string proposalCanonicalText;
    const std::string proposalIdentityHash;
    const RecommendationCampaignFollowUpProposalReviewDecision reviewDecision;
    const std::string reviewerIdentity;
    const std::string ratificationAuthorityRole;
    const RecommendationCampaignFollowUpProposalRatificationDecision decision;
    const std::string ratifierIdentity;
    const std::string ratificationBasis;

    RecommendationCampaignFollowUpProposalRatification(
        const RecommendationCampaignFollowUpProposalRatification&) = default;
    RecommendationCampaignFollowUpProposalRatification(
        RecommendationCampaignFollowUpProposalRatification&&) = default;
    bool operator==(
        const RecommendationCampaignFollowUpProposalRatification&) const = default;

private:
    RecommendationCampaignFollowUpProposalRatification(
        RecommendationCampaignFollowUpProposalRatificationIdentity identity,
        long long reviewEventId,
        int reviewContractVersion,
        std::string reviewCanonicalText,
        std::string reviewIdentityHash,
        long long followUpProposalId,
        int proposalContractVersion,
        std::string proposalCanonicalText,
        std::string proposalIdentityHash,
        RecommendationCampaignFollowUpProposalReviewDecision reviewDecision,
        std::string reviewerIdentity,
        std::string ratificationAuthorityRole,
        RecommendationCampaignFollowUpProposalRatificationDecision decision,
        std::string ratifierIdentity,
        std::string ratificationBasis);
    friend struct RecommendationCampaignFollowUpProposalRatificationBuilder;
};

RecommendationCampaignFollowUpProposalRatification
BuildRecommendationCampaignFollowUpProposalRatification(
    int ratificationContractVersion,
    long long reviewEventId,
    int reviewContractVersion,
    const std::string& reviewCanonicalText,
    const std::string& reviewIdentityHash,
    long long followUpProposalId,
    int proposalContractVersion,
    const std::string& proposalCanonicalText,
    const std::string& proposalIdentityHash,
    RecommendationCampaignFollowUpProposalReviewDecision reviewDecision,
    std::string reviewerIdentity,
    std::string ratificationAuthorityRole,
    RecommendationCampaignFollowUpProposalRatificationDecision decision,
    std::string ratifierIdentity,
    std::string ratificationBasis);

RecommendationCampaignFollowUpProposalRatification
BuildRecommendationCampaignFollowUpProposalRatification(
    long long reviewEventId,
    const RecommendationCampaignFollowUpProposalReview& review,
    std::string ratifierIdentity,
    std::string ratificationBasis);

void ValidateRecommendationCampaignFollowUpProposalRatification(
    const RecommendationCampaignFollowUpProposalRatification& ratification);

void ValidateRecommendationCampaignFollowUpProposalRatifierIdentity(
    const std::string& ratifierIdentity);

void ValidateRecommendationCampaignFollowUpProposalRatificationBasis(
    const std::string& ratificationBasis);

std::string RecommendationCampaignFollowUpProposalRatificationDecisionText(
    RecommendationCampaignFollowUpProposalRatificationDecision decision);

RecommendationCampaignFollowUpProposalRatificationDecision
RecommendationCampaignFollowUpProposalRatificationDecisionFromText(
    const std::string& text);

} // namespace EA::ExperimentRecommendation
