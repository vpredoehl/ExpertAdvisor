#pragma once

#include "ExperimentRecommendationCampaignFollowUpProposal.hpp"

#include <cstddef>
#include <string>

namespace EA::ExperimentRecommendation
{

inline constexpr int
    kRecommendationCampaignFollowUpProposalReviewContractVersion = 1;
inline constexpr std::size_t
    kMaximumRecommendationCampaignFollowUpProposalReviewerIdentityBytes =
        128U;
inline constexpr std::size_t
    kMaximumRecommendationCampaignFollowUpProposalReviewReasonTextBytes =
        4096U;
inline constexpr std::size_t
    kMaximumRecommendationCampaignFollowUpProposalReviewCanonicalTextBytes =
        kMaximumRecommendationCampaignFollowUpProposalCanonicalTextBytes +
        8192U;

enum class RecommendationCampaignFollowUpProposalReviewDecision
{
    approved,
    rejected
};

struct RecommendationCampaignFollowUpProposalReviewIdentity
{
    const int contractVersion;
    const std::string canonicalText;
    const std::string hash;

    RecommendationCampaignFollowUpProposalReviewIdentity(
        const RecommendationCampaignFollowUpProposalReviewIdentity&) =
        default;
    RecommendationCampaignFollowUpProposalReviewIdentity(
        RecommendationCampaignFollowUpProposalReviewIdentity&&) = default;
    bool operator==(
        const RecommendationCampaignFollowUpProposalReviewIdentity&) const =
        default;

private:
    RecommendationCampaignFollowUpProposalReviewIdentity(
        int contractVersion,
        std::string canonicalText,
        std::string hash);
    friend struct RecommendationCampaignFollowUpProposalReviewBuilder;
};

// Phase 6C is an immutable administrative fact only. None of these safety
// properties are caller-controlled or derived from an approved decision.
struct RecommendationCampaignFollowUpProposalReview
{
    static constexpr bool readOnly = true;
    static constexpr bool databaseFree = true;
    static constexpr bool persistent = false;
    static constexpr bool administrativeReview = true;
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

    const RecommendationCampaignFollowUpProposalReviewIdentity identity;
    const long long followUpProposalId;
    const int proposalContractVersion;
    const std::string proposalCanonicalText;
    const std::string proposalIdentityHash;
    const RecommendationCampaignFollowUpProposalReviewDecision decision;
    const std::string reviewerIdentity;
    const std::string reasonText;

    RecommendationCampaignFollowUpProposalReview(
        const RecommendationCampaignFollowUpProposalReview&) = default;
    RecommendationCampaignFollowUpProposalReview(
        RecommendationCampaignFollowUpProposalReview&&) = default;
    bool operator==(
        const RecommendationCampaignFollowUpProposalReview&) const = default;

private:
    RecommendationCampaignFollowUpProposalReview(
        RecommendationCampaignFollowUpProposalReviewIdentity identity,
        long long followUpProposalId,
        int proposalContractVersion,
        std::string proposalCanonicalText,
        std::string proposalIdentityHash,
        RecommendationCampaignFollowUpProposalReviewDecision decision,
        std::string reviewerIdentity,
        std::string reasonText);
    friend struct RecommendationCampaignFollowUpProposalReviewBuilder;
};

RecommendationCampaignFollowUpProposalReview
BuildRecommendationCampaignFollowUpProposalReview(
    int reviewContractVersion,
    long long followUpProposalId,
    int proposalContractVersion,
    const std::string& proposalCanonicalText,
    const std::string& proposalIdentityHash,
    RecommendationCampaignFollowUpProposalReviewDecision decision,
    std::string reviewerIdentity,
    std::string reasonText);

RecommendationCampaignFollowUpProposalReview
BuildRecommendationCampaignFollowUpProposalReview(
    long long followUpProposalId,
    const RecommendationCampaignFollowUpProposal& proposal,
    RecommendationCampaignFollowUpProposalReviewDecision decision,
    std::string reviewerIdentity,
    std::string reasonText);

void ValidateRecommendationCampaignFollowUpProposalReview(
    const RecommendationCampaignFollowUpProposalReview& review);

std::string RecommendationCampaignFollowUpProposalReviewDecisionText(
    RecommendationCampaignFollowUpProposalReviewDecision decision);

RecommendationCampaignFollowUpProposalReviewDecision
RecommendationCampaignFollowUpProposalReviewDecisionFromText(
    const std::string& text);

} // namespace EA::ExperimentRecommendation
