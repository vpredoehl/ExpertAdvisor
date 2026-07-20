#pragma once

#include "ExperimentRecommendationCampaignOutcomePolicy.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentRecommendation
{

inline constexpr int
    kRecommendationCampaignFollowUpProposalContractVersion = 1;
// The authoritative canonical text is deliberately bounded before immutable
// proposal construction. The byte ceiling is conservative enough for later
// persistence/review planning without granting any persistence capability.
inline constexpr std::size_t
    kMaximumRecommendationCampaignFollowUpProposalCanonicalTextBytes =
        1024U * 1024U;

enum class RecommendationCampaignFollowUpProposalReason
{
    EligibleFavorablePolicyDecision
};

// Narrow plain-data views used only to exercise the immutable Phase 5/6
// validation boundary. They cannot construct an assessment, policy decision,
// proposal identity, or proposal and add no database or execution capability.
struct RecommendationCampaignFollowUpProposalMemberValidationView
{
    int memberOrdinal = 0;
    long long materializationMemberId = 0;
    long long rankingMemberId = 0;
    long long recommendationId = 0;
    long long sourceExperimentId = 0;
    long long proposalId = 0;
    std::optional<long long> expectedExperimentId;
    bool operator==(
        const RecommendationCampaignFollowUpProposalMemberValidationView&)
        const = default;
};

struct RecommendationCampaignFollowUpProposalCampaignValidationView
{
    long long campaignApprovalId = 0;
    std::string identityCanonical;
    std::string identityHash;
    bool operator==(
        const RecommendationCampaignFollowUpProposalCampaignValidationView&)
        const = default;
};

struct RecommendationCampaignFollowUpProposalMaterializationValidationView
{
    long long materializationId = 0;
    long long campaignApprovalId = 0;
    std::string campaignIdentityHash;
    int contractVersion = 0;
    int memberCount = 0;
    std::string identityCanonical;
    std::string identityHash;
    bool operator==(
        const RecommendationCampaignFollowUpProposalMaterializationValidationView&)
        const = default;
};

struct RecommendationCampaignFollowUpProposalValidationView
{
    int assessmentContractVersion = 0;
    std::string assessmentCanonicalText;
    std::string assessmentIdentityHash;

    int policyContractVersion = 0;
    std::string policyCanonicalText;
    std::string policyIdentityHash;

    int policyDecisionContractVersion = 0;
    std::string policyDecisionCanonicalText;
    std::string policyDecisionIdentityHash;
    int decisionAssessmentContractVersion = 0;
    std::string decisionAssessmentCanonicalText;
    std::string decisionAssessmentIdentityHash;

    RecommendationCampaignFollowUpProposalCampaignValidationView
        assessmentCampaignIdentity;
    RecommendationCampaignFollowUpProposalCampaignValidationView
        decisionCampaignIdentity;
    RecommendationCampaignFollowUpProposalMaterializationValidationView
        assessmentMaterializationIdentity;
    RecommendationCampaignFollowUpProposalMaterializationValidationView
        decisionMaterializationIdentity;

    int assessmentSummaryMemberCount = 0;
    int decisionSummaryMemberCount = 0;
    std::vector<RecommendationCampaignFollowUpProposalMemberValidationView>
        assessmentMembers;
    std::vector<RecommendationCampaignFollowUpProposalMemberValidationView>
        decisionMembers;

    RecommendationCampaignOutcomePolicyEvidenceSufficiency
        evidenceSufficiency =
            RecommendationCampaignOutcomePolicyEvidenceSufficiency::
                Insufficient;
    RecommendationCampaignOutcomePolicyCampaignInterpretation
        campaignInterpretation =
            RecommendationCampaignOutcomePolicyCampaignInterpretation::
                Inconclusive;
    RecommendationCampaignOutcomePolicyFollowUpEligibility
        followUpEligibility =
            RecommendationCampaignOutcomePolicyFollowUpEligibility::
                NotEligible;
    bool followUpAuthorized = false;
};

struct RecommendationCampaignFollowUpProposalBuilder;

struct RecommendationCampaignFollowUpProposalIdentity
{
    const int contractVersion;
    const std::string canonicalText;
    const std::string hash;

    RecommendationCampaignFollowUpProposalIdentity(
        const RecommendationCampaignFollowUpProposalIdentity&) = default;
    RecommendationCampaignFollowUpProposalIdentity(
        RecommendationCampaignFollowUpProposalIdentity&&) = default;
    bool operator==(
        const RecommendationCampaignFollowUpProposalIdentity&) const =
        default;

private:
    RecommendationCampaignFollowUpProposalIdentity(
        int contractVersion,
        std::string canonicalText,
        std::string hash);
    friend struct RecommendationCampaignFollowUpProposalBuilder;
};

struct RecommendationCampaignFollowUpProposalMember
{
    const RecommendationCampaignOutcomeAssessmentMemberIdentity identity;

    RecommendationCampaignFollowUpProposalMember(
        const RecommendationCampaignFollowUpProposalMember&) = default;
    RecommendationCampaignFollowUpProposalMember(
        RecommendationCampaignFollowUpProposalMember&&) = default;
    bool operator==(
        const RecommendationCampaignFollowUpProposalMember&) const = default;

private:
    explicit RecommendationCampaignFollowUpProposalMember(
        RecommendationCampaignOutcomeAssessmentMemberIdentity identity);
    friend struct RecommendationCampaignFollowUpProposalBuilder;
};

struct RecommendationCampaignFollowUpProposalSummary
{
    const RecommendationCampaignOutcomePolicyEvidenceSufficiency
        evidenceSufficiency;
    const RecommendationCampaignOutcomePolicyCampaignInterpretation
        campaignInterpretation;
    const RecommendationCampaignOutcomePolicyFollowUpEligibility
        followUpEligibility;
    const bool followUpAuthorized;
    const int memberCount;
    const std::vector<RecommendationCampaignFollowUpProposalReason> reasons;

    RecommendationCampaignFollowUpProposalSummary(
        const RecommendationCampaignFollowUpProposalSummary&) = default;
    RecommendationCampaignFollowUpProposalSummary(
        RecommendationCampaignFollowUpProposalSummary&&) = default;
    bool operator==(
        const RecommendationCampaignFollowUpProposalSummary&) const = default;

private:
    RecommendationCampaignFollowUpProposalSummary(
        RecommendationCampaignOutcomePolicyEvidenceSufficiency
            evidenceSufficiency,
        RecommendationCampaignOutcomePolicyCampaignInterpretation
            campaignInterpretation,
        RecommendationCampaignOutcomePolicyFollowUpEligibility
            followUpEligibility,
        bool followUpAuthorized,
        int memberCount,
        std::vector<RecommendationCampaignFollowUpProposalReason> reasons);
    friend struct RecommendationCampaignFollowUpProposalBuilder;
};

struct RecommendationCampaignFollowUpProposal
{
    static constexpr bool readOnly = true;
    static constexpr bool databaseFree = true;
    static constexpr bool persistent = false;
    static constexpr bool advisory = true;
    static constexpr bool authoritative = false;
    static constexpr bool approved = false;
    static constexpr bool activated = false;
    static constexpr bool executionAuthorized = false;
    static constexpr bool followUpAuthorized = false;
    static constexpr bool schedulerWork = false;
    static constexpr bool declaresCampaignSuccess = false;

    const RecommendationCampaignFollowUpProposalIdentity identity;

    const int assessmentContractVersion;
    const std::string assessmentCanonicalText;
    const std::string assessmentIdentityHash;

    const int policyContractVersion;
    const std::string policyCanonicalText;
    const std::string policyIdentityHash;

    const int policyDecisionContractVersion;
    const std::string policyDecisionCanonicalText;
    const std::string policyDecisionIdentityHash;

    const RecommendationCampaignOutcomeAssessmentCampaignIdentity
        campaignIdentity;
    const RecommendationCampaignOutcomeAssessmentMaterializationIdentity
        materializationIdentity;
    const int memberCount;
    const RecommendationCampaignFollowUpProposalSummary summary;
    const std::vector<RecommendationCampaignFollowUpProposalMember> members;

    RecommendationCampaignFollowUpProposal(
        const RecommendationCampaignFollowUpProposal&) = default;
    RecommendationCampaignFollowUpProposal(
        RecommendationCampaignFollowUpProposal&&) = default;
    bool operator==(
        const RecommendationCampaignFollowUpProposal&) const = default;

private:
    RecommendationCampaignFollowUpProposal(
        RecommendationCampaignFollowUpProposalIdentity identity,
        int assessmentContractVersion,
        std::string assessmentCanonicalText,
        std::string assessmentIdentityHash,
        int policyContractVersion,
        std::string policyCanonicalText,
        std::string policyIdentityHash,
        int policyDecisionContractVersion,
        std::string policyDecisionCanonicalText,
        std::string policyDecisionIdentityHash,
        RecommendationCampaignOutcomeAssessmentCampaignIdentity
            campaignIdentity,
        RecommendationCampaignOutcomeAssessmentMaterializationIdentity
            materializationIdentity,
        int memberCount,
        RecommendationCampaignFollowUpProposalSummary summary,
        std::vector<RecommendationCampaignFollowUpProposalMember> members);
    friend struct RecommendationCampaignFollowUpProposalBuilder;
};

RecommendationCampaignFollowUpProposalValidationView
MakeRecommendationCampaignFollowUpProposalValidationView(
    const RecommendationCampaignOutcomeAssessment& assessment,
    const RecommendationCampaignOutcomePolicyDecision& policyDecision);

void ValidateRecommendationCampaignFollowUpProposalInput(
    const RecommendationCampaignFollowUpProposalValidationView& view);

void ValidateRecommendationCampaignFollowUpProposalCanonicalSize(
    std::size_t canonicalTextBytes);

RecommendationCampaignFollowUpProposal
BuildRecommendationCampaignFollowUpProposal(
    const RecommendationCampaignOutcomeAssessment& assessment,
    const RecommendationCampaignOutcomePolicyDecision& policyDecision);

std::string RecommendationCampaignFollowUpProposalReasonText(
    RecommendationCampaignFollowUpProposalReason value);

} // namespace EA::ExperimentRecommendation
