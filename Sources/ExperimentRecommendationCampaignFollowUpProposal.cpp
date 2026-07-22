#include "ExperimentRecommendationCampaignFollowUpProposal.hpp"

#include "ExperimentRecommendation.hpp"

#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace EA::ExperimentRecommendation
{
namespace
{

using Assessment = RecommendationCampaignOutcomeAssessment;
using CampaignView =
    RecommendationCampaignFollowUpProposalCampaignValidationView;
using Decision = RecommendationCampaignOutcomePolicyDecision;
using Eligibility = RecommendationCampaignOutcomePolicyFollowUpEligibility;
using Interpretation =
    RecommendationCampaignOutcomePolicyCampaignInterpretation;
using MaterializationView =
    RecommendationCampaignFollowUpProposalMaterializationValidationView;
using MemberView =
    RecommendationCampaignFollowUpProposalMemberValidationView;
using ProposalMember = RecommendationCampaignFollowUpProposalMember;
using ProposalReason = RecommendationCampaignFollowUpProposalReason;

static_assert(Assessment::readOnly);
static_assert(!Assessment::persistent);
static_assert(!Assessment::authoritative);
static_assert(!Assessment::declaresCampaignSuccess);
static_assert(Decision::readOnly);
static_assert(Decision::databaseFree);
static_assert(!Decision::persistent);
static_assert(Decision::advisory);
static_assert(!Decision::authoritative);
static_assert(!Decision::declaresCampaignSuccess);
static_assert(!Decision::followUpAuthorizing);

bool ValidCanonicalIdentity(
    const std::string& canonicalText,
    const std::string& hash)
{
    return !canonicalText.empty() &&
        canonicalText.find('\0') == std::string::npos &&
        hash == RecommendationCanonicalHash(canonicalText);
}

CampaignView CampaignValidationView(
    const RecommendationCampaignOutcomeAssessmentCampaignIdentity& identity)
{
    return {identity.campaignApprovalId, identity.identityCanonical,
        identity.identityHash};
}

MaterializationView MaterializationValidationView(
    const RecommendationCampaignOutcomeAssessmentMaterializationIdentity&
        identity)
{
    return {identity.materializationId, identity.campaignApprovalId,
        identity.campaignIdentityHash, identity.contractVersion,
        identity.memberCount, identity.identityCanonical,
        identity.identityHash};
}

MemberView MemberValidationView(
    const RecommendationCampaignOutcomeAssessmentMemberIdentity& identity)
{
    return {identity.memberOrdinal, identity.materializationMemberId,
        identity.rankingMemberId, identity.recommendationId,
        identity.sourceExperimentId, identity.proposalId,
        identity.expectedExperimentId};
}

bool ValidCampaignIdentity(const CampaignView& identity)
{
    return identity.campaignApprovalId > 0 &&
        ValidCanonicalIdentity(
            identity.identityCanonical, identity.identityHash);
}

bool ValidMaterializationIdentity(const MaterializationView& identity)
{
    return identity.materializationId > 0 &&
        identity.campaignApprovalId > 0 &&
        identity.contractVersion > 0 && identity.memberCount > 0 &&
        !identity.campaignIdentityHash.empty() &&
        ValidCanonicalIdentity(
            identity.identityCanonical, identity.identityHash);
}

std::string LengthPrefixed(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

std::string OptionalLongLong(const std::optional<long long>& value)
{
    return value ? std::to_string(*value) : "null";
}

std::string ProposalCanonicalText(
    int assessmentContractVersion,
    const std::string& assessmentCanonicalText,
    const std::string& assessmentIdentityHash,
    int policyContractVersion,
    const std::string& policyCanonicalText,
    const std::string& policyIdentityHash,
    int policyDecisionContractVersion,
    const std::string& policyDecisionCanonicalText,
    const std::string& policyDecisionIdentityHash,
    const RecommendationCampaignOutcomeAssessmentCampaignIdentity&
        campaignIdentity,
    const RecommendationCampaignOutcomeAssessmentMaterializationIdentity&
        materializationIdentity,
    const RecommendationCampaignFollowUpProposalSummary& summary,
    const std::vector<ProposalMember>& members)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_campaign_follow_up_proposal_v1"
        << ";proposal_contract_version="
        << kRecommendationCampaignFollowUpProposalContractVersion
        << ";read_only=true"
           ";database_free=true"
           ";persistent=false"
           ";advisory=true"
           ";authoritative=false"
           ";approved=false"
           ";activated=false"
           ";execution_authorized=false"
           ";follow_up_authorized=false"
           ";scheduler_work=false"
           ";declares_campaign_success=false"
        << ";assessment_contract_version="
        << assessmentContractVersion
        << ";assessment_canonical="
        << LengthPrefixed(assessmentCanonicalText)
        << ";assessment_hash=" << LengthPrefixed(assessmentIdentityHash)
        << ";policy_contract_version="
        << policyContractVersion
        << ";policy_canonical="
        << LengthPrefixed(policyCanonicalText)
        << ";policy_hash="
        << LengthPrefixed(policyIdentityHash)
        << ";policy_decision_contract_version="
        << policyDecisionContractVersion
        << ";policy_decision_canonical="
        << LengthPrefixed(policyDecisionCanonicalText)
        << ";policy_decision_hash="
        << LengthPrefixed(policyDecisionIdentityHash)
        << ";campaign_approval_id="
        << campaignIdentity.campaignApprovalId
        << ";campaign_identity="
        << LengthPrefixed(campaignIdentity.identityCanonical)
        << ";campaign_identity_hash="
        << LengthPrefixed(campaignIdentity.identityHash)
        << ";materialization_id="
        << materializationIdentity.materializationId
        << ";materialization_campaign_approval_id="
        << materializationIdentity.campaignApprovalId
        << ";materialization_campaign_identity_hash="
        << LengthPrefixed(materializationIdentity.campaignIdentityHash)
        << ";materialization_contract_version="
        << materializationIdentity.contractVersion
        << ";materialization_member_count="
        << materializationIdentity.memberCount
        << ";materialization_identity="
        << LengthPrefixed(materializationIdentity.identityCanonical)
        << ";materialization_identity_hash="
        << LengthPrefixed(materializationIdentity.identityHash)
        << ";member_count=" << members.size();
    for (const auto& member : members)
    {
        out << ";member=" << member.identity.memberOrdinal << ','
            << member.identity.materializationMemberId << ','
            << member.identity.rankingMemberId << ','
            << member.identity.recommendationId << ','
            << member.identity.sourceExperimentId << ','
            << member.identity.proposalId
            << ",expected_experiment_id="
            << OptionalLongLong(member.identity.expectedExperimentId);
    }
    out << ";evidence_sufficiency="
        << RecommendationCampaignOutcomePolicyEvidenceSufficiencyText(
               summary.evidenceSufficiency)
        << ";campaign_interpretation="
        << RecommendationCampaignOutcomePolicyCampaignInterpretationText(
               summary.campaignInterpretation)
        << ";follow_up_eligibility="
        << RecommendationCampaignOutcomePolicyFollowUpEligibilityText(
               summary.followUpEligibility)
        << ";decision_follow_up_authorized="
        << (summary.followUpAuthorized ? "true" : "false")
        << ";proposal_reason_count=" << summary.reasons.size();
    for (const auto reason : summary.reasons)
        out << ',' << RecommendationCampaignFollowUpProposalReasonText(reason);
    return out.str();
}

} // namespace

RecommendationCampaignFollowUpProposalValidationView
MakeRecommendationCampaignFollowUpProposalValidationView(
    const Assessment& assessment,
    const Decision& policyDecision)
{
    RecommendationCampaignFollowUpProposalValidationView view;
    view.assessmentContractVersion = assessment.identity.contractVersion;
    view.assessmentCanonicalText = assessment.identity.canonicalText;
    view.assessmentIdentityHash = assessment.identity.hash;
    view.policyContractVersion =
        policyDecision.policy.identity.contractVersion;
    view.policyCanonicalText = policyDecision.policy.identity.canonicalText;
    view.policyIdentityHash = policyDecision.policy.identity.hash;
    view.policyDecisionContractVersion =
        policyDecision.identity.contractVersion;
    view.policyDecisionCanonicalText = policyDecision.identity.canonicalText;
    view.policyDecisionIdentityHash = policyDecision.identity.hash;
    view.decisionAssessmentContractVersion =
        policyDecision.assessmentContractVersion;
    view.decisionAssessmentCanonicalText =
        policyDecision.assessmentCanonicalText;
    view.decisionAssessmentIdentityHash =
        policyDecision.assessmentIdentityHash;
    view.assessmentCampaignIdentity =
        CampaignValidationView(assessment.campaignIdentity);
    view.decisionCampaignIdentity =
        CampaignValidationView(policyDecision.campaignIdentity);
    view.assessmentMaterializationIdentity =
        MaterializationValidationView(assessment.materializationIdentity);
    view.decisionMaterializationIdentity =
        MaterializationValidationView(policyDecision.materializationIdentity);
    view.assessmentSummaryMemberCount = assessment.summary.memberCount;
    view.decisionSummaryMemberCount = policyDecision.summary.memberCount;
    view.assessmentMembers.reserve(assessment.members.size());
    for (const auto& member : assessment.members)
        view.assessmentMembers.push_back(MemberValidationView(member.identity));
    view.decisionMembers.reserve(policyDecision.members.size());
    for (const auto& member : policyDecision.members)
        view.decisionMembers.push_back(MemberValidationView(member.identity));
    view.evidenceSufficiency = policyDecision.summary.evidenceSufficiency;
    view.campaignInterpretation =
        policyDecision.summary.campaignInterpretation;
    view.followUpEligibility = policyDecision.summary.followUpEligibility;
    view.followUpAuthorized = policyDecision.summary.followUpAuthorized;
    return view;
}

void ValidateRecommendationCampaignFollowUpProposalInput(
    const RecommendationCampaignFollowUpProposalValidationView& view)
{
    if (view.assessmentContractVersion !=
            kRecommendationCampaignOutcomeAssessmentContractVersion ||
        view.decisionAssessmentContractVersion !=
            kRecommendationCampaignOutcomeAssessmentContractVersion)
        throw std::invalid_argument("assessment_contract_unsupported");
    if (view.policyContractVersion !=
        kRecommendationCampaignOutcomePolicyContractVersion)
        throw std::invalid_argument("policy_contract_unsupported");
    if (view.policyDecisionContractVersion !=
        kRecommendationCampaignOutcomePolicyDecisionContractVersion)
        throw std::invalid_argument("policy_decision_contract_unsupported");

    if (!ValidCanonicalIdentity(
            view.assessmentCanonicalText, view.assessmentIdentityHash))
        throw std::invalid_argument("assessment_identity_mismatch");
    if (!ValidCanonicalIdentity(
            view.policyCanonicalText, view.policyIdentityHash))
        throw std::invalid_argument("policy_identity_mismatch");
    if (!ValidCanonicalIdentity(view.policyDecisionCanonicalText,
            view.policyDecisionIdentityHash))
        throw std::invalid_argument("policy_decision_identity_mismatch");

    if (!ValidCampaignIdentity(view.assessmentCampaignIdentity) ||
        !ValidCampaignIdentity(view.decisionCampaignIdentity) ||
        view.assessmentCampaignIdentity != view.decisionCampaignIdentity)
        throw std::invalid_argument("campaign_identity_mismatch");
    if (!ValidMaterializationIdentity(
            view.assessmentMaterializationIdentity) ||
        !ValidMaterializationIdentity(view.decisionMaterializationIdentity) ||
        view.assessmentMaterializationIdentity !=
            view.decisionMaterializationIdentity ||
        view.assessmentMaterializationIdentity.campaignApprovalId !=
            view.assessmentCampaignIdentity.campaignApprovalId ||
        view.assessmentMaterializationIdentity.campaignIdentityHash !=
            view.assessmentCampaignIdentity.identityHash)
        throw std::invalid_argument("materialization_identity_mismatch");

    const std::size_t assessmentMemberCount =
        view.assessmentMembers.size();
    if (assessmentMemberCount == 0 ||
        view.decisionMembers.size() != assessmentMemberCount ||
        view.assessmentSummaryMemberCount !=
            static_cast<int>(assessmentMemberCount) ||
        view.decisionSummaryMemberCount !=
            static_cast<int>(assessmentMemberCount) ||
        view.assessmentMaterializationIdentity.memberCount !=
            static_cast<int>(assessmentMemberCount))
        throw std::invalid_argument("member_count_mismatch");
    if (view.assessmentMembers != view.decisionMembers)
        throw std::invalid_argument("member_identity_mismatch");
    std::set<long long> materializationMemberIds;
    std::set<long long> rankingMemberIds;
    std::set<long long> recommendationIds;
    std::set<long long> proposalIds;
    for (std::size_t index = 0; index < assessmentMemberCount; ++index)
    {
        const auto& member = view.assessmentMembers[index];
        if (member.memberOrdinal != static_cast<int>(index) + 1 ||
            member.materializationMemberId <= 0 ||
            member.rankingMemberId <= 0 || member.recommendationId <= 0 ||
            member.sourceExperimentId <= 0 || member.proposalId <= 0 ||
            (member.expectedExperimentId &&
                *member.expectedExperimentId <= 0) ||
            !materializationMemberIds
                 .insert(member.materializationMemberId).second ||
            !rankingMemberIds.insert(member.rankingMemberId).second ||
            !recommendationIds.insert(member.recommendationId).second ||
            !proposalIds.insert(member.proposalId).second)
            throw std::invalid_argument("member_identity_mismatch");
    }

    if (view.decisionAssessmentContractVersion !=
            view.assessmentContractVersion ||
        view.decisionAssessmentCanonicalText !=
            view.assessmentCanonicalText ||
        view.decisionAssessmentIdentityHash != view.assessmentIdentityHash)
        throw std::invalid_argument("assessment_identity_mismatch");

    if (view.followUpAuthorized)
        throw std::invalid_argument(
            "follow_up_authorization_must_be_false");
    if (view.followUpEligibility !=
        Eligibility::EligibleForOperatorReview)
        throw std::invalid_argument(
            "decision_not_eligible_for_operator_review");
    if (view.campaignInterpretation != Interpretation::Favorable)
        throw std::invalid_argument(
            "favorable_campaign_interpretation_required");
}

void ValidateRecommendationCampaignFollowUpProposalCanonicalSize(
    std::size_t canonicalTextBytes)
{
    if (canonicalTextBytes >
        kMaximumRecommendationCampaignFollowUpProposalCanonicalTextBytes)
        throw std::length_error("proposal_canonical_size_exceeded");
}

RecommendationCampaignFollowUpProposalIdentity::
    RecommendationCampaignFollowUpProposalIdentity(
        int contractVersionValue,
        std::string canonicalTextValue,
        std::string hashValue)
    : contractVersion(contractVersionValue),
      canonicalText(std::move(canonicalTextValue)),
      hash(std::move(hashValue))
{
    if (contractVersion !=
            kRecommendationCampaignFollowUpProposalContractVersion ||
        !ValidCanonicalIdentity(canonicalText, hash) ||
        canonicalText.size() >
            kMaximumRecommendationCampaignFollowUpProposalCanonicalTextBytes)
        throw std::logic_error(
            "recommendation_campaign_follow_up_proposal_identity_invariant_failed");
}

RecommendationCampaignFollowUpProposalMember::
    RecommendationCampaignFollowUpProposalMember(
        RecommendationCampaignOutcomeAssessmentMemberIdentity identityValue)
    : identity(std::move(identityValue))
{
}

RecommendationCampaignFollowUpProposalSummary::
    RecommendationCampaignFollowUpProposalSummary(
        RecommendationCampaignOutcomePolicyEvidenceSufficiency
            evidenceSufficiencyValue,
        Interpretation campaignInterpretationValue,
        Eligibility followUpEligibilityValue,
        bool followUpAuthorizedValue,
        int memberCountValue,
        std::vector<ProposalReason> reasonsValue)
    : evidenceSufficiency(evidenceSufficiencyValue),
      campaignInterpretation(campaignInterpretationValue),
      followUpEligibility(followUpEligibilityValue),
      followUpAuthorized(followUpAuthorizedValue),
      memberCount(memberCountValue),
      reasons(std::move(reasonsValue))
{
    if (campaignInterpretation != Interpretation::Favorable ||
        followUpEligibility != Eligibility::EligibleForOperatorReview ||
        followUpAuthorized || memberCount <= 0 ||
        reasons != std::vector<ProposalReason>{
            ProposalReason::EligibleFavorablePolicyDecision})
        throw std::logic_error(
            "recommendation_campaign_follow_up_proposal_summary_invariant_failed");
}

RecommendationCampaignFollowUpProposal::
    RecommendationCampaignFollowUpProposal(
        RecommendationCampaignFollowUpProposalIdentity identityValue,
        int assessmentContractVersionValue,
        std::string assessmentCanonicalTextValue,
        std::string assessmentIdentityHashValue,
        int policyContractVersionValue,
        std::string policyCanonicalTextValue,
        std::string policyIdentityHashValue,
        int policyDecisionContractVersionValue,
        std::string policyDecisionCanonicalTextValue,
        std::string policyDecisionIdentityHashValue,
        RecommendationCampaignOutcomeAssessmentCampaignIdentity
            campaignIdentityValue,
        RecommendationCampaignOutcomeAssessmentMaterializationIdentity
            materializationIdentityValue,
        int memberCountValue,
        RecommendationCampaignFollowUpProposalSummary summaryValue,
        std::vector<ProposalMember> membersValue)
    : identity(std::move(identityValue)),
      assessmentContractVersion(assessmentContractVersionValue),
      assessmentCanonicalText(std::move(assessmentCanonicalTextValue)),
      assessmentIdentityHash(std::move(assessmentIdentityHashValue)),
      policyContractVersion(policyContractVersionValue),
      policyCanonicalText(std::move(policyCanonicalTextValue)),
      policyIdentityHash(std::move(policyIdentityHashValue)),
      policyDecisionContractVersion(policyDecisionContractVersionValue),
      policyDecisionCanonicalText(
          std::move(policyDecisionCanonicalTextValue)),
      policyDecisionIdentityHash(
          std::move(policyDecisionIdentityHashValue)),
      campaignIdentity(std::move(campaignIdentityValue)),
      materializationIdentity(std::move(materializationIdentityValue)),
      memberCount(memberCountValue),
      summary(std::move(summaryValue)),
      members(std::move(membersValue))
{
    if (assessmentContractVersion !=
            kRecommendationCampaignOutcomeAssessmentContractVersion ||
        policyContractVersion !=
            kRecommendationCampaignOutcomePolicyContractVersion ||
        policyDecisionContractVersion !=
            kRecommendationCampaignOutcomePolicyDecisionContractVersion ||
        !ValidCanonicalIdentity(
            assessmentCanonicalText, assessmentIdentityHash) ||
        !ValidCanonicalIdentity(policyCanonicalText, policyIdentityHash) ||
        !ValidCanonicalIdentity(
            policyDecisionCanonicalText, policyDecisionIdentityHash) ||
        memberCount != static_cast<int>(members.size()) ||
        summary.memberCount != memberCount)
        throw std::logic_error(
            "recommendation_campaign_follow_up_proposal_invariant_failed");
    const std::string expectedCanonical = ProposalCanonicalText(
        assessmentContractVersion, assessmentCanonicalText,
        assessmentIdentityHash, policyContractVersion, policyCanonicalText,
        policyIdentityHash, policyDecisionContractVersion,
        policyDecisionCanonicalText, policyDecisionIdentityHash,
        campaignIdentity, materializationIdentity, summary, members);
    if (identity.canonicalText != expectedCanonical ||
        identity.hash != RecommendationCanonicalHash(expectedCanonical))
        throw std::logic_error(
            "recommendation_campaign_follow_up_proposal_identity_payload_mismatch");
}

struct RecommendationCampaignFollowUpProposalBuilder
{
    static RecommendationCampaignFollowUpProposal Build(
        const Assessment& assessment,
        const Decision& policyDecision)
    {
        ValidateRecommendationCampaignFollowUpProposalInput(
            MakeRecommendationCampaignFollowUpProposalValidationView(
                assessment, policyDecision));

        std::vector<ProposalMember> members;
        members.reserve(assessment.members.size());
        for (const auto& member : assessment.members)
            members.push_back(ProposalMember(member.identity));

        RecommendationCampaignFollowUpProposalSummary summary(
            policyDecision.summary.evidenceSufficiency,
            policyDecision.summary.campaignInterpretation,
            policyDecision.summary.followUpEligibility,
            policyDecision.summary.followUpAuthorized,
            static_cast<int>(members.size()),
            {ProposalReason::EligibleFavorablePolicyDecision});
        std::string canonical = ProposalCanonicalText(
            assessment.identity.contractVersion,
            assessment.identity.canonicalText, assessment.identity.hash,
            policyDecision.policy.identity.contractVersion,
            policyDecision.policy.identity.canonicalText,
            policyDecision.policy.identity.hash,
            policyDecision.identity.contractVersion,
            policyDecision.identity.canonicalText,
            policyDecision.identity.hash, assessment.campaignIdentity,
            assessment.materializationIdentity, summary, members);
        ValidateRecommendationCampaignFollowUpProposalCanonicalSize(
            canonical.size());
        RecommendationCampaignFollowUpProposalIdentity identity(
            kRecommendationCampaignFollowUpProposalContractVersion,
            canonical, RecommendationCanonicalHash(canonical));
        return RecommendationCampaignFollowUpProposal(
            std::move(identity), assessment.identity.contractVersion,
            assessment.identity.canonicalText, assessment.identity.hash,
            policyDecision.policy.identity.contractVersion,
            policyDecision.policy.identity.canonicalText,
            policyDecision.policy.identity.hash,
            policyDecision.identity.contractVersion,
            policyDecision.identity.canonicalText,
            policyDecision.identity.hash, assessment.campaignIdentity,
            assessment.materializationIdentity,
            static_cast<int>(members.size()), std::move(summary),
            std::move(members));
    }
};

RecommendationCampaignFollowUpProposal
BuildRecommendationCampaignFollowUpProposal(
    const Assessment& assessment,
    const Decision& policyDecision)
{
    return RecommendationCampaignFollowUpProposalBuilder::Build(
        assessment, policyDecision);
}

std::string RecommendationCampaignFollowUpProposalReasonText(
    RecommendationCampaignFollowUpProposalReason value)
{
    switch (value)
    {
        case ProposalReason::EligibleFavorablePolicyDecision:
            return "eligible_favorable_policy_decision";
    }
    throw std::invalid_argument(
        "recommendation_campaign_follow_up_proposal_reason_invalid");
}

} // namespace EA::ExperimentRecommendation
