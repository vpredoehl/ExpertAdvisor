#include "ExperimentRecommendationCampaignFollowUpProposalPreview.hpp"

#include <iomanip>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

std::string PreviewMachineText(const std::string& value)
{
    std::ostringstream escaped;
    escaped.imbue(std::locale::classic());
    escaped << std::uppercase << std::hex;
    for (const unsigned char ch : value)
    {
        const bool alphanumeric =
            (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9');
        const bool safe = alphanumeric || ch == '-' || ch == '_' ||
            ch == '.' || ch == ':' || ch == '/' || ch == ';';
        if (safe) escaped << static_cast<char>(ch);
        else
            escaped << '%' << std::setw(2) << std::setfill('0')
                    << static_cast<unsigned int>(ch);
    }
    return escaped.str() == "NULL" ? "%4E%55%4C%4C" : escaped.str();
}

void PrintSafety(std::ostream& output)
{
    output << "read_only=true,persisted=true,advisory=true,"
              "authoritative=false,operator_approved=false,activated=false,"
              "execution_authorized=false,follow_up_authorized=false,"
              "queued=false,scheduled=false,scheduler_started=false,"
              "scheduler_signaled=false,workers_started=false,"
              "experiments_created=false,experiments_modified=false";
}

} // namespace

void WriteRecommendationCampaignFollowUpProposalPreview(
    std::ostream& output,
    const PersistedRecommendationCampaignFollowUpProposal& persisted)
{
    const auto& proposal = persisted.proposal;
    output << "RECOMMENDATION_CAMPAIGN_FOLLOW_UP_PROPOSAL_PREVIEW"
           << ",follow_up_proposal_id=" << persisted.followUpProposalId
           << ",proposal_contract_version="
           << proposal.identity.contractVersion
           << ",proposal_identity_hash="
           << PreviewMachineText(proposal.identity.hash)
           << ",proposal_identity_canonical="
           << PreviewMachineText(proposal.identity.canonicalText)
           << ",assessment_contract_version="
           << proposal.assessmentContractVersion
           << ",assessment_identity_hash="
           << PreviewMachineText(proposal.assessmentIdentityHash)
           << ",policy_contract_version=" << proposal.policyContractVersion
           << ",policy_identity_hash="
           << PreviewMachineText(proposal.policyIdentityHash)
           << ",policy_decision_contract_version="
           << proposal.policyDecisionContractVersion
           << ",policy_decision_identity_hash="
           << PreviewMachineText(proposal.policyDecisionIdentityHash)
           << ",campaign_approval_id="
           << proposal.campaignIdentity.campaignApprovalId
           << ",campaign_identity_hash="
           << PreviewMachineText(proposal.campaignIdentity.identityHash)
           << ",materialization_id="
           << proposal.materializationIdentity.materializationId
           << ",materialization_identity_hash="
           << PreviewMachineText(
                  proposal.materializationIdentity.identityHash)
           << ",member_count=" << proposal.memberCount
           << ",evidence_sufficiency="
           << RecommendationCampaignOutcomePolicyEvidenceSufficiencyText(
                  proposal.summary.evidenceSufficiency)
           << ",campaign_interpretation="
           << RecommendationCampaignOutcomePolicyCampaignInterpretationText(
                  proposal.summary.campaignInterpretation)
           << ",follow_up_eligibility="
           << RecommendationCampaignOutcomePolicyFollowUpEligibilityText(
                  proposal.summary.followUpEligibility)
           << ",proposal_reason="
           << RecommendationCampaignFollowUpProposalReasonText(
                  proposal.summary.reasons.front())
           << ",created_at=" << PreviewMachineText(persisted.createdAt)
           << ',';
    PrintSafety(output);
    output << '\n';

    for (const auto& member : proposal.members)
    {
        output << "RECOMMENDATION_CAMPAIGN_FOLLOW_UP_PROPOSAL_PREVIEW_MEMBER"
               << ",follow_up_proposal_id=" << persisted.followUpProposalId
               << ",member_ordinal=" << member.identity.memberOrdinal
               << ",materialization_member_id="
               << member.identity.materializationMemberId
               << ",ranking_member_id=" << member.identity.rankingMemberId
               << ",recommendation_id="
               << member.identity.recommendationId
               << ",source_experiment_id="
               << member.identity.sourceExperimentId
               << ",conversion_proposal_id=" << member.identity.proposalId
               << ",expected_experiment_id=";
        if (member.identity.expectedExperimentId)
            output << *member.identity.expectedExperimentId;
        else output << "null";
        output << ',';
        PrintSafety(output);
        output << '\n';
    }
}

int RunPreviewRecommendationCampaignFollowUpProposalCommand(
    const std::string& connectionString,
    long long followUpProposalId,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        pqxx::connection connection{connectionString};
        pqxx::read_transaction transaction{connection};
        transaction.exec(
            "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
        if (!RecommendationCampaignFollowUpProposalSchemaExists(transaction))
            throw std::runtime_error(
                "recommendation_campaign_follow_up_proposal_schema_required");
        const auto persisted = FindRecommendationCampaignFollowUpProposal(
            transaction, followUpProposalId);
        if (!persisted)
        {
            errors << "RECOMMENDATION_CAMPAIGN_FOLLOW_UP_PROPOSAL_NOT_FOUND"
                   << ",follow_up_proposal_id=" << followUpProposalId << ',';
            PrintSafety(errors);
            errors << '\n';
            return 1;
        }
        WriteRecommendationCampaignFollowUpProposalPreview(
            output, *persisted);
        return 0;
    }
    catch (const std::invalid_argument& error)
    {
        errors << "RECOMMENDATION_CAMPAIGN_FOLLOW_UP_PROPOSAL_INVALID,error="
               << PreviewMachineText(error.what()) << ',';
        PrintSafety(errors);
        errors << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        errors << "RECOMMENDATION_CAMPAIGN_FOLLOW_UP_PROPOSAL_FAILED,error="
               << PreviewMachineText(error.what()) << ',';
        PrintSafety(errors);
        errors << '\n';
        return 2;
    }
}

} // namespace EA::ExperimentRecommendation
