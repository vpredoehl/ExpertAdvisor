#include "ExperimentRecommendationCampaignActivationRepository.hpp"

#include "ExperimentRecommendationCampaignHandoffRepository.hpp"
#include "ExperimentRecommendationConversionExecutionRepository.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

std::map<int, const RecommendationCampaignHandoffMember*> MembersByOrdinal(
    const RecommendationCampaignHandoff& handoff)
{
    std::map<int, const RecommendationCampaignHandoffMember*> result;
    for (const auto& member : handoff.members)
        if (!result.emplace(member.memberOrdinal, &member).second)
            throw std::runtime_error(
                "campaign_activation_duplicate_handoff_ordinal");
    return result;
}

} // namespace

bool RecommendationCampaignActivationSchemasExist(
    pqxx::transaction_base& transaction)
{
    return RecommendationCampaignHandoffSchemasExist(transaction);
}

std::optional<PersistedRecommendationCampaignMaterialization>
LoadRecommendationCampaignActivationMaterialization(
    pqxx::transaction_base& transaction,
    long long materializationId)
{
    if (materializationId <= 0)
        throw std::invalid_argument(
            "campaign_activation_materialization_id_invalid");
    return FindRecommendationCampaignMaterialization(
        transaction, materializationId);
}

void LockRecommendationCampaignActivations(
    pqxx::transaction_base& transaction,
    const PersistedRecommendationCampaignMaterialization& materialization)
{
    const auto handoff = FindRecommendationCampaignHandoff(
        transaction, materialization.materializationId);
    if (!handoff)
        throw std::runtime_error(
            "campaign_activation_materialization_disappeared");
    const auto byOrdinal = MembersByOrdinal(*handoff);
    std::vector<long long> executionIds;
    std::vector<long long> experimentIds;
    executionIds.reserve(materialization.members.size());
    experimentIds.reserve(materialization.members.size());
    for (const auto& stored : materialization.members)
    {
        const auto found = byOrdinal.find(stored.memberOrdinal);
        if (found == byOrdinal.end() ||
            found->second->conversionProposalId != stored.conversionProposalId ||
            !found->second->executionId)
            throw std::runtime_error(
                "campaign_activation_execution_required");
        const auto execution = FindRecommendationConversionExecution(
            transaction, *found->second->executionId);
        if (!execution || execution->proposalId != stored.conversionProposalId)
            throw std::runtime_error(
                "campaign_activation_execution_provenance_invalid");
        executionIds.push_back(execution->executionId);
        experimentIds.push_back(execution->experimentId);
    }

    std::sort(executionIds.begin(), executionIds.end());
    if (std::adjacent_find(executionIds.begin(), executionIds.end()) !=
        executionIds.end())
        throw std::runtime_error("campaign_activation_duplicate_execution_link");
    for (const auto executionId : executionIds)
        LockRecommendationConversionActivationSequence(
            transaction, executionId);

    std::sort(experimentIds.begin(), experimentIds.end());
    if (std::adjacent_find(experimentIds.begin(), experimentIds.end()) !=
        experimentIds.end())
        throw std::runtime_error("campaign_activation_duplicate_experiment_link");
    for (const auto experimentId : experimentIds)
        LockRecommendationConversionActivationExperiment(
            transaction, experimentId);
}

RecommendationCampaignActivationInput LoadRecommendationCampaignActivationInput(
    pqxx::transaction_base& transaction,
    const PersistedRecommendationCampaignMaterialization& materialization)
{
    const auto handoff = FindRecommendationCampaignHandoff(
        transaction, materialization.materializationId);
    if (!handoff)
        throw std::runtime_error(
            "campaign_activation_materialization_disappeared");
    const auto byOrdinal = MembersByOrdinal(*handoff);

    RecommendationCampaignActivationInput input;
    input.materializationId = materialization.materializationId;
    input.materializationContractVersion = materialization.contractVersion;
    input.materializationIdentityCanonical = materialization.identityCanonical;
    input.materializationIdentityHash = materialization.identityHash;
    input.selectedMemberCount = materialization.selectedMemberCount;
    input.materializationComplete = handoff->materializationComplete;
    input.workflowEvidenceConsistent =
        handoff->integrity == RecommendationCampaignHandoffIntegrity::consistent;
    input.members.reserve(materialization.members.size());

    for (const auto& stored : materialization.members)
    {
        RecommendationCampaignActivationMemberInput member;
        member.memberOrdinal = stored.memberOrdinal;
        member.materializationMemberId = stored.materializationMemberId;
        member.recommendationId = stored.recommendationId;
        member.proposalId = stored.conversionProposalId;
        const auto found = byOrdinal.find(stored.memberOrdinal);
        if (found == byOrdinal.end() ||
            found->second->conversionProposalId != stored.conversionProposalId ||
            found->second->recommendationId != stored.recommendationId ||
            found->second->sourceExperimentId != stored.sourceExperimentId)
        {
            input.workflowEvidenceConsistent = false;
            member.evidenceDiagnostic =
                "campaign_activation_handoff_member_invalid";
            input.members.push_back(std::move(member));
            continue;
        }
        const auto& observed = *found->second;
        if (!observed.executionId)
        {
            member.evidenceDiagnostic =
                "campaign_activation_execution_required";
            input.members.push_back(std::move(member));
            continue;
        }
        const auto assessment = AssessRecommendationConversionActivation(
            transaction, *observed.executionId, false);
        if (!assessment.execution ||
            assessment.execution->proposalId != stored.conversionProposalId ||
            assessment.execution->experimentId <= 0)
        {
            member.evidenceDiagnostic =
                "campaign_activation_execution_provenance_invalid";
            input.members.push_back(std::move(member));
            continue;
        }
        member.authorizationReviewDecisionId =
            assessment.execution->reviewDecisionId;
        member.executionId = assessment.execution->executionId;
        member.experimentId = assessment.execution->experimentId;
        member.experimentStatus = assessment.experimentStatus;
        member.experimentPhase = assessment.experimentPhase;
        if (assessment.activation)
            member.activationId = assessment.activation->activationId;
        if (observed.activationId != member.activationId)
        {
            member.evidenceDiagnostic =
                "campaign_activation_handoff_activation_mismatch";
        }
        else if (assessment.state ==
                 RecommendationConversionActivationAssessmentState::eligible)
        {
            member.evidenceState =
                RecommendationCampaignActivationEvidenceState::eligible;
        }
        else if (assessment.state ==
                 RecommendationConversionActivationAssessmentState::
                     existingIdentical)
        {
            member.evidenceState = RecommendationCampaignActivationEvidenceState::
                existingIdentical;
        }
        else
        {
            member.evidenceDiagnostic = assessment.reason.empty()
                ? "campaign_activation_member_evidence_invalid"
                : "campaign_" + assessment.reason;
        }
        input.members.push_back(std::move(member));
    }
    return input;
}

std::vector<PersistedRecommendationConversionActivation>
PersistRecommendationCampaignActivations(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignActivationPlan& plan)
{
    if (plan.state != RecommendationCampaignActivationPlanState::ready)
        throw std::invalid_argument("campaign_activation_plan_not_writable");
    std::vector<PersistedRecommendationConversionActivation> persisted;
    persisted.reserve(plan.members.size());
    for (const auto& member : plan.members)
    {
        if (member.action != RecommendationCampaignActivationMemberAction::activate)
            throw std::invalid_argument(
                "campaign_activation_partial_write_forbidden");
        RecommendationConversionActivationResult result;
        try
        {
            result = ActivateRecommendationConversionExecutionInTransaction(
                transaction, member.executionId);
        }
        catch (const pqxx::unique_violation&)
        {
            throw std::runtime_error(
                "campaign_activation_uniqueness_conflict");
        }
        if (result.outcome != RecommendationConversionActivationOutcome::activated ||
            !result.activation ||
            result.activation->executionId != member.executionId ||
            result.activation->proposalId != member.proposalId ||
            result.activation->reviewDecisionId !=
                member.authorizationReviewDecisionId ||
            result.activation->experimentId != member.experimentId)
            throw std::runtime_error(
                "campaign_activation_insert_did_not_activate");
        const auto verified = AssessRecommendationConversionActivation(
            transaction, member.executionId, false);
        if (verified.state !=
                RecommendationConversionActivationAssessmentState::
                    existingIdentical ||
            verified.experimentStatus !=
                kRecommendationConversionActivationResultingStatus ||
            verified.experimentPhase !=
                kRecommendationConversionActivationResultingPhase)
            throw std::runtime_error(
                "campaign_activation_post_state_invalid");
        persisted.push_back(std::move(*result.activation));
    }
    return persisted;
}

} // namespace EA::ExperimentRecommendation
