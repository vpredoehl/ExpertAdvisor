#include "ExperimentRecommendationCampaignStatus.hpp"

#include "ExperimentRecommendation.hpp"
#include "ExperimentRecommendationCampaignMaterialization.hpp"

#include <algorithm>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

void Diagnose(RecommendationCampaignStatusMember& member, std::string code)
{
    if (std::find(member.diagnosticCodes.begin(), member.diagnosticCodes.end(), code) ==
        member.diagnosticCodes.end())
        member.diagnosticCodes.push_back(std::move(code));
    member.consistency = RecommendationCampaignStatusConsistency::inconsistent;
    member.operationalState = RecommendationCampaignStatusOperationalState::inconsistent;
    member.terminalResult = RecommendationCampaignStatusTerminalResult::unknown;
}

bool IsKnownStatus(const std::string& value)
{
    return value == "pending" || value == "paused" || value == "running" ||
        value == "completed" || value == "failed" || value == "cancelled";
}

bool IsKnownPhase(const std::string& value)
{
    return value == "train" || value == "infer" || value == "analyze" ||
        value == "done";
}

RecommendationCampaignStatusMember ClassifyMember(
    const RecommendationCampaignStatusMemberInput& input)
{
    RecommendationCampaignStatusMember member;
    member.memberOrdinal = input.memberOrdinal;
    member.materializationMemberId = input.materializationMemberId;
    member.recommendationId = input.recommendationId;
    member.sourceExperimentId = input.sourceExperimentId;
    member.rankingMemberId = input.rankingMemberId;
    member.proposalId = input.proposalId;
    member.currentReviewDecisionId = input.currentReviewDecisionId;
    member.currentReviewDecision = input.currentReviewDecision;
    member.authorizationReviewDecisionId = input.authorizationReviewDecisionId;
    member.executionId = input.executionId;
    member.activationId = input.activationId;
    member.experimentId = input.experimentId;
    member.executionEvidenceCount = input.executionCount;
    member.activationEvidenceCount = input.activationCount;

    if (input.executionCount > 1)
        member.executionState = RecommendationCampaignStatusExecutionState::conflicting;
    else if (input.executionId)
        member.executionState = RecommendationCampaignStatusExecutionState::executed;
    if (input.activationCount > 1)
        member.activationState = RecommendationCampaignStatusActivationState::conflicting;
    else if (input.activationId)
        member.activationState = RecommendationCampaignStatusActivationState::activated;

    if (!input.workflowConsistent)
    {
        if (input.executionCount == 1 && input.executionId)
            member.executionState = RecommendationCampaignStatusExecutionState::malformed;
        if (input.activationCount == 1 && input.activationId)
            member.activationState = RecommendationCampaignStatusActivationState::malformed;
        for (const auto& code : input.workflowDiagnostics) Diagnose(member, code);
        if (input.workflowDiagnostics.empty())
            Diagnose(member, "campaign_status_workflow_inconsistent");
    }
    if (input.executionCount < 0 || input.executionCount > 1 ||
        (input.executionId.has_value() != (input.executionCount == 1)))
        Diagnose(member, "campaign_status_execution_cardinality_invalid");
    if (input.activationCount < 0 || input.activationCount > 1 ||
        (input.activationId.has_value() != (input.activationCount == 1)))
        Diagnose(member, "campaign_status_activation_cardinality_invalid");
    if (input.activationId && !input.executionId)
        Diagnose(member, "campaign_status_activation_without_execution");
    if (!input.executionId)
    {
        if (input.experiment || input.experimentId)
            Diagnose(member, "campaign_status_experiment_without_execution");
        return member;
    }
    if (!input.experiment || !input.experimentId)
    {
        Diagnose(member, "campaign_status_execution_experiment_missing");
        return member;
    }

    const auto& experiment = *input.experiment;
    member.symbol = experiment.symbol;
    member.predictionHorizon = experiment.predictionHorizon;
    member.status = experiment.status;
    member.phase = experiment.phase;
    member.currentEpoch = experiment.currentEpoch;
    member.targetEpochs = experiment.targetEpochs;
    member.workerPid = experiment.workerPid;
    member.currentOperation = experiment.currentOperation;
    member.workerStartedAt = experiment.workerStartedAt;
    member.startedAt = experiment.startedAt;
    member.completedAt = experiment.completedAt;
    member.exitCode = experiment.exitCode;
    member.errorMessage = experiment.errorMessage;
    member.modelId = experiment.modelId;
    if (!experiment.invocationIdentityCanonical.empty())
        member.invocationIdentityHash = RecommendationCanonicalHash(
            experiment.invocationIdentityCanonical);
    member.modelLinkCount = experiment.modelLinkCount;
    member.finalInferenceResultCount = experiment.inferenceResultCount;
    member.completedFinalInferenceResultCount =
        experiment.completedInferenceResultCount;
    member.failedFinalInferenceResultCount =
        experiment.failedInferenceResultCount;
    member.finalAnalysisResultCount = experiment.analysisResultCount;
    member.completedFinalAnalysisResultCount =
        experiment.completedAnalysisResultCount;
    member.failedFinalAnalysisResultCount = experiment.failedAnalysisResultCount;
    member.inferConfigured = experiment.inferConfigured;
    member.analyzeConfigured = experiment.inferConfigured;
    member.inferComplete = experiment.completedInferenceResultCount > 0;
    member.analyzeComplete = experiment.completedAnalysisResultCount > 0;
    member.trainComplete = experiment.modelId.has_value() &&
        experiment.modelLinkCount == 1;

    if (experiment.experimentId != *input.experimentId)
        Diagnose(member, "campaign_status_experiment_provenance_mismatch");
    if (!experiment.invocationProvenanceValid)
        Diagnose(member, "campaign_status_experiment_invocation_mismatch");
    if (experiment.symbol.empty() || experiment.predictionHorizon <= 0 ||
        experiment.targetEpochs <= 0 || experiment.duplicateNonce != 0 ||
        experiment.invocationMode != "recommendation_conversion")
        Diagnose(member, "campaign_status_experiment_identity_invalid");
    if (!IsKnownStatus(experiment.status))
        Diagnose(member, "campaign_status_experiment_status_unknown");
    if (!IsKnownPhase(experiment.phase))
        Diagnose(member, "campaign_status_experiment_phase_unknown");
    if (experiment.currentEpoch &&
        (*experiment.currentEpoch < 0 || *experiment.currentEpoch > experiment.targetEpochs))
        Diagnose(member, "campaign_status_current_epoch_invalid");
    if (experiment.inferStartPresent != experiment.inferEndPresent)
        Diagnose(member, "campaign_status_inference_configuration_invalid");
    if (experiment.modelLinkCount < 0 || experiment.modelLinkCount > 1)
        Diagnose(member, "campaign_status_model_link_conflict");
    if (experiment.inferenceResultCount < 0 ||
        experiment.completedInferenceResultCount < 0 ||
        experiment.failedInferenceResultCount < 0 ||
        experiment.completedInferenceResultCount +
                experiment.failedInferenceResultCount !=
            experiment.inferenceResultCount ||
        experiment.completedInferenceResultCount > 1 ||
        (experiment.completedInferenceResultCount > 0 &&
         experiment.failedInferenceResultCount > 0))
        Diagnose(member, "campaign_status_inference_evidence_invalid");
    if (experiment.analysisResultCount < 0 ||
        experiment.completedAnalysisResultCount < 0 ||
        experiment.failedAnalysisResultCount < 0 ||
        experiment.completedAnalysisResultCount +
                experiment.failedAnalysisResultCount !=
            experiment.analysisResultCount ||
        experiment.completedAnalysisResultCount > 1 ||
        experiment.failedAnalysisResultCount > 1 ||
        (experiment.completedAnalysisResultCount > 0 &&
         experiment.failedAnalysisResultCount > 0))
        Diagnose(member, "campaign_status_analysis_evidence_invalid");

    if (member.consistency == RecommendationCampaignStatusConsistency::inconsistent)
        return member;

    if (!input.activationId)
    {
        if (experiment.status == "paused" && experiment.phase == "train" &&
            !experiment.workerPid && !experiment.currentOperation &&
            !experiment.currentEpoch && !experiment.workerStartedAt &&
            !experiment.startedAt && !experiment.completedAt &&
            !experiment.exitCode && !experiment.errorMessage && !experiment.modelId)
            member.operationalState = RecommendationCampaignStatusOperationalState::paused;
        else
            Diagnose(member, "campaign_status_unactivated_lifecycle_invalid");
        return member;
    }

    if (experiment.status == "pending")
    {
        if (experiment.workerPid)
            Diagnose(member, "campaign_status_pending_worker_invalid");
        else if (experiment.phase == "train" || experiment.phase == "infer" ||
                 experiment.phase == "analyze")
            member.operationalState = RecommendationCampaignStatusOperationalState::pending;
        else
            Diagnose(member, "campaign_status_pending_phase_invalid");
    }
    else if (experiment.status == "running")
    {
        if (!experiment.workerStartedAt || !experiment.currentOperation ||
            *experiment.currentOperation != experiment.phase)
            Diagnose(member, "campaign_status_running_worker_evidence_invalid");
        else if (experiment.phase == "train")
            member.operationalState = RecommendationCampaignStatusOperationalState::runningTrain;
        else if (experiment.phase == "infer")
            member.operationalState = RecommendationCampaignStatusOperationalState::runningInfer;
        else if (experiment.phase == "analyze")
            member.operationalState = RecommendationCampaignStatusOperationalState::runningAnalyze;
        else
            Diagnose(member, "campaign_status_running_phase_invalid");
    }
    else if (experiment.status == "paused")
    {
        if (experiment.startedAt || experiment.currentEpoch || experiment.modelId)
            member.operationalState =
                RecommendationCampaignStatusOperationalState::pausedAfterProgress;
        else
            Diagnose(member, "campaign_status_activated_experiment_still_paused");
    }
    else if (experiment.status == "completed")
    {
        if (experiment.phase != "done" || !experiment.completedAt ||
            experiment.workerPid || !member.trainComplete ||
            !member.inferConfigured || !member.inferComplete ||
            !member.analyzeConfigured || !member.analyzeComplete ||
            (experiment.errorMessage && !experiment.errorMessage->empty()) ||
            (experiment.exitCode && *experiment.exitCode != 0))
            Diagnose(member, "campaign_status_completed_evidence_invalid");
        else
        {
            member.operationalState = RecommendationCampaignStatusOperationalState::completed;
            member.terminalResult = RecommendationCampaignStatusTerminalResult::succeeded;
        }
    }
    else if (experiment.status == "failed")
    {
        if (!experiment.completedAt || !experiment.errorMessage ||
            experiment.errorMessage->empty() || experiment.workerPid)
            Diagnose(member, "campaign_status_failed_evidence_invalid");
        else
        {
            member.operationalState = RecommendationCampaignStatusOperationalState::failed;
            member.terminalResult = RecommendationCampaignStatusTerminalResult::failed;
        }
    }
    else if (experiment.status == "cancelled")
    {
        if (!experiment.completedAt || experiment.workerPid)
            Diagnose(member, "campaign_status_cancelled_evidence_invalid");
        else
        {
            member.operationalState = RecommendationCampaignStatusOperationalState::cancelled;
            member.terminalResult = RecommendationCampaignStatusTerminalResult::cancelled;
        }
    }
    else if (experiment.status != "pending" && experiment.status != "running" &&
             experiment.status != "paused")
        Diagnose(member, "campaign_status_experiment_lifecycle_invalid");

    return member;
}

std::string OptionalLongLong(const std::optional<long long>& value)
{
    return value ? std::to_string(*value) : "null";
}

std::string OptionalInt(const std::optional<int>& value)
{
    return value ? std::to_string(*value) : "null";
}

std::string OptionalText(const std::optional<std::string>& value)
{
    return value ? std::to_string(value->size()) + ":" + *value : "null";
}

std::string SnapshotCanonical(
    const RecommendationCampaignStatusInput& input,
    const std::vector<RecommendationCampaignStatusMember>& members)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_campaign_status_snapshot_v1"
        << ";status_contract_version=" << kRecommendationCampaignStatusContractVersion
        << ";materialization_id=" << input.materializationId
        << ";materialization_contract_version=" << input.materializationContractVersion
        << ";materialization_identity_hash=" << input.materializationIdentityHash
        << ";member_count=" << members.size();
    for (const auto& member : members)
    {
        out << ";member=" << member.memberOrdinal << ','
            << member.materializationMemberId << ',' << member.recommendationId << ','
            << member.sourceExperimentId << ',' << member.rankingMemberId << ','
            << member.proposalId << ','
            << OptionalLongLong(member.currentReviewDecisionId) << ','
            << OptionalText(member.currentReviewDecision) << ','
            << OptionalLongLong(member.authorizationReviewDecisionId) << ','
            << OptionalLongLong(member.executionId) << ','
            << OptionalLongLong(member.activationId) << ','
            << OptionalLongLong(member.experimentId) << ','
            << member.executionEvidenceCount << ','
            << member.activationEvidenceCount << ','
            << RecommendationCampaignStatusExecutionStateText(member.executionState)
            << ',' << RecommendationCampaignStatusActivationStateText(member.activationState)
            << ',' << RecommendationCampaignStatusOperationalStateText(member.operationalState)
            << ',' << RecommendationCampaignStatusTerminalResultText(member.terminalResult)
            << ',' << OptionalText(member.symbol) << ','
            << OptionalInt(member.predictionHorizon) << ','
            << OptionalText(member.status) << ',' << OptionalText(member.phase)
            << ',' << OptionalInt(member.currentEpoch) << ','
            << OptionalInt(member.targetEpochs) << ',' << OptionalInt(member.workerPid)
            << ',' << OptionalText(member.currentOperation) << ','
            << OptionalText(member.workerStartedAt) << ','
            << OptionalText(member.startedAt) << ','
            << OptionalText(member.completedAt) << ','
            << OptionalInt(member.exitCode) << ','
            << OptionalText(member.errorMessage) << ','
            << OptionalLongLong(member.modelId) << ','
            << OptionalText(member.invocationIdentityHash) << ','
            << member.modelLinkCount << ','
            << member.finalInferenceResultCount << ','
            << member.completedFinalInferenceResultCount << ','
            << member.failedFinalInferenceResultCount << ','
            << member.finalAnalysisResultCount << ','
            << member.completedFinalAnalysisResultCount << ','
            << member.failedFinalAnalysisResultCount << ','
            << member.trainComplete << ',' << member.inferConfigured << ','
            << member.inferComplete << ',' << member.analyzeConfigured << ','
            << member.analyzeComplete << ','
            << RecommendationCampaignStatusConsistencyText(member.consistency);
        for (const auto& code : member.diagnosticCodes)
            out << ',' << code.size() << ':' << code;
    }
    return out.str();
}

RecommendationCampaignAggregateStatus AggregateState(
    const RecommendationCampaignStatusSnapshot& result)
{
    const int count = static_cast<int>(result.members.size());
    if (result.inconsistentCount > 0)
        return RecommendationCampaignAggregateStatus::inconsistent;
    if (result.executedCount == 0)
        return RecommendationCampaignAggregateStatus::notExecuted;
    if (result.executedCount < count)
        return RecommendationCampaignAggregateStatus::partiallyExecuted;
    if (result.activatedCount < count)
        return result.activatedCount == 0
            ? RecommendationCampaignAggregateStatus::executedPaused
            : RecommendationCampaignAggregateStatus::partiallyActivated;
    if (result.terminalCount == count)
    {
        if (result.completedCount == count)
            return RecommendationCampaignAggregateStatus::completed;
        if (result.cancelledCount == count)
            return RecommendationCampaignAggregateStatus::cancelled;
        if (result.failedCount > 0 && result.cancelledCount == 0)
            return RecommendationCampaignAggregateStatus::completedWithFailures;
        return RecommendationCampaignAggregateStatus::mixedTerminal;
    }
    if (result.runningCount > 0)
        return RecommendationCampaignAggregateStatus::inProgress;
    if (result.pendingCount > 0)
        return RecommendationCampaignAggregateStatus::queued;
    return RecommendationCampaignAggregateStatus::executedPaused;
}

} // namespace

RecommendationCampaignStatusRequest NormalizeRecommendationCampaignStatusRequest(
    const RecommendationCampaignStatusRequest& request)
{
    if (request.materializationId <= 0)
        throw std::invalid_argument("campaign_status_materialization_id_invalid");
    return request;
}

RecommendationCampaignStatusSnapshot BuildRecommendationCampaignStatusSnapshot(
    const RecommendationCampaignStatusRequest& request,
    RecommendationCampaignStatusInput input)
{
    RecommendationCampaignStatusSnapshot result;
    result.request = NormalizeRecommendationCampaignStatusRequest(request);
    if (input.materializationId != result.request.materializationId ||
        input.materializationContractVersion !=
            kRecommendationCampaignMaterializationContractVersion ||
        input.materializationIdentityCanonical.empty() ||
        input.materializationIdentityHash !=
            RecommendationCanonicalHash(input.materializationIdentityCanonical))
        throw std::invalid_argument("campaign_status_materialization_identity_invalid");
    if (input.selectedMemberCount <= 0 || input.members.empty())
        throw std::invalid_argument("campaign_status_zero_members");
    if (input.selectedMemberCount != static_cast<int>(input.members.size()))
        throw std::invalid_argument("campaign_status_member_count_mismatch");
    if (input.members.size() >
        static_cast<std::size_t>(kMaximumRecommendationCampaignStatusMembers))
        throw std::invalid_argument("campaign_status_member_limit_exceeded");
    if (input.observedAt.empty())
        throw std::invalid_argument("campaign_status_observed_at_required");

    std::sort(input.members.begin(), input.members.end(),
        [](const auto& left, const auto& right)
        {
            return left.memberOrdinal < right.memberOrdinal;
        });
    std::set<long long> memberIds;
    std::set<long long> proposalIds;
    result.members.reserve(input.members.size());
    for (std::size_t index = 0; index < input.members.size(); ++index)
    {
        const auto& member = input.members[index];
        if (member.memberOrdinal != static_cast<int>(index) + 1 ||
            member.materializationMemberId <= 0 || member.recommendationId <= 0 ||
            member.sourceExperimentId <= 0 || member.rankingMemberId <= 0 ||
            member.proposalId <= 0 ||
            !memberIds.insert(member.materializationMemberId).second ||
            !proposalIds.insert(member.proposalId).second)
            throw std::invalid_argument("campaign_status_materialization_members_invalid");
        result.members.push_back(ClassifyMember(member));
    }

    result.materializationContractVersion = input.materializationContractVersion;
    result.materializationIdentityHash = input.materializationIdentityHash;
    result.observedAt = input.observedAt;
    long long totalCurrent = 0;
    long long totalTarget = 0;
    bool allCurrentEpochsAvailable = true;
    bool allTargetEpochsAvailable = true;
    for (const auto& member : result.members)
    {
        if (member.executionState == RecommendationCampaignStatusExecutionState::executed)
            ++result.executedCount;
        if (member.activationState == RecommendationCampaignStatusActivationState::activated)
            ++result.activatedCount;
        switch (member.operationalState)
        {
            case RecommendationCampaignStatusOperationalState::proposalOnly:
                ++result.proposalOnlyCount; break;
            case RecommendationCampaignStatusOperationalState::paused:
            case RecommendationCampaignStatusOperationalState::pausedAfterProgress:
                ++result.pausedCount; break;
            case RecommendationCampaignStatusOperationalState::pending:
                ++result.pendingCount; break;
            case RecommendationCampaignStatusOperationalState::runningTrain:
                ++result.runningCount; ++result.runningTrainCount; break;
            case RecommendationCampaignStatusOperationalState::runningInfer:
                ++result.runningCount; ++result.runningInferCount; break;
            case RecommendationCampaignStatusOperationalState::runningAnalyze:
                ++result.runningCount; ++result.runningAnalyzeCount; break;
            case RecommendationCampaignStatusOperationalState::completed:
                ++result.completedCount; break;
            case RecommendationCampaignStatusOperationalState::failed:
                ++result.failedCount; break;
            case RecommendationCampaignStatusOperationalState::cancelled:
                ++result.cancelledCount; break;
            case RecommendationCampaignStatusOperationalState::inconsistent:
                ++result.inconsistentCount; break;
        }
        if (member.terminalResult != RecommendationCampaignStatusTerminalResult::notTerminal &&
            member.terminalResult != RecommendationCampaignStatusTerminalResult::unknown)
            ++result.terminalCount;
        if (member.workerPid &&
            (member.operationalState ==
                 RecommendationCampaignStatusOperationalState::runningTrain ||
             member.operationalState ==
                 RecommendationCampaignStatusOperationalState::runningInfer ||
             member.operationalState ==
                 RecommendationCampaignStatusOperationalState::runningAnalyze))
            ++result.activeWorkerCount;
        if (member.currentEpoch)
        {
            totalCurrent += *member.currentEpoch;
            result.minimumCurrentEpoch = result.minimumCurrentEpoch
                ? std::min(*result.minimumCurrentEpoch, *member.currentEpoch)
                : member.currentEpoch;
            result.maximumCurrentEpoch = result.maximumCurrentEpoch
                ? std::max(*result.maximumCurrentEpoch, *member.currentEpoch)
                : member.currentEpoch;
        }
        else allCurrentEpochsAvailable = false;
        if (member.targetEpochs) totalTarget += *member.targetEpochs;
        else allTargetEpochsAvailable = false;
    }
    result.nonterminalCount = static_cast<int>(result.members.size()) -
        result.terminalCount;
    if (allCurrentEpochsAvailable) result.totalCurrentEpochs = totalCurrent;
    if (allTargetEpochsAvailable) result.totalTargetEpochs = totalTarget;
    const double denominator = static_cast<double>(result.members.size());
    result.completedPercent = 100.0 * result.completedCount / denominator;
    result.terminalPercent = 100.0 * result.terminalCount / denominator;
    if (result.inconsistentCount > 0)
    {
        result.consistency = RecommendationCampaignStatusConsistency::inconsistent;
        result.diagnosticCodes.push_back("campaign_status_member_inconsistent");
    }
    else result.diagnosticCodes.push_back("campaign_status_snapshot_consistent");
    result.aggregateStatus = AggregateState(result);
    result.snapshotIdentityCanonical = SnapshotCanonical(input, result.members);
    result.snapshotIdentityHash = RecommendationCanonicalHash(
        result.snapshotIdentityCanonical);
    return result;
}

#define EA_STATUS_TEXT_FUNCTION(name, type, cases) \
std::string name(type value) { switch (value) { cases } throw std::invalid_argument(#name "_invalid"); }

EA_STATUS_TEXT_FUNCTION(RecommendationCampaignStatusExecutionStateText,
    RecommendationCampaignStatusExecutionState,
    case RecommendationCampaignStatusExecutionState::notExecuted: return "not_executed";
    case RecommendationCampaignStatusExecutionState::executed: return "executed";
    case RecommendationCampaignStatusExecutionState::malformed: return "malformed";
    case RecommendationCampaignStatusExecutionState::conflicting: return "conflicting";)
EA_STATUS_TEXT_FUNCTION(RecommendationCampaignStatusActivationStateText,
    RecommendationCampaignStatusActivationState,
    case RecommendationCampaignStatusActivationState::notActivated: return "not_activated";
    case RecommendationCampaignStatusActivationState::activated: return "activated";
    case RecommendationCampaignStatusActivationState::malformed: return "malformed";
    case RecommendationCampaignStatusActivationState::conflicting: return "conflicting";)
EA_STATUS_TEXT_FUNCTION(RecommendationCampaignStatusOperationalStateText,
    RecommendationCampaignStatusOperationalState,
    case RecommendationCampaignStatusOperationalState::proposalOnly: return "proposal_only";
    case RecommendationCampaignStatusOperationalState::paused: return "paused";
    case RecommendationCampaignStatusOperationalState::pending: return "pending";
    case RecommendationCampaignStatusOperationalState::runningTrain: return "running_train";
    case RecommendationCampaignStatusOperationalState::runningInfer: return "running_infer";
    case RecommendationCampaignStatusOperationalState::runningAnalyze: return "running_analyze";
    case RecommendationCampaignStatusOperationalState::completed: return "completed";
    case RecommendationCampaignStatusOperationalState::failed: return "failed";
    case RecommendationCampaignStatusOperationalState::cancelled: return "cancelled";
    case RecommendationCampaignStatusOperationalState::pausedAfterProgress: return "paused_after_progress";
    case RecommendationCampaignStatusOperationalState::inconsistent: return "inconsistent";)
EA_STATUS_TEXT_FUNCTION(RecommendationCampaignStatusTerminalResultText,
    RecommendationCampaignStatusTerminalResult,
    case RecommendationCampaignStatusTerminalResult::notTerminal: return "not_terminal";
    case RecommendationCampaignStatusTerminalResult::succeeded: return "succeeded";
    case RecommendationCampaignStatusTerminalResult::failed: return "failed";
    case RecommendationCampaignStatusTerminalResult::cancelled: return "cancelled";
    case RecommendationCampaignStatusTerminalResult::unknown: return "unknown";)
EA_STATUS_TEXT_FUNCTION(RecommendationCampaignStatusConsistencyText,
    RecommendationCampaignStatusConsistency,
    case RecommendationCampaignStatusConsistency::consistent: return "consistent";
    case RecommendationCampaignStatusConsistency::inconsistent: return "inconsistent";)
EA_STATUS_TEXT_FUNCTION(RecommendationCampaignAggregateStatusText,
    RecommendationCampaignAggregateStatus,
    case RecommendationCampaignAggregateStatus::notExecuted: return "not_executed";
    case RecommendationCampaignAggregateStatus::partiallyExecuted: return "partially_executed";
    case RecommendationCampaignAggregateStatus::executedPaused: return "executed_paused";
    case RecommendationCampaignAggregateStatus::partiallyActivated: return "partially_activated";
    case RecommendationCampaignAggregateStatus::queued: return "queued";
    case RecommendationCampaignAggregateStatus::inProgress: return "in_progress";
    case RecommendationCampaignAggregateStatus::completed: return "completed";
    case RecommendationCampaignAggregateStatus::completedWithFailures: return "completed_with_failures";
    case RecommendationCampaignAggregateStatus::cancelled: return "cancelled";
    case RecommendationCampaignAggregateStatus::mixedTerminal: return "mixed_terminal";
    case RecommendationCampaignAggregateStatus::inconsistent: return "inconsistent";)

#undef EA_STATUS_TEXT_FUNCTION

} // namespace EA::ExperimentRecommendation
