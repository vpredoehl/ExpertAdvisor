#include "ExperimentRecommendationCampaignStatusService.hpp"

#include "ExperimentRecommendationCampaignStatusRepository.hpp"
#include "ExperimentRecommendationService.hpp"

#include <iomanip>
#include <locale>
#include <ostream>

namespace EA::ExperimentRecommendation
{
namespace
{

template <typename Value>
void PrintOptional(std::ostream& output, const std::optional<Value>& value)
{
    if (value) output << *value;
    else output << "null";
}

void PrintOptionalText(
    std::ostream& output,
    const std::optional<std::string>& value)
{
    if (value) output << RecommendationMachineText(*value);
    else output << "null";
}

void PrintOptionalPercent(
    std::ostream& output,
    const std::optional<double>& value)
{
    if (!value)
    {
        output << "null";
        return;
    }
    const auto flags = output.flags();
    const auto precision = output.precision();
    output.imbue(std::locale::classic());
    output << std::fixed << std::setprecision(6) << *value;
    output.flags(flags);
    output.precision(precision);
}

void PrintDiagnostics(
    std::ostream& output,
    const std::vector<std::string>& diagnostics)
{
    if (diagnostics.empty())
    {
        output << "none";
        return;
    }
    for (std::size_t index = 0; index < diagnostics.size(); ++index)
    {
        if (index != 0) output << '|';
        output << RecommendationMachineText(diagnostics[index]);
    }
}

void PrintSafety(std::ostream& output)
{
    output << "transaction_read_only=true,snapshot_isolation=repeatable_read,"
              "advisory_locks_acquired=0,row_locks_acquired=0,rows_inserted=0,"
              "rows_updated=0,rows_deleted=0,sequences_advanced=0,"
              "scheduler_started=false,scheduler_signaled=false,"
              "scheduler_polled=false,workers_launched=false,"
              "automatic_follow_up=false,campaign_state_row_created=false,"
              "snapshot_advisory=true";
}

void PrintSnapshot(
    std::ostream& output,
    const RecommendationCampaignStatusSnapshot& snapshot)
{
    output << "RECOMMENDATION_CAMPAIGN_STATUS"
           << ",materialization_id=" << snapshot.request.materializationId
           << ",materialization_contract_version="
           << snapshot.materializationContractVersion
           << ",materialization_identity_hash="
           << RecommendationMachineText(snapshot.materializationIdentityHash)
           << ",status_contract_version="
           << kRecommendationCampaignStatusContractVersion
           << ",snapshot_identity_hash="
           << RecommendationMachineText(snapshot.snapshotIdentityHash)
           << ",observed_at=" << RecommendationMachineText(snapshot.observedAt)
           << ",aggregate_status="
           << RecommendationCampaignAggregateStatusText(snapshot.aggregateStatus)
           << ",consistency_status="
           << RecommendationCampaignStatusConsistencyText(snapshot.consistency)
           << ",diagnostic=";
    PrintDiagnostics(output, snapshot.diagnosticCodes);
    output << ",member_count=" << snapshot.members.size()
           << ",proposal_only_count=" << snapshot.proposalOnlyCount
           << ",executed_count=" << snapshot.executedCount
           << ",activated_count=" << snapshot.activatedCount
           << ",paused_count=" << snapshot.pausedCount
           << ",pending_count=" << snapshot.pendingCount
           << ",running_count=" << snapshot.runningCount
           << ",running_train_count=" << snapshot.runningTrainCount
           << ",running_infer_count=" << snapshot.runningInferCount
           << ",running_analyze_count=" << snapshot.runningAnalyzeCount
           << ",completed_count=" << snapshot.completedCount
           << ",failed_count=" << snapshot.failedCount
           << ",cancelled_count=" << snapshot.cancelledCount
           << ",inconsistent_count=" << snapshot.inconsistentCount
           << ",terminal_count=" << snapshot.terminalCount
           << ",nonterminal_count=" << snapshot.nonterminalCount
           << ",active_worker_count=" << snapshot.activeWorkerCount
           << ",minimum_current_epoch=";
    PrintOptional(output, snapshot.minimumCurrentEpoch);
    output << ",maximum_current_epoch=";
    PrintOptional(output, snapshot.maximumCurrentEpoch);
    output << ",total_current_epochs=";
    PrintOptional(output, snapshot.totalCurrentEpochs);
    output << ",total_target_epochs=";
    PrintOptional(output, snapshot.totalTargetEpochs);
    output << ",completed_percent=";
    PrintOptionalPercent(output, snapshot.completedPercent);
    output << ",terminal_percent=";
    PrintOptionalPercent(output, snapshot.terminalPercent);
    output << ',';
    PrintSafety(output);
    output << '\n';

    for (const auto& member : snapshot.members)
    {
        output << "RECOMMENDATION_CAMPAIGN_STATUS_MEMBER"
               << ",materialization_id=" << snapshot.request.materializationId
               << ",member_ordinal=" << member.memberOrdinal
               << ",materialization_member_id=" << member.materializationMemberId
               << ",recommendation_id=" << member.recommendationId
               << ",source_experiment_id=" << member.sourceExperimentId
               << ",ranking_member_id=" << member.rankingMemberId
               << ",proposal_id=" << member.proposalId
               << ",current_review_decision_id=";
        PrintOptional(output, member.currentReviewDecisionId);
        output << ",current_review_decision=";
        PrintOptionalText(output, member.currentReviewDecision);
        output << ",authorization_review_decision_id=";
        PrintOptional(output, member.authorizationReviewDecisionId);
        output << ",execution_id=";
        PrintOptional(output, member.executionId);
        output << ",activation_id=";
        PrintOptional(output, member.activationId);
        output << ",experiment_id=";
        PrintOptional(output, member.experimentId);
        output << ",execution_evidence_count=" << member.executionEvidenceCount
               << ",activation_evidence_count=" << member.activationEvidenceCount;
        output << ",symbol=";
        PrintOptionalText(output, member.symbol);
        output << ",prediction_horizon=";
        PrintOptional(output, member.predictionHorizon);
        output << ",execution_state="
               << RecommendationCampaignStatusExecutionStateText(
                      member.executionState)
               << ",activation_state="
               << RecommendationCampaignStatusActivationStateText(
                      member.activationState)
               << ",operational_state="
               << RecommendationCampaignStatusOperationalStateText(
                      member.operationalState)
               << ",terminal_result="
               << RecommendationCampaignStatusTerminalResultText(
                      member.terminalResult)
               << ",status=";
        PrintOptionalText(output, member.status);
        output << ",phase=";
        PrintOptionalText(output, member.phase);
        output << ",current_epoch=";
        PrintOptional(output, member.currentEpoch);
        output << ",target_epochs=";
        PrintOptional(output, member.targetEpochs);
        output << ",worker_pid=";
        PrintOptional(output, member.workerPid);
        output << ",current_operation=";
        PrintOptionalText(output, member.currentOperation);
        output << ",worker_started_at=";
        PrintOptionalText(output, member.workerStartedAt);
        output << ",started_at=";
        PrintOptionalText(output, member.startedAt);
        output << ",completed_at=";
        PrintOptionalText(output, member.completedAt);
        output << ",exit_code=";
        PrintOptional(output, member.exitCode);
        output << ",error_message=";
        PrintOptionalText(output, member.errorMessage);
        output << ",model_id=";
        PrintOptional(output, member.modelId);
        output << ",invocation_identity_hash=";
        PrintOptionalText(output, member.invocationIdentityHash);
        output << ",model_link_count=" << member.modelLinkCount
               << ",final_inference_result_count="
               << member.finalInferenceResultCount
               << ",completed_final_inference_result_count="
               << member.completedFinalInferenceResultCount
               << ",failed_final_inference_result_count="
               << member.failedFinalInferenceResultCount
               << ",final_analysis_result_count="
               << member.finalAnalysisResultCount
               << ",completed_final_analysis_result_count="
               << member.completedFinalAnalysisResultCount
               << ",failed_final_analysis_result_count="
               << member.failedFinalAnalysisResultCount;
        output << ",train_complete=" << (member.trainComplete ? "true" : "false")
               << ",infer_configured=" << (member.inferConfigured ? "true" : "false")
               << ",infer_complete=" << (member.inferComplete ? "true" : "false")
               << ",analyze_configured=" << (member.analyzeConfigured ? "true" : "false")
               << ",analyze_complete=" << (member.analyzeComplete ? "true" : "false")
               << ",consistency_status="
               << RecommendationCampaignStatusConsistencyText(member.consistency)
               << ",diagnostic=";
        PrintDiagnostics(output, member.diagnosticCodes);
        output << ',';
        PrintSafety(output);
        output << '\n';
    }
}

} // namespace

int RunRecommendationCampaignStatusCommand(
    const std::string& connectionString,
    const RecommendationCampaignStatusRequest& request,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        const auto normalized = NormalizeRecommendationCampaignStatusRequest(request);
        pqxx::connection connection{connectionString};
        PrintSnapshot(
            output, ReadRecommendationCampaignStatusSnapshot(connection, normalized));
        return 0;
    }
    catch (const std::exception& error)
    {
        errors << "RECOMMENDATION_CAMPAIGN_STATUS_FAILED"
               << ",materialization_id=" << request.materializationId
               << ",materialization_contract_version=null,"
                  "materialization_identity_hash=null,status_contract_version="
               << kRecommendationCampaignStatusContractVersion
               << ",snapshot_identity_hash=null,observed_at=null,"
                  "aggregate_status=inconsistent,consistency_status=inconsistent,"
                  "diagnostic="
               << RecommendationMachineText(error.what())
               << ",member_count=0,proposal_only_count=0,executed_count=0,"
                  "activated_count=0,paused_count=0,pending_count=0,"
                  "running_count=0,running_train_count=0,running_infer_count=0,"
                  "running_analyze_count=0,completed_count=0,failed_count=0,"
                  "cancelled_count=0,inconsistent_count=0,terminal_count=0,"
                  "nonterminal_count=0,active_worker_count=0,"
                  "minimum_current_epoch=null,maximum_current_epoch=null,"
                  "total_current_epochs=null,total_target_epochs=null,"
                  "completed_percent=null,terminal_percent=null,";
        PrintSafety(errors);
        errors << '\n';
        return 1;
    }
}

} // namespace EA::ExperimentRecommendation
