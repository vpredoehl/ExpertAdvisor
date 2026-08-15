#include "ExperimentRecommendationCampaignStatusRepository.hpp"

#include "ExperimentRecommendationRepository.hpp"
#include "ExperimentRecommendationCampaignMaterializationRepository.hpp"
#include "ExperimentRecommendationConversionRepository.hpp"
#include "ExperimentRecommendationConversionActivation.hpp"
#include "ExperimentRecommendationConversionExecutionRepository.hpp"
#include "ExperimentRecommendationConversionWorkflowRepository.hpp"
#include "ExperimentCurrentOperation.hpp"

#include <algorithm>
#include <map>
#include <sstream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

std::optional<std::string> OptionalText(
    const pqxx::row& row,
    const char* column)
{
    if (row[column].is_null()) return std::nullopt;
    return row[column].as<std::string>();
}

template <typename Value>
std::optional<Value> OptionalValue(
    const pqxx::row& row,
    const char* column)
{
    if (row[column].is_null()) return std::nullopt;
    return row[column].as<Value>();
}

std::map<long long, RecommendationCampaignStatusExperimentEvidence>
LoadExperimentEvidence(
    pqxx::transaction_base& transaction,
    const std::vector<long long>& experimentIds,
    const std::map<long long, std::string>& expectedInvocationByExperiment)
{
    if (experimentIds.empty()) return {};
    std::vector<long long> ids = experimentIds;
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
        throw std::runtime_error("campaign_status_duplicate_experiment_link");

    pqxx::params parameters;
    std::ostringstream placeholders;
    for (std::size_t index = 0; index < ids.size(); ++index)
    {
        if (index != 0) placeholders << ',';
        placeholders << '$' << index + 1;
        parameters.append(ids[index]);
    }
    const std::size_t timeZoneParameter = ids.size() + 1;
    parameters.append(kRecommendationDateTimeZone);
    const std::string query = std::string{R"SQL(
SELECT e.experiment_id,e.symbol,e.prediction_horizon,e.status,e.phase,
       e.target_epochs,e.current_epoch,e.worker_pid,e.current_operation,
       e.worker_started_at::text AS worker_started_at,
       e.started_at::text AS started_at,e.completed_at::text AS completed_at,
       e.exit_code,e.error_message,e.last_model_id,e.invocation_mode,
       e.duplicate_nonce,e.c_next_threshold,e.core_lr_mult,e.head_lr_mult,
       e.checkpoint_interval,e.donchian20_mode,
       to_char(e.train_start AT TIME ZONE )SQL"} +
        "$" + std::to_string(timeZoneParameter) +
        R"SQL(,'YYYY-MM-DD') AS train_start,
       to_char(e.train_end AT TIME ZONE )SQL" +
        "$" + std::to_string(timeZoneParameter) +
        R"SQL(,'YYYY-MM-DD') AS train_end,
       to_char(e.infer_start AT TIME ZONE )SQL" +
        "$" + std::to_string(timeZoneParameter) +
        R"SQL(,'YYYY-MM-DD') AS infer_start,
       to_char(e.infer_end AT TIME ZONE )SQL" +
        "$" + std::to_string(timeZoneParameter) +
        R"SQL(,'YYYY-MM-DD') AS infer_end,e.resume_model_id,
       e.infer_start IS NOT NULL AS infer_start_present,
       e.infer_end IS NOT NULL AS infer_end_present,
       (e.infer_start IS NOT NULL AND e.infer_end IS NOT NULL) AS infer_configured,
       CASE WHEN e.last_model_id IS NULL THEN 0 ELSE
         (SELECT count(*) FROM model mo
           WHERE mo.model_id=e.last_model_id AND mo.experiment_id=e.experiment_id)
       END AS model_link_count,
       CASE WHEN e.last_model_id IS NULL THEN 0 ELSE
         (SELECT count(*) FROM inference_eval_result ir
           WHERE ir.model_id=e.last_model_id AND ir.inference_scope='final'
             AND ir.checkpoint_eval_id IS NULL AND ir.symbol=e.symbol
             AND ir.prediction_horizon=e.prediction_horizon
             AND ir.threshold_logret=e.c_next_threshold
             AND ir.from_date=to_char(e.infer_start AT TIME ZONE )SQL" +
        "$" + std::to_string(timeZoneParameter) +
        R"SQL(,'YYYY-MM-DD')
             AND ir.to_date=to_char(e.infer_end AT TIME ZONE )SQL" +
        "$" + std::to_string(timeZoneParameter) +
        R"SQL(,'YYYY-MM-DD'))
       END AS inference_result_count,
       CASE WHEN e.last_model_id IS NULL THEN 0 ELSE
         (SELECT count(*) FROM inference_eval_result ir
           WHERE ir.model_id=e.last_model_id AND ir.inference_scope='final'
             AND ir.checkpoint_eval_id IS NULL AND ir.symbol=e.symbol
             AND ir.prediction_horizon=e.prediction_horizon
             AND ir.threshold_logret=e.c_next_threshold
             AND ir.from_date=to_char(e.infer_start AT TIME ZONE )SQL" +
        "$" + std::to_string(timeZoneParameter) +
        R"SQL(,'YYYY-MM-DD')
             AND ir.to_date=to_char(e.infer_end AT TIME ZONE )SQL" +
        "$" + std::to_string(timeZoneParameter) +
        R"SQL(,'YYYY-MM-DD')
             AND ir.status='completed')
       END AS completed_inference_result_count,
       CASE WHEN e.last_model_id IS NULL THEN 0 ELSE
         (SELECT count(*) FROM inference_eval_result ir
           WHERE ir.model_id=e.last_model_id AND ir.inference_scope='final'
             AND ir.checkpoint_eval_id IS NULL AND ir.symbol=e.symbol
             AND ir.prediction_horizon=e.prediction_horizon
             AND ir.threshold_logret=e.c_next_threshold
             AND ir.from_date=to_char(e.infer_start AT TIME ZONE )SQL" +
        "$" + std::to_string(timeZoneParameter) +
        R"SQL(,'YYYY-MM-DD')
             AND ir.to_date=to_char(e.infer_end AT TIME ZONE )SQL" +
        "$" + std::to_string(timeZoneParameter) +
        R"SQL(,'YYYY-MM-DD')
             AND ir.status='failed')
       END AS failed_inference_result_count,
       CASE WHEN e.last_model_id IS NULL THEN
         (SELECT count(*) FROM experiment_analysis_result ar
           WHERE ar.experiment_id=e.experiment_id AND ar.analysis_scope='final')
       ELSE
         (SELECT count(*) FROM experiment_analysis_result ar
           WHERE ar.experiment_id=e.experiment_id AND ar.model_id=e.last_model_id
             AND ar.analysis_scope='final') END
         AS analysis_result_count,
       CASE WHEN e.last_model_id IS NULL THEN 0 ELSE
         (SELECT count(*) FROM experiment_analysis_result ar
           WHERE ar.experiment_id=e.experiment_id AND ar.model_id=e.last_model_id
             AND ar.analysis_scope='final' AND ar.analysis_status='completed') END
         AS completed_analysis_result_count,
       CASE WHEN e.last_model_id IS NULL THEN 0 ELSE
         (SELECT count(*) FROM experiment_analysis_result ar
           WHERE ar.experiment_id=e.experiment_id AND ar.model_id=e.last_model_id
             AND ar.analysis_scope='final' AND ar.analysis_status='failed') END
         AS failed_analysis_result_count
FROM experiment e
WHERE e.experiment_id IN ()SQL" + placeholders.str() +
        ") ORDER BY e.experiment_id;";
    const pqxx::result rows = transaction.exec(query, parameters);

    std::map<long long, RecommendationCampaignStatusExperimentEvidence> result;
    for (const auto& row : rows)
    {
        RecommendationCampaignStatusExperimentEvidence value;
        value.experimentId = row["experiment_id"].as<long long>();
        value.symbol = row["symbol"].as<std::string>();
        value.predictionHorizon = row["prediction_horizon"].as<int>();
        value.status = row["status"].as<std::string>();
        value.phase = row["phase"].as<std::string>();
        value.targetEpochs = row["target_epochs"].as<int>();
        value.currentEpoch = OptionalValue<int>(row, "current_epoch");
        value.workerPid = OptionalValue<int>(row, "worker_pid");
        value.currentOperation =
            EA::ExperimentLifecycle::NormalizeOptionalPersistedCurrentOperation(
                OptionalText(row, "current_operation"));
        value.workerStartedAt = OptionalText(row, "worker_started_at");
        value.startedAt = OptionalText(row, "started_at");
        value.completedAt = OptionalText(row, "completed_at");
        value.exitCode = OptionalValue<int>(row, "exit_code");
        value.errorMessage = OptionalText(row, "error_message");
        value.modelId = OptionalValue<long long>(row, "last_model_id");
        value.invocationMode = row["invocation_mode"].is_null()
            ? std::string{} : row["invocation_mode"].as<std::string>();
        value.duplicateNonce = row["duplicate_nonce"].as<long long>();
        ExperimentInvocationConfiguration invocation;
        invocation.configuration.symbol = value.symbol;
        invocation.configuration.predictionHorizon = value.predictionHorizon;
        invocation.configuration.labelThreshold =
            row["c_next_threshold"].as<double>();
        invocation.configuration.coreLrMult =
            OptionalValue<double>(row, "core_lr_mult");
        invocation.configuration.headLrMult =
            OptionalValue<double>(row, "head_lr_mult");
        invocation.configuration.targetEpochs = value.targetEpochs;
        invocation.configuration.trainStartDate =
            row["train_start"].as<std::string>();
        invocation.configuration.trainEndDate = row["train_end"].as<std::string>();
        invocation.configuration.inferStartDate = OptionalText(row, "infer_start");
        invocation.configuration.inferEndDate = OptionalText(row, "infer_end");
        invocation.configuration.donchian20Mode = ParseDonchian20Mode(
            row["donchian20_mode"].as<std::string>());
        invocation.checkpointInterval = row["checkpoint_interval"].as<int>();
        invocation.resumeModelId = OptionalValue<long long>(row, "resume_model_id");
        const auto expected = expectedInvocationByExperiment.find(value.experimentId);
        if (expected == expectedInvocationByExperiment.end())
        {
            value.invocationIdentityCanonical =
                BuildRecommendationInvocationIdentity(invocation).canonicalText;
            value.invocationProvenanceValid = false;
        }
        else
        {
            const std::string semanticCanonical =
                RecommendationSemanticConfigurationFromInvocationCanonicalText(
                    expected->second);
            const auto semanticVersion =
                RecommendationSemanticConfigurationVersionFromCanonicalText(
                    semanticCanonical);
            if (!semanticVersion)
                throw std::runtime_error(
                    "invalid_persisted_recommendation_semantic_configuration_version");
            value.invocationIdentityCanonical =
                BuildRecommendationInvocationIdentity(invocation, *semanticVersion).canonicalText;
            value.invocationProvenanceValid =
                expected->second == value.invocationIdentityCanonical;
        }
        value.inferStartPresent = row["infer_start_present"].as<bool>();
        value.inferEndPresent = row["infer_end_present"].as<bool>();
        value.inferConfigured = row["infer_configured"].as<bool>();
        value.modelLinkCount = row["model_link_count"].as<int>();
        value.inferenceResultCount = row["inference_result_count"].as<int>();
        value.completedInferenceResultCount =
            row["completed_inference_result_count"].as<int>();
        value.failedInferenceResultCount =
            row["failed_inference_result_count"].as<int>();
        value.analysisResultCount = row["analysis_result_count"].as<int>();
        value.completedAnalysisResultCount =
            row["completed_analysis_result_count"].as<int>();
        value.failedAnalysisResultCount =
            row["failed_analysis_result_count"].as<int>();
        if (!result.emplace(value.experimentId, std::move(value)).second)
            throw std::runtime_error("campaign_status_duplicate_experiment_row");
    }
    return result;
}

RecommendationCampaignStatusInput LoadInput(
    pqxx::transaction_base& transaction,
    const PersistedRecommendationCampaignMaterialization& materialization)
{
    RecommendationCampaignStatusInput input;
    input.materializationId = materialization.materializationId;
    input.materializationContractVersion = materialization.contractVersion;
    input.materializationIdentityCanonical = materialization.identityCanonical;
    input.materializationIdentityHash = materialization.identityHash;
    input.selectedMemberCount = materialization.selectedMemberCount;
    input.observedAt = transaction.exec(
        "SELECT transaction_timestamp()::text;").one_row()[0].as<std::string>();

    std::vector<long long> proposalIds;
    proposalIds.reserve(materialization.members.size());
    for (const auto& member : materialization.members)
        proposalIds.push_back(member.conversionProposalId);
    const auto workflows = ListRecommendationConversionWorkflowsForProposals(
        transaction, proposalIds);
    const auto proposals = ListRecommendationConversionProposalsByIds(
        transaction, proposalIds);
    std::map<long long, const PersistedRecommendationConversionProposal*> byProposalEvidence;
    for (const auto& proposal : proposals)
        byProposalEvidence.emplace(proposal.proposalId, &proposal);
    std::map<long long, const RecommendationConversionWorkflowView*> byProposal;
    std::vector<long long> experimentIds;
    std::map<long long, std::string> expectedInvocationByExperiment;
    for (const auto& workflow : workflows)
    {
        if (!byProposal.emplace(workflow.proposalId, &workflow).second)
            throw std::runtime_error("campaign_status_duplicate_workflow");
        if (workflow.execution)
        {
            experimentIds.push_back(workflow.execution->experimentId);
            const auto proposal = byProposalEvidence.find(workflow.proposalId);
            if (proposal != byProposalEvidence.end())
                expectedInvocationByExperiment.emplace(
                    workflow.execution->experimentId,
                    proposal->second->proposal.proposedInvocationCanonical);
        }
    }
    const auto experiments = LoadExperimentEvidence(
        transaction, experimentIds, expectedInvocationByExperiment);

    input.members.reserve(materialization.members.size());
    for (const auto& stored : materialization.members)
    {
        RecommendationCampaignStatusMemberInput member;
        member.memberOrdinal = stored.memberOrdinal;
        member.materializationMemberId = stored.materializationMemberId;
        member.recommendationId = stored.recommendationId;
        member.sourceExperimentId = stored.sourceExperimentId;
        member.rankingMemberId = stored.rankingMemberId;
        member.proposalId = stored.conversionProposalId;
        const auto found = byProposal.find(stored.conversionProposalId);
        if (found == byProposal.end())
        {
            member.workflowConsistent = false;
            member.workflowDiagnostics.push_back("campaign_status_proposal_missing");
            input.members.push_back(std::move(member));
            continue;
        }
        const auto& workflow = *found->second;
        const auto proposal = byProposalEvidence.find(stored.conversionProposalId);
        if (workflow.recommendationId != stored.recommendationId ||
            workflow.sourceExperimentId != stored.sourceExperimentId ||
            workflow.proposalIdentityCanonical != stored.proposalIdentityCanonical ||
            workflow.proposalIdentityHash != stored.proposalIdentityHash ||
            proposal == byProposalEvidence.end() ||
            proposal->second->proposal.recommendationId != stored.recommendationId ||
            proposal->second->proposal.sourceExperimentId != stored.sourceExperimentId ||
            proposal->second->proposal.conversionIdentityCanonical !=
                stored.proposalIdentityCanonical ||
            proposal->second->proposal.conversionIdentityHash !=
                stored.proposalIdentityHash)
        {
            member.workflowConsistent = false;
            member.workflowDiagnostics.push_back(
                "campaign_status_proposal_provenance_invalid");
        }
        member.workflowConsistent = member.workflowConsistent &&
            workflow.derivation.integrity ==
                RecommendationConversionWorkflowIntegrity::consistent;
        member.workflowDiagnostics.insert(
            member.workflowDiagnostics.end(),
            workflow.derivation.diagnosticCodes.begin(),
            workflow.derivation.diagnosticCodes.end());
        member.executionCount = workflow.executionCount;
        member.activationCount = workflow.activationCount;
        if (workflow.latestReview)
        {
            member.currentReviewDecisionId = workflow.latestReview->reviewDecisionId;
            member.currentReviewDecision = workflow.latestReview->decision;
        }
        if (workflow.execution)
        {
            member.authorizationReviewDecisionId = workflow.execution->reviewDecisionId;
            member.executionId = workflow.execution->executionId;
            member.experimentId = workflow.execution->experimentId;
            if (proposal != byProposalEvidence.end())
            {
                const std::string expectedExecutionIdentity =
                    BuildRecommendationConversionExecutionIdentityCanonical(
                        *proposal->second, workflow.execution->reviewDecisionId);
                if (workflow.execution->identityCanonical !=
                        expectedExecutionIdentity ||
                    workflow.execution->identityHash !=
                        RecommendationCanonicalHash(expectedExecutionIdentity))
                {
                    member.workflowConsistent = false;
                    member.workflowDiagnostics.push_back(
                        "campaign_status_execution_identity_mismatch");
                }
            }
            const auto detail = experiments.find(workflow.execution->experimentId);
            if (detail != experiments.end()) member.experiment = detail->second;
        }
        if (workflow.activation)
        {
            member.activationId = workflow.activation->activationId;
            if (workflow.execution)
            {
                const auto expectedActivation =
                    BuildRecommendationConversionActivationIdentity({
                        workflow.execution->executionId,
                        workflow.execution->proposalId,
                        workflow.execution->reviewDecisionId,
                        workflow.execution->experimentId,
                        workflow.execution->identityHash});
                if (workflow.activation->identityCanonical !=
                        expectedActivation.canonicalText ||
                    workflow.activation->identityHash != expectedActivation.hash)
                {
                    member.workflowConsistent = false;
                    member.workflowDiagnostics.push_back(
                        "campaign_status_activation_identity_mismatch");
                }
            }
        }
        input.members.push_back(std::move(member));
    }
    return input;
}

} // namespace

bool RecommendationCampaignStatusSchemasExist(
    pqxx::transaction_base& transaction)
{
    return transaction.exec(R"SQL(
SELECT to_regclass('experiment_recommendation_campaign_materialization') IS NOT NULL
   AND to_regclass('experiment_recommendation_campaign_materialization_member') IS NOT NULL
   AND to_regclass('experiment_recommendation_conversion_proposal') IS NOT NULL
   AND to_regclass('experiment_recommendation_conversion_review_decision') IS NOT NULL
   AND to_regclass('experiment_recommendation_conversion_execution') IS NOT NULL
   AND to_regclass('experiment_recommendation_conversion_activation') IS NOT NULL
   AND to_regclass('experiment') IS NOT NULL
   AND to_regclass('model') IS NOT NULL
   AND to_regclass('inference_eval_result') IS NOT NULL
   AND to_regclass('experiment_analysis_result') IS NOT NULL
   AND NOT EXISTS (
     SELECT 1 FROM (VALUES
       ('experiment','experiment_id'),('experiment','symbol'),
       ('experiment','prediction_horizon'),('experiment','status'),
       ('experiment','phase'),('experiment','target_epochs'),
       ('experiment','current_epoch'),('experiment','worker_pid'),
       ('experiment','current_operation'),('experiment','worker_started_at'),
       ('experiment','started_at'),('experiment','completed_at'),
       ('experiment','exit_code'),('experiment','error_message'),
       ('experiment','last_model_id'),('experiment','invocation_mode'),
       ('experiment','duplicate_nonce'),('experiment','infer_start'),
       ('experiment','infer_end'),('experiment','c_next_threshold'),
       ('experiment','core_lr_mult'),('experiment','head_lr_mult'),
       ('experiment','checkpoint_interval'),('experiment','train_start'),
       ('experiment','train_end'),('experiment','resume_model_id'),
       ('model','model_id'),
       ('model','experiment_id'),('inference_eval_result','model_id'),
       ('inference_eval_result','symbol'),
       ('inference_eval_result','prediction_horizon'),
       ('inference_eval_result','threshold_logret'),
       ('inference_eval_result','from_date'),
       ('inference_eval_result','to_date'),
       ('inference_eval_result','status'),
       ('inference_eval_result','inference_scope'),
       ('inference_eval_result','checkpoint_eval_id'),
       ('experiment_analysis_result','experiment_id'),
       ('experiment_analysis_result','model_id'),
       ('experiment_analysis_result','analysis_status'),
       ('experiment_analysis_result','analysis_scope'),
       ('experiment_recommendation_conversion_proposal',
        'recommendation_conversion_proposal_id'),
       ('experiment_recommendation_conversion_proposal','recommendation_id'),
       ('experiment_recommendation_conversion_proposal','source_experiment_id'),
       ('experiment_recommendation_conversion_proposal','conversion_contract_version'),
       ('experiment_recommendation_conversion_proposal','changed_parameter'),
       ('experiment_recommendation_conversion_proposal','source_value_canonical'),
       ('experiment_recommendation_conversion_proposal','proposed_value_canonical'),
       ('experiment_recommendation_conversion_proposal','recommendation_semantic_hash'),
       ('experiment_recommendation_conversion_proposal','evaluation_identity_hash'),
       ('experiment_recommendation_conversion_proposal','evaluation_policy_hash'),
       ('experiment_recommendation_conversion_proposal','scoring_policy_hash'),
       ('experiment_recommendation_conversion_proposal','review_authorization_hash'),
       ('experiment_recommendation_conversion_proposal','ranking_snapshot_identity_hash'),
       ('experiment_recommendation_conversion_proposal','proposed_symbol'),
       ('experiment_recommendation_conversion_proposal','proposed_prediction_horizon'),
       ('experiment_recommendation_conversion_proposal','proposed_label_threshold'),
       ('experiment_recommendation_conversion_proposal','proposed_core_lr_mult'),
       ('experiment_recommendation_conversion_proposal','proposed_head_lr_mult'),
       ('experiment_recommendation_conversion_proposal','proposed_target_epochs'),
       ('experiment_recommendation_conversion_proposal','proposed_train_start_date'),
       ('experiment_recommendation_conversion_proposal','proposed_train_end_date'),
       ('experiment_recommendation_conversion_proposal','proposed_infer_start_date'),
       ('experiment_recommendation_conversion_proposal','proposed_infer_end_date'),
       ('experiment_recommendation_conversion_proposal','proposed_checkpoint_interval'),
       ('experiment_recommendation_conversion_proposal','proposed_resume_model_id'),
       ('experiment_recommendation_conversion_proposal','source_invocation_canonical'),
       ('experiment_recommendation_conversion_proposal','proposed_invocation_canonical'),
       ('experiment_recommendation_conversion_proposal','conversion_identity_canonical'),
       ('experiment_recommendation_conversion_proposal','conversion_identity_hash'),
       ('experiment_recommendation_conversion_proposal','conversion_hash_collision_ordinal'),
       ('experiment_recommendation_conversion_proposal','created_at'),
       ('experiment_recommendation_conversion_review_decision',
        'recommendation_conversion_review_decision_id'),
       ('experiment_recommendation_conversion_review_decision',
        'recommendation_conversion_proposal_id'),
       ('experiment_recommendation_conversion_review_decision','decision'),
       ('experiment_recommendation_conversion_review_decision','decided_at'),
       ('experiment_recommendation_conversion_execution',
        'recommendation_conversion_execution_id'),
       ('experiment_recommendation_conversion_execution',
        'recommendation_conversion_proposal_id'),
       ('experiment_recommendation_conversion_execution',
        'recommendation_conversion_review_decision_id'),
       ('experiment_recommendation_conversion_execution','experiment_id'),
       ('experiment_recommendation_conversion_execution','execution_contract_version'),
       ('experiment_recommendation_conversion_execution','authorization_decision'),
       ('experiment_recommendation_conversion_execution','execution_identity_canonical'),
       ('experiment_recommendation_conversion_execution','execution_identity_hash'),
       ('experiment_recommendation_conversion_execution','created_at'),
       ('experiment_recommendation_conversion_activation',
        'recommendation_conversion_activation_id'),
       ('experiment_recommendation_conversion_activation',
        'recommendation_conversion_execution_id'),
       ('experiment_recommendation_conversion_activation',
        'recommendation_conversion_proposal_id'),
       ('experiment_recommendation_conversion_activation',
        'recommendation_conversion_review_decision_id'),
       ('experiment_recommendation_conversion_activation','experiment_id'),
       ('experiment_recommendation_conversion_activation','activation_contract_version'),
       ('experiment_recommendation_conversion_activation','previous_status'),
       ('experiment_recommendation_conversion_activation','previous_phase'),
       ('experiment_recommendation_conversion_activation','resulting_status'),
       ('experiment_recommendation_conversion_activation','resulting_phase'),
       ('experiment_recommendation_conversion_activation','activation_identity_canonical'),
       ('experiment_recommendation_conversion_activation','activation_identity_hash'),
       ('experiment_recommendation_conversion_activation','created_at'),
       ('experiment_recommendation_campaign_materialization',
        'recommendation_campaign_materialization_id'),
       ('experiment_recommendation_campaign_materialization',
        'recommendation_campaign_approval_id'),
       ('experiment_recommendation_campaign_materialization','materialization_contract_version'),
       ('experiment_recommendation_campaign_materialization','approval_identity_hash'),
       ('experiment_recommendation_campaign_materialization',
        'recommendation_ranking_snapshot_id'),
       ('experiment_recommendation_campaign_materialization','ranking_snapshot_identity_hash'),
       ('experiment_recommendation_campaign_materialization','planning_policy_hash'),
       ('experiment_recommendation_campaign_materialization','campaign_plan_identity_hash'),
       ('experiment_recommendation_campaign_materialization','campaign_review_identity_hash'),
       ('experiment_recommendation_campaign_materialization','materialized_by'),
       ('experiment_recommendation_campaign_materialization','materialization_reason_text'),
       ('experiment_recommendation_campaign_materialization','selected_member_count'),
       ('experiment_recommendation_campaign_materialization',
        'initially_created_proposal_count'),
       ('experiment_recommendation_campaign_materialization',
        'initially_reused_proposal_count'),
       ('experiment_recommendation_campaign_materialization',
        'materialization_identity_canonical'),
       ('experiment_recommendation_campaign_materialization','materialization_identity_hash'),
       ('experiment_recommendation_campaign_materialization','created_at'),
       ('experiment_recommendation_campaign_materialization_member',
        'recommendation_campaign_materialization_member_id'),
       ('experiment_recommendation_campaign_materialization_member',
        'recommendation_campaign_materialization_id'),
       ('experiment_recommendation_campaign_materialization_member','member_ordinal'),
       ('experiment_recommendation_campaign_materialization_member',
        'recommendation_ranking_member_id'),
       ('experiment_recommendation_campaign_materialization_member','recommendation_id'),
       ('experiment_recommendation_campaign_materialization_member','source_experiment_id'),
       ('experiment_recommendation_campaign_materialization_member','ranking_position'),
       ('experiment_recommendation_campaign_materialization_member',
        'selected_member_identity_canonical'),
       ('experiment_recommendation_campaign_materialization_member',
        'selected_member_identity_hash'),
       ('experiment_recommendation_campaign_materialization_member',
        'recommendation_conversion_proposal_id'),
       ('experiment_recommendation_campaign_materialization_member',
        'proposal_identity_canonical'),
       ('experiment_recommendation_campaign_materialization_member','proposal_identity_hash'),
       ('experiment_recommendation_campaign_materialization_member','created_at')
     ) required(table_name,column_name)
     WHERE NOT EXISTS (
       SELECT 1 FROM information_schema.columns c
       WHERE c.table_schema=current_schema()
         AND c.table_name=required.table_name
         AND c.column_name=required.column_name));
)SQL").one_row()[0].as<bool>();
}

RecommendationCampaignStatusSnapshot LoadRecommendationCampaignStatusSnapshot(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignStatusRequest& request)
{
    const auto normalized = NormalizeRecommendationCampaignStatusRequest(request);
    if (!RecommendationCampaignStatusSchemasExist(transaction))
        throw std::runtime_error("campaign_status_schemas_required");
    const auto materialization = FindRecommendationCampaignMaterialization(
        transaction, normalized.materializationId);
    if (!materialization)
        throw std::runtime_error("campaign_status_materialization_not_found");
    return BuildRecommendationCampaignStatusSnapshot(
        normalized, LoadInput(transaction, *materialization));
}

RecommendationCampaignStatusSnapshot ReadRecommendationCampaignStatusSnapshot(
    pqxx::connection& connection,
    const RecommendationCampaignStatusRequest& request)
{
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    return LoadRecommendationCampaignStatusSnapshot(transaction, request);
}

} // namespace EA::ExperimentRecommendation
