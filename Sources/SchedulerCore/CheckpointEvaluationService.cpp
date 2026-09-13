#include "CheckpointEvaluationService.hpp"

#include <cmath>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace EA::SchedulerCore
{
namespace
{

using EA::ExperimentScheduler::CheckpointPolicyConfig;
using EA::ExperimentScheduler::CheckpointPolicyDecision;
using EA::ExperimentScheduler::CheckpointPolicyDecisionInputs;
using EA::ExperimentScheduler::CheckpointPolicyEvidenceIdentity;
using EA::ExperimentScheduler::CheckpointPolicyEvidenceWatermark;
using EA::ExperimentScheduler::CheckpointPolicySemanticHash;
using EA::ExperimentScheduler::DecideCheckpointPolicyPure;

bool ValidCheckpointPolicyScope(const std::string& value)
{
    return value == "symbol_horizon" || value == "horizon" ||
           value == "global";
}

bool ValidCheckpointPolicyStopMode(const std::string& value)
{
    return value == "next_checkpoint" ||
           value == "current_checkpoint_if_possible" ||
           value == "mark_pruned_when_not_running";
}

std::string FormatDouble(double value)
{
    std::ostringstream output;
    output << std::setprecision(17) << value;
    return output.str();
}

std::string OptionalDoubleText(const std::optional<double>& value)
{
    return value ? FormatDouble(*value) : "NULL";
}

std::string JoinRules(const std::vector<std::string>& rules)
{
    if (rules.empty())
        return "none";
    std::ostringstream output;
    for (std::size_t index = 0; index < rules.size(); ++index)
    {
        if (index != 0)
            output << '|';
        output << rules[index];
    }
    return output.str();
}

std::string CheckpointPolicyRuleText(const CheckpointPolicyConfig& config)
{
    std::ostringstream output;
    bool any = false;
    if (config.minLeaderScore)
    {
        output << "leader_score>=" << FormatDouble(*config.minLeaderScore);
        any = true;
    }
    if (config.minInferAccuracy)
    {
        if (any)
            output << '|';
        output << "infer_accuracy>=" << FormatDouble(*config.minInferAccuracy);
        any = true;
    }
    if (config.topN)
    {
        if (any)
            output << '|';
        output << "top_n<=" << *config.topN;
        any = true;
    }
    if (!any)
        output << "none";
    output << "|scope=" << config.scope
           << "|stop_mode=" << config.stopMode
           << "|grace_evals=" << config.graceEvals;
    return output.str();
}

void PrintEvaluationIdentity(
    std::ostream& output,
    const CheckpointEvaluationRecord& evaluation)
{
    output << ",parent_experiment_id=" << evaluation.parentExperimentId
           << ",checkpoint_eval_id=" << evaluation.checkpointEvalId
           << ",checkpoint_epoch=" << evaluation.checkpointEpoch
           << ",checkpoint_model_id=" << evaluation.checkpointModelId
           << ",symbol=" << evaluation.symbol
           << ",prediction_horizon=" << evaluation.predictionHorizon;
}

} // namespace

std::optional<std::string> CheckpointPolicyConfigurationError(
    const CheckpointPolicyConfig& config,
    bool requireRules)
{
    if (config.minLeaderScore &&
        (!std::isfinite(*config.minLeaderScore) ||
         *config.minLeaderScore <= 0.0))
    {
        return "min_leader_score_must_be_positive_and_finite";
    }
    if (config.minInferAccuracy &&
        (!std::isfinite(*config.minInferAccuracy) ||
         *config.minInferAccuracy <= 0.0))
    {
        return "min_infer_accuracy_must_be_positive_and_finite";
    }
    if (config.topN && *config.topN <= 0)
        return "top_n_must_be_positive";
    if (!ValidCheckpointPolicyScope(config.scope))
        return "invalid_scope";
    if (!ValidCheckpointPolicyStopMode(config.stopMode))
        return "invalid_stop_mode";
    if (config.graceEvals <= 0)
        return "grace_evals_must_be_positive";
    if (requireRules && !config.minLeaderScore &&
        !config.minInferAccuracy && !config.topN)
    {
        return "at_least_one_continue_rule_required";
    }
    return std::nullopt;
}

CheckpointPolicyDecisionContext PlanCheckpointPolicyDecision(
    const CheckpointEvaluationRecord& evaluation,
    const CheckpointPolicyConfig& config,
    const ValidatedCheckpointPolicyEvidence& evidence,
    CheckpointPolicyPopulation completedPopulation,
    CheckpointPolicyPopulation rankPopulation)
{
    const CheckpointPolicyDecisionInputs inputs{
        evaluation.checkpointEpoch,
        completedPopulation.count,
        evidence.leaderScore,
        evidence.inferenceAccuracy,
        rankPopulation.rankValue};
    return {
        DecideCheckpointPolicyPure(config, inputs),
        std::move(completedPopulation),
        std::move(rankPopulation)};
}

CheckpointPolicyEvidenceIdentity MakeCheckpointPolicyEvidenceIdentity(
    const CheckpointEvaluationRecord& evaluation,
    const ValidatedCheckpointPolicyEvidence& evidence,
    const CheckpointPolicyDecisionContext& context,
    const CheckpointPolicyConfig& config)
{
    CheckpointPolicyEvidenceIdentity identity;
    identity.checkpointEvalId = evaluation.checkpointEvalId;
    identity.parentExperimentId = evaluation.parentExperimentId;
    identity.checkpointModelId = evaluation.checkpointModelId;
    identity.checkpointEpoch = evaluation.checkpointEpoch;
    identity.observedCurrentEpoch = config.currentEpoch;
    identity.analysisId = evidence.analysisId;
    identity.inferenceEvalResultId = evidence.inferenceEvalResultId;
    identity.symbol = evaluation.symbol;
    identity.predictionHorizon = evaluation.predictionHorizon;
    identity.inferenceFromDate = evidence.inferenceFromDate;
    identity.inferenceToDate = evidence.inferenceToDate;
    identity.leaderScore = evidence.leaderScore;
    identity.inferenceAccuracy = evidence.inferenceAccuracy;
    identity.rankValue = config.topN ? context.decision.rankValue
                                     : std::nullopt;
    identity.rankScope = config.scope;
    identity.rankPopulationWatermark =
        config.topN ? context.rankPopulation.watermark
                    : "not_decision_bearing";
    identity.completedEvalCount = context.completedPopulation.count;
    identity.completedPopulationWatermark =
        context.completedPopulation.watermark;
    return identity;
}

CheckpointEvaluationService::CheckpointEvaluationService(
    CheckpointEvaluationOperations operations,
    std::ostream& output)
    : operations_{std::move(operations)}, output_{output}
{
    if (!operations_.schemaAvailable || !operations_.loadPolicy ||
        !operations_.reconcilePolicyIdentity || !operations_.loadEvidence ||
        !operations_.loadCompletedPopulation ||
        !operations_.loadRankPopulation || !operations_.persistDecision ||
        !operations_.applyStopRequest)
    {
        throw std::invalid_argument(
            "complete checkpoint evaluation operations required");
    }
}

CheckpointPolicyEvaluationResult CheckpointEvaluationService::evaluate(
    const CheckpointEvaluationRecord& evaluation)
{
    CheckpointPolicyEvaluationResult result;
    const auto skip = [&](std::string reason) {
        result.reason = std::move(reason);
        output_ << "CHECKPOINT_POLICY_SKIPPED";
        PrintEvaluationIdentity(output_, evaluation);
        output_ << ",reason=" << result.reason << std::endl;
        return result;
    };

    if (!operations_.schemaAvailable())
        return skip("migration_required");

    std::optional<CheckpointPolicyConfig> config =
        operations_.loadPolicy(evaluation.parentExperimentId);
    if (!config || !config->enabled)
    {
        return skip(config ? "disabled" : "parent_experiment_not_found");
    }
    *config = operations_.reconcilePolicyIdentity(evaluation, *config);

    if (!config->checkpointInferEnabled)
        return skip("checkpoint_infer_not_enabled");
    if (const std::optional<std::string> error =
            CheckpointPolicyConfigurationError(*config, true))
    {
        return skip("invalid_configuration:" + *error);
    }
    if (config->phase != "train" || config->status != "running")
        return skip("parent_not_running_train");

    const CheckpointPolicyEvidenceLoadResult loadedEvidence =
        operations_.loadEvidence(evaluation);
    if (!loadedEvidence.evidence)
        return skip(loadedEvidence.rejectionReason);
    const ValidatedCheckpointPolicyEvidence& evidence =
        *loadedEvidence.evidence;

    const std::string configuredRules = CheckpointPolicyRuleText(*config);
    output_ << "CHECKPOINT_POLICY_EVALUATING";
    PrintEvaluationIdentity(output_, evaluation);
    output_ << ",leader_score=" << OptionalDoubleText(evidence.leaderScore)
            << ",infer_accuracy="
            << OptionalDoubleText(evidence.inferenceAccuracy)
            << ",configured_rules=" << configuredRules << std::endl;

    CheckpointPolicyPopulation completedPopulation =
        operations_.loadCompletedPopulation(evaluation.parentExperimentId);
    CheckpointPolicyPopulation rankPopulation =
        operations_.loadRankPopulation(evaluation, *config);
    const CheckpointPolicyDecisionContext context =
        PlanCheckpointPolicyDecision(
            evaluation,
            *config,
            evidence,
            std::move(completedPopulation),
            std::move(rankPopulation));
    const CheckpointPolicyDecision& decision = context.decision;
    const CheckpointPolicyEvidenceIdentity evidenceIdentity =
        MakeCheckpointPolicyEvidenceIdentity(
            evaluation, evidence, context, *config);
    const std::string evidenceWatermark =
        CheckpointPolicyEvidenceWatermark(evidenceIdentity);
    const PersistedCheckpointPolicyDecision persisted =
        operations_.persistDecision(
            evaluation, *config, decision, evidence, evidenceIdentity);

    std::string stopActionResult = "not_requested";
    if (decision.decision == "continue")
        output_ << "CHECKPOINT_POLICY_CONTINUE";
    else if (decision.decision == "continue_grace")
        output_ << "CHECKPOINT_POLICY_CONTINUE_GRACE";
    else if (decision.decision == "stop_requested")
    {
        stopActionResult = operations_.applyStopRequest(
            evaluation,
            *config,
            decision,
            persisted,
            evidenceWatermark);
        output_ << (stopActionResult == "stop_request_applied"
                        ? "CHECKPOINT_POLICY_STOP_REQUESTED"
                        : "CHECKPOINT_POLICY_STOP_SUPERSEDED");
    }
    else
        output_ << "CHECKPOINT_POLICY_SKIPPED";

    PrintEvaluationIdentity(output_, evaluation);
    output_ << ",leader_score=" << OptionalDoubleText(evidence.leaderScore)
            << ",infer_accuracy="
            << OptionalDoubleText(evidence.inferenceAccuracy)
            << ",rank="
            << (decision.rankValue ? std::to_string(*decision.rankValue)
                                   : "NULL")
            << ",configured_rules=" << configuredRules
            << ",passed_rules=" << JoinRules(decision.passedRules)
            << ",failed_rules=" << JoinRules(decision.failedRules)
            << ",reason=" << decision.reason
            << ",decision_id=" << persisted.decisionId
            << ",decision_identity_status=" << persisted.identityStatus
            << ",decision_reused=" << (persisted.reused ? "1" : "0")
            << ",policy_revision=" << config->policyRevision
            << ",policy_hash=" << CheckpointPolicySemanticHash(*config)
            << ",analysis_id=" << evidence.analysisId
            << ",inference_eval_result_id="
            << evidence.inferenceEvalResultId
            << ",evidence_watermark=" << evidenceWatermark
            << ",rank_population_watermark="
            << context.rankPopulation.watermark
            << ",stop_action_result=" << stopActionResult;
    if (decision.requestedStopEpoch)
        output_ << ",requested_stop_epoch=" << *decision.requestedStopEpoch;
    output_ << std::endl;

    result.evaluated = true;
    result.decision = decision.decision;
    result.reason = decision.reason;
    return result;
}

} // namespace EA::SchedulerCore
