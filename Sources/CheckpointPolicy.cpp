#include "CheckpointPolicy.hpp"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <sstream>

namespace EA::ExperimentScheduler
{
namespace
{

std::string OptionalDoubleText(const std::optional<double>& value)
{
    if (!value.has_value())
        return "NULL";
    std::ostringstream out;
    out << std::setprecision(17) << *value;
    return out.str();
}

std::string OptionalIntText(const std::optional<int>& value)
{
    return value.has_value() ? std::to_string(*value) : "NULL";
}

}  // namespace

std::string CheckpointPolicyCanonicalText(const CheckpointPolicyConfig& config)
{
    std::ostringstream out;
    out << "enabled=" << (config.enabled ? "true" : "false")
        << "|min_leader_score=" << OptionalDoubleText(config.minLeaderScore)
        << "|min_infer_accuracy=" << OptionalDoubleText(config.minInferAccuracy)
        << "|top_n=" << OptionalIntText(config.topN)
        << "|rank_scope=" << config.scope
        << "|stop_mode=" << config.stopMode
        << "|grace_evals=" << config.graceEvals
        << "|checkpoint_interval=" << config.checkpointInterval
        << "|target_epochs=" << config.targetEpochs;
    return out.str();
}

std::string StableCheckpointPolicyHash(const std::string& canonicalText)
{
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const unsigned char ch : canonicalText)
    {
        hash ^= static_cast<std::uint64_t>(ch);
        hash *= UINT64_C(1099511628211);
    }
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

std::string CheckpointPolicySemanticHash(const CheckpointPolicyConfig& config)
{
    return StableCheckpointPolicyHash(CheckpointPolicyCanonicalText(config));
}

std::string CheckpointPolicyEvidenceCanonicalText(
    const CheckpointPolicyEvidenceIdentity& evidence)
{
    std::ostringstream out;
    out << "checkpoint_eval_id=" << evidence.checkpointEvalId
        << "|parent_experiment_id=" << evidence.parentExperimentId
        << "|checkpoint_model_id=" << evidence.checkpointModelId
        << "|checkpoint_epoch=" << evidence.checkpointEpoch
        << "|observed_current_epoch="
        << OptionalIntText(evidence.observedCurrentEpoch)
        << "|analysis_id=" << evidence.analysisId
        << "|inference_eval_result_id=" << evidence.inferenceEvalResultId
        << "|symbol=" << evidence.symbol
        << "|prediction_horizon=" << evidence.predictionHorizon
        << "|inference_from_date=" << evidence.inferenceFromDate
        << "|inference_to_date=" << evidence.inferenceToDate
        << "|leader_score=" << OptionalDoubleText(evidence.leaderScore)
        << "|inference_accuracy=" << OptionalDoubleText(evidence.inferenceAccuracy)
        << "|rank_value=" << OptionalIntText(evidence.rankValue)
        << "|rank_scope=" << evidence.rankScope
        << "|rank_population_watermark=" << evidence.rankPopulationWatermark
        << "|completed_eval_count=" << evidence.completedEvalCount
        << "|completed_population_watermark="
        << evidence.completedPopulationWatermark
        << "|checkpoint_status=" << evidence.checkpointStatus
        << "|checkpoint_phase=" << evidence.checkpointPhase
        << "|analysis_status=" << evidence.analysisStatus
        << "|inference_status=" << evidence.inferenceStatus
        << "|inference_scope=" << evidence.inferenceScope;
    return out.str();
}

std::string CheckpointPolicyEvidenceWatermark(
    const CheckpointPolicyEvidenceIdentity& evidence)
{
    return StableCheckpointPolicyHash(
        CheckpointPolicyEvidenceCanonicalText(evidence));
}

std::optional<int> RequestedCheckpointPolicyStopEpoch(
    int checkpointEpoch,
    const CheckpointPolicyConfig& config)
{
    const int currentEpoch = config.currentEpoch.value_or(checkpointEpoch);
    const int interval = std::max(1, config.checkpointInterval);
    const int latestObservedEpoch = std::max(checkpointEpoch, currentEpoch);
    int requested = ((latestObservedEpoch / interval) + 1) * interval;
    if (config.stopMode == "current_checkpoint_if_possible" &&
        currentEpoch <= checkpointEpoch)
    {
        requested = checkpointEpoch;
    }
    if (config.targetEpochs > 0 && requested >= config.targetEpochs)
        return std::nullopt;
    return requested;
}

CheckpointPolicyDecision DecideCheckpointPolicyPure(
    const CheckpointPolicyConfig& config,
    const CheckpointPolicyDecisionInputs& inputs)
{
    CheckpointPolicyDecision decision;
    decision.rankValue = inputs.rankValue;
    if (inputs.completedEvalCount < config.graceEvals)
    {
        decision.decision = "continue_grace";
        decision.reason =
            "completed_checkpoint_evals=" +
            std::to_string(inputs.completedEvalCount) +
            ";grace_evals=" + std::to_string(config.graceEvals);
        return decision;
    }

    if (config.minLeaderScore.has_value())
    {
        if (inputs.leaderScore.has_value() &&
            *inputs.leaderScore >= *config.minLeaderScore)
            decision.passedRules.push_back("leader_score");
        else
            decision.failedRules.push_back("leader_score");
    }
    if (config.minInferAccuracy.has_value())
    {
        if (inputs.inferenceAccuracy.has_value() &&
            *inputs.inferenceAccuracy >= *config.minInferAccuracy)
            decision.passedRules.push_back("infer_accuracy");
        else
            decision.failedRules.push_back("infer_accuracy");
    }
    if (config.topN.has_value())
    {
        if (inputs.rankValue.has_value() && *inputs.rankValue <= *config.topN)
            decision.passedRules.push_back("top_n");
        else
            decision.failedRules.push_back("top_n");
    }

    if (!decision.passedRules.empty())
    {
        decision.decision = "continue";
        decision.reason = "passed_rules=";
        for (std::size_t index = 0; index < decision.passedRules.size(); ++index)
        {
            if (index != 0)
                decision.reason += "|";
            decision.reason += decision.passedRules[index];
        }
        return decision;
    }
    if (decision.failedRules.empty())
    {
        decision.decision = "skipped";
        decision.reason = "no_policy_rules_configured";
        return decision;
    }

    decision.decision = "stop_requested";
    decision.reason = "failed_rules=";
    for (std::size_t index = 0; index < decision.failedRules.size(); ++index)
    {
        if (index != 0)
            decision.reason += "|";
        decision.reason += decision.failedRules[index];
    }
    decision.requestedStopEpoch =
        RequestedCheckpointPolicyStopEpoch(inputs.checkpointEpoch, config);
    if (!decision.requestedStopEpoch.has_value())
    {
        decision.decision = "skipped";
        decision.reason += ";no_future_safe_checkpoint";
    }
    return decision;
}

}  // namespace EA::ExperimentScheduler
