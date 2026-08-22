#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "../Sources/ContinuationPolicyInheritance.hpp"

using namespace EA::ExperimentScheduler;

namespace
{

ContinuationProfitabilityEvidence Profitability(
    long long observationId,
    const std::string& scope,
    std::uint64_t actionableCount,
    double aggregateReturn,
    std::optional<double> averageReturn)
{
    ContinuationProfitabilityEvidence evidence;
    evidence.observationId = observationId;
    evidence.observationIdentityHash =
        "fnv1a64:" + std::string(16, scope == "final" ? '1' : '2');
    evidence.inferenceEvalResultId = observationId + 100;
    evidence.inferenceScope = scope;
    if (scope == "checkpoint")
        evidence.checkpointEvalId = observationId + 200;
    evidence.metricDefinitionHash = "fnv1a64:3333333333333333";
    evidence.sourceContentHash = "fnv1a64:4444444444444444";
    evidence.predictionCount = actionableCount + 5;
    evidence.actionableCount = actionableCount;
    evidence.aggregateTerminalHorizonLogReturnSum = aggregateReturn;
    evidence.averageTerminalHorizonLogReturnPerActionablePrediction =
        averageReturn;
    return evidence;
}

ContinuationEvidence Source(
    long long analysisId,
    long long modelId,
    int epoch,
    double leaderScore,
    const std::string& scope,
    std::optional<long long> checkpointEvalId,
    std::optional<ContinuationProfitabilityEvidence> profitability)
{
    ContinuationEvidence evidence;
    evidence.analysisId = analysisId;
    evidence.modelId = modelId;
    evidence.completedEpoch = epoch;
    evidence.leaderScore = leaderScore;
    evidence.inferAccuracy = 0.8;
    evidence.analysisScope = scope;
    evidence.checkpointEvalId = checkpointEvalId;
    evidence.completedAt = std::to_string(epoch);
    evidence.updatedAt = std::to_string(epoch);
    if (profitability.has_value())
    {
        profitability->inferenceScope = scope;
        profitability->checkpointEvalId = checkpointEvalId;
        evidence.inferenceEvalResultId =
            profitability->inferenceEvalResultId;
    }
    evidence.profitability = std::move(profitability);
    evidence.profitabilityUnavailableReason = evidence.profitability.has_value()
        ? ""
        : "no_profitability_observation";
    return evidence;
}

} // namespace

int main()
{
    ContinuationPolicyConfig legacy;
    legacy.targetEpochs = 120;
    legacy.minEvals = 2;
    legacy.patience = 2;
    legacy.minLeaderScore = 0.5;
    legacy.sourceMode = "best_checkpoint";
    const std::string legacyCanonical =
        ContinuationPolicySemanticCanonicalText(legacy);
    const std::string legacyHash = ContinuationPolicySemanticHash(legacy);
    assert(legacyCanonical.find("profitability") == std::string::npos);
    assert(!ContinuationProfitabilityPolicyConfigured(legacy));

    ContinuationEvidence missing;
    missing.profitabilityUnavailableReason = "no_profitability_observation";
    auto gate = EvaluateContinuationProfitabilityGate(legacy, missing);
    assert(gate.passed && !gate.policyConfigured);
    assert(gate.reason == "profitability_policy_disabled");

    ContinuationEvidence observed;
    observed.profitability = Profitability(1, "final", 10, 0.25, 0.025);
    gate = EvaluateContinuationProfitabilityGate(legacy, observed);
    assert(gate.passed && gate.reason == "profitability_policy_disabled");
    assert(ContinuationPolicySemanticCanonicalText(legacy) == legacyCanonical);
    assert(ContinuationPolicySemanticHash(legacy) == legacyHash);

    ContinuationPolicyConfig policy = legacy;
    policy.minProfitabilityActionableCount = 10;
    policy.minProfitabilityAggregateTerminalHorizonLogReturnSum = 0.20;
    policy
        .minProfitabilityAverageTerminalHorizonLogReturnPerActionablePrediction =
        0.02;
    assert(ContinuationProfitabilityPolicyConfigured(policy));
    assert(ContinuationPolicySemanticHash(policy) != legacyHash);
    assert(ContinuationPolicySemanticCanonicalText(policy).find(
               "min_profitability_actionable_count=10") !=
           std::string::npos);

    gate = EvaluateContinuationProfitabilityGate(policy, observed);
    assert(gate.passed && gate.evidenceAvailable);
    assert(gate.actionableCountPassed == true);
    assert(gate.aggregateReturnPassed == true);
    assert(gate.averageReturnPassed == true);
    assert(gate.reason == "profitability_all_configured_requirements_passed");

    ContinuationEvidence tooFew = observed;
    tooFew.profitability->actionableCount = 9;
    gate = EvaluateContinuationProfitabilityGate(policy, tooFew);
    assert(!gate.passed && gate.actionableCountPassed == false);
    assert(gate.reason == "profitability_actionable_count_below_minimum");

    ContinuationEvidence lowAggregate = observed;
    lowAggregate.profitability->aggregateTerminalHorizonLogReturnSum = 0.19;
    gate = EvaluateContinuationProfitabilityGate(policy, lowAggregate);
    assert(!gate.passed && gate.aggregateReturnPassed == false);
    assert(gate.reason ==
           "profitability_aggregate_terminal_horizon_log_return_sum_below_minimum");

    ContinuationEvidence lowAverage = observed;
    lowAverage.profitability
        ->averageTerminalHorizonLogReturnPerActionablePrediction = 0.019;
    gate = EvaluateContinuationProfitabilityGate(policy, lowAverage);
    assert(!gate.passed && gate.averageReturnPassed == false);
    assert(gate.reason ==
           "profitability_average_terminal_horizon_log_return_per_actionable_prediction_below_minimum");

    gate = EvaluateContinuationProfitabilityGate(policy, missing);
    assert(!gate.passed && !gate.evidenceAvailable);
    assert(gate.actionableCountPassed == false);
    assert(gate.aggregateReturnPassed == false);
    assert(gate.averageReturnPassed == false);
    assert(gate.reason == "profitability_evidence_unavailable");

    ContinuationEvidence ambiguous = missing;
    ambiguous.profitabilityUnavailableReason =
        "ambiguous_profitability_observation";
    gate = EvaluateContinuationProfitabilityGate(policy, ambiguous);
    assert(!gate.passed && gate.reason == "profitability_evidence_unavailable");
    assert(ContinuationProfitabilityEvidenceLogFields(ambiguous).find(
               "profitability_unavailable_reason=ambiguous_profitability_observation") !=
           std::string::npos);

    ContinuationEvidence zeroActionable;
    zeroActionable.profitability = Profitability(2, "final", 0, 0.0, std::nullopt);
    gate = EvaluateContinuationProfitabilityGate(policy, zeroActionable);
    assert(gate.evidenceAvailable && !gate.passed);
    assert(gate.reason == "profitability_actionable_count_below_minimum");
    ContinuationPolicyConfig averageOnly = legacy;
    averageOnly
        .minProfitabilityAverageTerminalHorizonLogReturnPerActionablePrediction =
        0.0;
    gate = EvaluateContinuationProfitabilityGate(averageOnly, zeroActionable);
    assert(gate.evidenceAvailable && !gate.passed);
    assert(gate.reason ==
           "profitability_average_terminal_horizon_log_return_per_actionable_prediction_undefined");

    ContinuationPolicyConfig aggregateOnly = legacy;
    aggregateOnly.minProfitabilityAggregateTerminalHorizonLogReturnSum = 0.0;
    gate = EvaluateContinuationProfitabilityGate(aggregateOnly, zeroActionable);
    assert(gate.passed);

    const ContinuationPolicyUpdate parsed = ParseContinuationPolicyUpdate(
        "min_profitability_actionable_count=25,"
        "min_profitability_aggregate_terminal_horizon_log_return_sum=-0.01,"
        "min_profitability_average_terminal_horizon_log_return_per_actionable_prediction=0.002");
    ContinuationPolicyConfig parsedPolicy = legacy;
    ApplyContinuationPolicyUpdate(parsedPolicy, parsed);
    assert(parsedPolicy.minProfitabilityActionableCount == 25);
    assert(parsedPolicy.minProfitabilityAggregateTerminalHorizonLogReturnSum ==
           -0.01);
    assert(parsedPolicy
               .minProfitabilityAverageTerminalHorizonLogReturnPerActionablePrediction ==
           0.002);
    ApplyContinuationPolicyUpdate(
        parsedPolicy,
        ParseContinuationPolicyUpdate(
            "min_profitability_actionable_count=null,"
            "min_profitability_aggregate_terminal_horizon_log_return_sum=null,"
            "min_profitability_average_terminal_horizon_log_return_per_actionable_prediction=null"));
    assert(!ContinuationProfitabilityPolicyConfigured(parsedPolicy));
    assert(ContinuationPolicySemanticHash(parsedPolicy) == legacyHash);

    assert(ContinuationPolicyConfigurationError(policy, false) == std::nullopt);
    ContinuationPolicyConfig invalid = policy;
    invalid.minProfitabilityActionableCount = 0;
    assert(ContinuationPolicyConfigurationError(invalid, false) ==
           "min_profitability_actionable_count_must_be_positive");
    invalid = policy;
    invalid.minProfitabilityAggregateTerminalHorizonLogReturnSum =
        std::numeric_limits<double>::infinity();
    assert(ContinuationPolicyConfigurationError(invalid, false) ==
           "min_profitability_aggregate_terminal_horizon_log_return_sum_must_be_finite");
    invalid = policy;
    invalid
        .minProfitabilityAverageTerminalHorizonLogReturnPerActionablePrediction =
        std::numeric_limits<double>::quiet_NaN();
    assert(ContinuationPolicyConfigurationError(invalid, false) ==
           "min_profitability_average_terminal_horizon_log_return_per_actionable_prediction_must_be_finite");

    bool rejectedInvalidCount = false;
    try
    {
        (void)ParseContinuationPolicyUpdate(
            "min_profitability_actionable_count=0");
    }
    catch (const std::invalid_argument&)
    {
        rejectedInvalidCount = true;
    }
    assert(rejectedInvalidCount);
    bool rejectedNonFinite = false;
    try
    {
        (void)ParseContinuationPolicyUpdate(
            "min_profitability_aggregate_terminal_horizon_log_return_sum=nan");
    }
    catch (const std::invalid_argument&)
    {
        rejectedNonFinite = true;
    }
    assert(rejectedNonFinite);

    ContinuationPolicyConfig inheritedSource = policy;
    inheritedSource.source.targetEpochs = 100;
    inheritedSource.targetEpochs = 120;
    inheritedSource.minEvals = 1;
    inheritedSource.inheritToChild = true;
    inheritedSource.targetIncrement = 20;
    inheritedSource.maxTargetEpochs = 140;
    const ContinuationChildPolicyPlan inherited =
        PlanContinuationChildPolicy(inheritedSource);
    assert(inherited.inheritedPolicy.minProfitabilityActionableCount == 10);
    assert(inherited.inheritedPolicy
               .minProfitabilityAggregateTerminalHorizonLogReturnSum == 0.20);
    assert(inherited.inheritedPolicy
               .minProfitabilityAverageTerminalHorizonLogReturnPerActionablePrediction ==
           0.02);
    ContinuationPolicyConfig inheritedUnset = inheritedSource;
    inheritedUnset.minProfitabilityActionableCount.reset();
    inheritedUnset.minProfitabilityAggregateTerminalHorizonLogReturnSum.reset();
    inheritedUnset
        .minProfitabilityAverageTerminalHorizonLogReturnPerActionablePrediction
        .reset();
    const ContinuationChildPolicyPlan unsetPlan =
        PlanContinuationChildPolicy(inheritedUnset);
    assert(!ContinuationProfitabilityPolicyConfigured(
        unsetPlan.inheritedPolicy));

    ContinuationEvidence best = Source(
        10,
        100,
        80,
        0.9,
        "checkpoint",
        1000,
        Profitability(10, "checkpoint", 10, -1.0, -0.1));
    ContinuationEvidence latest = Source(
        11,
        101,
        100,
        0.8,
        "checkpoint",
        1001,
        Profitability(11, "checkpoint", 10, 1.0, 0.1));
    ContinuationEvidence finalSource = Source(
        12,
        102,
        100,
        0.7,
        "final",
        std::nullopt,
        Profitability(12, "final", 10, 0.5, 0.05));
    std::vector<ContinuationEvidence> sources{latest, finalSource, best};
    ContinuationPolicyConfig selection = legacy;
    selection.source.lastModelId = 102;
    selection.sourceMode = "best_checkpoint";
    assert(SelectContinuationSourceEvidence(selection, sources)->analysisId ==
           best.analysisId);
    selection.sourceMode = "latest_checkpoint";
    assert(SelectContinuationSourceEvidence(selection, sources)->analysisId ==
           latest.analysisId);
    selection.sourceMode = "final_model";
    assert(SelectContinuationSourceEvidence(selection, sources)->analysisId ==
           finalSource.analysisId);

    ContinuationPolicyConfig positiveAggregate = legacy;
    positiveAggregate.minProfitabilityAggregateTerminalHorizonLogReturnSum =
        0.0;
    selection.sourceMode = "best_checkpoint";
    const ContinuationEvidence selectedBest =
        *SelectContinuationSourceEvidence(selection, sources);
    assert(!EvaluateContinuationProfitabilityGate(
                positiveAggregate,
                selectedBest)
                .passed);
    assert(SelectContinuationSourceEvidence(selection, sources)->analysisId ==
           best.analysisId);

    const std::vector<ContinuationEvidence> distinct =
        DeduplicateContinuationEvidence(sources);
    const std::string baseWatermark = ContinuationEvidenceWatermark(distinct);
    best.profitability->aggregateTerminalHorizonLogReturnSum = 100.0;
    sources = {latest, finalSource, best};
    assert(ContinuationEvidenceWatermark(
               DeduplicateContinuationEvidence(sources)) == baseWatermark);

    ContinuationPolicyConfig trend;
    trend.trendMode = "non_degrading";
    trend.patience = 2;
    trend.maxDegradation = 1.0;
    std::optional<std::string> metric;
    std::optional<double> value;
    std::string reason;
    const ContinuationTrendResult before = EvaluateContinuationTrend(
        trend,
        distinct,
        metric,
        value,
        reason);
    std::optional<std::string> changedMetric;
    std::optional<double> changedValue;
    std::string changedReason;
    const ContinuationTrendResult after = EvaluateContinuationTrend(
        trend,
        DeduplicateContinuationEvidence(sources),
        changedMetric,
        changedValue,
        changedReason);
    assert(before == after && metric == changedMetric && value == changedValue &&
           reason == changedReason);

    const std::string disabledDiagnostics =
        ContinuationProfitabilityPolicyLogFields(
            legacy,
            EvaluateContinuationProfitabilityGate(legacy, missing));
    assert(disabledDiagnostics.find("profitability_policy=disabled") !=
           std::string::npos);
    const std::string enabledDiagnostics =
        ContinuationProfitabilityPolicyLogFields(
            policy,
            EvaluateContinuationProfitabilityGate(policy, observed));
    assert(enabledDiagnostics.find("profitability_policy=enabled") !=
           std::string::npos);
    assert(enabledDiagnostics.find("profitability_gate_passed=1") !=
           std::string::npos);

    std::cout << "ContinuationProfitabilityPolicyTests passed\n";
    return 0;
}
