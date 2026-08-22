#include <cassert>
#include <optional>
#include <string>
#include <vector>

#include "../Sources/ExperimentRecommendationCampaignPlanning.hpp"
#include "../Sources/ExperimentRecommendationCandidateGenerator.hpp"
#include "../Sources/ExperimentRecommendationEvaluation.hpp"
#include "../Sources/ExperimentRecommendationRanking.hpp"

using namespace EA::ExperimentRecommendation;

namespace
{

RecommendationSource::FinalProfitabilityEvidence Available(
    long long observationId,
    long long actionableCount,
    double aggregate,
    std::optional<double> average)
{
    RecommendationSource::FinalProfitabilityEvidence value;
    value.finalInferenceEvalResultId = 700 + observationId;
    value.profitabilityObservationId = observationId;
    value.inferenceStart = "2025-01-01";
    value.inferenceEnd = "2026-01-01";
    value.actionablePredictionCount = actionableCount;
    value.aggregateTerminalHorizonLogReturnSum = aggregate;
    value.averageTerminalHorizonLogReturnPerActionablePrediction = average;
    value.metricDefinitionHash = "fnv1a64:1111111111111111";
    value.sourceContentHash = "fnv1a64:2222222222222222";
    value.observationIdentityHash = "fnv1a64:3333333333333333";
    return value;
}

RecommendationSource::FinalProfitabilityEvidence Missing()
{
    RecommendationSource::FinalProfitabilityEvidence value;
    value.finalInferenceEvalResultId = 701;
    value.unavailableReason = "no_profitability_observation";
    return value;
}

RecommendationEvaluationInput EvaluationInput(long long id, double leader)
{
    RecommendationEvaluationInput input;
    auto& score = input.scoringInput;
    score.recommendationId = id;
    score.sourceExperimentId = 100 + id;
    score.sourcePredictionHorizon = 12;
    score.sourceRankWithinGroup = 1;
    score.sourceLeaderScore = leader;
    score.sourceInferenceAccuracy = 0.70;
    score.sourcePredictedNeutralProportion = 0.30;
    score.sourceEvidenceCount = 100;
    score.changedParameter = kCoreLrMult;
    score.sourceValueCanonical = "1";
    score.proposedValueCanonical = "1.25";
    score.absoluteDelta = 0.25;
    score.relativeDelta = 0.25;
    score.generationOrdinal = 1;
    score.structuralRank = 1;
    score.semanticCanonicalText = "semantic_" + std::to_string(id);
    input.recommendationSemanticHash = "semantic_hash_" + std::to_string(id);
    score.invocationCanonicalText = "invocation_" + std::to_string(id);
    score.recommendationPolicyCanonicalText = "policy_v1";
    score.duplicateType = "no_duplicate";
    score.recommendationStatus = "proposed";
    input.recommendationScanId = 20;
    input.sourceModelId = 30 + id;
    input.sourceAnalysisId = 40 + id;
    input.sourceSymbol = "eurusd";
    input.scanStatus = "completed";
    input.sourceExperimentStatus = "completed";
    input.sourceExperimentPhase = "done";
    input.currentSourceModelId = input.sourceModelId;
    input.currentSourceAnalysisId = input.sourceAnalysisId;
    input.currentSourceAnalysisStatus = "completed";
    input.currentSourceAnalysisScope = "final";
    return input;
}

RecommendationSource Source(
    const std::optional<RecommendationSource::FinalProfitabilityEvidence>&
        evidence)
{
    RecommendationSource source;
    source.experimentId = 1;
    source.modelId = 2;
    source.analysisId = 3;
    source.invocation.configuration.symbol = "eurusd";
    source.invocation.configuration.predictionHorizon = 12;
    source.invocation.configuration.labelThreshold = 0.001;
    source.invocation.configuration.coreLrMult = 1.0;
    source.invocation.configuration.headLrMult = 1.0;
    source.invocation.configuration.targetEpochs = 100;
    source.invocation.configuration.trainStartDate = "2024-01-01";
    source.invocation.configuration.trainEndDate = "2025-01-01";
    source.invocation.configuration.inferStartDate = "2025-01-01";
    source.invocation.configuration.inferEndDate = "2026-01-01";
    source.leaderScore = 0.8;
    source.inferenceAccuracy = 0.7;
    source.predictedNeutralProportion = 0.3;
    source.evidenceCount = 100;
    source.finalProfitabilityEvidence = evidence;
    return source;
}

RecommendationCampaignCandidateInput CampaignCandidate(
    long long id,
    int ordinal,
    const std::optional<RecommendationSource::FinalProfitabilityEvidence>&
        evidence)
{
    RecommendationCampaignCandidateInput value;
    value.rankingSnapshotId = 7;
    value.rankingSnapshotIdentityCanonical = "snapshot";
    value.rankingSnapshotIdentityHash =
        RecommendationCanonicalHash(value.rankingSnapshotIdentityCanonical);
    value.rankingPolicyCanonical =
        RecommendationRankingPolicyCanonicalText({});
    value.rankingPolicyHash =
        RecommendationCanonicalHash(value.rankingPolicyCanonical);
    value.rankingVersion = 1;
    value.rankingMemberId = 200 + id;
    value.rankingPosition = ordinal;
    value.rankingBucket = "advisory_ready";
    value.rankingScore = 0.75;
    value.recommendationId = id;
    value.sourceExperimentId = 100 + id;
    value.symbol = "eurusd";
    value.predictionHorizon = 12;
    value.family = kCoreLrMult;
    value.targetEpochs = 100;
    value.leaderScore = 0.8;
    value.inferenceAccuracy = 0.7;
    value.predictedNeutralProportion = 0.3;
    value.recommendationSemanticCanonical =
        "campaign_semantic_" + std::to_string(id);
    value.recommendationSemanticHash =
        RecommendationCanonicalHash(value.recommendationSemanticCanonical);
    value.recommendationInvocationCanonical =
        "campaign_invocation_" + std::to_string(id);
    value.recommendationInvocationHash =
        RecommendationCanonicalHash(value.recommendationInvocationCanonical);
    value.finalProfitabilityEvidence = evidence;
    return value;
}

} // namespace

int main()
{
    static_assert(kPhase3AProfitabilityScoringWeight == 0.0);
    static_assert(kPhase3AProfitabilityScoreContribution == 0.0);
    assert(kPhase3AProfitabilityScoringWeight == 0.0);
    assert(kPhase3AProfitabilityScoreContribution == 0.0);

    const auto positive = Available(1, 100, 1000.0, 10.0);
    const auto negative = Available(2, 100, -1000.0, -10.0);
    const auto zeroActionable = Available(3, 0, 0.0, std::nullopt);
    const auto missing = Missing();
    assert(!ValidateRecommendationFinalProfitabilityEvidence(positive));
    assert(!ValidateRecommendationFinalProfitabilityEvidence(negative));
    assert(!ValidateRecommendationFinalProfitabilityEvidence(zeroActionable));
    assert(!ValidateRecommendationFinalProfitabilityEvidence(missing));
    assert(zeroActionable.Available());
    assert(!missing.Available());
    assert(RecommendationFinalProfitabilityEvidenceCanonicalText(
               zeroActionable) !=
           RecommendationFinalProfitabilityEvidenceCanonicalText(missing));

    // Candidate discovery eligibility is independent of profitability
    // availability and sign.
    const RecommendationPolicy sourcePolicy;
    for (const auto& evidence : {
             std::optional<RecommendationSource::FinalProfitabilityEvidence>{
                 positive},
             std::optional<RecommendationSource::FinalProfitabilityEvidence>{
                 negative},
             std::optional<RecommendationSource::FinalProfitabilityEvidence>{
                 missing},
             std::optional<RecommendationSource::FinalProfitabilityEvidence>{
                 zeroActionable}})
        assert(EvaluateRecommendationSource(
                   sourcePolicy, Source(evidence)).eligible);

    const RecommendationEvaluationPolicy policy;
    const auto baselineInput = EvaluationInput(1, 0.8);
    const auto baseline = EvaluateExperimentRecommendation(
        policy, baselineInput);
    for (const auto& evidence : {positive, negative, missing, zeroActionable})
    {
        auto input = baselineInput;
        input.finalProfitabilityEvidence = evidence;
        const auto observed = EvaluateExperimentRecommendation(policy, input);
        assert(observed.eligibility == baseline.eligibility);
        assert(observed.disposition == baseline.disposition);
        assert(observed.finalScore == baseline.finalScore);
        assert(observed.rawPositiveScore == baseline.rawPositiveScore);
        assert(observed.rawPenaltyScore == baseline.rawPenaltyScore);
        assert(observed.rawTotalScore == baseline.rawTotalScore);
        assert(observed.components == baseline.components);
        assert(observed.evidenceCanonical == baseline.evidenceCanonical);
        assert(observed.evidenceHash == baseline.evidenceHash);
        assert(observed.evaluationIdentityCanonical ==
               baseline.evaluationIdentityCanonical);
        assert(observed.evaluationIdentityHash ==
               baseline.evaluationIdentityHash);
        assert(observed.profitabilityEvidenceCanonical !=
               baseline.profitabilityEvidenceCanonical);
    }

    // Profitability values chosen to reverse the leader-score order cannot
    // reverse evaluation ranking.
    auto highInput = EvaluationInput(10, 0.9);
    highInput.finalProfitabilityEvidence = negative;
    auto lowInput = EvaluationInput(11, 0.1);
    lowInput.finalProfitabilityEvidence = positive;
    const auto ranked = RankRecommendationEvaluations({
        EvaluateExperimentRecommendation(policy, lowInput),
        EvaluateExperimentRecommendation(policy, highInput)});
    assert(ranked[0].recommendationId == 10);
    assert(ranked[1].recommendationId == 11);

    // Different profitability cannot alter the existing semantic/hash tie
    // breaker when scores are tied.
    auto tieLeft = EvaluationInput(20, 0.8);
    auto tieRight = EvaluationInput(21, 0.8);
    const auto baselineTie = RankRecommendationEvaluations({
        EvaluateExperimentRecommendation(policy, tieLeft),
        EvaluateExperimentRecommendation(policy, tieRight)});
    tieLeft.finalProfitabilityEvidence = positive;
    tieRight.finalProfitabilityEvidence = negative;
    const auto profitabilityTie = RankRecommendationEvaluations({
        EvaluateExperimentRecommendation(policy, tieLeft),
        EvaluateExperimentRecommendation(policy, tieRight)});
    assert(baselineTie[0].recommendationId ==
           profitabilityTie[0].recommendationId);
    assert(baselineTie[1].recommendationId ==
           profitabilityTie[1].recommendationId);

    // Campaign planning receives the observational snapshot, but the plan
    // identity, selected recommendation, order, and decision remain exactly
    // those of the existing ranking evidence.
    RecommendationCampaignPlanInput campaign;
    campaign.policy.enabled = true;
    campaign.policy.maximumPerSourceExperiment.reset();
    campaign.scope.rankingSnapshotId = 7;
    campaign.rankingSnapshotIdentityCanonical = "snapshot";
    campaign.rankingSnapshotIdentityHash =
        RecommendationCanonicalHash("snapshot");
    campaign.generatedAt = "2026-08-22T00:00:00Z";
    campaign.candidates = {
        CampaignCandidate(30, 1, std::nullopt),
        CampaignCandidate(31, 2, std::nullopt)};
    const auto baselineCampaign = PlanRecommendationCampaign(campaign);
    campaign.candidates[0].finalProfitabilityEvidence = negative;
    campaign.candidates[1].finalProfitabilityEvidence = positive;
    const auto observedCampaign = PlanRecommendationCampaign(campaign);
    assert(observedCampaign.identityCanonical ==
           baselineCampaign.identityCanonical);
    assert(observedCampaign.identityHash == baselineCampaign.identityHash);
    assert(observedCampaign.summary.selectedCount ==
           baselineCampaign.summary.selectedCount);
    for (std::size_t index = 0;
         index < baselineCampaign.candidates.size(); ++index)
    {
        assert(observedCampaign.candidates[index].input.recommendationId ==
               baselineCampaign.candidates[index].input.recommendationId);
        assert(observedCampaign.candidates[index].decision ==
               baselineCampaign.candidates[index].decision);
        assert(observedCampaign.candidates[index].reasons ==
               baselineCampaign.candidates[index].reasons);
    }
    return 0;
}
