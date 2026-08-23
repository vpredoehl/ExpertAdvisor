#include "../Sources/ProfitabilityDistribution.hpp"

#include "../Sources/ExperimentRecommendation.hpp"
#include "../Sources/ExperimentRecommendationEvaluation.hpp"
#include "../Sources/ExperimentRecommendationRanking.hpp"
#include "../Sources/ExperimentRecommendationScoring.hpp"
#include "../Sources/InferenceProfitability.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <locale>
#include <string>
#include <vector>

using namespace EA::ExperimentRecommendation;

namespace
{

class CommaNumpunct : public std::numpunct<char>
{
protected:
    char do_decimal_point() const override { return ','; }
};

ProfitabilitySemanticIdentity Semantic(const std::string& canonical,
                                       int version = 1)
{
    return {canonical, RecommendationCanonicalHash(canonical), version};
}

ProfitabilityObservation Observation(long long id,
                                     double average,
                                     std::uint64_t actionable = 100)
{
    ProfitabilityObservation value;
    value.profitabilityObservationId = id;
    value.experimentId = 1000 + id;
    value.modelId = 2000 + id;
    value.inferenceEvalResultId = 3000 + id;
    value.metricDefinitionCanonical =
        EA::InferenceProfitability::kMetricDefinitionCanonical;
    value.metricDefinitionHash = RecommendationCanonicalHash(
        value.metricDefinitionCanonical);
    value.sourceContentHash = RecommendationCanonicalHash(
        "source_content_" + std::to_string(id));
    value.observationIdentityCanonical =
        "observation_identity_" + std::to_string(id);
    value.observationIdentityHash = RecommendationCanonicalHash(
        value.observationIdentityCanonical);
    value.inferenceScope = "final";
    value.inferenceStart = "2025-01-01";
    value.inferenceEnd = "2026-01-01";
    value.symbol = "audcadrmp";
    value.predictionHorizon = 12;
    value.modelInputWidth = 48;
    value.featureClassIdentity = Semantic("feature_class_v1;input=48");
    value.inferenceEvaluationSemanticIdentity = Semantic(
        "inference_evaluation_semantic_v1;label=first_hit;target=return");
    value.scoringSemanticIdentity = Semantic(
        "recommendation_scoring_semantic_v1");
    value.evaluationSemanticIdentity = Semantic(
        "recommendation_evaluation_semantic_v1");
    value.predictionCount = actionable + 10;
    value.actionableCount = actionable;
    const double aggregate = average * static_cast<double>(actionable);
    value.aggregateTerminalHorizonLogReturnSum = aggregate;
    if (aggregate > 0.0)
    {
        value.winningActionableCount = actionable;
        value.grossPositiveTerminalHorizonLogReturnSum = aggregate;
    }
    else if (aggregate < 0.0)
    {
        value.losingActionableCount = actionable;
        value.grossNegativeTerminalHorizonLogReturnSum = aggregate;
    }
    value.averageTerminalHorizonLogReturnPerActionablePrediction = average;
    return value;
}

ProfitabilityObservation ZeroActionable(long long id)
{
    ProfitabilityObservation value = Observation(id, 0.0, 1);
    value.predictionCount = 50;
    value.actionableCount = 0;
    value.winningActionableCount = 0;
    value.losingActionableCount = 0;
    value.grossPositiveTerminalHorizonLogReturnSum = 0.0;
    value.grossNegativeTerminalHorizonLogReturnSum = 0.0;
    value.aggregateTerminalHorizonLogReturnSum = 0.0;
    value.averageTerminalHorizonLogReturnPerActionablePrediction.reset();
    return value;
}

ProfitabilityNormalizationPolicy Policy(std::size_t minimum = 3)
{
    ProfitabilityNormalizationPolicy policy;
    policy.minimumAnalyzablePopulationSize = minimum;
    policy.supportHalfSaturationActionableCount = 100;
    return policy;
}

const ProfitabilityNormalizationResult& Result(
    const ProfitabilityDistributionAnalysis& analysis,
    long long observationId)
{
    const auto found = std::find_if(
        analysis.normalizationResults.begin(),
        analysis.normalizationResults.end(),
        [observationId](const auto& value) {
            return value.profitabilityObservationId == observationId;
        });
    assert(found != analysis.normalizationResults.end());
    return *found;
}

bool Close(double left, double right, double tolerance = 1e-12)
{
    return std::abs(left - right) <= tolerance;
}

void AssertInferenceStartDateValidity(const std::string& date, bool valid)
{
    auto observation = Observation(9000, 0.1);
    observation.inferenceStart = date;
    observation.inferenceEnd = "9999-12-31";
    const auto error = ValidateProfitabilityObservation(observation);
    if (valid)
        assert(!error);
    else
        assert(error == "invalid_inference_window");
}

RecommendationScoringInput ScoringInput(long long id)
{
    RecommendationScoringInput input;
    input.recommendationId = id;
    input.sourceExperimentId = 99;
    input.sourcePredictionHorizon = 12;
    input.sourceRankWithinGroup = 1;
    input.sourceLeaderScore = 0.8;
    input.sourceInferenceAccuracy = 0.7;
    input.sourcePredictedNeutralProportion = 0.3;
    input.sourceEvidenceCount = 1000;
    input.changedParameter = kCoreLrMult;
    input.sourceValueCanonical = "1";
    input.proposedValueCanonical = "1.1";
    input.absoluteDelta = 0.1;
    input.relativeDelta = 0.1;
    input.generationOrdinal = 1;
    input.structuralRank = 1;
    input.semanticCanonicalText = "same_semantic";
    input.invocationCanonicalText = "invocation_" + std::to_string(id);
    input.recommendationPolicyCanonicalText = "same_policy";
    input.duplicateType = "no_duplicate";
    input.recommendationStatus = "proposed";
    return input;
}

RecommendationRankingEvaluation RankingEvaluation(long long id, double score)
{
    const RecommendationEvaluationPolicy policy;
    RecommendationRankingEvaluation value;
    value.evaluationResultId = id;
    value.evaluationRunId = 500;
    value.recommendationId = 600 + id;
    value.recommendationScanId = 700;
    value.sourceExperimentId = 800 + id;
    value.sourceModelId = 900 + id;
    value.sourceAnalysisId = 1000 + id;
    value.symbol = "audcadrmp";
    value.horizon = 12;
    value.family = "core_lr_mult";
    value.sourceValueCanonical = "1";
    value.proposedValueCanonical = "1.1";
    value.recommendationSemanticHash = RecommendationCanonicalHash(
        "recommendation_" + std::to_string(id));
    value.evaluationIdentityCanonical =
        "evaluation_identity_" + std::to_string(id);
    value.evaluationIdentityHash = RecommendationEvaluationCanonicalHash(
        value.evaluationIdentityCanonical);
    value.evaluationPolicyCanonical =
        RecommendationEvaluationPolicyCanonicalText(policy);
    value.evaluationPolicyHash = RecommendationEvaluationPolicyHash(policy);
    value.evaluationVersion = policy.evaluationVersion;
    value.evaluatorVersion = policy.evaluatorVersion;
    value.scoringPolicyCanonical =
        RecommendationScoringPolicyCanonicalText(policy.scoringPolicy);
    value.scoringPolicyHash = RecommendationScoringPolicyHash(
        policy.scoringPolicy);
    value.scoringVersion = policy.scoringPolicy.scoringVersion;
    value.scoringSemanticIdentity =
        RecommendationScoringSemanticIdentityForPolicy(policy.scoringPolicy);
    value.evaluationSemanticIdentity =
        RecommendationEvaluationSemanticIdentityForPolicy(policy);
    value.eligibility = RecommendationEligibility::eligible;
    value.disposition = RecommendationEvaluationDisposition::advisoryReady;
    value.reasonCode = "advisory_ready";
    value.explanation = "synthetic";
    value.finalScore = score;
    value.components.push_back({
        "leader_quality", "leader_quality", "0.8", 0.8, 0.25, 0.2,
        false, "synthetic"});
    value.componentCount = 1;
    return value;
}

} // namespace

int main()
{
    const auto policy = Policy();
    assert(!ValidateProfitabilityNormalizationPolicy(policy));
    assert(RecommendationCanonicalHash(
        EA::InferenceProfitability::kMetricDefinitionCanonical) ==
        EA::InferenceProfitability::MetricDefinitionHash());
    assert(ProfitabilityNormalizationPolicyCanonicalText(policy).find(
        "profitability_weight=0;profitability_score_contribution=0") !=
        std::string::npos);

    // Inference dates use the strict YYYY-MM-DD representation and must be
    // real Gregorian calendar dates, independent of locale and timezone.
    AssertInferenceStartDateValidity("2025-01-01", true);
    AssertInferenceStartDateValidity("2025-01-31", true);
    AssertInferenceStartDateValidity("2025-04-30", true);
    AssertInferenceStartDateValidity("2025-12-31", true);
    AssertInferenceStartDateValidity("2025-02-28", true);
    AssertInferenceStartDateValidity("2025-02-29", false);
    AssertInferenceStartDateValidity("2024-02-29", true);
    AssertInferenceStartDateValidity("2100-02-29", false);
    AssertInferenceStartDateValidity("2000-02-29", true);
    AssertInferenceStartDateValidity("2025-04-31", false);
    AssertInferenceStartDateValidity("2025-00-10", false);
    AssertInferenceStartDateValidity("2025-13-10", false);
    AssertInferenceStartDateValidity("2025-01-00", false);
    AssertInferenceStartDateValidity("0000-01-01", false);
    AssertInferenceStartDateValidity("", false);
    AssertInferenceStartDateValidity("20250228", false);
    AssertInferenceStartDateValidity("2025-2-28", false);
    AssertInferenceStartDateValidity("2025-02-8", false);
    AssertInferenceStartDateValidity("2025/02/28", false);
    AssertInferenceStartDateValidity("2025-02-28Z", false);
    AssertInferenceStartDateValidity("202A-02-28", false);

    auto invalidEndDate = Observation(9001, 0.1);
    invalidEndDate.inferenceEnd = "2025-02-29";
    assert(ValidateProfitabilityObservation(invalidEndDate) ==
           "invalid_inference_window");
    auto equalWindow = Observation(9002, 0.1);
    equalWindow.inferenceEnd = equalWindow.inferenceStart;
    assert(ValidateProfitabilityObservation(equalWindow) ==
           "invalid_inference_window");
    auto decreasingWindow = Observation(9003, 0.1);
    decreasingWindow.inferenceEnd = "2024-12-31";
    assert(ValidateProfitabilityObservation(decreasingWindow) ==
           "invalid_inference_window");

    // Homogeneous valid population and descriptive statistics.
    std::vector<ProfitabilityObservation> homogeneous{
        Observation(1, -0.02, 10), Observation(2, 0.0, 100),
        Observation(3, 0.01, 100), Observation(4, 0.02, 200),
        Observation(5, 1.0, 1000)};
    const auto analysis = AnalyzeProfitabilityDistribution(homogeneous, policy);
    assert(analysis.summary.state == ProfitabilityPopulationState::valid);
    assert(analysis.summary.populationCount == 5);
    assert(analysis.summary.analyzablePopulationCount == 5);
    assert(analysis.summary.totalActionableCount == 1410);
    assert(analysis.summary.negativeCount == 1);
    assert(analysis.summary.zeroCount == 1);
    assert(analysis.summary.positiveCount == 3);
    assert(*analysis.summary.minimum == -0.02);
    assert(*analysis.summary.maximum == 1.0);
    assert(*analysis.summary.median == 0.01);
    assert(*analysis.summary.populationStandardDeviation > 0.0);
    assert(*analysis.summary.medianAbsoluteDeviation == 0.01);
    assert(analysis.summary.quantiles.size() == 5);
    assert(!analysis.summary.populationIdentityCanonical.empty());
    assert(!analysis.summary.populationIdentityHash.empty());
    assert(!analysis.summary.membershipHash.empty());

    // The transform is bounded, sign preserving, and robust to the magnitude
    // of the strong outlier.
    assert(*Result(analysis, 1).boundedCandidateMetric < 0.5);
    assert(*Result(analysis, 2).boundedCandidateMetric == 0.5);
    assert(*Result(analysis, 3).boundedCandidateMetric > 0.5);
    assert(*Result(analysis, 5).boundedCandidateMetric < 1.0);
    assert(*Result(analysis, 5).boundedCandidateMetric == 0.95);

    // Actionable support is separate, monotone, and half saturated at the
    // declared count. It never changes the candidate or a production score.
    assert(Close(Result(analysis, 1).supportReliability, 10.0 / 110.0));
    assert(Result(analysis, 2).supportReliability == 0.5);
    assert(Result(analysis, 5).supportReliability == 1000.0 / 1100.0);

    // Mixed metric definitions fail closed, even when each row is internally
    // valid and carries a self-consistent canonical/hash pair.
    auto mixedMetric = homogeneous;
    mixedMetric[4].metricDefinitionCanonical += ";alternate=true";
    mixedMetric[4].metricDefinitionHash = RecommendationCanonicalHash(
        mixedMetric[4].metricDefinitionCanonical);
    assert(AnalyzeProfitabilityDistribution(mixedMetric, policy).summary.reason ==
           "incompatible_profitability_population_identity");

    auto mixedWindow = homogeneous;
    mixedWindow[4].inferenceStart = "2025-02-01";
    assert(AnalyzeProfitabilityDistribution(mixedWindow, policy).summary.reason ==
           "incompatible_profitability_population_identity");

    auto mixedScope = homogeneous;
    mixedScope[4].inferenceScope = "checkpoint";
    assert(AnalyzeProfitabilityDistribution(mixedScope, policy).summary.reason ==
           "incompatible_profitability_population_identity");

    auto mixedHorizon = homogeneous;
    mixedHorizon[4].predictionHorizon = 24;
    assert(AnalyzeProfitabilityDistribution(mixedHorizon, policy).summary.reason ==
           "incompatible_profitability_population_identity");

    auto mixedSymbol = homogeneous;
    mixedSymbol[4].symbol = "eurusdrmp";
    assert(AnalyzeProfitabilityDistribution(mixedSymbol, policy).summary.reason ==
           "incompatible_profitability_population_identity");

    auto mixedEvaluationSemantics = homogeneous;
    mixedEvaluationSemantics[4].evaluationSemanticIdentity =
        Semantic("recommendation_evaluation_semantic_v2", 2);
    assert(AnalyzeProfitabilityDistribution(
        mixedEvaluationSemantics, policy).summary.reason ==
        "incompatible_profitability_population_identity");

    // Finite-value and internal-statistic enforcement.
    auto nonfinite = Observation(10, 0.1);
    nonfinite.averageTerminalHorizonLogReturnPerActionablePrediction =
        std::numeric_limits<double>::quiet_NaN();
    assert(ValidateProfitabilityObservation(nonfinite) ==
           "nonfinite_profitability_average");
    auto inconsistent = Observation(11, 0.1);
    inconsistent.aggregateTerminalHorizonLogReturnSum += 1.0;
    assert(ValidateProfitabilityObservation(inconsistent) ==
           "invalid_profitability_sums");

    // Zero actionable is present in membership and totals but has no fabricated
    // average, percentile, or bounded value.
    auto withZeroActionable = homogeneous;
    withZeroActionable.push_back(ZeroActionable(6));
    const auto zeroAnalysis = AnalyzeProfitabilityDistribution(
        withZeroActionable, policy);
    assert(zeroAnalysis.summary.populationCount == 6);
    assert(zeroAnalysis.summary.analyzablePopulationCount == 5);
    assert(zeroAnalysis.summary.zeroActionableCount == 1);
    const auto& zeroResult = Result(zeroAnalysis, 6);
    assert(zeroResult.state == ProfitabilityNormalizationState::zeroActionable);
    assert(!zeroResult.rawProfitabilityMetric);
    assert(!zeroResult.empiricalMidrankPercentile);
    assert(!zeroResult.boundedCandidateMetric);
    assert(zeroResult.supportReliability == 0.0);

    const auto noActionable = AnalyzeProfitabilityDistribution(
        {ZeroActionable(20), ZeroActionable(21), ZeroActionable(22)}, policy);
    assert(noActionable.summary.state ==
           ProfitabilityPopulationState::noAnalyzableObservations);
    assert(!noActionable.summary.mean);

    // Single-member and all-equal populations are explicit and deterministic.
    const auto single = AnalyzeProfitabilityDistribution(
        {Observation(30, 0.2)}, policy);
    assert(single.summary.state ==
           ProfitabilityPopulationState::insufficientPopulation);
    assert(Result(single, 30).state ==
           ProfitabilityNormalizationState::insufficientPopulation);
    assert(*Result(single, 30).empiricalMidrankPercentile == 0.5);
    assert(*Result(single, 30).boundedCandidateMetric == 0.75);
    assert(*single.summary.populationStandardDeviation == 0.0);

    const auto allEqual = AnalyzeProfitabilityDistribution(
        {Observation(31, 0.2), Observation(32, 0.2), Observation(33, 0.2)},
        policy);
    assert(allEqual.summary.state == ProfitabilityPopulationState::valid);
    assert(*allEqual.summary.populationStandardDeviation == 0.0);
    assert(*allEqual.summary.medianAbsoluteDeviation == 0.0);
    for (const auto& result : allEqual.normalizationResults)
    {
        assert(*result.empiricalMidrankPercentile == 0.5);
        assert(*result.boundedCandidateMetric == 0.75);
    }

    // Midrank tie handling does not depend on member identity or row order.
    const auto tied = AnalyzeProfitabilityDistribution(
        {Observation(40, 0.1), Observation(41, 0.1), Observation(42, 0.2)},
        policy);
    assert(Close(*Result(tied, 40).empiricalMidrankPercentile, 1.0 / 3.0));
    assert(Result(tied, 40).empiricalMidrankPercentile ==
           Result(tied, 41).empiricalMidrankPercentile);
    assert(Close(*Result(tied, 42).empiricalMidrankPercentile, 5.0 / 6.0));

    auto reversed = homogeneous;
    std::reverse(reversed.begin(), reversed.end());
    const auto reverseAnalysis = AnalyzeProfitabilityDistribution(
        reversed, policy);
    assert(reverseAnalysis.canonical == analysis.canonical);
    assert(reverseAnalysis.hash == analysis.hash);
    for (std::size_t index = 0;
         index < analysis.normalizationResults.size(); ++index)
        assert(reverseAnalysis.normalizationResults[index].canonical ==
               analysis.normalizationResults[index].canonical);

    // Empty input and malformed provenance fail closed rather than producing
    // an invented cohort.
    const auto empty = AnalyzeProfitabilityDistribution({}, policy);
    assert(empty.summary.state == ProfitabilityPopulationState::invalid);
    assert(empty.summary.reason == "empty_profitability_population");
    auto malformed = Observation(50, 0.1);
    malformed.sourceContentHash.clear();
    assert(AnalyzeProfitabilityDistribution({malformed}, policy).summary.reason ==
           "empty_profitability_observation_provenance");

    // Locale does not alter canonical policy, summary, or result output.
    const std::locale previous = std::locale();
    std::locale::global(std::locale(previous, new CommaNumpunct));
    const auto localeAnalysis = AnalyzeProfitabilityDistribution(
        homogeneous, policy);
    std::locale::global(previous);
    assert(localeAnalysis.canonical == analysis.canonical);

    // Phase 3C observability is disconnected from production scoring/ranking.
    // Scores and ranks are identical before and after analysis, and neither
    // production scoring input nor its policy has a profitability field.
    const RecommendationScoringPolicy scoringPolicy;
    assert(RecommendationScoringPolicyCanonicalText(scoringPolicy).find(
        "profitability") == std::string::npos);
    auto firstInput = ScoringInput(2);
    auto secondInput = ScoringInput(1);
    const auto firstScore = ScoreExperimentRecommendation(
        scoringPolicy, firstInput);
    const auto secondScore = ScoreExperimentRecommendation(
        scoringPolicy, secondInput);
    const std::vector<RankedRecommendationScore> baselineRanked =
        RankRecommendationScores({{firstInput, firstScore, 0, 0, 0},
                                  {secondInput, secondScore, 0, 0, 0}});
    (void)AnalyzeProfitabilityDistribution(homogeneous, policy);
    const auto firstScoreAfter = ScoreExperimentRecommendation(
        scoringPolicy, firstInput);
    const auto secondScoreAfter = ScoreExperimentRecommendation(
        scoringPolicy, secondInput);
    const std::vector<RankedRecommendationScore> rankedAfter =
        RankRecommendationScores({{firstInput, firstScoreAfter, 0, 0, 0},
                                  {secondInput, secondScoreAfter, 0, 0, 0}});
    assert(firstScore.finalScore == firstScoreAfter.finalScore);
    assert(secondScore.finalScore == secondScoreAfter.finalScore);
    assert(std::none_of(firstScore.components.begin(), firstScore.components.end(),
        [](const auto& component) {
            return component.componentName.find("profitability") !=
                std::string::npos;
        }));
    assert(baselineRanked.size() == rankedAfter.size());
    for (std::size_t index = 0; index < baselineRanked.size(); ++index)
    {
        assert(baselineRanked[index].input.recommendationId ==
               rankedAfter[index].input.recommendationId);
        assert(baselineRanked[index].scoreRank == rankedAfter[index].scoreRank);
        assert(baselineRanked[index].tieGroup == rankedAfter[index].tieGroup);
        assert(baselineRanked[index].rankingOrdinal ==
               rankedAfter[index].rankingOrdinal);
    }

    const RecommendationRankingPolicy rankingPolicy;
    const std::vector<RecommendationRankingEvaluation> rankingInput{
        RankingEvaluation(2, 0.70), RankingEvaluation(1, 0.80)};
    const auto phase3BBaseline = RankRecommendationEvaluationEvidence(
        rankingPolicy, rankingInput, 100);
    (void)AnalyzeProfitabilityDistribution(homogeneous, policy);
    const auto phase3BAfter = RankRecommendationEvaluationEvidence(
        rankingPolicy, rankingInput, 100);
    assert(phase3BBaseline.size() == phase3BAfter.size());
    for (std::size_t index = 0; index < phase3BBaseline.size(); ++index)
    {
        assert(phase3BBaseline[index].evaluation.evaluationResultId ==
               phase3BAfter[index].evaluation.evaluationResultId);
        assert(phase3BBaseline[index].bucketRank ==
               phase3BAfter[index].bucketRank);
        assert(phase3BBaseline[index].globalOrdinal ==
               phase3BAfter[index].globalOrdinal);
    }

    static_assert(ProfitabilityNormalizationResult::profitabilityWeight == 0.0);
    static_assert(
        ProfitabilityNormalizationResult::profitabilityScoreContribution == 0.0);
    assert(analysis.canonical.find(
        "profitability_weight=0;profitability_score_contribution=0") !=
        std::string::npos);
    for (const auto& result : analysis.normalizationResults)
    {
        assert(result.profitabilityWeight == 0.0);
        assert(result.profitabilityScoreContribution == 0.0);
        assert(result.canonical.find(
            "profitability_weight=0;profitability_score_contribution=0") !=
            std::string::npos);
    }
    return 0;
}
