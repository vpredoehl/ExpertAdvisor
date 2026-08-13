#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "../Sources/ExperimentRecommendationCandidateGenerator.hpp"

using namespace EA::ExperimentRecommendation;

namespace
{

EffectiveExperimentConfiguration SemanticConfiguration()
{
    EffectiveExperimentConfiguration configuration;
    configuration.symbol = "eurusd";
    configuration.predictionHorizon = 12;
    configuration.labelThreshold = 0.001;
    configuration.coreLrMult = 1.0;
    configuration.headLrMult = 5.0;
    configuration.targetEpochs = 120;
    configuration.trainStartDate = "2010-01-01";
    configuration.trainEndDate = "2025-01-01";
    configuration.inferStartDate = "2025-01-01";
    configuration.inferEndDate = "2026-01-01";
    return configuration;
}

RecommendationSource EligibleSource()
{
    RecommendationSource source;
    source.experimentId = 143;
    source.modelId = 645;
    source.analysisId = 9001;
    source.invocation.configuration = SemanticConfiguration();
    source.invocation.checkpointInterval = 20;
    source.invocation.resumeModelId = 41;
    source.leaderScore = 0.75;
    source.inferenceAccuracy = 0.65;
    source.predictedNeutralProportion = 0.40;
    source.evidenceCount = 5;
    return source;
}

RecommendationPolicy UnboundedPolicy()
{
    RecommendationPolicy policy;
    policy.minimumLeaderScore = 0.50;
    policy.minimumInferenceAccuracy = 0.55;
    policy.minimumEvidenceCount = 3;
    policy.maximumRecommendationsPerSource = 100;
    return policy;
}

std::size_t DifferenceCount(const EffectiveExperimentConfiguration& lhs,
                            const EffectiveExperimentConfiguration& rhs)
{
    std::size_t count = 0;
    count += lhs.symbol != rhs.symbol;
    count += lhs.predictionHorizon != rhs.predictionHorizon;
    count += lhs.labelThreshold != rhs.labelThreshold;
    count += lhs.coreLrMult != rhs.coreLrMult;
    count += lhs.headLrMult != rhs.headLrMult;
    count += lhs.targetEpochs != rhs.targetEpochs;
    count += lhs.trainStartDate != rhs.trainStartDate;
    count += lhs.trainEndDate != rhs.trainEndDate;
    count += lhs.inferStartDate != rhs.inferStartDate;
    count += lhs.inferEndDate != rhs.inferEndDate;
    return count;
}

std::string CandidateBytes(
    const RecommendationCandidateGenerationResult& result)
{
    std::string bytes;
    for (const auto& candidate : result.candidates)
    {
        bytes += RecommendationMutationParameterText(candidate.parameter);
        bytes += '|';
        bytes += CanonicalRecommendationDouble(candidate.proposedValue);
        bytes += '|';
        bytes += candidate.semanticIdentity.canonicalText;
        bytes += '|';
        bytes += candidate.semanticIdentity.hash;
        bytes += '|';
        bytes += candidate.invocationIdentity.canonicalText;
        bytes += '|';
        bytes += candidate.invocationIdentity.hash;
        bytes += '\n';
    }
    for (const auto& rejected : result.rejected)
    {
        bytes += "rejected|";
        bytes += RecommendationMutationParameterText(rejected.parameter);
        bytes += '|';
        bytes += rejected.proposedValue;
        bytes += '|';
        bytes += RecommendationCandidateRejectionReasonText(rejected.reason);
        bytes += '\n';
    }
    return bytes;
}

RecommendationPolicy SingleParameterPolicy(const char* parameter)
{
    RecommendationPolicy policy = UnboundedPolicy();
    policy.allowedParameters = {parameter};
    policy.coreLrOffsets.clear();
    policy.headLrOffsets.clear();
    policy.labelThresholdOffsets.clear();
    if (std::string{parameter} == kCoreLrMult)
        policy.coreLrOffsets = {-0.25, 0.25};
    else if (std::string{parameter} == kHeadLrMult)
        policy.headLrOffsets = {-0.5, 0.5};
    else if (std::string{parameter} == kLabelThreshold)
        policy.labelThresholdOffsets = {-0.0001, 0.0001};
    return policy;
}

} // namespace

int main()
{
    const RecommendationSource source = EligibleSource();
    const RecommendationPolicy policy = UnboundedPolicy();

    const auto eligible = EvaluateRecommendationSource(policy, source);
    assert(eligible.eligible);
    assert(eligible.reason == RecommendationSourceEligibilityReason::eligible);
    assert(RecommendationSourceEligibilityReasonText(eligible.reason) ==
           "eligible");

    RecommendationPolicy disabled = policy;
    disabled.enabled = false;
    assert(EvaluateRecommendationSource(disabled, source).reason ==
           RecommendationSourceEligibilityReason::policyDisabled);

    RecommendationSource changed = source;
    changed.experimentId = 0;
    assert(EvaluateRecommendationSource(policy, changed).reason ==
           RecommendationSourceEligibilityReason::invalidSourceExperimentId);
    changed = source;
    changed.leaderScore.reset();
    assert(EvaluateRecommendationSource(policy, changed).reason ==
           RecommendationSourceEligibilityReason::missingLeaderScore);
    changed = source;
    changed.inferenceAccuracy.reset();
    assert(EvaluateRecommendationSource(policy, changed).reason ==
           RecommendationSourceEligibilityReason::missingInferenceAccuracy);
    changed = source;
    changed.evidenceCount = 2;
    assert(EvaluateRecommendationSource(policy, changed).reason ==
           RecommendationSourceEligibilityReason::insufficientEvidence);
    changed = source;
    changed.leaderScore = 0.49;
    assert(EvaluateRecommendationSource(policy, changed).reason ==
           RecommendationSourceEligibilityReason::leaderScoreBelowMinimum);
    changed = source;
    changed.inferenceAccuracy = 0.54;
    assert(EvaluateRecommendationSource(policy, changed).reason ==
           RecommendationSourceEligibilityReason::inferenceAccuracyBelowMinimum);
    changed = source;
    changed.predictedNeutralProportion.reset();
    assert(EvaluateRecommendationSource(policy, changed).reason ==
           RecommendationSourceEligibilityReason::missingPredictedNeutralProportion);
    changed = source;
    changed.predictedNeutralProportion = 0.81;
    assert(EvaluateRecommendationSource(policy, changed).reason ==
           RecommendationSourceEligibilityReason::predictedNeutralProportionAboveMaximum);
    changed = source;
    changed.leaderScore = std::numeric_limits<double>::infinity();
    assert(EvaluateRecommendationSource(policy, changed).reason ==
           RecommendationSourceEligibilityReason::nonfiniteLeaderScore);
    changed = source;
    changed.inferenceAccuracy = std::numeric_limits<double>::quiet_NaN();
    assert(EvaluateRecommendationSource(policy, changed).reason ==
           RecommendationSourceEligibilityReason::nonfiniteInferenceAccuracy);
    changed = source;
    changed.predictedNeutralProportion =
        std::numeric_limits<double>::infinity();
    assert(EvaluateRecommendationSource(policy, changed).reason ==
           RecommendationSourceEligibilityReason::nonfinitePredictedNeutralProportion);
    changed = source;
    changed.predictedNeutralProportion = 1.1;
    assert(EvaluateRecommendationSource(policy, changed).reason ==
           RecommendationSourceEligibilityReason::predictedNeutralProportionOutOfRange);
    changed = source;
    changed.invocation.configuration.trainStartDate = "2010-02-30";
    assert(EvaluateRecommendationSource(policy, changed).reason ==
           RecommendationSourceEligibilityReason::invalidSemanticConfiguration);
    changed = source;
    changed.invocation.checkpointInterval = 0;
    assert(EvaluateRecommendationSource(policy, changed).reason ==
           RecommendationSourceEligibilityReason::invalidInvocationConfiguration);

    RecommendationPolicy neutralMetricOptional = policy;
    neutralMetricOptional.maximumPredictedNeutralProportion.reset();
    changed = source;
    changed.predictedNeutralProportion.reset();
    assert(EvaluateRecommendationSource(neutralMetricOptional, changed).eligible);

    RecommendationPolicy invalidPolicy = policy;
    invalidPolicy.maximumRecommendationsPerSource = 0;
    const auto invalidPolicyResult =
        GenerateRecommendationCandidates(invalidPolicy, source);
    assert(!invalidPolicyResult.eligibility.eligible);
    assert(invalidPolicyResult.eligibility.reason ==
           RecommendationSourceEligibilityReason::invalidPolicy);
    assert(invalidPolicyResult.candidates.empty());

    const RecommendationCandidateGenerationResult generated =
        GenerateRecommendationCandidates(policy, source);
    assert(generated.eligibility.eligible);
    assert(generated.candidates.size() == 6);
    assert(generated.counters.attempted == 6);
    assert(generated.counters.validBeforeDeduplication == 6);
    assert(generated.counters.emitted == 6);
    assert(generated.rejected.empty());
    assert(generated.collisions.empty());

    const std::vector<RecommendationMutationParameter> expectedParameters{
        RecommendationMutationParameter::coreLrMult,
        RecommendationMutationParameter::coreLrMult,
        RecommendationMutationParameter::headLrMult,
        RecommendationMutationParameter::headLrMult,
        RecommendationMutationParameter::labelThreshold,
        RecommendationMutationParameter::labelThreshold};
    for (std::size_t index = 0; index < generated.candidates.size(); ++index)
    {
        const auto& candidate = generated.candidates[index];
        assert(candidate.parameter == expectedParameters[index]);
        assert(DifferenceCount(source.invocation.configuration,
                               candidate.semanticIdentity.configuration) == 1);
        assert(candidate.invocationIdentity.invocation.checkpointInterval ==
               source.invocation.checkpointInterval);
        assert(candidate.invocationIdentity.invocation.resumeModelId ==
               source.invocation.resumeModelId);
        assert(candidate.semanticIdentity.configuration.targetEpochs ==
               source.invocation.configuration.targetEpochs);
        assert(candidate.semanticIdentity.configuration.symbol == "eurusd");
        assert(candidate.semanticIdentity.configuration.trainStartDate ==
               source.invocation.configuration.trainStartDate);
        assert(candidate.semanticIdentity.configuration.trainEndDate ==
               source.invocation.configuration.trainEndDate);
        assert(candidate.semanticIdentity.configuration.inferStartDate ==
               source.invocation.configuration.inferStartDate);
        assert(candidate.semanticIdentity.configuration.inferEndDate ==
               source.invocation.configuration.inferEndDate);
        assert(candidate.semanticIdentity.hash ==
               RecommendationCandidateHash(
                   candidate.semanticIdentity.configuration));
        assert(candidate.invocationIdentity.hash ==
               ExperimentInvocationHash(
                   candidate.invocationIdentity.invocation));
        assert(candidate.absoluteDelta > 0.0);
        assert(candidate.relativeDelta.has_value());
        assert(!candidate.horizonDelta.has_value());
    }
    assert(generated.candidates[0].proposedValue == 0.75);
    assert(generated.candidates[1].proposedValue == 1.25);
    assert(generated.candidates[2].proposedValue == 4.5);
    assert(generated.candidates[3].proposedValue == 5.5);
    assert(generated.candidates[4].proposedValue == 0.0009);
    assert(generated.candidates[5].proposedValue == 0.0011);

    RecommendationPolicy horizonPolicy = SingleParameterPolicy(kPredictionHorizon);
    horizonPolicy.allowHorizonChanges = true;
    horizonPolicy.permittedHorizons = {16, 8, 4, 12, 8};
    const auto horizons = GenerateRecommendationCandidates(horizonPolicy, source);
    assert(horizons.candidates.size() == 3);
    assert(horizons.candidates[0].proposedValue == 4.0);
    assert(horizons.candidates[1].proposedValue == 8.0);
    assert(horizons.candidates[2].proposedValue == 16.0);
    assert(horizons.candidates[0].horizonDelta == -8);
    assert(horizons.candidates[2].horizonDelta == 4);
    assert(horizons.counters.rejectedUnchanged == 1);

    assert(std::none_of(
        generated.candidates.begin(), generated.candidates.end(),
        [](const auto& candidate) {
            return candidate.parameter ==
                RecommendationMutationParameter::predictionHorizon;
        }));

    RecommendationSource nullable = source;
    nullable.invocation.configuration.coreLrMult.reset();
    const auto missingSource = GenerateRecommendationCandidates(
        SingleParameterPolicy(kCoreLrMult), nullable);
    assert(missingSource.candidates.empty());
    assert(missingSource.counters.rejectedMissingSourceValue == 2);

    RecommendationPolicy duplicateOffsets = SingleParameterPolicy(kCoreLrMult);
    duplicateOffsets.coreLrOffsets = {0.25, -0.25, 0.25, -0.25};
    const auto deduplicatedOffsets =
        GenerateRecommendationCandidates(duplicateOffsets, source);
    assert(deduplicatedOffsets.candidates.size() == 2);
    assert(deduplicatedOffsets.counters.attempted == 2);

    RecommendationPolicy reorderedOffsets = duplicateOffsets;
    reorderedOffsets.coreLrOffsets = {-0.25, 0.25};
    assert(CandidateBytes(GenerateRecommendationCandidates(
               duplicateOffsets, source)) ==
           CandidateBytes(GenerateRecommendationCandidates(
               reorderedOffsets, source)));

    RecommendationPolicy reorderedHorizons = horizonPolicy;
    reorderedHorizons.permittedHorizons = {8, 16, 12, 4};
    assert(CandidateBytes(horizons) == CandidateBytes(
        GenerateRecommendationCandidates(reorderedHorizons, source)));

    RecommendationPolicy invalidValues = SingleParameterPolicy(kCoreLrMult);
    invalidValues.coreLrOffsets = {
        -2.0, -1.0, std::numeric_limits<double>::denorm_min()};
    const auto rejectedValues =
        GenerateRecommendationCandidates(invalidValues, source);
    assert(rejectedValues.candidates.empty());
    assert(rejectedValues.counters.rejectedInvalidValue == 2);
    assert(rejectedValues.counters.rejectedUnchanged == 1);

    std::vector<GeneratedRecommendationCandidate> exactDuplicates{
        generated.candidates[0], generated.candidates[0]};
    const auto exactDedup = DeduplicateRecommendationCandidatesCanonical(
        std::move(exactDuplicates));
    assert(exactDedup.candidates.size() == 1);
    assert(exactDedup.rejected.size() == 1);
    assert(exactDedup.collisions.empty());

    GeneratedRecommendationCandidate collisionA = generated.candidates[0];
    GeneratedRecommendationCandidate collisionB = generated.candidates[1];
    collisionB.semanticIdentity.hash = collisionA.semanticIdentity.hash;
    const auto collision = DeduplicateRecommendationCandidatesCanonical(
        {collisionA, collisionB});
    assert(collision.candidates.size() == 2);
    assert(collision.rejected.empty());
    assert(collision.collisions.size() == 1);
    assert(collision.collisions[0].hash == collisionA.semanticIdentity.hash);

    RecommendationPolicy limited = policy;
    limited.maximumRecommendationsPerSource = 2;
    const auto limitedResult = GenerateRecommendationCandidates(limited, source);
    assert(limitedResult.candidates.size() == 2);
    assert(limitedResult.candidates[0].parameter ==
           RecommendationMutationParameter::coreLrMult);
    assert(limitedResult.candidates[1].parameter ==
           RecommendationMutationParameter::coreLrMult);
    assert(limitedResult.counters.rejectedByPerSourceLimit == 4);
    assert(limitedResult.rejected.size() == 4);

    assert(CandidateBytes(generated) == CandidateBytes(
        GenerateRecommendationCandidates(policy, source)));

    assert(RecommendationPolicyHash(RecommendationPolicy{}) ==
           "fnv1a64:5d38b796e2380a45");
    assert(RecommendationCandidateHash(SemanticConfiguration()) ==
           "fnv1a64:e55c515fcbe9a1ec");
    assert(ExperimentInvocationHash(source.invocation) ==
           "fnv1a64:c760e6edd2e4c425");

    return 0;
}
