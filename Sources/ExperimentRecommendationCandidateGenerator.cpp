#include "ExperimentRecommendationCandidateGenerator.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iterator>
#include <map>
#include <set>
#include <tuple>
#include <utility>

namespace EA::ExperimentRecommendation
{
namespace
{

template <typename T>
std::vector<T> SortedUnique(std::vector<T> values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

int ParameterOrder(RecommendationMutationParameter parameter)
{
    switch (parameter)
    {
        case RecommendationMutationParameter::coreLrMult: return 0;
        case RecommendationMutationParameter::headLrMult: return 1;
        case RecommendationMutationParameter::labelThreshold: return 2;
        case RecommendationMutationParameter::predictionHorizon: return 3;
    }
    return 4;
}

bool CandidateLess(const GeneratedRecommendationCandidate& lhs,
                   const GeneratedRecommendationCandidate& rhs)
{
    return std::tuple{
               ParameterOrder(lhs.parameter),
               lhs.proposedValue,
               lhs.semanticIdentity.canonicalText,
               lhs.invocationIdentity.canonicalText} <
           std::tuple{
               ParameterOrder(rhs.parameter),
               rhs.proposedValue,
               rhs.semanticIdentity.canonicalText,
               rhs.invocationIdentity.canonicalText};
}

bool RejectionLess(const RejectedRecommendationCandidate& lhs,
                   const RejectedRecommendationCandidate& rhs)
{
    return std::tuple{
               ParameterOrder(lhs.parameter),
               lhs.proposedValue,
               RecommendationCandidateRejectionReasonText(lhs.reason),
               lhs.semanticCanonicalText} <
           std::tuple{
               ParameterOrder(rhs.parameter),
               rhs.proposedValue,
               RecommendationCandidateRejectionReasonText(rhs.reason),
               rhs.semanticCanonicalText};
}

std::size_t SemanticDifferenceCount(
    const EffectiveExperimentConfiguration& lhs,
    const EffectiveExperimentConfiguration& rhs)
{
    std::size_t differences = 0;
    differences += lhs.symbol != rhs.symbol;
    differences += lhs.predictionHorizon != rhs.predictionHorizon;
    differences += lhs.labelThreshold != rhs.labelThreshold;
    differences += lhs.coreLrMult != rhs.coreLrMult;
    differences += lhs.headLrMult != rhs.headLrMult;
    differences += lhs.targetEpochs != rhs.targetEpochs;
    differences += lhs.trainStartDate != rhs.trainStartDate;
    differences += lhs.trainEndDate != rhs.trainEndDate;
    differences += lhs.inferStartDate != rhs.inferStartDate;
    differences += lhs.inferEndDate != rhs.inferEndDate;
    return differences;
}

std::string ProposalText(double value)
{
    if (!std::isfinite(value)) return "nonfinite";
    return CanonicalRecommendationDouble(value);
}

RejectedRecommendationCandidate Rejection(
    RecommendationMutationParameter parameter,
    const std::string& proposedValue,
    RecommendationCandidateRejectionReason reason,
    const RecommendationCandidateIdentity* identity = nullptr)
{
    RejectedRecommendationCandidate rejected;
    rejected.parameter = parameter;
    rejected.proposedValue = proposedValue;
    rejected.reason = reason;
    if (identity != nullptr)
    {
        rejected.semanticCanonicalText = identity->canonicalText;
        rejected.semanticHash = identity->hash;
    }
    return rejected;
}

GeneratedRecommendationCandidate BuildCandidate(
    RecommendationMutationParameter parameter,
    double sourceValue,
    double proposedValue,
    const RecommendationSource& source,
    const EffectiveExperimentConfiguration& configuration,
    std::optional<int> horizonDelta)
{
    GeneratedRecommendationCandidate candidate;
    candidate.parameter = parameter;
    candidate.sourceValue = sourceValue;
    candidate.proposedValue = proposedValue;
    candidate.absoluteDelta = std::fabs(proposedValue - sourceValue);
    if (sourceValue != 0.0)
        candidate.relativeDelta = candidate.absoluteDelta / std::fabs(sourceValue);
    candidate.horizonDelta = horizonDelta;
    candidate.semanticIdentity =
        BuildRecommendationCandidateIdentity(configuration);

    ExperimentInvocationConfiguration invocation = source.invocation;
    invocation.configuration = candidate.semanticIdentity.configuration;
    candidate.invocationIdentity =
        BuildRecommendationInvocationIdentity(invocation);
    return candidate;
}

void CountRejection(RecommendationCandidateGenerationCounters& counters,
                    RecommendationCandidateRejectionReason reason)
{
    switch (reason)
    {
        case RecommendationCandidateRejectionReason::missingSourceValue:
            ++counters.rejectedMissingSourceValue;
            break;
        case RecommendationCandidateRejectionReason::invalidCandidateValue:
            ++counters.rejectedInvalidValue;
            break;
        case RecommendationCandidateRejectionReason::unchangedCandidate:
            ++counters.rejectedUnchanged;
            break;
        case RecommendationCandidateRejectionReason::singleParameterRuleViolation:
            ++counters.rejectedSingleParameterRule;
            break;
        case RecommendationCandidateRejectionReason::duplicateCanonicalCandidate:
            ++counters.rejectedDuplicate;
            break;
        case RecommendationCandidateRejectionReason::perSourceLimit:
            ++counters.rejectedByPerSourceLimit;
            break;
    }
}

} // namespace

std::string RecommendationSourceEligibilityReasonText(
    RecommendationSourceEligibilityReason reason)
{
    switch (reason)
    {
        case RecommendationSourceEligibilityReason::eligible: return "eligible";
        case RecommendationSourceEligibilityReason::policyDisabled: return "policy_disabled";
        case RecommendationSourceEligibilityReason::invalidPolicy: return "invalid_policy";
        case RecommendationSourceEligibilityReason::invalidSourceExperimentId: return "invalid_source_experiment_id";
        case RecommendationSourceEligibilityReason::invalidSemanticConfiguration: return "invalid_semantic_configuration";
        case RecommendationSourceEligibilityReason::invalidInvocationConfiguration: return "invalid_invocation_configuration";
        case RecommendationSourceEligibilityReason::missingLeaderScore: return "missing_leader_score";
        case RecommendationSourceEligibilityReason::missingInferenceAccuracy: return "missing_inference_accuracy";
        case RecommendationSourceEligibilityReason::nonfiniteLeaderScore: return "nonfinite_leader_score";
        case RecommendationSourceEligibilityReason::nonfiniteInferenceAccuracy: return "nonfinite_inference_accuracy";
        case RecommendationSourceEligibilityReason::insufficientEvidence: return "insufficient_evidence";
        case RecommendationSourceEligibilityReason::leaderScoreBelowMinimum: return "leader_score_below_minimum";
        case RecommendationSourceEligibilityReason::inferenceAccuracyBelowMinimum: return "inference_accuracy_below_minimum";
        case RecommendationSourceEligibilityReason::missingPredictedNeutralProportion: return "missing_predicted_neutral_proportion";
        case RecommendationSourceEligibilityReason::nonfinitePredictedNeutralProportion: return "nonfinite_predicted_neutral_proportion";
        case RecommendationSourceEligibilityReason::predictedNeutralProportionOutOfRange: return "predicted_neutral_proportion_out_of_range";
        case RecommendationSourceEligibilityReason::predictedNeutralProportionAboveMaximum: return "predicted_neutral_proportion_above_maximum";
    }
    return "invalid_source_eligibility_reason";
}

std::string RecommendationMutationParameterText(
    RecommendationMutationParameter parameter)
{
    switch (parameter)
    {
        case RecommendationMutationParameter::coreLrMult: return kCoreLrMult;
        case RecommendationMutationParameter::headLrMult: return kHeadLrMult;
        case RecommendationMutationParameter::labelThreshold: return kLabelThreshold;
        case RecommendationMutationParameter::predictionHorizon: return kPredictionHorizon;
    }
    return "invalid_recommendation_mutation_parameter";
}

std::string RecommendationCandidateRejectionReasonText(
    RecommendationCandidateRejectionReason reason)
{
    switch (reason)
    {
        case RecommendationCandidateRejectionReason::missingSourceValue: return "missing_source_value";
        case RecommendationCandidateRejectionReason::invalidCandidateValue: return "invalid_candidate_value";
        case RecommendationCandidateRejectionReason::unchangedCandidate: return "unchanged_candidate";
        case RecommendationCandidateRejectionReason::singleParameterRuleViolation: return "single_parameter_rule_violation";
        case RecommendationCandidateRejectionReason::duplicateCanonicalCandidate: return "duplicate_canonical_candidate";
        case RecommendationCandidateRejectionReason::perSourceLimit: return "per_source_limit";
    }
    return "invalid_candidate_rejection_reason";
}

RecommendationSourceEligibilityResult EvaluateRecommendationSource(
    const RecommendationPolicy& policy,
    const RecommendationSource& source)
{
    const auto reject = [](RecommendationSourceEligibilityReason reason,
                           std::string detail = {}) {
        return RecommendationSourceEligibilityResult{
            false, reason, std::move(detail)};
    };

    if (!policy.enabled)
        return reject(RecommendationSourceEligibilityReason::policyDisabled);
    if (const auto error = ValidateRecommendationPolicy(policy))
        return reject(
            RecommendationSourceEligibilityReason::invalidPolicy, *error);
    if (source.experimentId <= 0)
        return reject(
            RecommendationSourceEligibilityReason::invalidSourceExperimentId);
    try
    {
        (void)BuildRecommendationCandidateIdentity(
            source.invocation.configuration);
    }
    catch (const std::exception& error)
    {
        return reject(
            RecommendationSourceEligibilityReason::invalidSemanticConfiguration,
            error.what());
    }
    try
    {
        (void)BuildRecommendationInvocationIdentity(source.invocation);
    }
    catch (const std::exception& error)
    {
        return reject(
            RecommendationSourceEligibilityReason::invalidInvocationConfiguration,
            error.what());
    }
    if (!source.leaderScore)
        return reject(RecommendationSourceEligibilityReason::missingLeaderScore);
    if (!source.inferenceAccuracy)
        return reject(
            RecommendationSourceEligibilityReason::missingInferenceAccuracy);
    if (!std::isfinite(*source.leaderScore))
        return reject(
            RecommendationSourceEligibilityReason::nonfiniteLeaderScore);
    if (!std::isfinite(*source.inferenceAccuracy))
        return reject(
            RecommendationSourceEligibilityReason::nonfiniteInferenceAccuracy);
    if (source.evidenceCount < policy.minimumEvidenceCount)
        return reject(
            RecommendationSourceEligibilityReason::insufficientEvidence);
    if (*source.leaderScore < policy.minimumLeaderScore)
        return reject(
            RecommendationSourceEligibilityReason::leaderScoreBelowMinimum);
    if (*source.inferenceAccuracy < policy.minimumInferenceAccuracy)
        return reject(
            RecommendationSourceEligibilityReason::inferenceAccuracyBelowMinimum);
    if (policy.maximumPredictedNeutralProportion &&
        !source.predictedNeutralProportion)
        return reject(
            RecommendationSourceEligibilityReason::missingPredictedNeutralProportion);
    if (source.predictedNeutralProportion)
    {
        if (!std::isfinite(*source.predictedNeutralProportion))
            return reject(
                RecommendationSourceEligibilityReason::nonfinitePredictedNeutralProportion);
        if (*source.predictedNeutralProportion < 0.0 ||
            *source.predictedNeutralProportion > 1.0)
            return reject(
                RecommendationSourceEligibilityReason::predictedNeutralProportionOutOfRange);
        if (policy.maximumPredictedNeutralProportion &&
            *source.predictedNeutralProportion >
                *policy.maximumPredictedNeutralProportion)
            return reject(
                RecommendationSourceEligibilityReason::predictedNeutralProportionAboveMaximum);
    }
    return RecommendationSourceEligibilityResult{
        true, RecommendationSourceEligibilityReason::eligible, {}};
}

RecommendationCandidateDeduplicationResult
DeduplicateRecommendationCandidatesCanonical(
    std::vector<GeneratedRecommendationCandidate> candidates)
{
    std::sort(candidates.begin(), candidates.end(), CandidateLess);
    RecommendationCandidateDeduplicationResult result;
    std::set<std::string> canonicalTexts;
    std::map<std::string, std::vector<std::string>> hashBuckets;

    for (GeneratedRecommendationCandidate& candidate : candidates)
    {
        const std::string& canonical = candidate.semanticIdentity.canonicalText;
        const std::string& hash = candidate.semanticIdentity.hash;
        if (!canonicalTexts.insert(canonical).second)
        {
            result.rejected.push_back(Rejection(
                candidate.parameter,
                ProposalText(candidate.proposedValue),
                RecommendationCandidateRejectionReason::duplicateCanonicalCandidate,
                &candidate.semanticIdentity));
            continue;
        }

        auto& bucket = hashBuckets[hash];
        for (const std::string& existingCanonical : bucket)
        {
            if (existingCanonical != canonical)
            {
                result.collisions.push_back(RecommendationCandidateHashCollision{
                    hash, existingCanonical, canonical});
            }
        }
        bucket.push_back(canonical);
        result.candidates.push_back(std::move(candidate));
    }

    std::sort(result.rejected.begin(), result.rejected.end(), RejectionLess);
    return result;
}

RecommendationCandidateGenerationResult GenerateRecommendationCandidates(
    const RecommendationPolicy& policy,
    const RecommendationSource& source)
{
    RecommendationCandidateGenerationResult result;
    result.eligibility = EvaluateRecommendationSource(policy, source);
    if (!result.eligibility.eligible)
        return result;

    const RecommendationCandidateIdentity sourceIdentity =
        BuildRecommendationCandidateIdentity(source.invocation.configuration);
    const EffectiveExperimentConfiguration& sourceConfiguration =
        sourceIdentity.configuration;
    std::vector<GeneratedRecommendationCandidate> generated;

    const auto reject = [&](RecommendationMutationParameter parameter,
                            const std::string& proposedValue,
                            RecommendationCandidateRejectionReason reason,
                            const RecommendationCandidateIdentity* identity = nullptr) {
        result.rejected.push_back(
            Rejection(parameter, proposedValue, reason, identity));
        CountRejection(result.counters, reason);
    };

    const auto emitDoubleMutations = [&](RecommendationMutationParameter parameter,
                                         const std::optional<double>& sourceOptional,
                                         const std::vector<double>& offsets) {
        for (double offset : SortedUnique(offsets))
        {
            ++result.counters.attempted;
            if (!sourceOptional)
            {
                reject(parameter,
                       "NULL+" + CanonicalRecommendationDouble(offset),
                       RecommendationCandidateRejectionReason::missingSourceValue);
                continue;
            }

            const double sourceValue = *sourceOptional;
            const double proposedValue = sourceValue + offset;
            if (!std::isfinite(proposedValue) || proposedValue <= 0.0)
            {
                reject(parameter, ProposalText(proposedValue),
                       RecommendationCandidateRejectionReason::invalidCandidateValue);
                continue;
            }
            if (proposedValue == sourceValue)
            {
                reject(parameter, ProposalText(proposedValue),
                       RecommendationCandidateRejectionReason::unchangedCandidate);
                continue;
            }

            EffectiveExperimentConfiguration configuration = sourceConfiguration;
            if (parameter == RecommendationMutationParameter::coreLrMult)
                configuration.coreLrMult = proposedValue;
            else
                configuration.headLrMult = proposedValue;

            if (SemanticDifferenceCount(sourceConfiguration, configuration) != 1)
            {
                reject(parameter, ProposalText(proposedValue),
                       RecommendationCandidateRejectionReason::singleParameterRuleViolation);
                continue;
            }
            generated.push_back(BuildCandidate(
                parameter, sourceValue, proposedValue, source, configuration,
                std::nullopt));
            ++result.counters.validBeforeDeduplication;
        }
    };

    if (policy.allowedParameters.contains(kCoreLrMult))
        emitDoubleMutations(
            RecommendationMutationParameter::coreLrMult,
            sourceConfiguration.coreLrMult,
            policy.coreLrOffsets);
    if (policy.allowedParameters.contains(kHeadLrMult))
        emitDoubleMutations(
            RecommendationMutationParameter::headLrMult,
            sourceConfiguration.headLrMult,
            policy.headLrOffsets);

    if (policy.allowedParameters.contains(kLabelThreshold))
    {
        for (double offset : SortedUnique(policy.labelThresholdOffsets))
        {
            ++result.counters.attempted;
            const double sourceValue = sourceConfiguration.labelThreshold;
            const double proposedValue = sourceValue + offset;
            if (!std::isfinite(proposedValue) || proposedValue <= 0.0)
            {
                reject(RecommendationMutationParameter::labelThreshold,
                       ProposalText(proposedValue),
                       RecommendationCandidateRejectionReason::invalidCandidateValue);
                continue;
            }
            if (proposedValue == sourceValue)
            {
                reject(RecommendationMutationParameter::labelThreshold,
                       ProposalText(proposedValue),
                       RecommendationCandidateRejectionReason::unchangedCandidate);
                continue;
            }
            EffectiveExperimentConfiguration configuration = sourceConfiguration;
            configuration.labelThreshold = proposedValue;
            if (SemanticDifferenceCount(sourceConfiguration, configuration) != 1)
            {
                reject(RecommendationMutationParameter::labelThreshold,
                       ProposalText(proposedValue),
                       RecommendationCandidateRejectionReason::singleParameterRuleViolation);
                continue;
            }
            generated.push_back(BuildCandidate(
                RecommendationMutationParameter::labelThreshold,
                sourceValue, proposedValue, source, configuration,
                std::nullopt));
            ++result.counters.validBeforeDeduplication;
        }
    }

    if (policy.allowHorizonChanges &&
        policy.allowedParameters.contains(kPredictionHorizon))
    {
        for (int proposedHorizon : SortedUnique(policy.permittedHorizons))
        {
            ++result.counters.attempted;
            const int sourceHorizon = sourceConfiguration.predictionHorizon;
            if (proposedHorizon == sourceHorizon)
            {
                reject(RecommendationMutationParameter::predictionHorizon,
                       std::to_string(proposedHorizon),
                       RecommendationCandidateRejectionReason::unchangedCandidate);
                continue;
            }
            if (proposedHorizon <= 0)
            {
                reject(RecommendationMutationParameter::predictionHorizon,
                       std::to_string(proposedHorizon),
                       RecommendationCandidateRejectionReason::invalidCandidateValue);
                continue;
            }
            EffectiveExperimentConfiguration configuration = sourceConfiguration;
            configuration.predictionHorizon = proposedHorizon;
            if (SemanticDifferenceCount(sourceConfiguration, configuration) != 1)
            {
                reject(RecommendationMutationParameter::predictionHorizon,
                       std::to_string(proposedHorizon),
                       RecommendationCandidateRejectionReason::singleParameterRuleViolation);
                continue;
            }
            generated.push_back(BuildCandidate(
                RecommendationMutationParameter::predictionHorizon,
                static_cast<double>(sourceHorizon),
                static_cast<double>(proposedHorizon), source, configuration,
                proposedHorizon - sourceHorizon));
            ++result.counters.validBeforeDeduplication;
        }
    }

    RecommendationCandidateDeduplicationResult deduplicated =
        DeduplicateRecommendationCandidatesCanonical(std::move(generated));
    result.counters.rejectedDuplicate += deduplicated.rejected.size();
    result.counters.hashCollisions = deduplicated.collisions.size();
    result.rejected.insert(
        result.rejected.end(),
        std::make_move_iterator(deduplicated.rejected.begin()),
        std::make_move_iterator(deduplicated.rejected.end()));
    result.collisions = std::move(deduplicated.collisions);
    result.candidates = std::move(deduplicated.candidates);

    const std::size_t limit = static_cast<std::size_t>(
        policy.maximumRecommendationsPerSource);
    if (result.candidates.size() > limit)
    {
        for (std::size_t index = limit; index < result.candidates.size(); ++index)
        {
            const GeneratedRecommendationCandidate& candidate =
                result.candidates[index];
            reject(candidate.parameter, ProposalText(candidate.proposedValue),
                   RecommendationCandidateRejectionReason::perSourceLimit,
                   &candidate.semanticIdentity);
        }
        result.candidates.resize(limit);
    }
    std::sort(result.rejected.begin(), result.rejected.end(), RejectionLess);
    result.counters.emitted = result.candidates.size();
    return result;
}

} // namespace EA::ExperimentRecommendation
