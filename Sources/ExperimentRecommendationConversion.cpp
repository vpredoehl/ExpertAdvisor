#include "ExperimentRecommendationConversion.hpp"

#include <charconv>
#include <cmath>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

std::string LengthText(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

RecommendationConversionResult Rejected(RecommendationConversionReason reason)
{
    return RecommendationConversionResult{
        RecommendationConversionEligibilityResult{
            false, reason, RecommendationConversionReasonExplanation(reason)},
        std::nullopt};
}

bool CanonicalMatchesHash(const std::string& canonical,
                          const std::string& hash)
{
    return !canonical.empty() && hash == RecommendationCanonicalHash(canonical);
}

bool IsBlockedDisposition(RecommendationEvaluationDisposition disposition)
{
    return disposition ==
               RecommendationEvaluationDisposition::blockedPendingDuplicate ||
        disposition ==
               RecommendationEvaluationDisposition::blockedActiveDuplicate ||
        disposition == RecommendationEvaluationDisposition::completedDuplicate;
}

std::optional<RecommendationMutationParameter> ParseMutationParameter(
    const std::string& value)
{
    if (value == kCoreLrMult)
        return RecommendationMutationParameter::coreLrMult;
    if (value == kHeadLrMult)
        return RecommendationMutationParameter::headLrMult;
    if (value == kLabelThreshold)
        return RecommendationMutationParameter::labelThreshold;
    if (value == kPredictionHorizon)
        return RecommendationMutationParameter::predictionHorizon;
    return std::nullopt;
}

struct ParsedMutationValue
{
    double doubleValue = 0.0;
    int integerValue = 0;
};

enum class ProposedValueValidation
{
    valid,
    malformed,
    outsideRange
};

ProposedValueValidation ParseProposedValue(
    RecommendationMutationParameter parameter,
    const std::string& canonical,
    ParsedMutationValue& parsed)
{
    if (parameter == RecommendationMutationParameter::predictionHorizon)
    {
        if (canonical.empty() || canonical.front() == '+' ||
            (canonical.size() > 1 && canonical.front() == '0'))
            return ProposedValueValidation::malformed;
        long long value = 0;
        const auto result = std::from_chars(
            canonical.data(), canonical.data() + canonical.size(), value, 10);
        if (result.ec == std::errc::result_out_of_range)
            return ProposedValueValidation::outsideRange;
        if (result.ec != std::errc{} ||
            result.ptr != canonical.data() + canonical.size())
            return ProposedValueValidation::malformed;
        if (value <= 0 || value > std::numeric_limits<int>::max())
            return ProposedValueValidation::outsideRange;
        if (canonical != std::to_string(value))
            return ProposedValueValidation::malformed;
        parsed.integerValue = static_cast<int>(value);
        parsed.doubleValue = static_cast<double>(value);
        return ProposedValueValidation::valid;
    }

    double value = 0.0;
    const auto result = std::from_chars(
        canonical.data(), canonical.data() + canonical.size(), value,
        std::chars_format::general);
    if (result.ec == std::errc::result_out_of_range)
        return ProposedValueValidation::outsideRange;
    if (result.ec != std::errc{} ||
        result.ptr != canonical.data() + canonical.size() ||
        !std::isfinite(value))
        return ProposedValueValidation::malformed;
    if (value <= 0.0) return ProposedValueValidation::outsideRange;
    if (canonical != CanonicalRecommendationDouble(value))
        return ProposedValueValidation::malformed;
    parsed.doubleValue = value;
    return ProposedValueValidation::valid;
}

std::optional<std::string> SourceValueCanonical(
    RecommendationMutationParameter parameter,
    const EffectiveExperimentConfiguration& source)
{
    switch (parameter)
    {
        case RecommendationMutationParameter::coreLrMult:
            if (!source.coreLrMult) return std::nullopt;
            return CanonicalRecommendationDouble(*source.coreLrMult);
        case RecommendationMutationParameter::headLrMult:
            if (!source.headLrMult) return std::nullopt;
            return CanonicalRecommendationDouble(*source.headLrMult);
        case RecommendationMutationParameter::labelThreshold:
            return CanonicalRecommendationDouble(source.labelThreshold);
        case RecommendationMutationParameter::predictionHorizon:
            return std::to_string(source.predictionHorizon);
    }
    return std::nullopt;
}

void ApplyMutation(RecommendationMutationParameter parameter,
                   const ParsedMutationValue& value,
                   EffectiveExperimentConfiguration& configuration)
{
    switch (parameter)
    {
        case RecommendationMutationParameter::coreLrMult:
            configuration.coreLrMult = value.doubleValue;
            break;
        case RecommendationMutationParameter::headLrMult:
            configuration.headLrMult = value.doubleValue;
            break;
        case RecommendationMutationParameter::labelThreshold:
            configuration.labelThreshold = value.doubleValue;
            break;
        case RecommendationMutationParameter::predictionHorizon:
            configuration.predictionHorizon = value.integerValue;
            break;
    }
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
    differences += lhs.donchian20Mode != rhs.donchian20Mode;
    return differences;
}

std::string AuthorizationCanonical(
    const RecommendationConversionReviewAuthorization& authorization)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_conversion_authorization_v1"
        << ";recommendation_id=" << authorization.recommendationId
        << ";latest_action="
        << RecommendationReviewActionText(authorization.latestAction)
        << ";resulting_status="
        << RecommendationStatusText(authorization.resultingStatus)
        << ";review_event="
        << LengthText(authorization.authorizationCanonical);
    return out.str();
}

std::string ConversionIdentityCanonical(
    const RecommendationConversionRequest& request,
    RecommendationMutationParameter parameter,
    const std::string& sourceValue,
    const std::string& proposedValue,
    const std::string& sourceInvocationCanonical,
    const std::string& proposedInvocationCanonical)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_conversion_identity_v1"
        << ";contract_version=" << kRecommendationConversionContractVersion
        << ";recommendation_id=" << request.recommendationId
        << ";source_experiment_id=" << request.sourceExperimentId
        << ";recommendation_semantic="
        << LengthText(request.recommendationSemanticCanonical)
        << ";evaluation_identity="
        << LengthText(request.evaluation.evaluationIdentityCanonical)
        << ";evaluation_policy="
        << LengthText(request.evaluation.evaluationPolicyCanonical)
        << ";scoring_policy="
        << LengthText(request.score.scoringPolicyCanonical)
        << ";manual_authorization="
        << LengthText(AuthorizationCanonical(request.reviewAuthorization))
        << ";changed_parameter="
        << RecommendationMutationParameterText(parameter)
        << ";source_value=" << LengthText(sourceValue)
        << ";proposed_value=" << LengthText(proposedValue)
        << ";source_invocation=" << LengthText(sourceInvocationCanonical)
        << ";proposed_invocation=" << LengthText(proposedInvocationCanonical);
    return out.str();
}

bool ValidOptionalRanking(
    const std::optional<RecommendationConversionRankingProvenance>& ranking)
{
    if (!ranking) return true;
    try
    {
        (void)RecommendationRankingBucketText(ranking->bucket);
    }
    catch (const std::invalid_argument&)
    {
        return false;
    }
    return ranking->bucketRank > 0 &&
        CanonicalMatchesHash(
            ranking->snapshotIdentityCanonical,
            ranking->snapshotIdentityHash) &&
        CanonicalMatchesHash(
            ranking->memberIdentityCanonical,
            ranking->memberIdentityHash);
}

} // namespace

std::string RecommendationConversionEvidenceStateText(
    RecommendationConversionEvidenceState value)
{
    switch (value)
    {
        case RecommendationConversionEvidenceState::missing: return "missing";
        case RecommendationConversionEvidenceState::pending: return "pending";
        case RecommendationConversionEvidenceState::completed: return "completed";
        case RecommendationConversionEvidenceState::failed: return "failed";
    }
    throw std::invalid_argument("invalid_recommendation_conversion_evidence_state");
}

std::string RecommendationConversionReasonText(
    RecommendationConversionReason value)
{
    switch (value)
    {
        case RecommendationConversionReason::eligible: return "eligible";
        case RecommendationConversionReason::recommendationMissing: return "recommendation_missing";
        case RecommendationConversionReason::recommendationLifecycleNotConvertible: return "recommendation_lifecycle_not_convertible";
        case RecommendationConversionReason::manualAuthorizationMissing: return "manual_authorization_missing";
        case RecommendationConversionReason::manualAuthorizationSupersededOrIneffective: return "manual_authorization_superseded_or_ineffective";
        case RecommendationConversionReason::evaluationMissingOrIncomplete: return "evaluation_missing_or_incomplete";
        case RecommendationConversionReason::evaluationInvalid: return "evaluation_invalid";
        case RecommendationConversionReason::scoreMissingOrIncomplete: return "score_missing_or_incomplete";
        case RecommendationConversionReason::scoreInvalid: return "score_invalid";
        case RecommendationConversionReason::recommendationBlocked: return "recommendation_blocked";
        case RecommendationConversionReason::mutationMissing: return "mutation_missing";
        case RecommendationConversionReason::unsupportedMutationFamily: return "unsupported_mutation_family";
        case RecommendationConversionReason::malformedProposedValue: return "malformed_proposed_value";
        case RecommendationConversionReason::proposedValueOutsideAllowedRange: return "proposed_value_outside_allowed_range";
        case RecommendationConversionReason::sourceConfigurationIncomplete: return "source_configuration_incomplete";
        case RecommendationConversionReason::sourceValueMismatch: return "source_value_mismatch";
        case RecommendationConversionReason::proposedValueEqualsSourceValue: return "proposed_value_equals_source_value";
        case RecommendationConversionReason::multipleMutationsDetected: return "multiple_mutations_detected";
        case RecommendationConversionReason::duplicateConversionIdentity: return "duplicate_conversion_identity";
        case RecommendationConversionReason::inconsistentProvenance: return "inconsistent_provenance";
        case RecommendationConversionReason::inconsistentSourceExperimentIdentity: return "inconsistent_source_experiment_identity";
    }
    throw std::invalid_argument("invalid_recommendation_conversion_reason");
}

std::string RecommendationConversionReasonExplanation(
    RecommendationConversionReason value)
{
    switch (value)
    {
        case RecommendationConversionReason::eligible:
            return "The reviewed recommendation can produce a deterministic proposed experiment specification.";
        case RecommendationConversionReason::recommendationMissing:
            return "The recommendation does not exist in the supplied evidence.";
        case RecommendationConversionReason::recommendationLifecycleNotConvertible:
            return "Only an approved recommendation is convertible by this contract.";
        case RecommendationConversionReason::manualAuthorizationMissing:
            return "Explicit effective human review authorization is required.";
        case RecommendationConversionReason::manualAuthorizationSupersededOrIneffective:
            return "The supplied review action is not the latest effective approval.";
        case RecommendationConversionReason::evaluationMissingOrIncomplete:
            return "A completed recommendation evaluation is required.";
        case RecommendationConversionReason::evaluationInvalid:
            return "The completed recommendation evaluation is not valid for conversion.";
        case RecommendationConversionReason::scoreMissingOrIncomplete:
            return "A completed recommendation score is required.";
        case RecommendationConversionReason::scoreInvalid:
            return "The completed recommendation score is invalid.";
        case RecommendationConversionReason::recommendationBlocked:
            return "The recommendation evaluation is blocked by duplicate evidence.";
        case RecommendationConversionReason::mutationMissing:
            return "Exactly one supported recommendation mutation is required.";
        case RecommendationConversionReason::unsupportedMutationFamily:
            return "The recommendation mutation family is not supported.";
        case RecommendationConversionReason::malformedProposedValue:
            return "The proposed value is not strict canonical numeric text.";
        case RecommendationConversionReason::proposedValueOutsideAllowedRange:
            return "The proposed value is outside the existing recommendation-domain range.";
        case RecommendationConversionReason::sourceConfigurationIncomplete:
            return "The supplied source configuration cannot authoritatively derive the proposal.";
        case RecommendationConversionReason::sourceValueMismatch:
            return "The recommendation source value does not match the supplied source configuration.";
        case RecommendationConversionReason::proposedValueEqualsSourceValue:
            return "The proposed value does not change the source configuration.";
        case RecommendationConversionReason::multipleMutationsDetected:
            return "The request represents more than one mutation.";
        case RecommendationConversionReason::duplicateConversionIdentity:
            return "The exact canonical conversion identity already exists in the supplied evidence.";
        case RecommendationConversionReason::inconsistentProvenance:
            return "The supplied recommendation, review, evaluation, score, or ranking provenance is inconsistent.";
        case RecommendationConversionReason::inconsistentSourceExperimentIdentity:
            return "The recommendation and source configuration identify different source experiments.";
    }
    throw std::invalid_argument("invalid_recommendation_conversion_reason");
}

RecommendationConversionResult BuildProposedExperimentSpecification(
    const RecommendationConversionRequest& request)
{
    if (!request.recommendationExists || request.recommendationId <= 0)
        return Rejected(RecommendationConversionReason::recommendationMissing);
    if (request.sourceExperimentId <= 0 ||
        request.recommendationSourceExperimentId <= 0 ||
        request.sourceExperimentId != request.recommendationSourceExperimentId)
        return Rejected(
            RecommendationConversionReason::inconsistentSourceExperimentIdentity);

    const auto& authorization = request.reviewAuthorization;
    if (!authorization.present)
        return Rejected(
            RecommendationConversionReason::manualAuthorizationMissing);
    if (authorization.recommendationId != request.recommendationId ||
        authorization.latestAction != RecommendationReviewAction::approve ||
        authorization.resultingStatus != RecommendationStatus::approved ||
        !authorization.latestActionEffective || authorization.superseded)
        return Rejected(
            RecommendationConversionReason::manualAuthorizationSupersededOrIneffective);
    if (request.recommendationStatus != RecommendationStatus::approved)
        return Rejected(
            RecommendationConversionReason::recommendationLifecycleNotConvertible);

    const auto& evaluation = request.evaluation;
    if (evaluation.state != RecommendationConversionEvidenceState::completed)
        return Rejected(
            RecommendationConversionReason::evaluationMissingOrIncomplete);
    if (!evaluation.valid)
        return Rejected(RecommendationConversionReason::evaluationInvalid);
    if (evaluation.recommendationId != request.recommendationId ||
        evaluation.sourceExperimentId != request.sourceExperimentId)
        return Rejected(RecommendationConversionReason::inconsistentProvenance);
    if (IsBlockedDisposition(evaluation.disposition))
        return Rejected(RecommendationConversionReason::recommendationBlocked);
    if (evaluation.eligibility != RecommendationEligibility::eligible ||
        evaluation.disposition !=
            RecommendationEvaluationDisposition::advisoryReady)
        return Rejected(RecommendationConversionReason::evaluationInvalid);

    const auto& score = request.score;
    if (score.state != RecommendationConversionEvidenceState::completed)
        return Rejected(
            RecommendationConversionReason::scoreMissingOrIncomplete);
    if (!score.valid || !score.finalScore ||
        !std::isfinite(*score.finalScore))
        return Rejected(RecommendationConversionReason::scoreInvalid);
    if (score.recommendationId != request.recommendationId)
        return Rejected(RecommendationConversionReason::inconsistentProvenance);
    if (evaluation.scoringPolicyHash != score.scoringPolicyHash)
        return Rejected(RecommendationConversionReason::inconsistentProvenance);

    if (!CanonicalMatchesHash(
            authorization.authorizationCanonical,
            authorization.authorizationHash) ||
        !CanonicalMatchesHash(
            evaluation.evaluationIdentityCanonical,
            evaluation.evaluationIdentityHash) ||
        !CanonicalMatchesHash(
            evaluation.evaluationPolicyCanonical,
            evaluation.evaluationPolicyHash) ||
        !CanonicalMatchesHash(
            score.scoringPolicyCanonical,
            score.scoringPolicyHash) ||
        !ValidOptionalRanking(request.ranking))
        return Rejected(RecommendationConversionReason::inconsistentProvenance);

    RecommendationInvocationIdentity sourceIdentity;
    try
    {
        sourceIdentity = BuildRecommendationInvocationIdentity(
            request.sourceInvocation);
    }
    catch (const std::exception&)
    {
        return Rejected(
            RecommendationConversionReason::sourceConfigurationIncomplete);
    }

    if (request.mutations.empty())
        return Rejected(RecommendationConversionReason::mutationMissing);
    if (request.mutations.size() != 1)
        return Rejected(
            RecommendationConversionReason::multipleMutationsDetected);
    const RecommendationConversionMutation& mutation = request.mutations.front();
    const auto parameter = ParseMutationParameter(mutation.family);
    if (!parameter)
        return Rejected(
            RecommendationConversionReason::unsupportedMutationFamily);

    ParsedMutationValue parsed;
    switch (ParseProposedValue(*parameter, mutation.proposedValueCanonical, parsed))
    {
        case ProposedValueValidation::malformed:
            return Rejected(
                RecommendationConversionReason::malformedProposedValue);
        case ProposedValueValidation::outsideRange:
            return Rejected(
                RecommendationConversionReason::proposedValueOutsideAllowedRange);
        case ProposedValueValidation::valid:
            break;
    }

    std::optional<std::string> actualSourceValue;
    try
    {
        actualSourceValue = SourceValueCanonical(
            *parameter, sourceIdentity.invocation.configuration);
    }
    catch (const std::exception&)
    {
        return Rejected(
            RecommendationConversionReason::sourceConfigurationIncomplete);
    }
    if (!actualSourceValue)
        return Rejected(
            RecommendationConversionReason::sourceConfigurationIncomplete);
    if (mutation.sourceValueCanonical != *actualSourceValue)
        return Rejected(RecommendationConversionReason::sourceValueMismatch);
    if (mutation.proposedValueCanonical == *actualSourceValue)
        return Rejected(
            RecommendationConversionReason::proposedValueEqualsSourceValue);

    ExperimentInvocationConfiguration proposedInvocation = sourceIdentity.invocation;
    ApplyMutation(*parameter, parsed, proposedInvocation.configuration);
    RecommendationInvocationIdentity baseProposedIdentity;
    try
    {
        baseProposedIdentity = BuildRecommendationInvocationIdentity(
            proposedInvocation);
    }
    catch (const std::exception&)
    {
        return Rejected(
            RecommendationConversionReason::proposedValueOutsideAllowedRange);
    }
    const RecommendationCandidateIdentity baseProposedSemanticIdentity =
        BuildRecommendationCandidateIdentity(
            baseProposedIdentity.invocation.configuration);
    if (request.recommendationSemanticCanonical !=
            baseProposedSemanticIdentity.canonicalText ||
        request.recommendationSemanticHash != baseProposedSemanticIdentity.hash ||
        request.recommendationInvocationCanonical !=
            baseProposedIdentity.canonicalText ||
        request.recommendationInvocationHash != baseProposedIdentity.hash)
        return Rejected(RecommendationConversionReason::inconsistentProvenance);

    const bool campaignArmChangesMode = request.campaignDonchian20Mode &&
        *request.campaignDonchian20Mode !=
            sourceIdentity.invocation.configuration.donchian20Mode;
    if (request.campaignDonchian20Mode)
    {
        try
        {
            (void)Donchian20ModeText(*request.campaignDonchian20Mode);
        }
        catch (const std::invalid_argument&)
        {
            return Rejected(RecommendationConversionReason::inconsistentProvenance);
        }
        proposedInvocation.configuration.donchian20Mode =
            *request.campaignDonchian20Mode;
    }
    const std::size_t expectedSemanticDifferences =
        1 + (campaignArmChangesMode ? 1 : 0);
    if (SemanticDifferenceCount(
            sourceIdentity.invocation.configuration,
            proposedInvocation.configuration) != expectedSemanticDifferences)
        return Rejected(
            RecommendationConversionReason::multipleMutationsDetected);

    RecommendationInvocationIdentity proposedIdentity;
    try
    {
        proposedIdentity = BuildRecommendationInvocationIdentity(
            proposedInvocation);
    }
    catch (const std::exception&)
    {
        return Rejected(
            RecommendationConversionReason::proposedValueOutsideAllowedRange);
    }
    const RecommendationCandidateIdentity proposedSemanticIdentity =
        BuildRecommendationCandidateIdentity(
            proposedIdentity.invocation.configuration);
    const std::string identityCanonical = ConversionIdentityCanonical(
        request, *parameter, *actualSourceValue,
        mutation.proposedValueCanonical, sourceIdentity.canonicalText,
        proposedIdentity.canonicalText);
    for (const std::string& existing : request.existingConversionCanonicals)
    {
        if (existing.empty())
            return Rejected(RecommendationConversionReason::inconsistentProvenance);
        if (existing == identityCanonical)
            return Rejected(
                RecommendationConversionReason::duplicateConversionIdentity);
    }

    ProposedExperimentSpecification proposal;
    proposal.sourceExperimentId = request.sourceExperimentId;
    proposal.recommendationId = request.recommendationId;
    proposal.proposedInvocation = proposedIdentity.invocation;
    proposal.sourceInvocationCanonical = sourceIdentity.canonicalText;
    proposal.proposedInvocationCanonical = proposedIdentity.canonicalText;
    proposal.changedParameter = *parameter;
    proposal.sourceValueCanonical = *actualSourceValue;
    proposal.proposedValueCanonical = mutation.proposedValueCanonical;
    proposal.recommendationSemanticHash = proposedSemanticIdentity.hash;
    proposal.evaluationIdentityHash = evaluation.evaluationIdentityHash;
    proposal.evaluationPolicyHash = evaluation.evaluationPolicyHash;
    proposal.scoringPolicyHash = score.scoringPolicyHash;
    proposal.reviewAuthorizationHash = authorization.authorizationHash;
    if (request.ranking)
        proposal.rankingSnapshotIdentityHash =
            request.ranking->snapshotIdentityHash;
    proposal.conversionIdentityCanonical = identityCanonical;
    proposal.conversionIdentityHash =
        RecommendationCanonicalHash(identityCanonical);
    return RecommendationConversionResult{
        RecommendationConversionEligibilityResult{
            true, RecommendationConversionReason::eligible,
            RecommendationConversionReasonExplanation(
                RecommendationConversionReason::eligible)},
        std::move(proposal)};
}

} // namespace EA::ExperimentRecommendation
