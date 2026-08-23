#include "ExperimentRecommendationRanking.hpp"

#include "ExperimentRecommendation.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>

namespace EA::ExperimentRecommendation
{
namespace
{

std::string LengthText(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

std::string StableHash(const std::string& canonical)
{
    std::uint64_t value = 14695981039346656037ULL;
    for (const unsigned char byte : canonical)
    {
        value ^= static_cast<std::uint64_t>(byte);
        value *= 1099511628211ULL;
    }
    constexpr char digits[] = "0123456789abcdef";
    std::array<char, 16> encoded{};
    for (std::size_t index = 0; index < encoded.size(); ++index)
    {
        const unsigned shift = static_cast<unsigned>(
            (encoded.size() - index - 1U) * 4U);
        encoded[index] = digits[(value >> shift) & 0x0fU];
    }
    return "fnv1a64:" + std::string(encoded.begin(), encoded.end());
}

bool ParseCanonicalUnsigned(std::string_view text,
                            std::uint64_t maximum,
                            bool allowZero,
                            std::uint64_t& value)
{
    if (text.empty() || (text.size() > 1 && text.front() == '0')) return false;
    value = 0;
    const auto result = std::from_chars(
        text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size() &&
           value <= maximum && (allowZero || value > 0);
}

bool ValidRecommendationRankingMembershipCanonicalText(
    const std::string& canonical)
{
    constexpr std::string_view header =
        "experiment_recommendation_ranking_membership_v1;count=";
    if (!canonical.starts_with(header)) return false;
    std::size_t position = header.size();
    const std::size_t countEnd = canonical.find(';', position);
    const std::string_view countText{canonical.data() + position,
        (countEnd == std::string::npos ? canonical.size() : countEnd) - position};
    std::uint64_t count = 0;
    if (!ParseCanonicalUnsigned(
            countText, kMaximumRecommendationRankingInputs, true, count))
        return false;
    if (count == 0) return countEnd == std::string::npos;
    if (countEnd == std::string::npos) return false;
    position = countEnd;

    std::set<long long> resultIds;
    std::optional<std::pair<std::string, long long>> previous;
    for (std::uint64_t ordinal = 0; ordinal < count; ++ordinal)
    {
        const std::string identityPrefix = ";member[" +
            std::to_string(ordinal) + "].evaluation_identity=";
        if (canonical.compare(position, identityPrefix.size(), identityPrefix) != 0)
            return false;
        position += identityPrefix.size();
        const std::size_t lengthEnd = canonical.find(':', position);
        if (lengthEnd == std::string::npos) return false;
        const std::string_view lengthText{canonical.data() + position,
                                          lengthEnd - position};
        std::uint64_t identityLength = 0;
        if (!ParseCanonicalUnsigned(
                lengthText, canonical.size(), false, identityLength))
            return false;
        position = lengthEnd + 1;
        if (identityLength > canonical.size() - position) return false;
        const std::string identity = canonical.substr(
            position, static_cast<std::size_t>(identityLength));
        position += static_cast<std::size_t>(identityLength);

        const std::string resultPrefix = ";member[" +
            std::to_string(ordinal) + "].evaluation_result_id=";
        if (canonical.compare(position, resultPrefix.size(), resultPrefix) != 0)
            return false;
        position += resultPrefix.size();
        const std::size_t resultEnd = ordinal + 1 < count
            ? canonical.find(';', position) : canonical.size();
        if (resultEnd == std::string::npos) return false;
        const std::string_view resultText{canonical.data() + position,
                                          resultEnd - position};
        std::uint64_t parsedResultId = 0;
        if (!ParseCanonicalUnsigned(
                resultText, static_cast<std::uint64_t>(
                    std::numeric_limits<long long>::max()),
                false, parsedResultId))
            return false;
        const long long resultId = static_cast<long long>(parsedResultId);
        if (!resultIds.insert(resultId).second) return false;
        const std::pair<std::string, long long> current{identity, resultId};
        if (previous && !(*previous < current)) return false;
        previous = current;
        position = resultEnd;
    }
    return position == canonical.size();
}

int DispositionPriority(RecommendationEvaluationDisposition disposition)
{
    switch (disposition)
    {
        case RecommendationEvaluationDisposition::advisoryReady: return 0;
        case RecommendationEvaluationDisposition::blockedPendingDuplicate: return 0;
        case RecommendationEvaluationDisposition::blockedActiveDuplicate: return 1;
        case RecommendationEvaluationDisposition::completedDuplicate: return 2;
        case RecommendationEvaluationDisposition::insufficientEvidence: return 0;
        case RecommendationEvaluationDisposition::staleSourceEvidence: return 1;
        case RecommendationEvaluationDisposition::unsupportedRecommendationFamily:
            return 2;
        case RecommendationEvaluationDisposition::invalidPersistedEvidence: return 3;
    }
    throw std::invalid_argument("invalid_recommendation_ranking_disposition");
}

int BucketPriority(RecommendationRankingBucket bucket)
{
    switch (bucket)
    {
        case RecommendationRankingBucket::advisoryReady: return 0;
        case RecommendationRankingBucket::blocked: return 1;
        case RecommendationRankingBucket::nonActionable: return 2;
    }
    throw std::invalid_argument("invalid_recommendation_ranking_bucket");
}

bool ValidScopeText(const std::string& value)
{
    if (value.empty() || value.size() > 128) return false;
    const auto asciiWhitespace = [](unsigned char byte) {
        return byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r' ||
               byte == '\f' || byte == '\v';
    };
    if (asciiWhitespace(static_cast<unsigned char>(value.front())) ||
        asciiWhitespace(static_cast<unsigned char>(value.back())))
        return false;
    return std::none_of(value.begin(), value.end(), [](unsigned char byte) {
        return byte < 0x20 || byte == 0x7f;
    });
}

std::optional<std::string> TopComponent(
    const std::vector<RecommendationScoreComponent>& components,
    bool penalty)
{
    const RecommendationScoreComponent* best = nullptr;
    for (const auto& component : components)
    {
        if (component.penalty != penalty) continue;
        if (!std::isfinite(component.weightedContribution))
            throw std::invalid_argument("invalid_recommendation_ranking_component");
        if (best == nullptr ||
            component.weightedContribution > best->weightedContribution ||
            (component.weightedContribution == best->weightedContribution &&
             component.componentName < best->componentName))
            best = &component;
    }
    return best ? std::optional<std::string>{best->componentName} : std::nullopt;
}

void ValidateEvaluation(const RecommendationRankingEvaluation& value)
{
    if (value.evaluationResultId <= 0 || value.evaluationRunId <= 0 ||
        value.recommendationId <= 0 || value.recommendationScanId <= 0 ||
        value.sourceExperimentId <= 0 || value.horizon <= 0 ||
        value.symbol.empty() || value.family.empty() ||
        value.recommendationSemanticHash.empty() ||
        value.evaluationIdentityCanonical.empty() ||
        value.evaluationIdentityHash.empty() ||
        value.evaluationPolicyCanonical.empty() ||
        value.evaluationPolicyHash.empty() || value.evaluationVersion <= 0 ||
        value.evaluatorVersion <= 0 || value.scoringPolicyCanonical.empty() ||
        value.scoringPolicyHash.empty() || value.scoringVersion <= 0 ||
        value.reasonCode.empty() || value.explanation.empty() ||
        value.componentCount < 0 || value.missingEvidenceCount < 0 ||
        value.componentCount != static_cast<int>(value.components.size()))
        throw std::invalid_argument("invalid_recommendation_ranking_evidence");
    const bool ready = value.disposition ==
        RecommendationEvaluationDisposition::advisoryReady;
    if (ready != value.finalScore.has_value() ||
        ready != (value.eligibility == RecommendationEligibility::eligible) ||
        ready != !value.components.empty())
        throw std::invalid_argument("invalid_recommendation_ranking_result_shape");
    if (value.finalScore && (!std::isfinite(*value.finalScore) ||
        *value.finalScore < 0.0 || *value.finalScore > 1.0))
        throw std::invalid_argument("nonfinite_recommendation_ranking_score");
    if (value.components.size() >
        static_cast<std::size_t>(kMaximumRecommendationComparisonComponents))
        throw std::invalid_argument("recommendation_ranking_component_limit_exceeded");
    std::set<std::string> componentNames;
    for (const auto& component : value.components)
    {
        if (component.componentName.empty() ||
            !componentNames.insert(component.componentName).second ||
            !std::isfinite(component.normalizedValue) ||
            !std::isfinite(component.weight) ||
            !std::isfinite(component.weightedContribution) ||
            component.normalizedValue < 0.0 ||
            component.normalizedValue > 1.0 || component.weight < 0.0 ||
            component.weightedContribution < 0.0)
            throw std::invalid_argument("invalid_recommendation_ranking_component");
    }
}

std::string ScoreTieBreak(const RecommendationRankingEvaluation& value)
{
    if (!value.finalScore)
        return "priority:" + std::to_string(DispositionPriority(value.disposition));
    return "score:" + CanonicalRecommendationDouble(*value.finalScore);
}

template <typename Value>
void AssignSorted(const std::set<Value>& source, std::vector<Value>& target)
{
    target.assign(source.begin(), source.end());
}

} // namespace

std::optional<std::string> ValidateRecommendationRankingPolicy(
    const RecommendationRankingPolicy& policy)
{
    if (policy.rankingVersion != 1)
        return "unsupported_recommendation_ranking_version";
    return std::nullopt;
}

std::string RecommendationRankingPolicyCanonicalText(
    const RecommendationRankingPolicy& policy)
{
    if (const auto error = ValidateRecommendationRankingPolicy(policy))
        throw std::invalid_argument(*error);
    return "experiment_recommendation_ranking_policy_v1;ranking_version=1;"
        "bucket_order=advisory_ready,blocked,non_actionable;"
        "ready_order=score_desc,semantic_hash,evaluation_hash,result_id;"
        "blocked_order=pending,active,completed,semantic_hash,evaluation_hash,result_id;"
        "non_actionable_order=insufficient,stale,unsupported,invalid,semantic_hash,evaluation_hash,result_id;"
        "inclusion=all_persisted_dispositions_before_global_limit;"
        "limit_application=global_after_order;grouping=single_scope;"
        "component_summary=per_penalty_class,contribution_desc,component_name;"
        "comparison=matching_evaluation_policy,scoring_policy,evaluator,family";
}

std::string RecommendationRankingCanonicalHash(const std::string& canonical)
{
    return StableHash(canonical);
}

std::string RecommendationRankingScopeTypeText(
    RecommendationRankingScopeType value)
{
    switch (value)
    {
        case RecommendationRankingScopeType::evaluationRun: return "evaluation_run";
        case RecommendationRankingScopeType::recommendationScan: return "recommendation_scan";
        case RecommendationRankingScopeType::symbol: return "symbol";
        case RecommendationRankingScopeType::horizon: return "horizon";
        case RecommendationRankingScopeType::family: return "family";
        case RecommendationRankingScopeType::symbolHorizon: return "symbol_horizon";
        case RecommendationRankingScopeType::global: return "global";
    }
    throw std::invalid_argument("invalid_recommendation_ranking_scope_type");
}

std::optional<std::string> ValidateRecommendationRankingScope(
    const RecommendationRankingScope& scope)
{
    const bool run = scope.evaluationRunId.has_value();
    const bool scan = scope.recommendationScanId.has_value();
    const bool symbol = scope.symbol.has_value();
    const bool horizon = scope.horizon.has_value();
    const bool family = scope.family.has_value();
    if ((run && *scope.evaluationRunId <= 0) ||
        (scan && *scope.recommendationScanId <= 0) ||
        (symbol && !ValidScopeText(*scope.symbol)) ||
        (horizon && *scope.horizon <= 0) ||
        (family && !ValidScopeText(*scope.family)))
        return "invalid_recommendation_ranking_scope_value";
    bool valid = false;
    switch (scope.type)
    {
        case RecommendationRankingScopeType::evaluationRun:
            valid = run && !scan && !symbol && !horizon && !family; break;
        case RecommendationRankingScopeType::recommendationScan:
            valid = !run && scan && !symbol && !horizon && !family; break;
        case RecommendationRankingScopeType::symbol:
            valid = !run && !scan && symbol && !horizon && !family; break;
        case RecommendationRankingScopeType::horizon:
            valid = !run && !scan && !symbol && horizon && !family; break;
        case RecommendationRankingScopeType::family:
            valid = !run && !scan && !symbol && !horizon && family; break;
        case RecommendationRankingScopeType::symbolHorizon:
            valid = !run && !scan && symbol && horizon && !family; break;
        case RecommendationRankingScopeType::global:
            valid = !run && !scan && !symbol && !horizon && !family; break;
    }
    return valid ? std::nullopt
                 : std::optional<std::string>{"ambiguous_recommendation_ranking_scope"};
}

std::string RecommendationRankingScopeCanonicalText(
    const RecommendationRankingScope& scope)
{
    if (const auto error = ValidateRecommendationRankingScope(scope))
        throw std::invalid_argument(*error);
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_ranking_scope_v1;type="
        << RecommendationRankingScopeTypeText(scope.type)
        << ";evaluation_run_id="
        << (scope.evaluationRunId ? std::to_string(*scope.evaluationRunId) : "NULL")
        << ";recommendation_scan_id="
        << (scope.recommendationScanId ?
            std::to_string(*scope.recommendationScanId) : "NULL")
        << ";symbol=" << (scope.symbol ? LengthText(*scope.symbol) : "NULL")
        << ";horizon=" << (scope.horizon ? std::to_string(*scope.horizon) : "NULL")
        << ";family=" << (scope.family ? LengthText(*scope.family) : "NULL");
    return out.str();
}

std::string RecommendationRankingScopeValueText(
    const RecommendationRankingScope& scope)
{
    if (const auto error = ValidateRecommendationRankingScope(scope))
        throw std::invalid_argument(*error);
    switch (scope.type)
    {
        case RecommendationRankingScopeType::evaluationRun:
            return std::to_string(*scope.evaluationRunId);
        case RecommendationRankingScopeType::recommendationScan:
            return std::to_string(*scope.recommendationScanId);
        case RecommendationRankingScopeType::symbol: return *scope.symbol;
        case RecommendationRankingScopeType::horizon:
            return std::to_string(*scope.horizon);
        case RecommendationRankingScopeType::family: return *scope.family;
        case RecommendationRankingScopeType::symbolHorizon:
            return *scope.symbol + ":" + std::to_string(*scope.horizon);
        case RecommendationRankingScopeType::global: return "ALL";
    }
    throw std::invalid_argument("invalid_recommendation_ranking_scope_type");
}

std::string RecommendationRankingBucketText(RecommendationRankingBucket value)
{
    switch (value)
    {
        case RecommendationRankingBucket::advisoryReady: return "advisory_ready";
        case RecommendationRankingBucket::blocked: return "blocked";
        case RecommendationRankingBucket::nonActionable: return "non_actionable";
    }
    throw std::invalid_argument("invalid_recommendation_ranking_bucket");
}

std::optional<RecommendationRankingBucket> ParseRecommendationRankingBucket(
    const std::string& value)
{
    if (value == "advisory_ready") return RecommendationRankingBucket::advisoryReady;
    if (value == "blocked") return RecommendationRankingBucket::blocked;
    if (value == "non_actionable") return RecommendationRankingBucket::nonActionable;
    return std::nullopt;
}

std::string RecommendationRankingPopulationSemanticStateText(
    RecommendationRankingPopulationSemanticState value)
{
    switch (value)
    {
        case RecommendationRankingPopulationSemanticState::verifiedHomogeneous:
            return "verified_homogeneous";
        case RecommendationRankingPopulationSemanticState::empty:
            return "empty";
        case RecommendationRankingPopulationSemanticState::legacyHeterogeneous:
            return "legacy_heterogeneous";
        case RecommendationRankingPopulationSemanticState::legacyUnverified:
            return "legacy_unverified";
    }
    throw std::invalid_argument("invalid_ranking_population_semantic_state");
}

RecommendationRankingPopulationSemanticValidation
ValidateRecommendationRankingPopulationSemantics(
    const std::vector<RecommendationRankingEvaluation>& evaluations)
{
    RecommendationRankingPopulationSemanticValidation result;
    if (evaluations.empty())
    {
        result.state = RecommendationRankingPopulationSemanticState::empty;
        result.reason = "empty_population";
        return result;
    }

    std::set<long long> resultIds;
    std::set<long long> runIds;
    std::set<std::string> scoringHashes;
    std::set<std::string> evaluationHashes;
    std::set<std::pair<int, std::string>> scoringIdentities;
    std::set<std::pair<int, std::string>> evaluationIdentities;
    bool invalidScoring = false;
    bool invalidEvaluation = false;
    for (const auto& value : evaluations)
    {
        resultIds.insert(value.evaluationResultId);
        runIds.insert(value.evaluationRunId);
        if (!value.scoringSemanticIdentity.hash.empty())
            scoringHashes.insert(value.scoringSemanticIdentity.hash);
        if (!value.evaluationSemanticIdentity.hash.empty())
            evaluationHashes.insert(value.evaluationSemanticIdentity.hash);

        if (value.evaluationResultId <= 0 || value.evaluationRunId <= 0 ||
            ValidateRecommendationScoringSemanticIdentity(
                value.scoringSemanticIdentity, value.scoringPolicyCanonical,
                value.scoringPolicyHash, value.scoringVersion))
        {
            invalidScoring = true;
            continue;
        }
        scoringIdentities.emplace(value.scoringSemanticIdentity.version,
                                  value.scoringSemanticIdentity.canonical);
        if (value.evaluationIdentityCanonical.empty() ||
            value.evaluationIdentityHash != RecommendationEvaluationCanonicalHash(
                value.evaluationIdentityCanonical) ||
            ValidateRecommendationEvaluationSemanticIdentity(
                value.evaluationSemanticIdentity,
                value.evaluationPolicyCanonical, value.evaluationPolicyHash,
                value.evaluationVersion, value.evaluatorVersion,
                value.scoringSemanticIdentity, value.scoringPolicyCanonical,
                value.scoringPolicyHash, value.scoringVersion))
        {
            invalidEvaluation = true;
            continue;
        }
        evaluationIdentities.emplace(value.evaluationSemanticIdentity.version,
                                     value.evaluationSemanticIdentity.canonical);
    }
    result.distinctScoringIdentityCount =
        static_cast<int>(scoringIdentities.size());
    result.distinctEvaluationIdentityCount =
        static_cast<int>(evaluationIdentities.size());
    AssignSorted(resultIds, result.evaluationResultIds);
    AssignSorted(runIds, result.evaluationRunIds);
    AssignSorted(scoringHashes, result.scoringSemanticHashes);
    AssignSorted(evaluationHashes, result.evaluationSemanticHashes);

    if (invalidScoring)
    {
        result.state = RecommendationRankingPopulationSemanticState::legacyUnverified;
        result.reason = "invalid_scoring_semantic_provenance";
        return result;
    }
    if (invalidEvaluation)
    {
        result.state = RecommendationRankingPopulationSemanticState::legacyUnverified;
        result.reason = "invalid_evaluation_semantic_provenance";
        return result;
    }
    if (scoringIdentities.size() != 1U)
    {
        result.state =
            RecommendationRankingPopulationSemanticState::legacyHeterogeneous;
        result.reason = "heterogeneous_scoring_semantics";
        return result;
    }
    if (evaluationIdentities.size() != 1U)
    {
        result.state =
            RecommendationRankingPopulationSemanticState::legacyHeterogeneous;
        result.reason = "heterogeneous_evaluation_semantics";
        return result;
    }
    result.state =
        RecommendationRankingPopulationSemanticState::verifiedHomogeneous;
    result.reason = "verified_homogeneous";
    result.scoringIdentity = evaluations.front().scoringSemanticIdentity;
    result.evaluationIdentity = evaluations.front().evaluationSemanticIdentity;
    return result;
}

RecommendationRankingBucket RecommendationRankingBucketForDisposition(
    RecommendationEvaluationDisposition disposition)
{
    switch (disposition)
    {
        case RecommendationEvaluationDisposition::advisoryReady:
            return RecommendationRankingBucket::advisoryReady;
        case RecommendationEvaluationDisposition::blockedPendingDuplicate:
        case RecommendationEvaluationDisposition::blockedActiveDuplicate:
        case RecommendationEvaluationDisposition::completedDuplicate:
            return RecommendationRankingBucket::blocked;
        case RecommendationEvaluationDisposition::insufficientEvidence:
        case RecommendationEvaluationDisposition::staleSourceEvidence:
        case RecommendationEvaluationDisposition::unsupportedRecommendationFamily:
        case RecommendationEvaluationDisposition::invalidPersistedEvidence:
            return RecommendationRankingBucket::nonActionable;
    }
    throw std::invalid_argument("invalid_recommendation_ranking_disposition");
}

std::vector<RecommendationRankingMember> RankRecommendationEvaluationEvidence(
    const RecommendationRankingPolicy& policy,
    const std::vector<RecommendationRankingEvaluation>& evaluations,
    int outputLimit)
{
    if (const auto error = ValidateRecommendationRankingPolicy(policy))
        throw std::invalid_argument(*error);
    if (outputLimit <= 0 || outputLimit > kMaximumRecommendationRankingMembers)
        throw std::invalid_argument("recommendation_ranking_limit_invalid");
    if (evaluations.size() >
        static_cast<std::size_t>(kMaximumRecommendationRankingInputs))
        throw std::invalid_argument("recommendation_ranking_input_limit_exceeded");
    const auto semantics =
        ValidateRecommendationRankingPopulationSemantics(evaluations);
    if (!semantics.acceptableForNewSnapshot())
        throw std::invalid_argument(semantics.reason);
    std::vector<RecommendationRankingMember> members;
    members.reserve(evaluations.size());
    std::set<long long> evaluationResultIds;
    for (const auto& evaluation : evaluations)
    {
        ValidateEvaluation(evaluation);
        if (!evaluationResultIds.insert(evaluation.evaluationResultId).second)
            throw std::invalid_argument(
                "duplicate_recommendation_ranking_evaluation_result");
        RecommendationRankingMember member;
        member.evaluation = evaluation;
        member.bucket = RecommendationRankingBucketForDisposition(
            evaluation.disposition);
        member.tieBreakPrimary = ScoreTieBreak(evaluation);
        member.tieBreakSemanticHash = evaluation.recommendationSemanticHash;
        member.tieBreakEvaluationHash = evaluation.evaluationIdentityHash;
        member.inclusionReason = member.bucket ==
            RecommendationRankingBucket::advisoryReady
            ? "included_advisory_ready" : "included_for_advisory_inspection";
        if (member.bucket != RecommendationRankingBucket::advisoryReady)
            member.blockReason = evaluation.reasonCode;
        member.topPositiveComponent = TopComponent(evaluation.components, false);
        member.topPenaltyComponent = TopComponent(evaluation.components, true);
        members.push_back(std::move(member));
    }
    std::sort(members.begin(), members.end(), [](const auto& left,
                                                  const auto& right) {
        if (left.bucket != right.bucket)
            return BucketPriority(left.bucket) < BucketPriority(right.bucket);
        if (left.bucket == RecommendationRankingBucket::advisoryReady &&
            left.evaluation.finalScore != right.evaluation.finalScore)
            return *left.evaluation.finalScore > *right.evaluation.finalScore;
        if (left.bucket != RecommendationRankingBucket::advisoryReady &&
            left.evaluation.disposition != right.evaluation.disposition)
            return DispositionPriority(left.evaluation.disposition) <
                   DispositionPriority(right.evaluation.disposition);
        return std::tie(left.evaluation.recommendationSemanticHash,
                        left.evaluation.evaluationIdentityHash,
                        left.evaluation.evaluationResultId) <
               std::tie(right.evaluation.recommendationSemanticHash,
                        right.evaluation.evaluationIdentityHash,
                        right.evaluation.evaluationResultId);
    });
    if (members.size() > static_cast<std::size_t>(outputLimit))
        members.resize(static_cast<std::size_t>(outputLimit));
    int bucketRanks[3] = {0, 0, 0};
    for (std::size_t index = 0; index < members.size(); ++index)
    {
        members[index].globalOrdinal = static_cast<int>(index) + 1;
        const int bucket = BucketPriority(members[index].bucket);
        members[index].bucketRank = ++bucketRanks[bucket];
    }
    return members;
}

std::string RecommendationRankingMembershipCanonicalText(
    const std::vector<RecommendationRankingEvaluation>& evaluations)
{
    if (evaluations.size() >
        static_cast<std::size_t>(kMaximumRecommendationRankingInputs))
        throw std::invalid_argument("recommendation_ranking_input_limit_exceeded");
    const auto semantics =
        ValidateRecommendationRankingPopulationSemantics(evaluations);
    if (!semantics.acceptableForNewSnapshot())
        throw std::invalid_argument(semantics.reason);
    std::vector<std::pair<std::string, long long>> identities;
    identities.reserve(evaluations.size());
    std::set<long long> evaluationResultIds;
    for (const auto& value : evaluations)
    {
        ValidateEvaluation(value);
        if (!evaluationResultIds.insert(value.evaluationResultId).second)
            throw std::invalid_argument(
                "duplicate_recommendation_ranking_evaluation_result");
        identities.emplace_back(value.evaluationIdentityCanonical,
                                value.evaluationResultId);
    }
    std::sort(identities.begin(), identities.end());
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_ranking_membership_v1;count="
        << identities.size();
    for (std::size_t index = 0; index < identities.size(); ++index)
        out << ";member[" << index << "].evaluation_identity="
            << LengthText(identities[index].first)
            << ";member[" << index << "].evaluation_result_id="
            << identities[index].second;
    return out.str();
}

std::string RecommendationRankingSnapshotIdentityCanonicalText(
    const RecommendationRankingPolicy& policy,
    const RecommendationRankingScope& scope,
    int outputLimit,
    const std::vector<RecommendationRankingEvaluation>& evaluations)
{
    if (outputLimit <= 0 || outputLimit > kMaximumRecommendationRankingMembers)
        throw std::invalid_argument("recommendation_ranking_limit_invalid");
    const std::string membership =
        RecommendationRankingMembershipCanonicalText(evaluations);
    const auto semantics =
        ValidateRecommendationRankingPopulationSemantics(evaluations);
    return RecommendationRankingSnapshotIdentityCanonicalTextFromMembership(
        policy, scope, outputLimit, membership, semantics);
}

std::string RecommendationRankingSnapshotIdentityCanonicalTextFromMembership(
    const RecommendationRankingPolicy& policy,
    const RecommendationRankingScope& scope,
    int outputLimit,
    const std::string& membershipCanonical,
    const RecommendationRankingPopulationSemanticValidation& semantics)
{
    if (!ValidRecommendationRankingMembershipCanonicalText(membershipCanonical))
        throw std::invalid_argument(
            "invalid_recommendation_ranking_membership_canonical");
    if (outputLimit <= 0 || outputLimit > kMaximumRecommendationRankingMembers)
        throw std::invalid_argument("recommendation_ranking_limit_invalid");
    if (!semantics.acceptableForNewSnapshot())
        throw std::invalid_argument(semantics.reason.empty()
            ? "invalid_ranking_population_semantics" : semantics.reason);
    const bool empty = semantics.state ==
        RecommendationRankingPopulationSemanticState::empty;
    if (empty != !semantics.scoringIdentity.has_value() ||
        empty != !semantics.evaluationIdentity.has_value() ||
        (!empty && (semantics.distinctScoringIdentityCount != 1 ||
                    semantics.distinctEvaluationIdentityCount != 1)))
        throw std::invalid_argument("invalid_ranking_population_semantic_shape");
    const std::string policyText =
        RecommendationRankingPolicyCanonicalText(policy);
    const std::string scopeText = RecommendationRankingScopeCanonicalText(scope);
    return "experiment_recommendation_ranking_snapshot_identity_v2;policy=" +
        LengthText(policyText) + ";scope=" + LengthText(scopeText) +
        ";limit=" + std::to_string(outputLimit) + ";population_state=" +
        RecommendationRankingPopulationSemanticStateText(semantics.state) +
        ";scoring_semantic=" + (empty ? "NULL" :
            LengthText(semantics.scoringIdentity->canonical)) +
        ";evaluation_semantic=" + (empty ? "NULL" :
            LengthText(semantics.evaluationIdentity->canonical)) +
        ";membership=" +
        LengthText(membershipCanonical);
}

std::string RecommendationComparisonStateText(
    RecommendationComparisonState value)
{
    switch (value)
    {
        case RecommendationComparisonState::comparable: return "comparable";
        case RecommendationComparisonState::incomparablePolicyVersion:
            return "incomparable_policy_version";
        case RecommendationComparisonState::incomparableEvaluatorVersion:
            return "incomparable_evaluator_version";
        case RecommendationComparisonState::incomparableScoringSemantics:
            return "incomparable_scoring_semantics";
        case RecommendationComparisonState::incomparableEvaluationSemantics:
            return "incomparable_evaluation_semantics";
        case RecommendationComparisonState::invalidScoringProvenance:
            return "invalid_scoring_provenance";
        case RecommendationComparisonState::invalidEvaluationProvenance:
            return "invalid_evaluation_provenance";
        case RecommendationComparisonState::incomparableMissingScore:
            return "incomparable_missing_score";
        case RecommendationComparisonState::incomparableScope:
            return "incomparable_scope";
        case RecommendationComparisonState::incomparableFamily:
            return "incomparable_family";
        case RecommendationComparisonState::identicalScoreTie:
            return "identical_score_tie";
        case RecommendationComparisonState::leftRankedHigher:
            return "left_ranked_higher";
        case RecommendationComparisonState::rightRankedHigher:
            return "right_ranked_higher";
    }
    throw std::invalid_argument("invalid_recommendation_comparison_state");
}

RecommendationComparisonResult CompareRecommendationEvaluations(
    const RecommendationRankingEvaluation& left,
    const RecommendationRankingEvaluation& right,
    std::optional<int> leftRank,
    std::optional<int> rightRank,
    bool sameScope)
{
    ValidateEvaluation(left);
    ValidateEvaluation(right);
    if ((leftRank && *leftRank <= 0) || (rightRank && *rightRank <= 0))
        throw std::invalid_argument("invalid_recommendation_comparison_rank");
    RecommendationComparisonResult result;
    result.leftEvaluationResultId = left.evaluationResultId;
    result.rightEvaluationResultId = right.evaluationResultId;
    result.leftRecommendationSemanticHash = left.recommendationSemanticHash;
    result.rightRecommendationSemanticHash = right.recommendationSemanticHash;
    result.leftEvaluationPolicyHash = left.evaluationPolicyHash;
    result.rightEvaluationPolicyHash = right.evaluationPolicyHash;
    result.leftScoringPolicyHash = left.scoringPolicyHash;
    result.rightScoringPolicyHash = right.scoringPolicyHash;
    result.leftScoringVersion = left.scoringVersion;
    result.rightScoringVersion = right.scoringVersion;
    result.leftEvaluatorVersion = left.evaluatorVersion;
    result.rightEvaluatorVersion = right.evaluatorVersion;
    result.leftScore = left.finalScore;
    result.rightScore = right.finalScore;
    result.leftRank = leftRank;
    result.rightRank = rightRank;
    if (!sameScope || leftRank.has_value() != rightRank.has_value())
    {
        result.state = RecommendationComparisonState::incomparableScope;
        result.explanation = "Rank positions are not from the same ranking snapshot.";
        return result;
    }
    const auto leftScoringError = ValidateRecommendationScoringSemanticIdentity(
        left.scoringSemanticIdentity, left.scoringPolicyCanonical,
        left.scoringPolicyHash, left.scoringVersion);
    const auto rightScoringError = ValidateRecommendationScoringSemanticIdentity(
        right.scoringSemanticIdentity, right.scoringPolicyCanonical,
        right.scoringPolicyHash, right.scoringVersion);
    if (leftScoringError || rightScoringError)
    {
        result.state = RecommendationComparisonState::invalidScoringProvenance;
        result.explanation = "At least one scoring semantic identity is malformed or internally inconsistent.";
        return result;
    }
    const auto leftEvaluationError =
        ValidateRecommendationEvaluationSemanticIdentity(
            left.evaluationSemanticIdentity, left.evaluationPolicyCanonical,
            left.evaluationPolicyHash, left.evaluationVersion,
            left.evaluatorVersion, left.scoringSemanticIdentity,
            left.scoringPolicyCanonical, left.scoringPolicyHash,
            left.scoringVersion);
    const auto rightEvaluationError =
        ValidateRecommendationEvaluationSemanticIdentity(
            right.evaluationSemanticIdentity, right.evaluationPolicyCanonical,
            right.evaluationPolicyHash, right.evaluationVersion,
            right.evaluatorVersion, right.scoringSemanticIdentity,
            right.scoringPolicyCanonical, right.scoringPolicyHash,
            right.scoringVersion);
    const bool invalidEvaluationResultIdentity =
        left.evaluationIdentityCanonical.empty() ||
        right.evaluationIdentityCanonical.empty() ||
        left.evaluationIdentityHash != RecommendationEvaluationCanonicalHash(
            left.evaluationIdentityCanonical) ||
        right.evaluationIdentityHash != RecommendationEvaluationCanonicalHash(
            right.evaluationIdentityCanonical);
    if (leftEvaluationError || rightEvaluationError ||
        invalidEvaluationResultIdentity)
    {
        result.state = RecommendationComparisonState::invalidEvaluationProvenance;
        result.explanation = "At least one evaluation semantic identity is malformed or internally inconsistent.";
        return result;
    }
    if (left.scoringSemanticIdentity != right.scoringSemanticIdentity)
    {
        result.state = RecommendationComparisonState::incomparableScoringSemantics;
        result.explanation = "The evaluations use different scoring semantics.";
        return result;
    }
    if (left.evaluationSemanticIdentity != right.evaluationSemanticIdentity)
    {
        result.state = RecommendationComparisonState::incomparableEvaluationSemantics;
        result.explanation = "The evaluations use different evaluation semantics.";
        return result;
    }
    if (left.family != right.family)
    {
        result.state = RecommendationComparisonState::incomparableFamily;
        result.explanation = "The recommendation families are not directly comparable.";
        return result;
    }
    if (!left.finalScore || !right.finalScore)
    {
        result.state = RecommendationComparisonState::incomparableMissingScore;
        result.explanation = "At least one evaluation has no advisory score.";
        return result;
    }
    result.scoreDelta = *left.finalScore - *right.finalScore;
    if (leftRank && rightRank) result.rankDelta = *leftRank - *rightRank;

    std::map<std::string, std::pair<std::optional<RecommendationScoreComponent>,
                                   std::optional<RecommendationScoreComponent>>> aligned;
    for (const auto& component : left.components)
        aligned[component.componentName].first = component;
    for (const auto& component : right.components)
        aligned[component.componentName].second = component;
    if (aligned.size() >
        static_cast<std::size_t>(kMaximumRecommendationComparisonComponents))
        throw std::invalid_argument("recommendation_comparison_component_limit_exceeded");
    double positiveMagnitude = 0.0;
    double penaltyMagnitude = 0.0;
    for (const auto& [name, pair] : aligned)
    {
        RecommendationComponentDifference difference;
        difference.componentName = name;
        if (pair.first)
        {
            difference.leftContribution = pair.first->weightedContribution;
            difference.penalty = pair.first->penalty;
        }
        if (pair.second)
        {
            difference.rightContribution = pair.second->weightedContribution;
            if (!pair.first) difference.penalty = pair.second->penalty;
            else if (pair.first->penalty != pair.second->penalty)
                throw std::invalid_argument("recommendation_comparison_penalty_mismatch");
        }
        if (difference.leftContribution && difference.rightContribution)
            difference.contributionDelta = *difference.leftContribution -
                                           *difference.rightContribution;
        if (difference.contributionDelta)
        {
            const double magnitude = std::fabs(*difference.contributionDelta);
            if (difference.penalty && magnitude > penaltyMagnitude)
            {
                penaltyMagnitude = magnitude;
                result.largestPenaltyDifference = name;
            }
            else if (!difference.penalty && magnitude > positiveMagnitude)
            {
                positiveMagnitude = magnitude;
                result.largestPositiveDifference = name;
            }
        }
        result.componentDifferences.push_back(std::move(difference));
    }

    if (leftRank && rightRank && *leftRank != *rightRank)
    {
        result.state = *leftRank < *rightRank
            ? RecommendationComparisonState::leftRankedHigher
            : RecommendationComparisonState::rightRankedHigher;
        result.explanation = *left.finalScore == *right.finalScore
            ? "Scores tie; the persisted deterministic identity tie-break decides rank."
            : "Persisted advisory score ordering decides rank.";
    }
    else if (*left.finalScore == *right.finalScore)
    {
        result.state = RecommendationComparisonState::identicalScoreTie;
        result.explanation = "The persisted final advisory scores are identical.";
    }
    else
    {
        result.state = RecommendationComparisonState::comparable;
        result.explanation = *left.finalScore > *right.finalScore
            ? "The left evaluation has the higher persisted advisory score."
            : "The right evaluation has the higher persisted advisory score.";
    }
    return result;
}

} // namespace EA::ExperimentRecommendation
