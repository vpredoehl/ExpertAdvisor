#include "ExperimentReplicationComparison.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace EA::ExperimentReplicationComparison
{
namespace
{

using IdentityMap =
    std::map<std::string, std::optional<std::string>, std::less<>>;

std::string Number(double value)
{
    char buffer[128] {};
    const auto converted = std::to_chars(
        std::begin(buffer), std::end(buffer), value,
        std::chars_format::general);
    if (converted.ec != std::errc{})
        throw std::runtime_error("experiment_replication_number_format_failed");
    return std::string(buffer, converted.ptr);
}

std::string OptionalNumber(const std::optional<double>& value)
{
    return value ? Number(*value) : "NULL";
}

std::string MachineText(std::string value)
{
    for (char& character : value)
    {
        const bool safe =
            (character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') ||
            character == '_' || character == '-' || character == '.';
        if (!safe) character = '_';
    }
    return value.empty() ? "EMPTY" : value;
}

std::string Reasons(const std::vector<std::string>& reasons)
{
    if (reasons.empty()) return "NONE";
    std::ostringstream output;
    for (std::size_t index = 0; index < reasons.size(); ++index)
    {
        if (index != 0) output << '|';
        output << MachineText(reasons[index]);
    }
    return output.str();
}

void AddReason(std::vector<std::string>& reasons, std::string reason)
{
    if (std::find(reasons.begin(), reasons.end(), reason) == reasons.end())
        reasons.push_back(std::move(reason));
}

IdentityMap MakeIdentityMap(const std::vector<Pair::IdentityField>& fields)
{
    IdentityMap result;
    for (const auto& field : fields)
    {
        if (!result.emplace(field.name, field.value).second)
            throw std::invalid_argument(
                "replication_pair_scientific_identity_duplicated");
    }
    return result;
}

bool PairScientificallyComparable(const Pair::ComparisonResult& pair)
{
    return pair.invalidReasons.empty() &&
        pair.unexpectedDifferences.empty() &&
        (pair.status == Pair::Status::ComparableComplete ||
         pair.status == Pair::Status::ComparableIncomplete);
}

bool IsMaterializationField(std::string_view field)
{
    static const std::set<std::string, std::less<>> fields {
        "model_input_width",
        "hidden_size",
        "layer_count",
        "window_size",
        "model_metadata_schema_version",
        "train_configuration_schema_version",
        "normalization_version",
        "class_weight_down",
        "class_weight_neutral",
        "class_weight_up",
        "optimizer_metadata_schema_version",
        "optimizer_type",
        "optimizer_first_moment_buffer_count",
        "optimizer_second_moment_buffer_count",
        "persisted_core_lr_mult",
        "persisted_head_weight_lr_mult",
        "persisted_head_bias_lr_mult",
        "label_rule_id",
        "target_type",
        "target_scale",
        "target_bias",
        "target_use_zscore",
        "target_mean",
        "target_standard_deviation",
        "model_input_metadata_schema_version",
        "model_input_semantic_layout_version",
        "persisted_training_symbol",
        "persisted_training_start",
        "persisted_training_end",
        "input_width_expansion_canonical",
        "training_execution_identity",
        "inference_execution_identity",
    };
    return fields.contains(field);
}

bool IsExecutionIdentityField(std::string_view field)
{
    return field == "training_execution_identity" ||
        field == "inference_execution_identity";
}

std::optional<std::string> RequiredConfiguredValue(
    const IdentityMap& identity,
    std::string_view field)
{
    const auto found = identity.find(field);
    if (found == identity.end() || !found->second ||
        *found->second == "NULL")
        return std::nullopt;
    return found->second;
}

void CompareIntervention(const Pair::ComparisonResult& reference,
                         const Pair::ComparisonResult& candidate,
                         std::size_t ordinal,
                         Result& result)
{
    if (reference.intentionalDifferences == candidate.intentionalDifferences)
        return;
    AddReason(result.compatibilityReasons,
              "pair_" + std::to_string(ordinal) +
                  "_intentional_intervention_mismatch");
    result.compatibility = Compatibility::Incompatible;
}

void CompareRoleIdentities(
    std::string_view role,
    const std::vector<IdentityMap>& identities,
    Result& result)
{
    std::set<std::string, std::less<>> names;
    for (const auto& identity : identities)
        for (const auto& [name, unused] : identity)
        {
            (void)unused;
            names.insert(name);
        }

    for (const std::string& name : names)
    {
        if (name == "fresh_initialization_seed") continue;
        if (name == "configured_model_input_width" ||
            name == "configured_model_input_semantic_layout_version")
            continue;

        std::optional<std::string> expected;
        bool unavailable = false;
        for (const auto& identity : identities)
        {
            const auto found = identity.find(name);
            if (found == identity.end() || !found->second)
            {
                unavailable = true;
                continue;
            }
            if (!expected) expected = *found->second;
            else if (*expected != *found->second)
            {
                AddReason(result.compatibilityReasons,
                          std::string(role) + "_identity_mismatch:" + name);
                result.compatibility = Compatibility::Incompatible;
            }
        }
        if (unavailable && !IsMaterializationField(name))
        {
            AddReason(result.compatibilityReasons,
                      std::string(role) + "_identity_unavailable:" + name);
            if (result.compatibility != Compatibility::Incompatible)
                result.compatibility =
                    Compatibility::UndeterminedDueToMissingEvidence;
        }
    }

    // These immutable experiment columns are authoritative before a final
    // model exists. A legacy NULL may fall back to final-model metadata, but
    // if neither source exists compatibility is genuinely undetermined.
    for (const auto& [configured, materialized] : {
             std::pair<std::string_view, std::string_view>{
                 "configured_model_input_width", "model_input_width"},
             {"configured_model_input_semantic_layout_version",
              "model_input_semantic_layout_version"}})
    {
        std::optional<std::string> expected;
        for (const auto& identity : identities)
        {
            auto value = RequiredConfiguredValue(identity, configured);
            if (!value)
            {
                const auto fallback = identity.find(materialized);
                if (fallback != identity.end()) value = fallback->second;
            }
            if (!value)
            {
                AddReason(result.compatibilityReasons,
                          std::string(role) + "_identity_unavailable:" +
                              std::string(configured));
                if (result.compatibility != Compatibility::Incompatible)
                    result.compatibility =
                        Compatibility::UndeterminedDueToMissingEvidence;
                continue;
            }
            if (!expected) expected = *value;
            else if (*expected != *value)
            {
                AddReason(result.compatibilityReasons,
                          std::string(role) + "_identity_mismatch:" +
                              std::string(configured));
                result.compatibility = Compatibility::Incompatible;
            }
        }
    }
}

std::optional<std::string> Seed(const IdentityMap& identity)
{
    const auto found = identity.find("fresh_initialization_seed");
    if (found == identity.end() || !found->second ||
        *found->second == "NULL")
        return std::nullopt;
    return found->second;
}

std::optional<std::string> IdentityValue(
    const std::vector<Pair::IdentityField>& fields,
    std::string_view name)
{
    const auto found = std::find_if(
        fields.begin(), fields.end(), [name](const Pair::IdentityField& field)
        {
            return field.name == name;
        });
    if (found == fields.end()) return std::nullopt;
    return found->value;
}

Pair::ComparisonResult ConfigurationOnly(Pair::ComparisonResult pair)
{
    std::erase_if(pair.armAScientificIdentity, [](const Pair::IdentityField& field)
    {
        return IsExecutionIdentityField(field.name);
    });
    std::erase_if(pair.armBScientificIdentity, [](const Pair::IdentityField& field)
    {
        return IsExecutionIdentityField(field.name);
    });
    std::erase_if(pair.unexpectedDifferences,
                  [](const Pair::IdentityDifference& difference)
    {
        return IsExecutionIdentityField(difference.field);
    });
    if (pair.status == Pair::Status::IncompatibleScientificIdentity &&
        pair.unexpectedDifferences.empty())
        pair.status = pair.incompleteReasons.empty()
            ? Pair::Status::ComparableComplete
            : Pair::Status::ComparableIncomplete;
    return pair;
}

EvidenceCompatibility ConfiguredPairCompatibility(
    const Pair::ComparisonResult& pair,
    std::vector<std::string>& reasons)
{
    if (!pair.invalidReasons.empty())
    {
        reasons = pair.invalidReasons;
        return EvidenceCompatibility::InvalidEvidence;
    }
    for (const auto& difference : pair.unexpectedDifferences)
        if (!IsExecutionIdentityField(difference.field))
            reasons.push_back("unexpected_difference:" + difference.field);
    return reasons.empty() ? EvidenceCompatibility::Compatible
                           : EvidenceCompatibility::Incompatible;
}

EvidenceCompatibility ExecutionPairCompatibility(
    const Pair::ComparisonResult& pair,
    std::vector<std::string>& reasons)
{
    for (const std::string_view field : {"training_execution_identity",
                                         "inference_execution_identity"})
    {
        const auto armA = IdentityValue(pair.armAScientificIdentity, field);
        const auto armB = IdentityValue(pair.armBScientificIdentity, field);
        if (!armA || !armB)
        {
            reasons.push_back(std::string(field) + "_unavailable");
            continue;
        }
        if (*armA != *armB)
            reasons.push_back(std::string(field) + "_mismatch");
    }
    if (std::any_of(reasons.begin(), reasons.end(), [](const std::string& reason)
        {
            return reason.ends_with("_mismatch");
        }))
        return EvidenceCompatibility::Incompatible;
    return reasons.empty() ? EvidenceCompatibility::Compatible
                           : EvidenceCompatibility::UndeterminedDueToMissingEvidence;
}

EvidenceCompatibility EvidenceCompatibilityFrom(Compatibility compatibility)
{
    switch (compatibility)
    {
        case Compatibility::Compatible: return EvidenceCompatibility::Compatible;
        case Compatibility::Incompatible: return EvidenceCompatibility::Incompatible;
        case Compatibility::UndeterminedDueToMissingEvidence:
            return EvidenceCompatibility::UndeterminedDueToMissingEvidence;
    }
    throw std::invalid_argument("unknown replication compatibility");
}

std::optional<std::string> HomogeneousSymbol(const Result& result)
{
    std::optional<std::string> expected;
    for (const Pair::ComparisonResult& pair : result.pairs)
        for (const auto* identity : {&pair.armAScientificIdentity,
                                     &pair.armBScientificIdentity})
        {
            const auto found = std::find_if(
                identity->begin(), identity->end(),
                [](const Pair::IdentityField& field)
                {
                    return field.name == "symbol";
                });
            if (found == identity->end() || !found->value) return std::nullopt;
            if (!expected) expected = *found->value;
            else if (*expected != *found->value) return std::nullopt;
        }
    return expected;
}

std::optional<std::string> HomogeneousIdentity(
    const Result& result, std::string_view name)
{
    std::optional<std::string> expected;
    for (const Pair::ComparisonResult& pair : result.pairs)
        for (const auto* identity : {&pair.armAScientificIdentity,
                                     &pair.armBScientificIdentity})
        {
            const auto found = std::find_if(
                identity->begin(), identity->end(),
                [name](const Pair::IdentityField& field)
                {
                    return field.name == name;
                });
            if (found == identity->end() || !found->value) return std::nullopt;
            if (!expected) expected = *found->value;
            else if (*expected != *found->value) return std::nullopt;
        }
    return expected;
}

void CompareCrossContextRoleIdentities(
    std::string_view role, const std::vector<IdentityMap>& identities,
    FamilyReport& report)
{
    static const std::set<std::string, std::less<>> allowedContextDimensions{
        "symbol", "prediction_horizon", "persisted_training_symbol",
        "fresh_initialization_seed"};
    std::set<std::string, std::less<>> names;
    for (const auto& identity : identities)
        for (const auto& [name, unused] : identity)
        {
            (void)unused;
            if (!allowedContextDimensions.contains(name)) names.insert(name);
        }
    for (const auto& name : names)
    {
        std::optional<std::string> expected;
        for (const auto& identity : identities)
        {
            const auto found = identity.find(name);
            if (found == identity.end() || !found->second)
            {
                AddReason(report.crossContextReasons,
                          "cross_context_" + std::string(role) +
                              "_identity_unavailable:" + name);
                if (report.crossContextCompatibility != Compatibility::Incompatible)
                    report.crossContextCompatibility =
                        Compatibility::UndeterminedDueToMissingEvidence;
                continue;
            }
            if (!expected) expected = *found->second;
            else if (*expected != *found->second)
            {
                AddReason(report.crossContextReasons,
                          "cross_context_" + std::string(role) +
                              "_identity_mismatch:" + name);
                report.crossContextCompatibility = Compatibility::Incompatible;
            }
        }
    }
}

void AddCrossContextMetrics(FamilyReport& report)
{
    for (const auto& definition : Pair::MetricDefinitions())
    {
        CrossContextMetricAggregate aggregate;
        aggregate.name = definition.name;
        long double sum = 0.0L;
        for (const auto& family : report.families)
        {
            const auto found = std::find_if(
                family.replication.strictCompletedSubset.metrics.begin(),
                family.replication.strictCompletedSubset.metrics.end(),
                [&](const MetricAggregate& metric) {
                    return metric.name == definition.name;
                });
            if (found == family.replication.strictCompletedSubset.metrics.end() ||
                !found->descriptiveMean || !std::isfinite(*found->descriptiveMean))
                throw std::invalid_argument(
                    std::string{"cross_context_family_metric_unavailable:"} +
                        std::string{definition.name});
            const double value = *found->descriptiveMean;
            aggregate.familyDescriptiveMeans.push_back(value);
            ++aggregate.compatibleContextFamilyCount;
            sum += static_cast<long double>(value);
            if (value > 0.0) ++aggregate.positiveFamilyMeanCount;
            else if (value < 0.0) ++aggregate.negativeFamilyMeanCount;
            else ++aggregate.zeroFamilyMeanCount;
        }
        if (aggregate.compatibleContextFamilyCount != report.families.size() ||
            !std::isfinite(sum))
            throw std::invalid_argument("cross_context_family_metric_count_invalid");
        aggregate.unweightedDescriptiveMeanOfFamilyMeans = static_cast<double>(
            sum / static_cast<long double>(aggregate.compatibleContextFamilyCount));
        report.crossContextMetrics.push_back(std::move(aggregate));
    }
}

bool IsUnambiguousControlAblation(
    const Pair::ComparisonResult& pair)
{
    return pair.intentionalDifferences.size() == 1 &&
        pair.intentionalDifferences.front().field == "feature_ablation_mask" &&
        pair.intentionalDifferences.front().armA.empty() &&
        !pair.intentionalDifferences.front().armB.empty();
}

MetricAggregate AggregateMetric(
    const std::vector<Pair::ComparisonResult>& pairs,
    const Pair::MetricDefinition& definition)
{
    MetricAggregate result;
    result.name = definition.name;
    result.pairCount = pairs.size();
    result.pairDeltas.reserve(pairs.size());
    double sum = 0.0;
    for (const auto& pair : pairs)
    {
        const auto& delta = pair.*definition.member;
        result.pairDeltas.push_back(delta.armBMinusArmA);
        if (!delta.armBMinusArmA) continue;
        const double value = *delta.armBMinusArmA;
        if (!std::isfinite(value))
            throw std::invalid_argument("replication_metric_nonfinite");
        sum += value;
        if (!std::isfinite(sum))
            throw std::invalid_argument("replication_metric_sum_nonfinite");
        ++result.availableCount;
        if (!result.minimum || value < *result.minimum) result.minimum = value;
        if (!result.maximum || value > *result.maximum) result.maximum = value;
        if (value > 0.0) ++result.positiveCount;
        else if (value < 0.0) ++result.negativeCount;
        else ++result.zeroCount;
    }
    if (result.availableCount != 0)
        result.descriptiveMean =
            sum / static_cast<double>(result.availableCount);
    return result;
}

std::string Deltas(const std::vector<std::optional<double>>& values)
{
    std::ostringstream output;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index != 0) output << '|';
        output << OptionalNumber(values[index]);
    }
    return output.str();
}

std::string OptionalText(const std::optional<std::string>& value)
{
    return value ? MachineText(*value) : "NULL";
}

void RenderAggregateMetrics(std::ostringstream& output,
                            std::string_view record,
                            const std::vector<MetricAggregate>& metrics)
{
    for (const auto& metric : metrics)
        output << record
               << ",metric=" << metric.name
               << ",n_pairs=" << metric.pairCount
               << ",n_available=" << metric.availableCount
               << ",pair_deltas=" << Deltas(metric.pairDeltas)
               << ",descriptive_mean="
               << OptionalNumber(metric.descriptiveMean)
               << ",minimum=" << OptionalNumber(metric.minimum)
               << ",maximum=" << OptionalNumber(metric.maximum)
               << ",count_positive=" << metric.positiveCount
               << ",count_zero=" << metric.zeroCount
               << ",count_negative=" << metric.negativeCount
               << '\n';
}

std::string SeedSet(const Result& result)
{
    std::string rendered;
    for (const auto& dimension : result.replicationDimensions)
    {
        if (!dimension.armASeed) return "UNAVAILABLE";
        if (!rendered.empty()) rendered.push_back(':');
        rendered += *dimension.armASeed;
    }
    return rendered.empty() ? "UNAVAILABLE" : rendered;
}

void RenderCrossContextMetrics(std::ostringstream& output,
                               const std::vector<CrossContextMetricAggregate>& metrics)
{
    for (const auto& metric : metrics)
        output << "EXPERIMENT_REPLICATION_CROSS_CONTEXT_METRIC"
               << ",metric=" << metric.name
               << ",compatible_context_family_count="
               << metric.compatibleContextFamilyCount
               << ",family_descriptive_means="
               << Deltas(metric.familyDescriptiveMeans)
               << ",count_positive_family_mean="
               << metric.positiveFamilyMeanCount
               << ",count_zero_family_mean=" << metric.zeroFamilyMeanCount
               << ",count_negative_family_mean="
               << metric.negativeFamilyMeanCount
               << ",unweighted_descriptive_mean_of_family_means="
               << OptionalNumber(metric.unweightedDescriptiveMeanOfFamilyMeans)
               << ",raw_pair_pooling=false"
               << ",statistical_independence=not_inferred\n";
}

std::optional<double> ConfiguredObservationDelta(const Pair::MetricDelta& metric)
{
    if (!metric.armA || !metric.armB || !std::isfinite(*metric.armA) ||
        !std::isfinite(*metric.armB))
        return std::nullopt;
    return *metric.armB - *metric.armA;
}

Result CompareCore(std::vector<Pair::ComparisonResult> pairs)
{
    if (pairs.size() < 2)
        throw std::invalid_argument(
            "experiment replication comparison requires at least two pairs");

    Result result;
    result.pairs = std::move(pairs);
    result.compatibility = Compatibility::Compatible;

    std::set<long long> experimentIds;
    std::vector<IdentityMap> armAIdentities;
    std::vector<IdentityMap> armBIdentities;
    armAIdentities.reserve(result.pairs.size());
    armBIdentities.reserve(result.pairs.size());

    for (std::size_t index = 0; index < result.pairs.size(); ++index)
    {
        const auto& pair = result.pairs[index];
        const std::size_t ordinal = index + 1;
        if (pair.experimentAId <= 0 || pair.experimentBId <= 0 ||
            pair.experimentAId == pair.experimentBId ||
            !experimentIds.insert(pair.experimentAId).second ||
            !experimentIds.insert(pair.experimentBId).second)
            throw std::invalid_argument(
                "replication experiment IDs must be globally unique");
        if (!PairScientificallyComparable(pair))
        {
            AddReason(result.compatibilityReasons,
                      "pair_" + std::to_string(ordinal) +
                          "_not_scientifically_comparable");
            result.compatibility = Compatibility::Incompatible;
        }
        if (index != 0)
            CompareIntervention(result.pairs.front(), pair, ordinal, result);
        armAIdentities.push_back(MakeIdentityMap(pair.armAScientificIdentity));
        armBIdentities.push_back(MakeIdentityMap(pair.armBScientificIdentity));
    }

    CompareRoleIdentities("arm_a", armAIdentities, result);
    CompareRoleIdentities("arm_b", armBIdentities, result);

    std::set<std::string> seeds;
    bool everySeedAvailable = true;
    for (std::size_t index = 0; index < result.pairs.size(); ++index)
    {
        ReplicationDimension dimension;
        dimension.pairOrdinal = index + 1;
        dimension.armASeed = Seed(armAIdentities[index]);
        dimension.armBSeed = Seed(armBIdentities[index]);
        if (!dimension.armASeed || !dimension.armBSeed)
            everySeedAvailable = false;
        else
        {
            seeds.insert(*dimension.armASeed);
            if (*dimension.armASeed != *dimension.armBSeed)
            {
                AddReason(result.compatibilityReasons,
                          "pair_" + std::to_string(index + 1) +
                              "_seed_mismatch");
                result.compatibility = Compatibility::Incompatible;
            }
        }
        result.replicationDimensions.push_back(std::move(dimension));
    }
    if (everySeedAvailable)
        result.seedReplicationMode = seeds.size() > 1
            ? SeedReplicationMode::DifferentSeeds
            : SeedReplicationMode::SameSeedRepeatedPairs;

    if (result.compatibility == Compatibility::Compatible)
        for (const auto& definition : Pair::MetricDefinitions())
            result.metrics.push_back(AggregateMetric(result.pairs, definition));
    return result;
}

} // namespace

Result Compare(std::vector<Pair::ComparisonResult> pairs)
{
    Result result = CompareCore(std::move(pairs));

    std::vector<Pair::ComparisonResult> configuredPairs;
    configuredPairs.reserve(result.pairs.size());
    result.pairCompatibility.reserve(result.pairs.size());
    std::vector<Pair::ComparisonResult> strictPairs;
    for (std::size_t index = 0; index < result.pairs.size(); ++index)
    {
        const Pair::ComparisonResult& pair = result.pairs[index];
        PairCompatibilityAssessment assessment;
        assessment.configuredScientificIdentity = ConfiguredPairCompatibility(
            pair, assessment.configuredScientificIdentityReasons);
        assessment.executionProvenance = ExecutionPairCompatibility(
            pair, assessment.executionProvenanceReasons);
        assessment.strictCompletedCompatible =
            assessment.configuredScientificIdentity ==
                EvidenceCompatibility::Compatible &&
            assessment.executionProvenance == EvidenceCompatibility::Compatible &&
            pair.status == Pair::Status::ComparableComplete;
        configuredPairs.push_back(ConfigurationOnly(pair));
        if (assessment.configuredScientificIdentity ==
            EvidenceCompatibility::Compatible)
            ++result.strictCompletedSubset.totalConfiguredPairCount;
        if (assessment.strictCompletedCompatible)
        {
            ++result.strictCompletedSubset.strictCompletedCompatiblePairCount;
            strictPairs.push_back(pair);
        }
        else
        {
            StrictSubsetExclusion exclusion;
            exclusion.pairOrdinal = index + 1;
            exclusion.experimentAId = pair.experimentAId;
            exclusion.experimentBId = pair.experimentBId;
            exclusion.armASeed = Seed(MakeIdentityMap(pair.armAScientificIdentity));
            exclusion.armBSeed = Seed(MakeIdentityMap(pair.armBScientificIdentity));
            if (assessment.configuredScientificIdentity !=
                EvidenceCompatibility::Compatible)
                exclusion.reasons.push_back(
                    "configured_scientific_identity_" +
                    EvidenceCompatibilityText(
                        assessment.configuredScientificIdentity));
            if (assessment.executionProvenance !=
                EvidenceCompatibility::Compatible)
                exclusion.reasons.push_back(
                    "execution_provenance_" + EvidenceCompatibilityText(
                        assessment.executionProvenance));
            if (pair.status != Pair::Status::ComparableComplete)
                exclusion.reasons.push_back("completed_result_" +
                                            Pair::StatusText(pair.status));
            result.strictCompletedSubset.exclusions.push_back(std::move(exclusion));
        }
        result.pairCompatibility.push_back(std::move(assessment));
    }

    const Result configured = CompareCore(std::move(configuredPairs));
    result.configuredScientificIdentity =
        EvidenceCompatibilityFrom(configured.compatibility);
    result.configuredScientificIdentityReasons = configured.compatibilityReasons;

    if (strictPairs.size() < 2)
    {
        result.strictCompletedSubset.aggregateCompatibility =
            Compatibility::UndeterminedDueToMissingEvidence;
        result.strictCompletedSubset.aggregateReasons.push_back(
            "fewer_than_two_strict_completed_compatible_pairs");
    }
    else
    {
        const Result strict = CompareCore(std::move(strictPairs));
        result.strictCompletedSubset.aggregateCompatibility = strict.compatibility;
        result.strictCompletedSubset.aggregateReasons = strict.compatibilityReasons;
        if (strict.compatibility == Compatibility::Compatible)
            result.strictCompletedSubset.metrics = strict.metrics;
    }
    return result;
}

FamilyReport CompareFamilies(
    std::vector<std::vector<Pair::ComparisonResult>> families)
{
    if (families.size() < 2)
        throw std::invalid_argument(
            "experiment replication family report requires at least two families");

    FamilyReport report;
    std::set<long long> experimentIds;
    std::set<std::string> symbols;
    report.families.reserve(families.size());
    for (auto& family : families)
    {
        Result replication = Compare(std::move(family));
        for (const Pair::ComparisonResult& pair : replication.pairs)
            if (!experimentIds.insert(pair.experimentAId).second ||
                !experimentIds.insert(pair.experimentBId).second)
                throw std::invalid_argument(
                    "replication family report experiment IDs must be globally unique");

        FamilyResult result;
        result.homogeneousSymbol = HomogeneousSymbol(replication);
        result.homogeneousPredictionHorizon = HomogeneousIdentity(
            replication, "prediction_horizon");
        if (result.homogeneousSymbol) symbols.insert(*result.homogeneousSymbol);
        result.replication = std::move(replication);
        report.families.push_back(std::move(result));
    }
    report.distinctHomogeneousSymbolCount = symbols.size();
    report.crossContextCompatibility = Compatibility::Compatible;
    std::vector<IdentityMap> armAContexts;
    std::vector<IdentityMap> armBContexts;
    for (std::size_t index = 0; index < report.families.size(); ++index)
    {
        const Result& family = report.families[index].replication;
        const std::string prefix = "context_" + std::to_string(index + 1) + "_";
        if (!report.families[index].homogeneousSymbol ||
            !report.families[index].homogeneousPredictionHorizon)
        {
            AddReason(report.crossContextReasons, prefix + "context_identity_unavailable");
            if (report.crossContextCompatibility != Compatibility::Incompatible)
                report.crossContextCompatibility =
                    Compatibility::UndeterminedDueToMissingEvidence;
        }
        if (family.compatibility != Compatibility::Compatible ||
            family.configuredScientificIdentity != EvidenceCompatibility::Compatible ||
            family.strictCompletedSubset.aggregateCompatibility != Compatibility::Compatible ||
            family.strictCompletedSubset.strictCompletedCompatiblePairCount !=
                family.pairs.size() ||
            family.seedReplicationMode != SeedReplicationMode::DifferentSeeds)
        {
            AddReason(report.crossContextReasons, prefix +
                "within_context_replication_contract_not_satisfied");
            report.crossContextCompatibility = Compatibility::Incompatible;
        }
        if (!IsUnambiguousControlAblation(family.pairs.front()))
        {
            AddReason(report.crossContextReasons, prefix +
                "control_ablation_arm_semantics_ambiguous");
            report.crossContextCompatibility = Compatibility::Incompatible;
        }
        if (index != 0 &&
            family.pairs.front().intentionalDifferences !=
                report.families.front().replication.pairs.front().intentionalDifferences)
        {
            AddReason(report.crossContextReasons,
                      prefix + "intentional_intervention_mismatch");
            report.crossContextCompatibility = Compatibility::Incompatible;
        }
        armAContexts.push_back(MakeIdentityMap(
            family.pairs.front().armAScientificIdentity));
        armBContexts.push_back(MakeIdentityMap(
            family.pairs.front().armBScientificIdentity));
    }
    CompareCrossContextRoleIdentities("arm_a", armAContexts, report);
    CompareCrossContextRoleIdentities("arm_b", armBContexts, report);
    if (report.crossContextCompatibility == Compatibility::Compatible)
        AddCrossContextMetrics(report);
    return report;
}

std::string Render(const Result& result)
{
    std::ostringstream output;
    output << "EXPERIMENT_REPLICATION_COMPARISON"
           << ",version=2"
           << ",pair_count=" << result.pairs.size()
           << ",pair_order=argument_order"
           << ",arm_order=argument_order"
           << ",delta_sign_convention=arm_b_minus_arm_a"
           << ",aggregation=unweighted_descriptive_available_pair_deltas"
           << '\n';
    for (std::size_t index = 0; index < result.pairs.size(); ++index)
    {
        output << "EXPERIMENT_REPLICATION_PAIR"
               << ",ordinal=" << index + 1
               << ",experiment_a_id=" << result.pairs[index].experimentAId
               << ",experiment_b_id=" << result.pairs[index].experimentBId
               << ",status=" << Pair::StatusText(result.pairs[index].status)
               << '\n';
        output << Pair::RenderSummary(result.pairs[index]);
        const PairCompatibilityAssessment& assessment =
            result.pairCompatibility[index];
        output << "EXPERIMENT_REPLICATION_PAIR_COMPATIBILITY"
               << ",ordinal=" << index + 1
               << ",configured_scientific_identity="
               << EvidenceCompatibilityText(
                      assessment.configuredScientificIdentity)
               << ",configured_reasons="
               << Reasons(assessment.configuredScientificIdentityReasons)
               << ",execution_provenance="
               << EvidenceCompatibilityText(assessment.executionProvenance)
               << ",execution_reasons="
               << Reasons(assessment.executionProvenanceReasons)
               << ",strict_completed_compatible="
               << (assessment.strictCompletedCompatible ? "true" : "false")
               << '\n';
        if (assessment.configuredScientificIdentity ==
            EvidenceCompatibility::Compatible)
            for (const auto& definition : Pair::MetricDefinitions())
            {
                const Pair::MetricDelta& metric =
                    result.pairs[index].*definition.member;
                output << "EXPERIMENT_REPLICATION_CONFIGURED_PAIR_METRIC"
                       << ",pair_ordinal=" << index + 1
                       << ",metric=" << definition.name
                       << ",arm_a=" << OptionalNumber(metric.armA)
                       << ",arm_b=" << OptionalNumber(metric.armB)
                       << ",arm_b_minus_arm_a="
                       << OptionalNumber(ConfiguredObservationDelta(metric))
                       << ",execution_provenance="
                       << EvidenceCompatibilityText(
                              assessment.executionProvenance)
                       << '\n';
            }
    }
    for (const auto& dimension : result.replicationDimensions)
        output << "EXPERIMENT_REPLICATION_DIMENSION"
               << ",pair_ordinal=" << dimension.pairOrdinal
               << ",field=fresh_initialization_seed"
               << ",arm_a=" << OptionalText(dimension.armASeed)
               << ",arm_b=" << OptionalText(dimension.armBSeed)
               << '\n';
    output << "EXPERIMENT_REPLICATION_COMPATIBILITY"
           << ",state=" << CompatibilityText(result.compatibility)
           << ",reasons=" << Reasons(result.compatibilityReasons)
           << ",seed_replication_mode="
           << SeedReplicationModeText(result.seedReplicationMode)
           << ",statistical_independence=not_inferred"
           << '\n';
    output << "EXPERIMENT_REPLICATION_CONFIGURED_FAMILY_COMPATIBILITY"
           << ",state="
           << EvidenceCompatibilityText(result.configuredScientificIdentity)
           << ",reasons=" << Reasons(result.configuredScientificIdentityReasons)
           << ",aggregate=not_performed"
           << ",configuration_does_not_establish_execution_equivalence=true"
           << '\n';
    if (result.compatibility == Compatibility::Compatible)
    {
        RenderAggregateMetrics(output, "EXPERIMENT_REPLICATION_METRIC",
                               result.metrics);
    }
    else
        output << "EXPERIMENT_REPLICATION_AGGREGATE"
               << ",suppressed=true"
               << ",reason=replication_compatibility_not_established\n";
    output << "EXPERIMENT_REPLICATION_STRICT_COMPLETED_SUBSET"
           << ",total_configured_pairs="
           << result.strictCompletedSubset.totalConfiguredPairCount
           << ",strict_completed_compatible_pairs="
           << result.strictCompletedSubset.strictCompletedCompatiblePairCount
           << ",excluded_or_caveated_pairs="
           << result.strictCompletedSubset.exclusions.size()
           << ",aggregate_state="
           << CompatibilityText(result.strictCompletedSubset.aggregateCompatibility)
           << ",aggregate_reasons="
           << Reasons(result.strictCompletedSubset.aggregateReasons)
           << ",aggregation=unweighted_descriptive_paired_deltas"
           << ",statistical_independence=not_inferred\n";
    for (const auto& exclusion : result.strictCompletedSubset.exclusions)
        output << "EXPERIMENT_REPLICATION_STRICT_COMPLETED_EXCLUSION"
               << ",pair_ordinal=" << exclusion.pairOrdinal
               << ",experiment_a_id=" << exclusion.experimentAId
               << ",experiment_b_id=" << exclusion.experimentBId
               << ",arm_a_seed=" << OptionalText(exclusion.armASeed)
               << ",arm_b_seed=" << OptionalText(exclusion.armBSeed)
               << ",reasons=" << Reasons(exclusion.reasons) << '\n';
    if (result.strictCompletedSubset.aggregateCompatibility ==
        Compatibility::Compatible)
        RenderAggregateMetrics(output,
                               "EXPERIMENT_REPLICATION_STRICT_COMPLETED_METRIC",
                               result.strictCompletedSubset.metrics);
    output << "EXPERIMENT_REPLICATION_RESULT"
           << ",compatibility=" << CompatibilityText(result.compatibility)
           << ",subjective_winner=NONE"
           << ",read_only=true\n";
    return output.str();
}

std::string RenderFamilyReport(const FamilyReport& report)
{
    std::ostringstream output;
    output << "EXPERIMENT_REPLICATION_FAMILY_REPORT"
           << ",version=2"
           << ",family_count=" << report.families.size()
           << ",cross_context_aggregation="
           << (report.crossContextCompatibility == Compatibility::Compatible
                   ? "unweighted_descriptive_family_means" : "suppressed")
           << ",raw_pair_pooling=false"
           << ",statistical_independence=not_inferred"
           << ",read_only=true\n";
    for (std::size_t index = 0; index < report.families.size(); ++index)
    {
        const FamilyResult& family = report.families[index];
        output << "EXPERIMENT_REPLICATION_FAMILY_BEGIN"
               << ",ordinal=" << index + 1
               << ",symbol="
               << (family.homogeneousSymbol
                       ? MachineText(*family.homogeneousSymbol)
                       : "UNAVAILABLE_OR_MIXED")
               << ",symbol_homogeneous="
               << (family.homogeneousSymbol ? "true" : "false")
               << ",prediction_horizon="
               << (family.homogeneousPredictionHorizon
                       ? *family.homogeneousPredictionHorizon
                       : "UNAVAILABLE_OR_MIXED")
               << ",prediction_horizon_homogeneous="
               << (family.homogeneousPredictionHorizon ? "true" : "false")
               << ",pair_count=" << family.replication.pairs.size()
               << ",seed_set=" << SeedSet(family.replication)
               << ",replication_dimension=fresh_initialization_seed"
               << ",seed_replication_mode="
               << SeedReplicationModeText(
                      family.replication.seedReplicationMode)
               << ",compatibility="
               << CompatibilityText(family.replication.compatibility)
               << '\n';
        output << Render(family.replication);
        output << "EXPERIMENT_REPLICATION_FAMILY_END"
               << ",ordinal=" << index + 1 << '\n';
    }
    output << "EXPERIMENT_REPLICATION_FAMILY_CROSS_GROUP"
           << ",family_count=" << report.families.size()
           << ",distinct_homogeneous_symbol_count="
           << report.distinctHomogeneousSymbolCount
           << ",cross_context_compatibility="
           << CompatibilityText(report.crossContextCompatibility)
           << ",cross_context_reasons=" << Reasons(report.crossContextReasons)
           << ",cross_context_aggregation="
           << (report.crossContextCompatibility == Compatibility::Compatible
                   ? "unweighted_descriptive_family_means" : "suppressed")
           << ",heterogeneity=preserved_by_context_family_boundaries"
           << ",subjective_winner=NONE"
           << ",read_only=true\n";
    if (report.crossContextCompatibility == Compatibility::Compatible)
        RenderCrossContextMetrics(output, report.crossContextMetrics);
    return output.str();
}

std::string CompatibilityText(Compatibility value)
{
    switch (value)
    {
        case Compatibility::Compatible: return "compatible";
        case Compatibility::Incompatible: return "incompatible";
        case Compatibility::UndeterminedDueToMissingEvidence:
            return "undetermined_due_to_missing_evidence";
    }
    throw std::invalid_argument("unknown replication compatibility");
}

std::string EvidenceCompatibilityText(EvidenceCompatibility value)
{
    switch (value)
    {
        case EvidenceCompatibility::Compatible: return "compatible";
        case EvidenceCompatibility::Incompatible: return "incompatible";
        case EvidenceCompatibility::UndeterminedDueToMissingEvidence:
            return "undetermined_due_to_missing_evidence";
        case EvidenceCompatibility::InvalidEvidence: return "invalid_evidence";
    }
    throw std::invalid_argument("unknown evidence compatibility");
}

std::string SeedReplicationModeText(SeedReplicationMode value)
{
    switch (value)
    {
        case SeedReplicationMode::DifferentSeeds:
            return "different_seed_replications";
        case SeedReplicationMode::SameSeedRepeatedPairs:
            return "same_seed_repeated_pairs";
        case SeedReplicationMode::UnavailableOrNotApplicable:
            return "unavailable_or_not_applicable";
    }
    throw std::invalid_argument("unknown seed replication mode");
}

} // namespace EA::ExperimentReplicationComparison
